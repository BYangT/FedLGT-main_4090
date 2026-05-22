# # ulearn/recovery_train.py
#
# import torch
# from torch.nn import functional as F
# from torch.utils.data import Subset, DataLoader
# from tqdm import tqdm
# from optim_schedule import WarmupLinearSchedule  # 如果你用到了 warmup
# import utils.evaluate as evaluate               # 如果后面想算指标可以用
# import logging
# import copy
#
#
# def train_net_recovery_no_target(
#     net_id,
#     model,
#     train_dataloader,
#     valid_dataloader,
#     epochs,
#     args,
#     device="cpu",
#     g_model=None,
#     emb_feat=None,
#     clip_model=None,
#     forget_cls=None,
#     teacher_model=None,          # 👈 新增：遗忘后（固定）的 teacher 模型
#     lambda_logit_cons=0.0,       # 👈 新增：目标类 logit 一致性约束权重
# ):
#     """
#     恢复训练：每个客户端本地训练 epochs 轮，但
#       1) 跳过目标类正样本；
#       2) loss 中不对目标类这一列做 BCE；
#       3) 额外加一个约束：当前模型在目标类上的 logit
#          尽量贴近 teacher 的 logit（只在目标类为 0 的样本上）。
#     """
#     logger = logging.getLogger()
#     logger.info(f'[Recovery] Training network {net_id}')
#
#     # ====== 优化器 & scheduler，完全照你原来 train_net 的配置 ====== #
#     if args.optim == 'adam':
#         optimizer = torch.optim.Adam(
#             filter(lambda p: p.requires_grad, model.parameters()),
#             lr=args.lr
#         )
#     elif args.optim == 'adamw':
#         optimizer = torch.optim.AdamW(
#             filter(lambda p: p.requires_grad, model.parameters()),
#             lr=args.lr
#         )
#     else:
#         optimizer = torch.optim.SGD(
#             filter(lambda p: p.requires_grad, model.parameters()),
#             lr=args.lr, momentum=0.9, weight_decay=1e-4
#         )
#
#     if args.warmup_scheduler:
#         step_scheduler = None
#         scheduler_warmup = WarmupLinearSchedule(optimizer, 1, 300000)
#     else:
#         scheduler_warmup = None
#         if args.scheduler_type == 'plateau':
#             step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer, mode='min', factor=0.1, patience=5
#             )
#         elif args.scheduler_type == 'step':
#             step_scheduler = torch.optim.lr_scheduler.StepLR(
#                 optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
#             )
#         else:
#             step_scheduler = None
#
#     model.to(device)
#     if teacher_model is not None:
#         teacher_model.to(device)
#         teacher_model.eval()
#
#     num_labels = args.num_labels
#     # 非目标类的列索引（后面只对这些类计算 BCE）
#     if forget_cls is not None:
#         keep_cls_idx = [c for c in range(num_labels) if c != forget_cls]
#     else:
#         keep_cls_idx = list(range(num_labels))
#
#     for epoch in range(epochs):
#         model.train()
#         desc = f"[Recovery] Client {net_id} Epoch {epoch}"
#         pbar = tqdm(train_dataloader, desc=desc, ncols=100)
#
#         running_loss = 0.0
#         n_steps = 0
#
#         for batch in pbar:
#             images = batch['image'].float().to(device)   # (B, C, H, W)
#             labels = batch['labels'].float().to(device)  # (B, L)
#             mask   = batch['mask'].float().to(device)    # (B, L)
#
#             # ====== 1) 恢复阶段不使用目标类正样本 ====== #
#             if forget_cls is not None:
#                 pos_idx = (labels[:, forget_cls] == 1)
#                 keep_idx = ~pos_idx
#                 if keep_idx.sum() == 0:
#                     # 这一批全是目标类正样本，整批跳过
#                     continue
#
#                 images = images[keep_idx]
#                 labels = labels[keep_idx]
#                 mask   = mask[keep_idx]
#
#             if images.size(0) == 0:
#                 continue
#
#             optimizer.zero_grad()
#
#             # ====== 2) 当前模型前向 ====== #
#             logits, _, _ = model(
#                 images,
#                 mask,
#                 args.learn_emb_type,
#                 emb_feat,
#                 clip_model,
#             )   # logits: (B_kept, L)
#
#             # ====== 3) 分类 BCE：只对非目标类列计算 ====== #
#             logits_keep = logits[:, keep_cls_idx]       # (B, L-1)
#             labels_keep = labels[:, keep_cls_idx]       # (B, L-1)
#
#             loss_matrix = F.binary_cross_entropy_with_logits(
#                 logits_keep, labels_keep, reduction='none'
#             )   # (B, L-1)
#             loss_cls = loss_matrix.mean()
#
#             # ====== 4) 目标类 logit 一致性约束（当前 vs teacher） ====== #
#             loss_logit_cons = torch.tensor(0.0, device=device)
#             if (teacher_model is not None) and (lambda_logit_cons > 0) and (forget_cls is not None):
#                 with torch.no_grad():
#                     t_logits, _, _ = teacher_model(
#                         images,
#                         mask,
#                         args.learn_emb_type,
#                         emb_feat,
#                         clip_model,
#                     )  # (B, L)
#
#                 z_cur = logits[:, forget_cls]       # (B,)
#                 z_tch = t_logits[:, forget_cls]     # (B,)
#
#                 # 只在目标类标签为 0 的样本上约束，防止“救回”正样本
#                 neg_mask = (labels[:, forget_cls] < 0.5)
#                 if neg_mask.any():
#                     z_cur = z_cur[neg_mask]
#                     z_tch = z_tch[neg_mask]
#                     loss_logit_cons = F.mse_loss(z_cur, z_tch)
#                 else:
#                     loss_logit_cons = torch.tensor(0.0, device=device)
#
#             # ====== 5) 总 loss ====== #
#             loss = loss_cls + lambda_logit_cons * loss_logit_cons
#
#             # 反向 + 更新
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
#             optimizer.step()
#
#             running_loss += loss.item()
#             n_steps += 1
#             avg_loss = running_loss / max(n_steps, 1)
#
#             pbar.set_postfix(
#                 loss=f"{avg_loss:.4f}",
#                 cls=f"{loss_cls.item():.4f}",
#                 logit=f"{loss_logit_cons.item():.4f}",
#             )
#
#         # ====== scheduler 更新 ====== #
#         if step_scheduler is not None:
#             if args.scheduler_type == 'step':
#                 step_scheduler.step(epoch)
#             elif args.scheduler_type == 'plateau':
#                 step_scheduler.step(avg_loss)
#
#         logger.info(
#             f"[Recovery] net {net_id} epoch {epoch}: "
#             f"loss={avg_loss:.4f}, "
#             f"loss_cls={loss_cls.item():.4f}, "
#             f"loss_logit_cons={loss_logit_cons.item():.4f}"
#         )
#
#     logger.info(f"[Recovery] Training complete for net {net_id}")
#
#
# def federated_recovery_no_target_samples(
#     args,
#     global_model,
#     nets,
#     train_dl_global,
#     partition_idx_map,
#     device,
#     emb_feat,
#     clip_model,
#     forget_cls,
#     recovery_rounds=3,
#     teacher_model=None,          # 👈 新增：遗忘后固定 teacher
#     lambda_logit_cons=0.0,       # 👈 新增：目标类 logit 一致性约束权重
# ):
#     """
#     联邦恢复阶段：
#     - 每一轮：
#       1) 广播当前全局模型到所有客户端
#       2) 每个客户端用自己的本地数据恢复训练（跳过目标类正样本，且不对目标类做 BCE）
#       3) 服务器做一次 FedAvg 聚合
#     teacher_model: 遗忘后的全局模型（固定不更新），用于 logit 一致性约束。
#     """
#     from scipy.special import softmax
#
#     n_parties = args.n_parties
#
#     if teacher_model is not None:
#         teacher_model.to(device)
#         teacher_model.eval()
#         for p in teacher_model.parameters():
#             p.requires_grad_(False)
#
#     for r in range(recovery_rounds):
#         print(f"\n[Recovery-Round {r}] Step 1: 广播当前全局模型到各客户端 ...")
#         global_state = global_model.state_dict()
#
#         # 广播权重
#         for cid in range(n_parties):
#             nets[cid].load_state_dict(global_state)
#             nets[cid].to(device)
#
#         # ===== Step 2: 各客户端本地恢复训练 =====
#         net_dataidx_map = {}
#         for cid in range(n_parties):
#             sub_dst = Subset(train_dl_global.dataset, partition_idx_map[cid])
#             train_dl_local = DataLoader(
#                 sub_dst,
#                 batch_size=args.batch_size,
#                 shuffle=True,
#                 num_workers=args.workers,
#                 drop_last=False
#             )
#             net_dataidx_map[cid] = len(sub_dst)
#
#             train_net_recovery_no_target(
#                 net_id=cid,
#                 model=nets[cid],
#                 train_dataloader=train_dl_local,
#                 valid_dataloader=None,
#                 epochs=args.epochs,
#                 args=args,
#                 device=device,
#                 g_model=global_model,
#                 emb_feat=emb_feat,
#                 clip_model=clip_model,
#                 forget_cls=forget_cls,
#                 teacher_model=teacher_model,
#                 lambda_logit_cons=lambda_logit_cons,
#             )
#
#             nets[cid].to('cpu')
#
#         # ===== Step 3: 服务器端 FedAvg 聚合 =====
#         print(f"[Recovery-Round {r}] Step 3: 服务器聚合各客户端恢复后的模型 ...")
#
#         total_points = sum(net_dataidx_map.values())
#         fed_avg_freqs = [net_dataidx_map[cid] / total_points for cid in range(n_parties)]
#
#         new_global_state = {}
#         for cid in range(n_parties):
#             state_c = nets[cid].state_dict()
#             w = fed_avg_freqs[cid]
#             for k, v in state_c.items():
#                 v = v.float()
#                 if k not in new_global_state:
#                     new_global_state[k] = v * w
#                 else:
#                     new_global_state[k] += v * w
#
#         global_model.load_state_dict(new_global_state)
#         global_model.to(device)
#
#         print(f"[Recovery-Round {r}] FedAvg 聚合完成")
#
#     return global_model

# fed_recovery_no_target.py

# ulearn/recovery_train.py 里添加 / 替换

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def train_net_recovery_no_target(
    net_id,
    model,
    train_dataloader,
    valid_dataloader,
    epochs,
    args,
    device,
    g_model,
    emb_feat,
    clip_model,
    forget_cls,
    teacher_model=None,
    lambda_logit_cons: float = 0.0,
):
    """
    本地恢复训练：
    - 只用非目标类样本 (y_forget == 0)
    - Loss 只对非目标类做 BCE（目标类完全不参与）
    - 只更新 backbone + 非目标类的 head（目标类 head 冻结）
    - 可选：用 teacher_model 在目标类上加一个一致性约束，避免被重新学回来
    """

    model.to(device)
    model.train()

    # ===== 0) 恢复阶段自己的超参（不要沿用原训练） =====
    rec_epochs = getattr(args, "recovery_epochs", epochs)      # 建议 3~5
    rec_lr     = getattr(args, "recovery_lr", 5e-4)            # 可以比遗忘阶段稍大
    alpha_pos  = getattr(args, "recovery_alpha_pos", 2.0)      # 正样本权重，>1 提升 Recall

    # ===== 1) 只训练 backbone + 非目标类 head =====
    for p in model.parameters():
        p.requires_grad = True          # 先都打开

    # 冻结 output_linear 的目标类那一行
    W = model.output_linear.weight      # (num_labels, D_emb)
    b = model.output_linear.bias        # (num_labels,)

    W.requires_grad = True
    b.requires_grad = True

    # hook：把目标类那一行的梯度强制为 0，只更新其它类
    mask_W = torch.ones_like(W)
    mask_W[forget_cls] = 0.0

    def hook_W(grad):
        return grad * mask_W

    handle_W = W.register_hook(hook_W)

    mask_b = torch.ones_like(b)
    mask_b[forget_cls] = 0.0

    def hook_b(grad):
        return grad * mask_b

    handle_b = b.register_hook(hook_b)

    # label_query / label_lt 之类的，如果你想更激进，也可以只冻结目标类那一行
    if hasattr(model, "label_lt"):
        L = model.label_lt.weight       # (num_labels, hidden)
        L.requires_grad = True
        mask_L = torch.ones_like(L)
        mask_L[forget_cls] = 0.0

        def hook_L(grad):
            return grad * mask_L

        handle_L = L.register_hook(hook_L)
    else:
        handle_L = None

    # ===== 2) Teacher：固定，用于目标类 logit 一致性（防止恢复） =====
    if teacher_model is not None and lambda_logit_cons > 0.0:
        teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    # ===== 3) 优化器只看 requires_grad=True 的参数 =====
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=rec_lr,
    )

    non_target_idx = [i for i in range(args.num_labels) if i != forget_cls]

    for ep in range(rec_epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"[Recovery] client {net_id} | ep {ep}", ncols=100)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            images = batch["image"].float().to(device)
            labels = batch["labels"].float().to(device)
            mask   = batch["mask"].float().to(device)
            mask_in = mask.clone()

            # 只保留“目标类为 0 的样本”（不含 Person 正样本）
            keep_idx = (labels[:, forget_cls] == 0)
            if keep_idx.sum() == 0:
                continue

            images = images[keep_idx]
            labels = labels[keep_idx]
            mask_in = mask_in[keep_idx]

            optimizer.zero_grad()

            # ---- 当前模型输出 ----
            logits, _, _ = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=False,
            )
            # 非目标类 logits / labels
            logits_nt = logits[:, non_target_idx]          # (B, L-1)
            labels_nt = labels[:, non_target_idx]          # (B, L-1)

            # BCE（非目标类）：对正样本加权，提升 Recall
            bce_nt = F.binary_cross_entropy_with_logits(
                logits_nt, labels_nt, reduction="none"
            )  # (B, L-1)

            pos_mask = (labels_nt == 1)
            neg_mask = (labels_nt == 0)

            if pos_mask.any():
                loss_pos = (bce_nt[pos_mask] * alpha_pos).mean()
            else:
                loss_pos = torch.tensor(0.0, device=device)

            if neg_mask.any():
                loss_neg = bce_nt[neg_mask].mean()
            else:
                loss_neg = torch.tensor(0.0, device=device)

            loss_bce_nt = loss_pos + loss_neg

            # ---- teacher 一致性约束（仅目标类）可选 ----
            loss_cons = torch.tensor(0.0, device=device)
            if teacher_model is not None and lambda_logit_cons > 0.0:
                with torch.no_grad():
                    t_logits, _, _ = teacher_model(
                        images,
                        mask_in,
                        args.learn_emb_type,
                        emb_feat,
                        clip_model,
                        return_label_emb=False,
                    )
                # 只看目标类 logit，鼓励与遗忘后保持一致（通常都很负）
                logit_f     = logits[:, forget_cls]
                t_logit_f   = t_logits[:, forget_cls]
                loss_cons = F.mse_loss(logit_f, t_logit_f)

            loss = loss_bce_nt + lambda_logit_cons * loss_cons

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / step

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "bce_nt": f"{loss_bce_nt.item():.3f}",
                "cons": f"{loss_cons.item():.3f}",
            })

    # 清理 hook
    handle_W.remove()
    handle_b.remove()
    if handle_L is not None:
        handle_L.remove()

# ulearn/recovery_train.py 里原来的 federated_recovery_no_target_samples 替换为：

from torch.utils.data import DataLoader, Subset
import torch

def federated_recovery_no_target_samples(
    args,
    global_model,
    nets,
    train_dl_global,
    partition_idx_map,
    device,
    emb_feat,
    clip_model,
    forget_cls,
    recovery_rounds=3,
    teacher_model=None,
    lambda_logit_cons=0.0,
):
    n_parties = args.n_parties

    # 恢复阶段的超参数（没有就用默认）
    rec_epochs = getattr(args, "recovery_epochs", 3)
    rec_bs     = getattr(args, "recovery_batch_size", args.batch_size)
    rec_lr     = getattr(args, "recovery_lr", 5e-4)

    # 准备 teacher：通常是“遗忘后”的 global_model
    if teacher_model is not None:
        teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    for r in range(recovery_rounds):
        print(f"\n[Recovery-Round {r}] Step 1: 广播当前全局模型到各客户端 ...")
        global_state = global_model.state_dict()

        for cid in range(n_parties):
            nets[cid].load_state_dict(global_state)
            nets[cid].to(device)

        # ===== Step 2: 各客户端本地恢复训练（只用非目标类样本） =====
        net_dataidx_map = {}
        for cid in range(n_parties):
            # 1) 先按客户端划分
            all_idxs = partition_idx_map[cid]

            # 2) 再过滤掉“目标类为 1 的样本”
            keep_idxs = []
            ds = train_dl_global.dataset
            for idx in all_idxs:
                sample = ds[idx]
                # sample["labels"]: (L,)
                lbl = sample["labels"]
                # 有的 Dataset 返回 numpy，有的返回 tensor，这里做一下兼容
                if isinstance(lbl, torch.Tensor):
                    y = lbl
                else:
                    y = torch.tensor(lbl)
                if y[forget_cls].item() == 0:
                    keep_idxs.append(idx)

            if len(keep_idxs) == 0:
                print(f"[Recovery-Round {r}] client {cid}: 没有非目标类样本，跳过恢复")
                continue

            sub_dst = Subset(ds, keep_idxs)
            train_dl_local = DataLoader(
                sub_dst,
                batch_size=rec_bs,
                shuffle=True,
                num_workers=args.workers,
                drop_last=False
            )
            net_dataidx_map[cid] = len(keep_idxs)

            train_net_recovery_no_target(
                net_id=cid,
                model=nets[cid],
                train_dataloader=train_dl_local,
                valid_dataloader=None,
                epochs=rec_epochs,           # ✅ 用恢复专用 epoch
                args=args,
                device=device,
                g_model=global_model,
                emb_feat=emb_feat,
                clip_model=clip_model,
                forget_cls=forget_cls,
                teacher_model=teacher_model,
                lambda_logit_cons=lambda_logit_cons,    # 如果你在里面用到了
            )

            nets[cid].to("cpu")

        # 没有任何客户端参与恢复，直接退出
        if len(net_dataidx_map) == 0:
            print(f"[Recovery-Round {r}] 没有客户端参与恢复，提前结束")
            break

        # ===== Step 3: FedAvg 聚合 =====
        print(f"[Recovery-Round {r}] Step 3: 服务器聚合各客户端恢复后的模型 ...")

        total_points = sum(net_dataidx_map.values())
        fed_avg_freqs = {
            cid: net_dataidx_map[cid] / total_points
            for cid in net_dataidx_map.keys()
        }

        new_global_state = {}
        for cid, freq in fed_avg_freqs.items():
            state_c = nets[cid].state_dict()
            for k, v in state_c.items():
                v = v.float()
                if k not in new_global_state:
                    new_global_state[k] = v * freq
                else:
                    new_global_state[k] += v * freq

        global_model.load_state_dict(new_global_state)
        global_model.to(device)

        print(f"[Recovery-Round {r}] FedAvg 聚合完成")

    return global_model

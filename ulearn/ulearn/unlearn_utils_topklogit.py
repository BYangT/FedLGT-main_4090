# ulearn/unlearn_utils_topklogit.py
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def collect_topk_dims_for_class(
    model,
    dataloader,
    forget_cls: int,
    K: int,
    device,
    args,
    emb_feat,
    clip_model,
):
    """
    作用：在【label embedding】上为某个类别统计 “top-K 判别维度”。

    假设 model(..., return_label_emb=True) 返回：
        logits, _, _, label_emb
    其中 label_emb: (B, L, D)

    思路：
      - 从 label_emb[:, forget_cls, :] 取出该类 embedding；
      - 正样本取 abs 平均：pos_mean
      - 负样本取 abs 平均：neg_mean
      - score = pos_mean - neg_mean
      - 选择 score 最大的 K 个维度，作为 top-K 判别维度
    """

    model.eval()
    model.to(device)

    pos_sum = None
    neg_sum = None
    pos_cnt = 0
    neg_cnt = 0

    for batch in tqdm(dataloader, desc=f"[TopK] collect cls {forget_cls}"):
        images = batch["image"].float().to(device)
        labels = batch["labels"].float().to(device)
        mask   = batch["mask"].float().to(device)
        mask_in = mask.clone()

        # logits: (B,L) ; label_emb: (B,L,D)
        logits, _, _, label_emb = model(
            images,
            mask_in,
            args.learn_emb_type,
            emb_feat,
            clip_model,
            return_label_emb=True,
        )
        # 取该类别的 embedding: (B, D)
        z_f = label_emb[:, forget_cls, :]

        pos_idx = (labels[:, forget_cls] == 1)
        neg_idx = (labels[:, forget_cls] == 0)

        if pos_idx.any():
            z_pos = z_f[pos_idx]          # (N_pos, D)
            v = z_pos.abs().sum(dim=0)    # (D,)
            if pos_sum is None:
                pos_sum = v
            else:
                pos_sum += v
            pos_cnt += z_pos.size(0)

        if neg_idx.any():
            z_neg = z_f[neg_idx]          # (N_neg, D)
            v = z_neg.abs().sum(dim=0)
            if neg_sum is None:
                neg_sum = v
            else:
                neg_sum += v
            neg_cnt += z_neg.size(0)

    if pos_cnt == 0 or neg_cnt == 0:
        raise RuntimeError(
            f"[TopK] pos_cnt={pos_cnt}, neg_cnt={neg_cnt}, 无法为类 {forget_cls} 统计 top-K 维度"
        )

    pos_mean = pos_sum / pos_cnt
    neg_mean = neg_sum / neg_cnt
    score = pos_mean - neg_mean        # (D,)

    D = score.numel()
    K_use = min(K, D)
    topk_vals, topk_idx = torch.topk(score, k=K_use, largest=True)

    return topk_idx.to(device), score.to(device)


def unlearn_one_class_on_model_topk_logit(
    model,
    dataloader,
    forget_cls: int,
    topk_idx: torch.Tensor,
    device,
    args,
    emb_feat,
    clip_model,
    epochs: int = 1,
    lambda_keep: float = 1.0,
    lambda_forget_logit: float = 5.0,
    lambda_forget_feat: float = 1.0,
    lr: float = 1e-4,
):
    """
    “top-K 维 + logits 惩罚” 单模型遗忘版本（label embedding 版）

    - loss_keep:
        对【非目标类】做正常 BCE，保护其它 L-1 个类别；
    - loss_forget_logit:
        对目标类的 logit 强行拉向 0（目标 label=0），
        会直接压低 AP / P / R；
    - loss_forget_feat:
        对目标类正样本的 label embedding 中 top-K 维做 L2 收缩，
        抹掉判别维度。

    参数：
      topk_idx: collect_topk_dims_for_class 得到的 (K,) 维索引
    """

    model.train()
    model.to(device)

    topk_idx = topk_idx.to(device)
    D = topk_idx.numel()  # 实际 K，用于 log 打印而已

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"[TopK+Logit] cls {forget_cls} | ep {ep}", ncols=100)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            images = batch["image"].float().to(device)
            labels = batch["labels"].float().to(device)
            mask   = batch["mask"].float().to(device)
            mask_in = mask.clone()

            optimizer.zero_grad()

            logits, _, _, label_emb = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )
            # logits: (B,L)
            # label_emb: (B,L,D_emb)
            B, L = logits.shape
            _, _, D_emb = label_emb.shape

            # ===== 1) 保护其它类别的 BCE =====
            bce_all = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )   # (B,L)

            keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
            keep_mask[:, forget_cls] = False
            if keep_mask.sum() > 0:
                loss_keep = bce_all[keep_mask].mean()
            else:
                loss_keep = torch.tensor(0.0, device=device)

            # ===== 2) 目标类 logits 惩罚：强制该 logit ≈ 0 =====
            logits_f = logits[:, forget_cls]      # (B,)
            target_zero = torch.zeros_like(logits_f)
            loss_forget_logit = F.binary_cross_entropy_with_logits(
                logits_f, target_zero, reduction="mean"
            )

            # ===== 3) top-K 判别维度 L2 shrink（仅正样本） =====
            pos_idx = (labels[:, forget_cls] == 1)
            if pos_idx.any():
                z_f = label_emb[:, forget_cls, :]       # (B,D_emb)
                z_pos = z_f[pos_idx]                    # (N_pos, D_emb)
                # 只取 top-K 维度
                z_pos_topk = z_pos[:, topk_idx]         # (N_pos, K)
                loss_forget_feat = (z_pos_topk ** 2).mean()
            else:
                loss_forget_feat = torch.tensor(0.0, device=device)

            # ===== 4) 总 loss =====
            loss = (
                lambda_keep * loss_keep +
                lambda_forget_logit * loss_forget_logit +
                lambda_forget_feat * loss_forget_feat
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / step

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "keep": f"{loss_keep.item():.3f}",
                "logit": f"{loss_forget_logit.item():.3f}",
                "feat": f"{loss_forget_feat.item():.3f}",
                "K": D,
            })

    return model
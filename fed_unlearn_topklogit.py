

import random
import torch
import gc
import copy
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

from ulearn.unlearn_utils_topklogit import (
    collect_topk_dims_for_class,
    estimate_target_subspace,
    rerank_topk_dims_with_gradient,
    unlearn_one_class_on_model,
)


def compute_client_pos_count(
        train_dl_global,
        partition_idx_map,
        forget_cls: int,
        batch_size: int,
        num_workers: int,
        device,
):
    """
    统计每个客户端的正样本数。
    """
    client_pos_count = {}

    for cid, idxs in partition_idx_map.items():
        sub_dst = Subset(train_dl_global.dataset, idxs)
        loader = DataLoader(
            sub_dst,
            # 统计时不需要梯度，Batch size 可以大一点
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        pos_cnt = 0
        for batch in loader:
            labels = batch["labels"].float().to(device)
            pos_cnt += int((labels[:, forget_cls] == 1).sum().item())

        client_pos_count[cid] = pos_cnt

        # 每次循环后清理一下 GPU 缓存
        del labels
        torch.cuda.empty_cache()

    return client_pos_count

# 现用方法
def federated_unlearn_one_class_topklogit(
        args,
        global_model,
        nets,
        train_dl_global,
        partition_idx_map,
        device,
        emb_feat,
        clip_model,
        forget_cls: int,
        K: int = 64,
        unlearn_rounds: int = 5,
        client_frac: float = 1.,
        unlearn_epochs: int = 3,
        unlearn_lr: float = 1e-4,
        lambda_keep: float = 1.0,
        lambda_forget_logit: float = 20.0,
        lambda_forget_feat: float = 1.0,
        min_pos: int = 10,
        mode: str = None,
):
    # ========= 0) 统计每个客户端中目标类的正样本数量 =========
    torch.cuda.empty_cache()
    print(f"lambda_forget_feat:{lambda_forget_feat}")
    client_pos_count = compute_client_pos_count(
        train_dl_global=train_dl_global,
        partition_idx_map=partition_idx_map,
        forget_cls=forget_cls,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device,
    )

    print(f"=== [TopK+Logit-Fed] client_pos_count (forget_cls = {forget_cls}) ===")
    count_print = 0
    for cid, c in client_pos_count.items():
        if count_print < 5:
            print(f"  client {cid}: {c} pos samples")
        count_print += 1
    if len(client_pos_count) > 5: print("  ...")

    candidate_clients = [cid for cid, c in client_pos_count.items() if c >= min_pos]
    if len(candidate_clients) == 0:
        print(f"[Warn] No client has enough positive samples for class {forget_cls}, skip unlearning.")
        return global_model

    # ========= 1) 客户端本地统计 score，server 做加权求 global top-K =========
    print(f"[TopK+Logit-Fed] Collect client-wise scores ...")

    total_pos = sum(client_pos_count[cid] for cid in candidate_clients)
    global_score = None

    # 临时将 global_model 放 CPU
    global_model.cpu()
    global_state_dict = copy.deepcopy(global_model.state_dict())

    for cid in candidate_clients:
        sub_dst = Subset(train_dl_global.dataset, partition_idx_map[cid])
        loader = DataLoader(
            sub_dst,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=False,
        )

        # === 显存优化：上车 ===
        local_model = nets[cid]
        local_model.load_state_dict(global_state_dict)
        local_model.to(device)

        topk_idx_local, score_local = collect_topk_dims_for_class(
            model=local_model,
            dataloader=loader,
            forget_cls=forget_cls,
            K=K,
            device=device,
            args=args,
            emb_feat=emb_feat,
            clip_model=clip_model,
        )

        # === 显存优化：下车 ===
        local_model.cpu()
        torch.cuda.empty_cache()
        gc.collect()

        # 累加分数 (在 CPU 上进行)
        w = client_pos_count[cid] / float(total_pos)
        score_local = score_local.cpu()

        if global_score is None:
            global_score = w * score_local
        else:
            global_score += w * score_local

    # ========= 1.5) 低算力重排：top-M 候选 -> 小样本梯度敏感度 -> 去冗余 top-K =========
    K_eff = min(K, global_score.numel())
    candidate_mul = max(1.0, float(getattr(args, "topk_candidate_mul", 2.0)))
    candidate_M = min(global_score.numel(), max(K_eff, int(round(K_eff * candidate_mul))))
    candidate_val, candidate_idx = torch.topk(global_score, k=candidate_M)

    rerank_client_n = max(1, int(getattr(args, "topk_rerank_clients", 2)))
    rerank_batch_n = max(1, int(getattr(args, "topk_rerank_batches", 3)))
    rerank_alpha = float(getattr(args, "topk_rerank_alpha", 0.5))
    rerank_beta = float(getattr(args, "topk_rerank_beta", 0.5))
    rerank_gamma = float(getattr(args, "topk_rerank_gamma", 0.2))

    rerank_clients = sorted(
        candidate_clients,
        key=lambda cid: client_pos_count[cid],
        reverse=True,
    )[:rerank_client_n]
    rerank_dataset = ConcatDataset([
        Subset(train_dl_global.dataset, partition_idx_map[cid]) for cid in rerank_clients
    ])
    rerank_loader = DataLoader(
        rerank_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    global_model.load_state_dict(global_state_dict)
    topk_idx_global, topk_weight_global = rerank_topk_dims_with_gradient(
        model=global_model,
        dataloader=rerank_loader,
        forget_cls=forget_cls,
        candidate_idx=candidate_idx,
        candidate_score=candidate_val,
        final_k=K_eff,
        device=device,
        args=args,
        emb_feat=emb_feat,
        clip_model=clip_model,
        max_batches=rerank_batch_n,
        alpha=rerank_alpha,
        beta=rerank_beta,
        gamma=rerank_gamma,
    )
    global_model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[TopK+Logit-Fed] Global reranked top-{K_eff} dims calculated from candidate_M={candidate_M}.")

    subspace_batches = max(1, int(getattr(args, "subspace_batches", rerank_batch_n)))
    subspace_rank = max(1, int(getattr(args, "subspace_rank", min(16, K_eff))))
    global_model.load_state_dict(global_state_dict)
    topk_subspace_basis, topk_subspace_center = estimate_target_subspace(
        model=global_model,
        dataloader=rerank_loader,
        forget_cls=forget_cls,
        topk_idx=topk_idx_global,
        topk_weights=topk_weight_global,
        device=device,
        args=args,
        emb_feat=emb_feat,
        clip_model=clip_model,
        max_batches=subspace_batches,
        subspace_rank=subspace_rank,
    )
    global_model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # ========= 2) 多轮联邦遗忘 =========
    for ur in range(unlearn_rounds):
        print(f"\n[TopK+Logit-Fed][Unlearn-Round {ur}] Start ...")

        # 2.1 选客户端
        if client_frac >= 1.0:
            selected_clients = list(candidate_clients)
        else:
            m = max(1, int(len(candidate_clients) * client_frac))
            selected_clients = random.sample(candidate_clients, m)

        # 2.2 准备聚合容器
        # 初始化：完整复制 global_model，保留 position_ids 等 LongTensor 的原值
        new_global_state = copy.deepcopy(global_model.state_dict())

        # 将 float 类型的参数（权重、偏置）清零，准备累加
        for k, v in new_global_state.items():
            if v.is_floating_point():
                v.zero_()

        total_data_points = sum(len(partition_idx_map[cid]) for cid in nets.keys())

        # 备份一份 CPU 版的 global 状态，用于没被选中的客户端
        global_state_dict = {k: v.cpu() for k, v in global_model.state_dict().items()}

        all_clients = list(nets.keys())

        # 使用流式聚合，避免内存爆炸
        for cid in tqdm(all_clients, desc=f"Round {ur} Processing", ncols=100):
            weight_factor = len(partition_idx_map[cid]) / total_data_points

            # --- 情况 A: 选中的客户端 ---
            if cid in selected_clients:
                if client_pos_count.get(cid, 0) == 0:
                    # 没正样本，当作没选中处理
                    for k, v in global_state_dict.items():
                        if v.is_floating_point():  # 只累加浮点数
                            new_global_state[k] += v * weight_factor
                    continue

                sub_dst = Subset(train_dl_global.dataset, partition_idx_map[cid])
                unlearn_bs = getattr(args, "unlearn_batch_size", max(1, args.batch_size // 2))
                local_loader = DataLoader(
                    sub_dst, batch_size=unlearn_bs, shuffle=True,
                    num_workers=args.workers, drop_last=False
                )

                # === 上车 ===
                local_model = nets[cid]
                local_model.load_state_dict(global_state_dict)
                local_model.to(device)

                # 训练
                local_model = unlearn_one_class_on_model(
                    model=local_model,
                    dataloader=local_loader,
                    forget_cls=forget_cls,
                    topk_idx=topk_idx_global,
                    topk_weights=topk_weight_global,
                    topk_subspace_basis=topk_subspace_basis,
                    topk_subspace_center=topk_subspace_center,
                    device=device,
                    args=args,
                    emb_feat=emb_feat,
                    clip_model=clip_model,
                    epochs=unlearn_epochs,
                    lambda_keep=lambda_keep,
                    lambda_forget_logit=lambda_forget_logit,
                    lambda_forget_feat=lambda_forget_feat,
                    lr=unlearn_lr,
                )

                # === 下车 & 累加 ===
                local_state = {k: v.cpu() for k, v in local_model.state_dict().items()}

                for k, v in local_state.items():
                    # 只聚合浮点数参数
                    if v.is_floating_point():
                        new_global_state[k] += v * weight_factor

                # 清理
                local_model.cpu()
                del local_state
                torch.cuda.empty_cache()

            # --- 情况 B: 没选中的客户端 ---
            else:
                for k, v in global_state_dict.items():
                    # 只聚合浮点数参数
                    if v.is_floating_point():
                        new_global_state[k] += v * weight_factor

        # 2.3 更新全局模型
        global_model.load_state_dict(new_global_state)
        gc.collect()

    print("[TopK+Logit-Fed] Federated unlearning finished.")

    # 【核心修复】返回前，强制将全局模型搬回 GPU！
    # 这样 run_epoch 里的 images.cuda() 就能和模型匹配了
    global_model.to(device)

    return global_model

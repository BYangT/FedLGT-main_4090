import torch
import copy
import gc
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# 如果你有 utils.evaluate 或其他引用，保留它们
# from run_epoch import run_epoch

def federated_recovery_simple(
        args,
        global_model,
        nets,
        train_dl_global,
        partition_idx_map,
        device,
        emb_feat,
        clip_model,
        forget_cls: int,
        recovery_rounds: int = 2,
        recovery_epochs: int = 1,  # 每个客户端恢复训练几轮
        recovery_lr: float = 1e-4,  # 恢复学习率
):
    """
    [Memory Optimized] 联邦恢复训练
    修复了 OOM 问题和 Float/Long 聚合报错问题。
    """
    print(f"\n[Fed-Recovery] Start recovery for {recovery_rounds} rounds...")

    # 1. 准备工作：计算总数据量
    total_data_points = sum(len(partition_idx_map[cid]) for cid in nets.keys())

    # 2. 临时将 global_model 放 CPU，用的时候再考
    global_model.cpu()

    for rnd in range(recovery_rounds):
        print(f"\n[Fed-Recovery] Round {rnd + 1}/{recovery_rounds}")

        # 备份 global 状态 (CPU)
        global_state_dict = copy.deepcopy(global_model.state_dict())

        # 初始化聚合容器 (保留 LongTensor 原值，避免类型错误)
        new_global_state = copy.deepcopy(global_state_dict)
        for k, v in new_global_state.items():
            if v.is_floating_point():
                v.zero_()

        all_clients = list(nets.keys())

        # 3. 遍历客户端 (流式处理：上车->训练->下车->聚合)
        for cid in tqdm(all_clients, desc=f"Recovery Round {rnd}", ncols=100):
            weight_factor = len(partition_idx_map[cid]) / total_data_points

            # --- A. 数据准备 ---
            sub_dst = Subset(train_dl_global.dataset, partition_idx_map[cid])
            # 恢复训练 batch size 保持安全值
            rec_bs = getattr(args, "recovery_batch_size", args.batch_size)

            local_loader = DataLoader(
                sub_dst, batch_size=rec_bs, shuffle=True,
                num_workers=args.workers, drop_last=False
            )

            # --- B. 模型上车 (CPU -> GPU) ---
            local_model = nets[cid]
            local_model.load_state_dict(global_state_dict)  # 同步全局参数
            local_model.to(device)
            local_model.train()

            # 定义优化器
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, local_model.parameters()),
                lr=recovery_lr,
                momentum=0.9,
                weight_decay=1e-4
            )

            # --- C. 本地训练 (Masked Loss) ---
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

            for epoch in range(recovery_epochs):
                for batch in local_loader:
                    images = batch["image"].to(device)
                    # 确保 label 也是 float 类型
                    labels = batch["labels"].float().to(device)

                    # 获取 mask (如果有)
                    mask_in = batch.get("mask", None)
                    if mask_in is not None:
                        mask_in = mask_in.to(device)

                    # 前向传播
                    # 注意：这里假设 CTranModel 的 forward 接收这些参数
                    # 如果你的 forward 参数不同，请根据 run_epoch.py 修改这里
                    pred, _, _ = local_model(
                        images, mask_in, args.learn_emb_type, emb_feat, clip_model
                    )

                    # 计算 Loss
                    loss_mat = criterion(pred, labels)  # (B, num_classes)

                    # 【关键】屏蔽目标类 (forget_cls) 的 Loss
                    # 我们希望模型恢复对"其他类"的记忆，而不强化"目标类"
                    mask_loss = torch.ones_like(loss_mat)
                    mask_loss[:, forget_cls] = 0.0

                    loss = (loss_mat * mask_loss).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # --- D. 模型下车 & 流式聚合 ---
            # 转回 CPU
            local_state = {k: v.cpu() for k, v in local_model.state_dict().items()}

            for k, v in local_state.items():
                # 【关键修复】只聚合浮点数，跳过 LongTensor
                if v.is_floating_point():
                    new_global_state[k] += v * weight_factor

            # --- E. 清理显存 ---
            local_model.cpu()  # 确保移出 GPU
            del local_state, optimizer, images, labels, pred, loss, mask_loss
            torch.cuda.empty_cache()

        # 4. 更新全局模型
        global_model.load_state_dict(new_global_state)
        gc.collect()  # 强制 GC

    print("[Fed-Recovery] Done.")

    # 5. 最后把全局模型搬回 GPU，以便主程序进行 test
    global_model.to(device)

    return global_model
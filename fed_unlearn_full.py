# fed_unlearn_full.py
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from ulearn.unlearn_utils import collect_topk_dims_for_class, unlearn_one_class_on_model, \
    collect_topk_dims_for_class_vis, unlearn_one_class_on_model_vis
from ulearn.unlearn_utils_ferrari import unlearn_one_class_on_model_ferrari


def compute_client_pos_count(
    train_dl_global,
    partition_idx_map,
    forget_cls: int,
    batch_size: int,
    num_workers: int,
    device,
):
    """
    统计：每个客户端中，目标类别 forget_cls 的正样本数量（label=1 的个数）

    返回：
        client_pos_count: dict[cid] = int（该客户端中 forget_cls 为 1 的图像数量）
    """
    client_pos_count = {}

    for cid, idxs in partition_idx_map.items():
        # 取这个客户端的数据子集
        sub_dst = torch.utils.data.Subset(train_dl_global.dataset, idxs)
        loader = DataLoader(
            sub_dst,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        pos_cnt = 0
        for batch in tqdm(loader, desc=f"[Count pos] client {cid}", ncols=100):
            labels = batch["labels"].float().to(device)  # (B, L)
            # 这一批里，forget_cls 维度是 1 的样本个数
            pos_cnt += int((labels[:, forget_cls] == 1).sum().item())

        client_pos_count[cid] = pos_cnt

    return client_pos_count


def build_client_loader_voc(net_id, args, train_dl_global, partition_idx_map):
    """
    根据 fed_main 的写法，构造某个客户端 net_id 的本地 DataLoader。
    这里只处理 coco/voc 这条分支。
    """
    sub_dst = torch.utils.data.Subset(
        train_dl_global.dataset,
        partition_idx_map[net_id]
    )
    train_dl_local = torch.utils.data.DataLoader(
        sub_dst,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False
    )
    return train_dl_local, len(sub_dst)


import random

def federated_unlearn_one_class_full(
    args,
    global_model,
    nets,
    train_dl_global,
    partition_idx_map,
    device,
    emb_feat,
    clip_model,
    forget_cls,
    K,
    unlearn_rounds=1,
    client_frac=1.0,
    unlearn_epochs=1,
    unlearn_lr=1e-4,
    lambda_keep=1.0,
    lambda_forget_logit=1.0,
    lambda_forget_feat=1.0,
):
    # ========= 1) 先在全体客户端上统计“目标类正样本数量” =========
    client_pos_count = compute_client_pos_count(
        train_dl_global=train_dl_global,
        partition_idx_map=partition_idx_map,
        forget_cls=forget_cls,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device,
    )

    print("=== client_pos_count (forget_cls = {}) ===".format(forget_cls))
    for cid, c in client_pos_count.items():
        print(f"  client {cid}: {c} pos samples")

    # 设一个阈值：正样本数 >= min_pos 的客户端才有资格参与遗忘
    min_pos = 10  # 你可以先试 10 或 20，根据统计结果再调
    candidate_clients = [cid for cid, c in client_pos_count.items() if c >= min_pos]

    if len(candidate_clients) == 0:
        print("[Warn] No client has enough positive samples for class {}, skip unlearning.".format(forget_cls))
        return global_model

    print(f"Clients eligible for unlearning (pos_count >= {min_pos}): {candidate_clients}")

    # ========= 2) 先全局算一份 top-K 维度（不变） =========
    # ========= 2) 联邦版：每个客户端本地算 score，再在服务器端聚合出全局 top-K =========
    from torch.utils.data import Subset, DataLoader
    from ulearn.unlearn_utils import collect_topk_dims_for_class

    print(f"[TopK] start federated collect for class {forget_cls}, K={K}")

    agg_score = None  # 用来累加各客户端的维度得分（shape: [hidden_dim]）

    for cid in range(args.n_parties):
        # 1) 取出这个客户端对应的全局样本索引
        idxs = partition_idx_map[cid]  # 例如 array([...])

        # 2) 构造这个客户端的本地数据集 / dataloader
        sub_dataset = Subset(train_dl_global.dataset, idxs)
        local_loader = DataLoader(
            sub_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # 统计用，不需要 shuffle
            num_workers=args.workers,
            drop_last=False,
        )

        # # 3) 在「这个客户端的数据」上统计该类的维度得分
        # #    注意：这里可以直接用 global_model（此时各客户端模型还没分化），
        # #    你也可以先把 global_state load 给某个临时 model 再传进去，本质一样。
        # _, local_score = collect_topk_dims_for_class(
        #     model=global_model,
        #     dataloader=local_loader,
        #     forget_cls=forget_cls,
        #     K=K,  # 这里返回的 topk_idx 我们先不用，主要用 score
        #     device=device,
        #     args=args,
        #     emb_feat=emb_feat,
        #     clip_model=clip_model,
        # )
        # # local_score 形状大致是 [hidden_dim]，是每个维度的“类专属程度得分”


        topk_idx, local_score = collect_topk_dims_for_class_vis(
            model=global_model,
            dataloader=local_loader,  # federated 版的话也可以先每个 client 算再累加
            forget_cls=forget_cls,
            K=K,
            device=device,
            args=args,
            emb_feat=emb_feat,
            clip_model=clip_model,
        )

        local_score = local_score.detach().cpu()

        if agg_score is None:
            agg_score = local_score.clone()
        else:
            agg_score += local_score  # 这里是“加和”；如果想平均，可以最后除以 args.n_parties

    # 4) 服务器端根据聚合后的 agg_score 选出全局 top-K 维度
    K_use = min(K, agg_score.numel())
    topk_vals, topk_idx = torch.topk(agg_score, K_use, largest=True)

    print(f"[TopK-Fed] global dims for class {forget_cls}: {topk_idx.cpu().numpy()}")
    # 如果后面还想用 score，就用 agg_score
    score = agg_score

    # ========= 3) 若有多轮联邦遗忘 =========
    for ur in range(unlearn_rounds):
        print(f"[Unlearn-Round {ur}] Start ...")

        # ---- 3.1 从有资格的客户端中抽取一部分参与本轮遗忘 ----
        if client_frac >= 1.0:
            selected_clients = list(candidate_clients)
        else:
            m = max(1, int(len(candidate_clients) * client_frac))
            selected_clients = random.sample(candidate_clients, m)

        print(f"[Unlearn-Round {ur}] selected clients: {selected_clients}")

        # 保存每个客户端遗忘后的 state_dict（参与聚合用）
        client_states = {}

        # ---- 3.2 对“选中的客户端”做遗忘微调 ----
        for cid in selected_clients:
            print(f"[Unlearn-Round {ur}] client {cid} do unlearning ...")
            # ⭐ 第一步：给遗忘阶段单独设置一个更小的 batch_size
            unlearn_bs = getattr(args, "unlearn_batch_size", max(1, args.batch_size // 2))
            # 准备该客户端的数据 loader（和 local_train_net 里一致）
            sub_dst = torch.utils.data.Subset(train_dl_global.dataset, partition_idx_map[cid])
            local_loader = torch.utils.data.DataLoader(
                sub_dst,
                batch_size=unlearn_bs,      # ⭐ 这里用 unlearn_bs
                shuffle=True,
                num_workers=args.workers,
                drop_last=False,
            )

            # 把全局模型参数拷贝到这个客户端
            nets[cid].load_state_dict(global_model.state_dict())
            nets[cid].to(device)

            # 调用你之前写好的单客户端遗忘函数
            from ulearn.unlearn_utils import unlearn_one_class_on_model

            # nets[cid] = unlearn_one_class_on_model(
            #     model=nets[cid],
            #     dataloader=local_loader,
            #     forget_cls=forget_cls,
            #     topk_idx=topk_idx,
            #     device=device,
            #     args=args,
            #     emb_feat=emb_feat,
            #     clip_model=clip_model,
            #     epochs=unlearn_epochs,
            #     lambda_keep=lambda_keep,
            #     lambda_forget_logit=lambda_forget_logit,
            #     lambda_forget_feat=lambda_forget_feat,
            #     lr=unlearn_lr,
            # )
            nets[cid] = unlearn_one_class_on_model_vis(
                model=nets[cid],
                dataloader=local_loader,
                forget_cls=forget_cls,
                topk_idx=topk_idx,  # 用视觉 topK
                device=device,
                args=args,
                emb_feat=emb_feat,
                clip_model=clip_model,
                epochs=unlearn_epochs,
                lambda_keep=lambda_keep,
                lambda_vis=args.lambda_vis,  # 你在 args 里加一个
                lr=unlearn_lr,
            )

            # 遗忘完，把参数搬到 CPU 存起来
            client_states[cid] = {k: v.cpu() for k, v in nets[cid].state_dict().items()}

        # ---- 3.3 对“没选中的客户端”什么都不做，直接用原全局参数 ----
        for cid in nets.keys():
            if cid not in selected_clients:
                # 这些客户端视为“没参与遗忘”，贡献的权重就是原来的 global_model
                client_states[cid] = {k: v.cpu() for k, v in global_model.state_dict().items()}

        # ---- 3.4 FedAvg 聚合得到新的全局模型 ----
        print(f"[Unlearn-Round {ur}] FedAvg aggregation ...")
        # 和你训练时一样，用样本数做加权
        total_data_points = sum(len(partition_idx_map[cid]) for cid in nets.keys())
        fed_avg_freqs = {
            cid: len(partition_idx_map[cid]) / total_data_points
            for cid in nets.keys()
        }

        new_global_state = {}
        for i, (cid, state_dict) in enumerate(client_states.items()):
            w = fed_avg_freqs[cid]
            if i == 0:
                # 先用第一个客户端初始化
                new_global_state = {k: v.clone() * w for k, v in state_dict.items()}
            else:
                for k in state_dict:
                    new_global_state[k] += state_dict[k] * w

        # 把 new_global_state 加载回 global_model
        global_model.load_state_dict(new_global_state)
        global_model.to(device)

        print(f"[Unlearn-Round {ur}] Done.")

    return global_model
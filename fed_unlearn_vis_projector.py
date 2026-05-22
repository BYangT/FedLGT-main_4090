# fed_unlearn_vis_projector.py

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ulearn.projector_subspace import build_vis_subspace_for_class
from ulearn.unlearn_utils_projector import (
    unlearn_one_class_on_model_vis_projector,
)


def federated_unlearn_one_class_vis_projector(
    args,
    global_model,
    nets,
    train_dl_global,
    partition_idx_map,
    device,
    emb_feat,
    clip_model,
    forget_cls: int,
    r: int = 8,               # 子空间维度 rank r
    unlearn_rounds: int = 1,  # 联邦遗忘轮数
    unlearn_epochs: int = 1,  # 每个客户端本地遗忘 epoch 数
    unlearn_lr: float = 1e-4,
    lambda_keep: float = 1.0, # 保护其他类 BCE 权重
    lambda_dir: float = 5.0,  # 子空间方向抹除权重（传给 lambda_vis）
):
    """
    极简版联邦 PROJECTOR 可视特征遗忘：

    对每一轮 ur：
      1）server 把 global_model 参数下发到所有客户端；
      2）每个客户端 cid：
            - 用自己的数据 Subset(dataset, partition_idx_map[cid]) 构造 U_c；
            - 在 U_c 上做 PROJECTOR 忘却，得到本地 unlearned 模型；
      3）server 对所有客户端模型做一次 FedAvg，更新 global_model。
    """

    for ur in range(unlearn_rounds):
        print(f"\n[Vis-Projector][Unlearn-Round {ur}] Start ...")

        client_states = {}

        # ===== 1) 所有客户端：下发 global_model，局部构造 U_c + 忘却 =====
        for cid in nets.keys():
            print(f"[Unlearn-Round {ur}] client {cid} do vis-projector unlearning ...")

            # 该客户端的数据子集
            sub_dst = Subset(train_dl_global.dataset, partition_idx_map[cid])

            # 1.1 用不打乱的 loader 在本客户端上估计 U_c
            U_loader = DataLoader(
                sub_dst,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                drop_last=False,
            )

            # 下发当前 global_model 参数
            nets[cid].load_state_dict(global_model.state_dict())
            nets[cid].to(device)

            # 有些客户端可能没有目标类正样本，会在 build_vis_subspace_for_class 里抛异常
            try:
                print(f"[Unlearn-Round {ur}] client {cid} build local subspace U_c (rank={r}) ...")
                U_c = build_vis_subspace_for_class(
                    model=nets[cid],
                    dataloader=U_loader,
                    forget_cls=forget_cls,
                    device=device,
                    args=args,
                    emb_feat=emb_feat,
                    clip_model=clip_model,
                    rank=r,
                    max_pos_images=2000,
                    save_path=f"./ulearn/U_vis_cls{forget_cls}_client{cid}_r{r}.pt",
                )
                print(f"[Unlearn-Round {ur}] client {cid} U_c shape: {U_c.shape}")
            except RuntimeError as e:
                # 没有正样本 / 负样本之类情况：直接跳过忘却，沿用 global_model
                print(f"[Unlearn-Round {ur}] client {cid}: build U_c failed ({e}), "
                      f"use global_model without unlearning.")
                client_states[cid] = {k: v.cpu() for k, v in global_model.state_dict().items()}
                continue

            # 1.2 用打乱的 loader 进行本地方向抹除训练
            unlearn_bs = getattr(args, "unlearn_batch_size", max(1, args.batch_size // 2))
            local_loader = DataLoader(
                sub_dst,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                drop_last=False,
            )

            nets[cid] = unlearn_one_class_on_model_vis_projector(
                model=nets[cid],
                dataloader=local_loader,
                forget_cls=forget_cls,
                U=U_c,                     # 本客户端自己的子空间 U_c
                device=device,
                args=args,
                emb_feat=emb_feat,
                clip_model=clip_model,
                epochs=unlearn_epochs,
                lambda_keep=lambda_keep,
                lambda_vis=lambda_dir,
                lr=unlearn_lr,
            )

            # 保存本地忘却后的参数
            client_states[cid] = {k: v.cpu() for k, v in nets[cid].state_dict().items()}

        # ===== 2) FedAvg 聚合所有客户端模型，得到新的 global_model =====
        print(f"[Unlearn-Round {ur}] FedAvg aggregation ...")
        total_data_points = sum(len(partition_idx_map[cid]) for cid in nets.keys())
        fed_avg_freqs = {
            cid: len(partition_idx_map[cid]) / total_data_points
            for cid in nets.keys()
        }

        new_global_state = None
        for i, (cid, state_dict) in enumerate(client_states.items()):
            w = fed_avg_freqs[cid]
            if i == 0:
                new_global_state = {k: v.clone() * w for k, v in state_dict.items()}
            else:
                for k in state_dict:
                    new_global_state[k] += state_dict[k] * w

        global_model.load_state_dict(new_global_state)
        global_model.to(device)
        print(f"[Unlearn-Round {ur}] Round {ur} Done.")

    print("[Vis-Projector] Federated unlearning finished.")
    return global_model
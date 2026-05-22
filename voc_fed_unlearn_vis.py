# voc_fed_unlearn_vis.py

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import clip

from config_args import get_args
from load_data import get_data
from fed_main import init_nets           # 只复用 init_nets，其他我们自己写
import utils.evaluate as evaluate
from run_epoch import run_epoch
from ulearn.perclass_metrics_utils import summarize_before_after
from ulearn.unlearn_utils_projector import (
    unlearn_one_class_on_model_vis_projector,
)

# ======================= 基础设置 =======================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VOC_CLASSES = [
    'Aeroplane',
    'Bicycle',
    'Bird',
    'Boat',
    'Bottle',
    'Bus',
    'Car',
    'Cat',
    'Chair',
    'Cow',
    'Diningtable',
    'Dog',
    'Horse',
    'Motorbike',
    'Person',
    'Pottedplant',
    'Sheep',
    'Sofa',
    'Train',
    'Tvmonitor'
]


def build_clip_embeddings_for_voc(device):
    label_space = VOC_CLASSES

    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    prompt = [f"The photo contains {x}." for x in label_space]
    with torch.no_grad():
        label_text = clip.tokenize(prompt).to(device)
        label_text_features = clip_model.encode_text(label_text)
        label_text_features = label_text_features / label_text_features.norm(dim=1, keepdim=True)

    # state embedding: positive / negative
    state_prompt = ['positive', 'negative']
    state_text = clip.tokenize(state_prompt).to(device)
    with torch.no_grad():
        weight = clip_model.encode_text(state_text)
        weight = weight / weight.norm(dim=1, keepdim=True)
        weight = torch.cat((torch.zeros(512).view(1, -1).to(device), weight), 0)

    return clip_model, label_text_features, weight


# ======================= FedAvg 训练 =======================

def train_one_client(
    args,
    model,
    train_loader,
    device,
    global_model,
    emb_feat,
    clip_model,
):
    """
    极简版单客户端训练：跑 args.epochs 个 epoch。
    """
    model.to(device)

    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

    for ep in range(args.epochs):
        desc = f"[FedAvg] Client train ep {ep}"
        run_epoch(
            args,
            model,
            train_loader,
            optimizer=optimizer,
            epoch=ep,
            desc=desc,
            train=True,
            warmup_scheduler=None,
            global_model=global_model,
            emb_feat=emb_feat,
            clip_model=clip_model,
        )

    model.to("cpu")
    torch.cuda.empty_cache()
    return model


def fedavg_train_voc(
    args,
    global_model,
    nets,
    train_dl_global,
    test_dl_global,
    device,
    emb_feat,
    clip_model,
):
    """
    使用 VOC 数据做 FedAvg 训练，返回训练好的 global_model 和 partition_idx_map。
    """
    n_train = len(train_dl_global.dataset)
    idxs = np.random.permutation(n_train)
    batch_idxs = np.array_split(idxs, args.n_parties)
    partition_idx_map = {i: batch_idxs[i] for i in range(args.n_parties)}

    global_para = global_model.state_dict()

    for comm_round in range(args.comm_round):
        print(f"\n====== FedAvg Round {comm_round} / {args.comm_round} ======")

        # 1) 下发全局参数
        for cid in range(args.n_parties):
            nets[cid].load_state_dict(global_para)

        # 2) 各客户端本地训练
        net_data_count = {}
        for cid in range(args.n_parties):
            sub_dst = Subset(train_dl_global.dataset, partition_idx_map[cid])
            train_dl_local = DataLoader(
                sub_dst,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=False,
            )
            net_data_count[cid] = len(sub_dst)

            print(f"[FedAvg] Train client {cid}, data size = {len(sub_dst)}")
            nets[cid] = train_one_client(
                args,
                nets[cid],
                train_dl_local,
                device,
                global_model,
                emb_feat,
                clip_model,
            )

        # 3) FedAvg 聚合
        total_pts = sum(net_data_count.values())
        freqs = {cid: net_data_count[cid] / total_pts for cid in range(args.n_parties)}

        new_global = None
        for i, cid in enumerate(range(args.n_parties)):
            w = freqs[cid]
            state = nets[cid].state_dict()
            if i == 0:
                new_global = {k: v.clone() * w for k, v in state.items()}
            else:
                for k in state:
                    new_global[k] += state[k] * w

        global_para = new_global
        global_model.load_state_dict(global_para)
        global_model.to(device)

        # 4) 每隔若干轮测试一下
        if (comm_round % 2 == 0) or (comm_round == args.comm_round - 1):
            all_preds, all_targs, all_masks, all_ids, tl, tl_unk = run_epoch(
                args,
                global_model,
                test_dl_global,
                optimizer=None,
                epoch=0,
                desc=f"Testing_round_{comm_round}",
                train=False,
                warmup_scheduler=None,
                global_model=global_model,
                emb_feat=emb_feat,
                clip_model=clip_model,
            )
            test_metrics = evaluate.compute_metrics(
                args,
                all_preds, all_targs, all_masks,
                tl, tl_unk,
                0, 1,
                verbose=False,
            )
            print(
                f"[FedAvg][Round {comm_round}] "
                f"mAP={test_metrics['mAP']:.4f}, "
                f"O_mAP={test_metrics['O_mAP']:.4f}, "
                f"CF1={test_metrics['CF1']:.4f}, "
                f"OF1={test_metrics['OF1']:.4f}"
            )

    global_model.to(device)
    return global_model, partition_idx_map


# ======================= 联邦 PROJECTOR 忘却 =======================

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
    lambda_dir: float = 10.0, # 子空间方向抹除权重（传给 lambda_vis）
):
    """
    极简版联邦 PROJECTOR 可视特征遗忘：

    对每一轮 ur：
      1）server 把 global_model 参数下发到所有客户端；
      2）每个客户端 cid：
            - 用自己的数据 Subset(dataset, partition_idx_map[cid]) 构造局部子空间 U_c；
            - 在 U_c 上做 PROJECTOR 忘却，得到本地 unlearned 模型；
      3）server 对所有客户端模型做一次 FedAvg，更新 global_model。
    """

    from ulearn.projector_subspace import build_vis_subspace_for_class

    for ur in range(unlearn_rounds):
        print(f"\n[Vis-Projector][Unlearn-Round {ur}] Start ...")

        client_states = {}

        # ===== 1) 所有客户端：下发 global_model，局部构造 U_c + 忘却 =====
        for cid in nets.keys():
            print(f"[Unlearn-Round {ur}] client {cid} do vis-projector unlearning ...")

            sub_dst = Subset(train_dl_global.dataset, partition_idx_map[cid])

            # 1.1 用不打乱的 loader 在本客户端上估计 U_c
            U_loader = DataLoader(
                sub_dst,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                drop_last=False,
            )

            nets[cid].load_state_dict(global_model.state_dict())
            nets[cid].to(device)

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
                # 没有正/负样本就跳过：直接沿用 global_model
                print(
                    f"[Unlearn-Round {ur}] client {cid}: build U_c failed ({e}), "
                    f"use global_model without unlearning."
                )
                client_states[cid] = {k: v.cpu() for k, v in global_model.state_dict().items()}
                continue

            # 1.2 用打乱的 loader 进行本地方向抹除训练
            unlearn_bs = getattr(args, "unlearn_batch_size", max(1, args.batch_size // 2))
            local_loader = DataLoader(
                sub_dst,
                batch_size=unlearn_bs,   # ✅ 用专门的 unlearn_bs
                shuffle=True,            # ✅ 念在训练要打乱
                num_workers=args.workers,
                drop_last=False,
            )

            nets[cid] = unlearn_one_class_on_model_vis_projector(
                model=nets[cid],
                dataloader=local_loader,
                forget_cls=forget_cls,
                U=U_c,
                device=device,
                args=args,
                emb_feat=emb_feat,
                clip_model=clip_model,
                epochs=unlearn_epochs,
                lambda_keep=lambda_keep,
                lambda_vis=lambda_dir,
                lr=unlearn_lr,
            )

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


# ======================= 主程序 =======================

if __name__ == "__main__":
    # 1) 基本参数
    args = get_args(argparse.ArgumentParser())
    args.dataset = 'voc'
    args.num_labels = 20
    args.dataroot = '/code/Fed/data'
    args.learn_emb_type = 'clip'
    args.scale_size = 256
    args.crop_size = 224
    args.device = 'cuda:0'

    # 视觉忘却相关兜底超参
    if not hasattr(args, "lambda_vis"):
        args.lambda_vis = 1.0
    if not hasattr(args, "unlearn_epochs"):
        args.unlearn_epochs = 1
    if not hasattr(args, "unlearn_lr"):
        args.unlearn_lr = 1e-4
    if not hasattr(args, "comm_round"):
        args.comm_round = 5          # 你可以根据需要改大一些
    if not hasattr(args, "epochs"):
        args.epochs = 1
    if not hasattr(args, "batch_size"):
        args.batch_size = 16
    if not hasattr(args, "workers"):
        args.workers = 4
    if not hasattr(args, "n_parties"):
        args.n_parties = 5           # 客户端数量，可以根据实验改

    seed = args.init_seed if hasattr(args, "init_seed") else 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"Seed: {seed}")

    # 2) 数据
    train_dl_global, valid_dl_global, test_dl_global = get_data(args)

    # 3) CLIP embedding
    clip_model, label_text_features, state_weight = build_clip_embeddings_for_voc(device)

    # 4) 初始化全局模型 + 客户端模型
    nets, _, _ = init_nets(
        args,
        is_global=False,
        state_weight=state_weight,
        label_weight=label_text_features,
    )
    global_models, _, _ = init_nets(
        args,
        is_global=True,
        state_weight=state_weight,
        label_weight=label_text_features,
    )
    global_model = global_models[0]

    # ========== FedAvg 训练 ==========
    print("==== Start FedAvg training on VOC ====")
    global_model, partition_idx_map = fedavg_train_voc(
        args,
        global_model,
        nets,
        train_dl_global,
        test_dl_global,
        device,
        emb_feat=label_text_features,
        clip_model=clip_model,
    )

    # 训练结束后评估（unlearning 前）
    print("\n==== Evaluate BEFORE federated unlearning ====")
    all_preds_before, all_targs_before, all_masks_before, all_ids_before, tl_b, tl_unk_b = run_epoch(
        args,
        global_model,
        test_dl_global,
        optimizer=None,
        epoch=0,
        desc="Testing_before_unlearn",
        train=False,
        warmup_scheduler=None,
        global_model=global_model,
        emb_feat=label_text_features,
        clip_model=clip_model,
    )
    before_metrics = evaluate.compute_metrics(
        args,
        all_preds_before, all_targs_before, all_masks_before,
        tl_b, tl_unk_b,
        0, 1,
        verbose=False,
    )
    print("mAP(before):   {:.3f}".format(before_metrics['mAP']))
    print("O_mAP(before): {:.3f}".format(before_metrics['O_mAP']))
    print("CF1(before):   {:.3f}".format(before_metrics['CF1']))
    print("OF1(before):   {:.3f}".format(before_metrics['OF1']))

    # ========== 联邦 PROJECTOR 忘却 ==========
    forget_cls = 14  # Person
    print(f"\n==== Start FEDERATED VIS-PROJECTOR unlearning for class {forget_cls} ({VOC_CLASSES[forget_cls]}) ====")

    global_model = federated_unlearn_one_class_vis_projector(
        args=args,
        global_model=global_model,
        nets=nets,
        train_dl_global=train_dl_global,
        partition_idx_map=partition_idx_map,
        device=device,
        emb_feat=label_text_features,
        clip_model=clip_model,
        forget_cls=forget_cls,
        r=32,
        unlearn_rounds=1,
        unlearn_epochs=1,
        unlearn_lr=1e-4,
        lambda_keep=1.0,
        lambda_dir=70.0,   # 建议先用 5~10，太大会把所有类都抹掉
    )

    # ========== 联邦忘却后评估 ==========
    print("\n==== Evaluate AFTER federated unlearning ====")
    all_preds_after, all_targs_after, all_masks_after, all_ids_after, tl_a, tl_unk_a = run_epoch(
        args,
        global_model,
        test_dl_global,
        optimizer=None,
        epoch=0,
        desc="Testing_after_unlearn",
        train=False,
        warmup_scheduler=None,
        global_model=global_model,
        emb_feat=label_text_features,
        clip_model=clip_model,
    )
    after_metrics = evaluate.compute_metrics(
        args,
        all_preds_after, all_targs_after, all_masks_after,
        tl_a, tl_unk_a,
        0, 1,
        verbose=False,
    )
    print("mAP(after):   {:.3f}".format(after_metrics['mAP']))
    print("O_mAP(after): {:.3f}".format(after_metrics['O_mAP']))
    print("CF1(after):   {:.3f}".format(after_metrics['CF1']))
    print("OF1(after):   {:.3f}".format(after_metrics['OF1']))

    # ========== per-class before/after 指标 ==========
    os.makedirs("./ulearn", exist_ok=True)
    csv_path = f"./ulearn/voc_fed_vis_unlearn_cls{forget_cls}_perclass_metrics.csv"

    per_before, per_after = summarize_before_after(
        all_preds_before, all_targs_before,
        all_preds_after,  all_targs_after,
        class_names=VOC_CLASSES,
        forget_cls=forget_cls,
        csv_path=csv_path,
        threshold=0.5,
    )
    print(f"[Per-class] metrics saved to: {csv_path}")

    c = forget_cls
    name = VOC_CLASSES[c]

    ap_before = per_before[c]['AP']
    ap_after = per_after[c]['AP']
    p_before = per_before[c]['P']
    p_after = per_after[c]['P']
    r_before = per_before[c]['R']
    r_after = per_after[c]['R']
    f1_before = per_before[c]['F1']
    f1_after = per_after[c]['F1']

    print(f"\nClass {c} ({name}):")
    print(f"  AP  {ap_before:.3f} -> {ap_after:.3f} (Δ={ap_after - ap_before:+.3f})")
    print(f"  P   {p_before:.3f}  -> {p_after:.3f}  (Δ={p_after - p_before:+.3f})")
    print(f"  R   {r_before:.3f}  -> {r_after:.3f}  (Δ={r_after - r_before:+.3f})")
    print(f"  F1  {f1_before:.3f} -> {f1_after:.3f} (Δ={f1_after - f1_before:+.3f})")
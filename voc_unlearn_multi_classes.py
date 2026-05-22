#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
voc_unlearn_multi_classes.py

在已经训练好的 VOC CTran 全局模型上，对多个类别同时做“特征维度 + logit 遗忘”，
并输出每一类 AP / P / R / F1 的前后变化到 CSV 表格中。
"""

import os
import copy
import argparse
import numpy as np
import csv

import torch
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from tqdm import tqdm

import clip

from config_args import get_args
from load_data import get_data
from models import CTranModel
from run_epoch import run_epoch
import utils.evaluate as evaluate  # 主要用里面的整体指标（可选）


# -------------------------
# VOC 类别名字（与训练顺序一致）
# -------------------------
VOC_CLASSES = [
    'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
    'Bus', 'Car', 'Cat', 'Chair', 'Cow',
    'Diningtable', 'Dog', 'Horse', 'Motorbike', 'Person',
    'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor'
]


# -------------------------
# 1. 构造 VOC 的 CLIP 文本嵌入
# -------------------------
def build_clip_embeddings_for_voc(device, clip_model):
    """
    构造：
    - label_text_features: (20, 512)，每个 VOC 类别的 CLIP text embedding
    - state_weight: (3, 512)，[0 行全 0，占位；1/2 行是 positive/negative]
    """
    # --- label_text_features ---
    label_space = VOC_CLASSES
    prompt = []
    for item in label_space:
        prompt.append(f'The photo contains {item}.')
    with torch.no_grad():
        label_text = clip.tokenize(prompt).to(device)
        label_text_features = clip_model.encode_text(label_text)
        label_text_features = label_text_features / label_text_features.norm(
            dim=1, keepdim=True
        )

    # --- state_weight ---
    state_prompt = ['positive', 'negative']
    state_text = clip.tokenize(state_prompt).to(device)
    with torch.no_grad():
        weight = clip_model.encode_text(state_text)
        weight = weight / weight.norm(dim=1, keepdim=True)
        # 前面加一行全 0（padding_idx=0 用）
        weight = torch.cat((torch.zeros(512).view(1, -1).to(device), weight), 0)

    return label_text_features, weight


# -------------------------
# 2. 单类：统计 top-K 关键维度
# -------------------------
def collect_topk_dims_for_class(
    model,
    dataloader,
    forget_cls: int,        # 要遗忘的类别 index（0~L-1）
    K: int,                 # 选多少个维度，比如 64
    device,
    args,
    emb_feat,
    clip_model,
):
    """
    对单个类别 forget_cls，统计在 label embedding 里最“专属”的 K 个维度：
    score = pos_mean - neg_mean，取 score 最大的 K 维。
    """
    model.eval()
    model.to(device)

    hidden_dim = 512
    pos_sum = torch.zeros(hidden_dim, device=device)
    pos_cnt = 0
    neg_sum = torch.zeros(hidden_dim, device=device)
    neg_cnt = 0

    with torch.no_grad():
        # ⚠️ 注意：这里改成拿到 batch，再从 batch 里取 image/labels/mask
        for batch in tqdm(dataloader, desc=f"Collect top-K stats (cls={forget_cls})"):
            # 你的 run_epoch 里基本是这样：
            if isinstance(batch, dict):
                # 按你 VOC Dataset 的 key 来，通常是这几个：
                images = batch['image'].float().to(device)   # (B, C, H, W)
                labels = batch['labels'].float().to(device)  # (B, L)
                mask   = batch['mask'].float().to(device)    # (B, L)
                # img_ids = batch.get('img_ids', None)  # 如果需要的话
            else:
                # 如果 dataset 返回的是元组 (images, labels, mask, img_ids)
                if len(batch) == 4:
                    images, labels, mask, img_ids = batch
                elif len(batch) == 3:
                    images, labels, mask = batch
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")
                images = images.float().to(device)
                labels = labels.float().to(device)
                mask   = mask.float().to(device)

            mask_in = mask.clone()

            # forward，拿到 label_embeddings
            logits, _, _, label_emb = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,  # ⭐ 你已经在 CTranModel 里加了这个参数
            )
            # label_emb: (B, L, hidden_dim)
            z_f = label_emb[:, forget_cls, :]  # (B, hidden_dim)

            pos_idx = (labels[:, forget_cls] == 1)
            neg_idx = (labels[:, forget_cls] == 0)

            if pos_idx.any():
                z_pos = z_f[pos_idx]
                pos_sum += z_pos.abs().sum(dim=0)
                pos_cnt += z_pos.size(0)

            if neg_idx.any():
                z_neg = z_f[neg_idx]
                neg_sum += z_neg.abs().sum(dim=0)
                neg_cnt += z_neg.size(0)

    pos_mean = pos_sum / (pos_cnt + 1e-6)
    neg_mean = neg_sum / (neg_cnt + 1e-6)
    score = pos_mean - neg_mean

    topk_vals, topk_idx = torch.topk(score, k=K, largest=True)
    return topk_idx, score


# -------------------------
# 3. 多类：封装 top-K 统计
# -------------------------
def collect_topk_dims_for_classes(
    model,
    dataloader,
    forget_cls_list,
    K: int,
    device,
    args,
    emb_feat,
    clip_model,
):
    """
    对多个类别依次调用 collect_topk_dims_for_class，
    返回：
        dims_dict  : {cls_id: np.array(topK_idx)}
        score_dict : {cls_id: torch.Tensor(score)}  # 可用来画图
    """
    if isinstance(forget_cls_list, int):
        forget_cls_list = [forget_cls_list]

    dims_dict = {}
    score_dict = {}

    for c in forget_cls_list:
        print(f"\n[TopK] Collecting class {c} ({VOC_CLASSES[c]}) ...")
        topk_idx, score = collect_topk_dims_for_class(
            model, dataloader, c, K, device, args, emb_feat, clip_model
        )
        dims_dict[c] = topk_idx.detach().cpu().numpy()
        score_dict[c] = score.detach().cpu()
        print(f"Class {c} top-{K} dims: {dims_dict[c]}")

    return dims_dict, score_dict


# -------------------------
# 4. 多类遗忘：unlearn_multi_class_on_model
# -------------------------
def unlearn_multi_class_on_model(
    model,
    global_model_old,
    train_loader,
    forget_classes,
    dims_dict,
    device,
    args,
    emb_feat,
    clip_model,
    alpha=1.0,
    beta=1.0,
    epochs=1,
):
    """
    在单机上对多个类别同时做“特征维度 + logit”遗忘微调。

    loss = alpha * (loss_erase + loss_logit) + beta * keep_loss
    """
    if isinstance(forget_classes, int):
        forget_classes = [forget_classes]

    model.to(device)
    global_model_old.to(device)
    model.train()
    global_model_old.eval()
    for p in global_model_old.parameters():
        p.requires_grad = False

    num_labels = args.num_labels

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, momentum=0.9, weight_decay=1e-4
    )

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Unlearn multi {forget_classes} | epoch {epoch}")
        running_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            # -------- 兼容 dict / tuple 两种 batch 形式 --------
            if isinstance(batch, dict):
                images = batch['image'].float().to(device)   # (B, C, H, W)
                labels = batch['labels'].float().to(device)  # (B, L)
                mask   = batch['mask'].float().to(device)    # (B, L)
                # img_ids = batch.get('img_ids', None)
            else:
                # 如果是 (images, labels, mask, img_ids)
                if len(batch) == 4:
                    images, labels, mask, img_ids = batch
                elif len(batch) == 3:
                    images, labels, mask = batch
                else:
                    raise ValueError(f"Unexpected batch format in unlearn_multi_class_on_model: {type(batch)}")
                images = images.float().to(device)
                labels = labels.float().to(device)
                mask   = mask.float().to(device)
            # -------------------------------------------------

            mask_in = mask.clone()

            optimizer.zero_grad()

            # 新模型前向（带 label_emb）
            preds_new, _, _, label_emb = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )
            # preds_new: (B, L)
            # label_emb: (B, L, hidden_dim=512)

            # 旧模型前向（保护其它类）
            with torch.no_grad():
                preds_old, _, _ = global_model_old(
                    images,
                    mask_in,
                    args.learn_emb_type,
                    emb_feat,
                    clip_model,
                )

            loss_erase = 0.0
            loss_logit = 0.0

            for c in forget_classes:
                cls_emb = label_emb[:, c, :]          # (B, 512)
                dims = torch.as_tensor(
                    dims_dict[c], device=device, dtype=torch.long
                )
                forget_emb = cls_emb[:, dims]         # (B, K)

                # 特征维度往 0 拉
                loss_erase = loss_erase + (forget_emb ** 2).mean()
                # logit 压低
                logits_c = preds_new[:, c]            # (B,)
                loss_logit = loss_logit + logits_c.mean()

            if len(forget_classes) > 0:
                loss_erase = loss_erase / len(forget_classes)
                loss_logit = loss_logit / len(forget_classes)

            # 保护其它类
            keep_classes = [j for j in range(num_labels) if j not in forget_classes]
            if len(keep_classes) > 0:
                keep_classes = torch.as_tensor(
                    keep_classes, device=device, dtype=torch.long
                )
                new_others = preds_new[:, keep_classes]
                old_others = preds_old[:, keep_classes]
                keep_loss = ((new_others - old_others) ** 2).mean()
            else:
                keep_loss = 0.0

            loss_forget = loss_erase + loss_logit
            loss = alpha * loss_forget + beta * keep_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({
                "loss":   f"{running_loss / (batch_idx + 1):.4f}",
                "erase":  f"{loss_erase.item():.4f}",
                "logit":  f"{loss_logit.item():.4f}",
                "keep":   f"{float(keep_loss):.4f}",
            })

    return model


# -------------------------
# 5. 计算每类 AP / P / R / F1
# -------------------------
def compute_per_class_metrics_from_raw(all_preds, all_targs, threshold=0.5):
    """
    使用 sigmoid(logits) + sklearn 来计算每个类别的
    AP, Precision, Recall, F1 （都是 per-class）。
    """
    probs = torch.sigmoid(all_preds).detach().cpu().numpy()
    targs = all_targs.detach().cpu().numpy()
    num_classes = targs.shape[1]

    AP = np.zeros(num_classes)
    P = np.zeros(num_classes)
    R = np.zeros(num_classes)
    F1 = np.zeros(num_classes)

    for c in range(num_classes):
        y_true = targs[:, c]
        y_score = probs[:, c]

        # AP
        if y_true.max() == y_true.min():
            AP[c] = np.nan  # 这个类全是 0 或全是 1，AP 不好定义
        else:
            AP[c] = average_precision_score(y_true, y_score)

        # 二值预测
        y_pred = (y_score >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        P[c], R[c], F1[c] = p, r, f1

    return AP, P, R, F1


# -------------------------
# 6. 主流程
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    args = get_args(parser)

    # 确保是 VOC
    args.dataset = 'voc'
    args.num_labels = 20
    args.test_known = 0

    # ======= 这里改你要忘的类别 =======
    forget_classes = [1, 14]
    # top-K 维度数
    K = 64
    # 遗忘微调轮数
    unlearn_epochs = 1
    # =================================

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Use device: {device}")

    # 随机种子
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 数据
    train_dl_global, valid_dl_global, test_dl_global = get_data(args)

    # CLIP 与文本嵌入
    clip_model, _ = clip.load("ViT-B/16", device=device)
    label_text_features, state_weight = build_clip_embeddings_for_voc(device, clip_model)

    # 模型
    model = CTranModel(
        args.num_labels,
        args.use_lmt,
        args.pos_emb,
        args.layers,
        args.heads,
        args.dropout,
        args.no_x_features,
        state_weight=state_weight,
        label_weight=label_text_features,
    ).to(device)

    # 载入全局模型 ckpt
    assert args.ckpt_path != '', "请通过 --ckpt_path 指定全局模型路径"
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Loaded checkpoint from {args.ckpt_path}")

    # 旧模型（保护用）
    global_model_old = copy.deepcopy(model)
    for p in global_model_old.parameters():
        p.requires_grad = False
    global_model_old.eval()

    # ========= 1) 遗忘前评估 =========
    print("==== Before unlearning ====")
    all_preds_b, all_targs_b, all_masks_b, all_ids_b, test_loss_b, test_loss_unk_b = run_epoch(
        args, model, test_dl_global, None, 1, 'Testing',
        global_model=model, emb_feat=label_text_features, clip_model=clip_model
    )
    # 整体指标（可选）
    test_metrics_b = evaluate.compute_metrics(
        args, all_preds_b, all_targs_b, all_masks_b,
        test_loss_b, test_loss_unk_b, 0, 1, verbose=False
    )
    print(f"mAP(before):   {test_metrics_b['mAP']:.3f}")
    print(f"O_mAP(before): {test_metrics_b['O_mAP']:.3f}")
    print(f"CF1(before):   {test_metrics_b['CF1']:.3f}")
    print(f"OF1(before):   {test_metrics_b['OF1']:.3f}")

    AP_b, P_b, R_b, F1_b = compute_per_class_metrics_from_raw(all_preds_b, all_targs_b)

    # ========= 2) 统计多类 top-K 维度 =========
    print(f"Collecting top-{K} dims for classes: {forget_classes}")
    dims_dict, score_dict = collect_topk_dims_for_classes(
        model, train_dl_global, forget_classes, K,
        device, args, emb_feat=label_text_features, clip_model=clip_model
    )

    # ========= 3) 多类遗忘 =========
    print("==== Start multi-class unlearning ====")
    model = unlearn_multi_class_on_model(
        model,
        global_model_old,
        train_dl_global,
        forget_classes,
        dims_dict,
        device,
        args,
        emb_feat=label_text_features,
        clip_model=clip_model,
        alpha=1.0,
        beta=1.0,
        epochs=unlearn_epochs,
    )

    # ========= 4) 遗忘后评估 =========
    print("==== After unlearning ====")
    all_preds_a, all_targs_a, all_masks_a, all_ids_a, test_loss_a, test_loss_unk_a = run_epoch(
        args, model, test_dl_global, None, 1, 'Testing',
        global_model=model, emb_feat=label_text_features, clip_model=clip_model
    )
    test_metrics_a = evaluate.compute_metrics(
        args, all_preds_a, all_targs_a, all_masks_a,
        test_loss_a, test_loss_unk_a, 0, 1, verbose=False
    )
    print(f"mAP(after):   {test_metrics_a['mAP']:.3f}")
    print(f"O_mAP(after): {test_metrics_a['O_mAP']:.3f}")
    print(f"CF1(after):   {test_metrics_a['CF1']:.3f}")
    print(f"OF1(after):   {test_metrics_a['OF1']:.3f}")

    AP_a, P_a, R_a, F1_a = compute_per_class_metrics_from_raw(all_preds_a, all_targs_a)

    # ========= 5) 写 CSV：每类前后变化 =========
    os.makedirs(args.results_dir, exist_ok=True)
    forget_tag = "_".join(str(c) for c in forget_classes)
    csv_path = os.path.join(
        args.results_dir,
        f"voc_multi_unlearn_cls_{forget_tag}.csv"
    )
    print(f"Saving per-class metrics diff to: {csv_path}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "cls_idx", "cls_name",
            "AP_before", "AP_after", "dAP",
            "P_before",  "P_after",  "dP",
            "R_before",  "R_after",  "dR",
            "F1_before", "F1_after", "dF1",
        ])
        num_classes = len(VOC_CLASSES)
        for i in range(num_classes):
            name = VOC_CLASSES[i]
            apb, apa = AP_b[i], AP_a[i]
            pb, pa = P_b[i], P_a[i]
            rb, ra = R_b[i], R_a[i]
            f1b, f1a = F1_b[i], F1_a[i]

            def fmt(x):
                return "" if np.isnan(x) else f"{x:.3f}"

            writer.writerow([
                i, name,
                fmt(apb), fmt(apa), fmt(apa - apb if not np.isnan(apb) and not np.isnan(apa) else np.nan),
                fmt(pb),  fmt(pa),  fmt(pa - pb),
                fmt(rb),  fmt(ra),  fmt(ra - rb),
                fmt(f1b), fmt(f1a), fmt(f1a - f1b),
            ])

    print("Done.")


if __name__ == '__main__':
    main()
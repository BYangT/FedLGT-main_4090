#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对已训练好的 VOC 联邦全局模型做“按类别遗忘”（只动分类头，不用蒸馏）。

用法示例：
  # 忘掉 Person 类（名字）
  python voc_unlearn_one_class.py \
      --dataset voc --dataroot /code/Fed/data \
      --ckpt_path /code/Fed/results/xxx/voc.3layer....round_40.pt \
      --forget_name Person \
      --unlearn_epochs 3 --unlearn_lr 1e-4

  # 忘掉 index=14 的类别（等价于 Person）
  python voc_unlearn_one_class.py \
      --dataset voc --dataroot /code/Fed/data \
      --ckpt_path /code/Fed/results/xxx/voc.3layer....round_40.pt \
      --forget_idx 14
"""

import os
import csv
import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

import clip

from config_args import get_args
from load_data import get_data
from models import CTranModel
from run_epoch import run_epoch
import utils.evaluate as evaluate


# VOC 20 类名字（一定要和你训练时的顺序一致）
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
    """
    按 fed_main 里的写法，构造：
    - label_text_features: (20, 512)
    - state_weight: (3, 512)  [0 行是全 0，占位；1/2 行是 positive/negative]
    """
    # --- label_text_features ---
    label_space = VOC_CLASSES
    prompt = []
    for item in label_space:
        prompt.append(f'The photo contains {item}.')
    with torch.no_grad():
        label_text = clip.tokenize(prompt).to(device)
        label_text_features = clip_model.encode_text(label_text)
        label_text_features = label_text_features / label_text_features.norm(dim=1, keepdim=True)

    # --- state_weight ---
    state_prompt = ['positive', 'negative']
    state_text = clip.tokenize(state_prompt).to(device)
    with torch.no_grad():
        weight = clip_model.encode_text(state_text)
        weight = weight / weight.norm(dim=1, keepdim=True)
        # 前面加一行全 0（padding_idx=0 用）
        weight = torch.cat((torch.zeros(512).view(1, -1).to(device), weight), 0)
    return label_text_features, weight


def compute_per_class_metrics(y_true, y_logits, threshold=0.5):
    """
    计算每个类别的 AP / Precision / Recall / F1
    y_true:   (N, C) 0/1
    y_logits: (N, C) 原始 logits
    返回 dict: {class_idx: {'AP_before':..., ...}} 在外层脚本控制命名
    """
    y_true = y_true.astype(np.int32)
    # 概率
    y_score = 1.0 / (1.0 + np.exp(-y_logits))  # sigmoid

    N, C = y_true.shape
    metrics = []

    for c in range(C):
        t = y_true[:, c]
        s = y_score[:, c]

        # --- AP ---
        # 如果该类在 GT 中一个正样本都没有，就设为 0
        if t.sum() == 0:
            ap = 0.0
        else:
            # 按分数从大到小排序
            order = np.argsort(-s)
            t_sorted = t[order]

            tp = (t_sorted == 1).astype(np.float64)
            fp = (t_sorted == 0).astype(np.float64)
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)

            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
            # 只在真正是正样本的位置累加 precision
            ap = (precision * tp).sum() / max(tp.sum(), 1.0)

        # --- P / R / F1（用 0.5 阈值二值化） ---
        pred = (s >= threshold).astype(np.int32)

        tp = np.logical_and(pred == 1, t == 1).sum()
        fp = np.logical_and(pred == 1, t == 0).sum()
        fn = np.logical_and(pred == 0, t == 1).sum()

        if tp + fp == 0:
            prec = 0.0
        else:
            prec = tp / (tp + fp)
        if tp + fn == 0:
            rec = 0.0
        else:
            rec = tp / (tp + fn)
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        metrics.append((ap, prec, rec, f1))

    return metrics  # list of (AP, P, R, F1) 按类索引顺序


if __name__ == "__main__":
    # 1) 先在 parser 里加“遗忘相关”的参数，再交给 get_args 扩展
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget_name", type=str, default=None,
                        help="要遗忘的 VOC 类名（如 Person）。若同时给了 forget_idx，以 forget_idx 为准。")
    parser.add_argument("--forget_idx", type=int, default=None,
                        help="要遗忘的类别 index（0-19）。")
    parser.add_argument("--unlearn_epochs", type=int, default=3,
                        help="遗忘微调的 epoch 数（只训分类头）。")
    parser.add_argument("--unlearn_lr", type=float, default=1e-4,
                        help="遗忘阶段的学习率。")
    parser.add_argument("--out_csv", type=str, default="voc_unlearn_per_class.csv",
                        help="保存每类指标的 csv 文件名（会放到 results_dir 下）。")

    # eval=True 避免 get_args 里询问是否 overwrite 等逻辑
    args = get_args(parser, eval=True)

    # 强制使用 VOC
    args.dataset = "voc"
    args.num_labels = 20

    if args.ckpt_path is None or args.ckpt_path == "":
        raise ValueError("请通过 --ckpt_path 指定已训练好的 VOC 全局模型路径 (.pt 文件)。")

    # 2) 确定要忘哪一类
    if args.forget_idx is not None:
        forget_cls = int(args.forget_idx)
    elif args.forget_name is not None:
        if args.forget_name not in VOC_CLASSES:
            raise ValueError(f"forget_name={args.forget_name} 不在 VOC_CLASSES 里，请检查拼写。")
        forget_cls = VOC_CLASSES.index(args.forget_name)
    else:
        raise ValueError("请至少指定 --forget_idx 或 --forget_name 之一。")

    forget_name = VOC_CLASSES[forget_cls]
    print(f"[INFO] 将遗忘的类别: idx={forget_cls}, name={forget_name}")

    # 3) 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 4) 数据加载（全局 DataLoader）
    print("[INFO] 加载 VOC 数据...")
    train_dl_global, valid_dl_global, test_dl_global = get_data(args)

    # 5) 加载 CLIP 模型（和 fed_main 一致）
    print("[INFO] 加载 CLIP 模型并构造文本特征...")
    global clip_model  # 为了在 build_clip_embeddings_for_voc 中使用
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    label_text_features, state_weight = build_clip_embeddings_for_voc(device)

    # 6) 构建 CTran 模型并加载 ckpt
    print("[INFO] 构建 CTran 模型并加载权重...")
    model = CTranModel(
        args.num_labels,
        args.use_lmt,
        args.pos_emb,
        args.layers,
        args.heads,
        args.dropout,
        args.no_x_features,
        state_weight=state_weight,
        label_weight=label_text_features
    ).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 复制一份“遗忘前”的模型用于对比（teacher-only for evaluation, 不参与训练）
    model_before = copy.deepcopy(model).to(device)
    model_before.eval()

    # 7) 遗忘前：在 test_dl_global 上评估一次整体指标 & 收集 per-class
    print("[INFO] 遗忘前评估...")
    all_preds_b, all_targs_b, all_masks_b, all_ids_b, test_loss_b, test_loss_unk_b = run_epoch(
        args, model_before, test_dl_global, None, 0, 'Testing-Before',
        train=False, global_model=model_before, emb_feat=label_text_features, clip_model=clip_model
    )
    metrics_before_global = evaluate.compute_metrics(
        args, all_preds_b, all_targs_b, all_masks_b, test_loss_b, test_loss_unk_b, 0, 1, verbose=False
    )
    print("[Before] mAP={:.3f}, O_mAP={:.3f}, CF1={:.3f}, OF1={:.3f}".format(
        metrics_before_global['mAP'], metrics_before_global['O_mAP'],
        metrics_before_global['CF1'], metrics_before_global['OF1']
    ))

    # per-class：转成 numpy
    y_true_before = all_targs_b.numpy()
    y_logits_before = all_preds_b.numpy()
    per_class_before = compute_per_class_metrics(y_true_before, y_logits_before)

    # 8) 遗忘阶段：只训练分类头（output_linear）
    # 8) 遗忘阶段：只训练分类头
    print("[INFO] 开始遗忘微调（只训练分类头）...")
    model.train()
    # 先全部冻结
    for name, p in model.named_parameters():
        p.requires_grad = False

    # 再根据 head 类型只解冻对应部分
    if getattr(model, "use_ml_head", False):
        print("[INFO] use_ml_head=True，只解冻 decoder 作为分类头。")
        for p in model.decoder.parameters():
            p.requires_grad = True
    else:
        print("[INFO] use_ml_head=False，只解冻 output_linear 作为分类头。")
        for p in model.output_linear.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.unlearn_lr
    )

    lambda_forget = 1.0
    lambda_keep = 1.0

    for epoch in range(args.unlearn_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_dl_global:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device).float()
            mask = batch['mask'].to(device)

            # CTran 前向
            logits, _, _ = model(images, mask, args.learn_emb_type, label_text_features, clip_model)

            # 构造“修改后的标签”：目标类强制为 0
            targets_mod = labels.clone()
            targets_mod[:, forget_cls] = 0.0

            bce_all = F.binary_cross_entropy_with_logits(
                logits, targets_mod, reduction='none'
            )  # (B, C)

            # 其它类正常训练
            keep_idx = [i for i in range(args.num_labels) if i != forget_cls]
            loss_keep = bce_all[:, keep_idx].mean()

            # 对原先 label=1 的样本，强行把目标类压到 0（反向学习）
            pos_idx = (labels[:, forget_cls] == 1)
            if pos_idx.any():
                loss_forget = bce_all[pos_idx, forget_cls].mean()
            else:
                loss_forget = torch.tensor(0.0, device=device)

            loss = lambda_forget * loss_forget + lambda_keep * loss_keep

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"[Epoch {epoch}] unlearn_loss = {epoch_loss / max(n_batches,1):.4f}")

    # 9) 遗忘后评估
    print("[INFO] 遗忘后评估...")
    model.eval()
    all_preds_a, all_targs_a, all_masks_a, all_ids_a, test_loss_a, test_loss_unk_a = run_epoch(
        args, model, test_dl_global, None, 0, 'Testing-After',
        train=False, global_model=model, emb_feat=label_text_features, clip_model=clip_model
    )
    metrics_after_global = evaluate.compute_metrics(
        args, all_preds_a, all_targs_a, all_masks_a, test_loss_a, test_loss_unk_a, 0, 1, verbose=False
    )
    print("[After ] mAP={:.3f}, O_mAP={:.3f}, CF1={:.3f}, OF1={:.3f}".format(
        metrics_after_global['mAP'], metrics_after_global['O_mAP'],
        metrics_after_global['CF1'], metrics_after_global['OF1']
    ))

    y_true_after = all_targs_a.numpy()
    y_logits_after = all_preds_a.numpy()
    per_class_after = compute_per_class_metrics(y_true_after, y_logits_after)

    # 10) 写 CSV 表格：每行一个类别，包含前/后的 AP, P, R, F1
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_csv)

    print(f"[INFO] 将 per-class 指标写入 {out_path}")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_idx", "class_name",
            "AP_before", "AP_after",
            "P_before", "P_after",
            "R_before", "R_after",
            "F1_before", "F1_after"
        ])
        for c in range(args.num_labels):
            name = VOC_CLASSES[c]
            ap_b, p_b, r_b, f1_b = per_class_before[c]
            ap_a, p_a, r_a, f1_a = per_class_after[c]

            writer.writerow([
                c, name,
                f"{ap_b:.3f}", f"{ap_a:.3f}",
                f"{p_b:.3f}", f"{p_a:.3f}",
                f"{r_b:.3f}", f"{r_a:.3f}",
                f"{f1_b:.3f}", f"{f1_a:.3f}",
            ])

    print("[INFO] 完成：遗忘类别 = {} (idx={})".format(forget_name, forget_cls))
import argparse
import os
import csv

import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from config_args import get_args
from load_data import get_data
from fed_main import init_nets
from run_epoch import run_epoch
import utils.evaluate as evaluate

import clip


# ----- VOC 20 类名称 -----
VOC_CLASSES = ['Aeroplane',
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
               'Tvmonitor']


def build_clip_embeddings_for_voc(device):
    """
    仿照 fed_main 里对 VOC 的写法，构造：
    - label_text_features: (20, 512)
    - state_weight: (3, 512)  [0 行是全 0，占位；1/2 行是 positive/negative]
    """
    label_space = VOC_CLASSES

    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    # label_text_features
    prompt = [f"The photo contains {item}." for item in label_space]
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


def compute_per_class_metrics(all_preds_tensor, all_targs_tensor, threshold=0.5):
    """
    根据 logits + targets，计算每个类别的 AP / P / R / F1
    返回一个列表，长度 = num_labels，每个元素是 dict。
    """
    # all_preds 是 logits，先过 sigmoid 变成概率
    probs = torch.sigmoid(all_preds_tensor).detach().cpu().numpy()   # (N, L)
    targets = all_targs_tensor.detach().cpu().numpy()                # (N, L)

    num_labels = probs.shape[1]
    per_class = []

    for c in range(num_labels):
        y_true = targets[:, c]
        y_score = probs[:, c]

        # AP
        if y_true.sum() == 0:
            ap = 0.0
        else:
            ap = average_precision_score(y_true, y_score)

        # 概率 → 0/1 预测
        y_pred = (y_score >= threshold).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        per_class.append({
            "AP":  float(ap),
            "P":   float(p),
            "R":   float(r),
            "F1":  float(f1),
        })

    return per_class


def save_before_after_csv(per_before, per_after, csv_path, forget_cls=None):
    """
    per_before / per_after: compute_per_class_metrics 的返回值
    csv_path: 要保存的 csv 路径
    forget_cls: 要遗忘的类别 index，可选，用来标一下
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cls_idx", "cls_name",
            "AP_before", "AP_after", "ΔAP",
            "P_before",  "P_after",  "ΔP",
            "R_before",  "R_after",  "ΔR",
            "F1_before", "F1_after", "ΔF1",
            "is_forget_cls"
        ])

        num_labels = len(per_before)
        for c in range(num_labels):
            name = VOC_CLASSES[c] if c < len(VOC_CLASSES) else f"class_{c}"
            b = per_before[c]
            a = per_after[c]

            writer.writerow([
                c, name,
                f"{b['AP']:.3f}",  f"{a['AP']:.3f}",  f"{(a['AP']  - b['AP']):+.3f}",
                f"{b['P']:.3f}",   f"{a['P']:.3f}",   f"{(a['P']   - b['P']):+.3f}",
                f"{b['R']:.3f}",   f"{a['R']:.3f}",   f"{(a['R']   - b['R']):+.3f}",
                f"{b['F1']:.3f}",  f"{a['F1']:.3f}",  f"{(a['F1']  - b['F1']):+.3f}",
                1 if (forget_cls is not None and c == forget_cls) else 0
            ])


if __name__ == "__main__":
    # ====== 1. 基本配置 ======
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_args(argparse.ArgumentParser())
    args.dataset = 'voc'
    args.num_labels = 20
    args.dataroot = '/code/Fed/data'   # ⚠️ 若路径不同，请改这里
    args.device = str(device)

    # 这里用你训练时的设置：ctran / clip 都可以，保持一致就行
    # 比如你之前是 ctran：
    #   Current embedding use:ctran
    # 那就保持默认的 ctran
    # args.learn_emb_type = 'ctran'

    # ====== 2. 数据：测试集 loader ======
    train_dl_global, valid_dl_global, test_dl_global = get_data(args)

    # ====== 3. CLIP embedding（与 fed_main 一致） ======
    clip_model, label_text_features, state_weight = build_clip_embeddings_for_voc(device)

    # ====== 4. 初始化两个模型（before / after）并加载各自 ckpt ======
    # ⚠️ 这里改成你自己的两个模型路径：
    ckpt_before_path = "/code/Fed/results/your_before_model.pt"
    ckpt_after_path  = "/code/Fed/results/your_after_unlearn_model.pt"

    # is_global=True → 只建一个全局模型
    nets_before, _, _ = init_nets(
        args,
        is_global=True,
        state_weight=state_weight,
        label_weight=label_text_features
    )
    model_before = nets_before[0]
    ckpt_before = torch.load(ckpt_before_path, map_location=device)
    model_before.load_state_dict(ckpt_before['state_dict'])
    model_before.to(device)

    nets_after, _, _ = init_nets(
        args,
        is_global=True,
        state_weight=state_weight,
        label_weight=label_text_features
    )
    model_after = nets_after[0]
    ckpt_after = torch.load(ckpt_after_path, map_location=device)
    model_after.load_state_dict(ckpt_after['state_dict'])
    model_after.to(device)

    # ====== 5. 分别跑一遍测试，拿到 logits / targets ======
    print("==== Evaluate BEFORE unlearning model ====")
    all_preds_before, all_targs_before, all_masks_before, all_ids_before, test_loss_before, test_loss_unk_before = run_epoch(
        args,
        model_before,
        test_dl_global,
        optimizer=None,
        epoch=0,
        desc='Testing_before',
        train=False,
        warmup_scheduler=None,
        global_model=model_before,
        emb_feat=label_text_features,
        clip_model=clip_model
    )
    before_metrics = evaluate.compute_metrics(
        args,
        all_preds_before, all_targs_before, all_masks_before,
        test_loss_before, test_loss_unk_before,
        0, 1,
        verbose=False
    )
    print("mAP(before):   {:.3f}".format(before_metrics['mAP']))
    print("O_mAP(before): {:.3f}".format(before_metrics['O_mAP']))
    print("CF1(before):   {:.3f}".format(before_metrics['CF1']))
    print("OF1(before):   {:.3f}".format(before_metrics['OF1']))

    print("==== Evaluate AFTER unlearning model ====")
    all_preds_after, all_targs_after, all_masks_after, all_ids_after, test_loss_after, test_loss_unk_after = run_epoch(
        args,
        model_after,
        test_dl_global,
        optimizer=None,
        epoch=0,
        desc='Testing_after',
        train=False,
        warmup_scheduler=None,
        global_model=model_after,
        emb_feat=label_text_features,
        clip_model=clip_model
    )
    after_metrics = evaluate.compute_metrics(
        args,
        all_preds_after, all_targs_after, all_masks_after,
        test_loss_after, test_loss_unk_after,
        0, 1,
        verbose=False
    )
    print("mAP(after):   {:.3f}".format(after_metrics['mAP']))
    print("O_mAP(after): {:.3f}".format(after_metrics['O_mAP']))
    print("CF1(after):   {:.3f}".format(after_metrics['CF1']))
    print("OF1(after):   {:.3f}".format(after_metrics['OF1']))

    # ====== 6. 计算所有类别的 AP / P / R / F1（before & after）并写 CSV ======
    per_before = compute_per_class_metrics(all_preds_before, all_targs_before, threshold=0.5)
    per_after  = compute_per_class_metrics(all_preds_after,  all_targs_after,  threshold=0.5)

    # 你遗忘的那个类别 index（比如 Person=14）
    forget_cls = 14

    csv_path = f"voc_perclass_before_after_cls{forget_cls}.csv"
    save_before_after_csv(per_before, per_after, csv_path, forget_cls=forget_cls)
    print(f"Per-class metrics saved to: {csv_path}")

    # 顺便打印一下遗忘类那一行
    c = forget_cls
    print(f"Class {c} ({VOC_CLASSES[c]}):")
    print(
        f"  AP  {per_before[c]['AP']:.3f} -> {per_after[c]['AP']:.3f} "
        f"(Δ={per_after[c]['AP'] - per_before[c]['AP']:+.3f})"
    )
    print(
        f"  P   {per_before[c]['P']:.3f} -> {per_after[c]['P']:.3f} "
        f"(Δ={per_after[c]['P'] - per_before[c]['P']:+.3f})"
    )
    print(
        f"  R   {per_before[c]['R']:.3f} -> {per_after[c]['R']:.3f} "
        f"(Δ={per_after[c]['R'] - per_before[c]['R']:+.3f})"
    )
    print(
        f"  F1  {per_before[c]['F1']:.3f} -> {per_after[c]['F1']:.3f} "
        f"(Δ={per_after[c]['F1'] - per_before[c]['F1']:+.3f})"
    )
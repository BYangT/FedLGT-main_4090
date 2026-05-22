# voc_unlearn_one_class_topklogit.py

import argparse
import torch
import numpy as np
import clip
import os

from config_args import get_args
from load_data import get_data
from fed_main import init_nets
import utils.evaluate as evaluate
from run_epoch import run_epoch
from ulearn.perclass_metrics_utils import summarize_before_after

from ulearn.unlearn_utils_topklogit import (
    collect_topk_dims_for_class,
    unlearn_one_class_on_model_topk_logit,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VOC_CLASSES = [
    'Aeroplane','Bicycle','Bird','Boat','Bottle',
    'Bus','Car','Cat','Chair','Cow',
    'Diningtable','Dog','Horse','Motorbike','Person',
    'Pottedplant','Sheep','Sofa','Train','Tvmonitor'
]

def build_clip_embeddings_for_voc(device):
    label_space = VOC_CLASSES
    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    prompt = [f"The photo contains {x}." for x in label_space]
    with torch.no_grad():
        label_text = clip.tokenize(prompt).to(device)
        label_text_features = clip_model.encode_text(label_text)
        label_text_features = label_text_features / label_text_features.norm(dim=1, keepdim=True)

    state_prompt = ['positive', 'negative']
    state_text = clip.tokenize(state_prompt).to(device)
    with torch.no_grad():
        weight = clip_model.encode_text(state_text)
        weight = weight / weight.norm(dim=1, keepdim=True)
        weight = torch.cat((torch.zeros(512).view(1, -1).to(device), weight), 0)

    return clip_model, label_text_features, weight


if __name__ == "__main__":
    args = get_args(argparse.ArgumentParser())
    args.dataset = 'voc'
    args.num_labels = 20
    args.train_known_labels = args.num_labels
    args.dataroot = '/code/Fed/data'
    args.learn_emb_type = 'clip'
    args.scale_size = 256
    args.crop_size = 224
    args.device = 'cuda:0'

    # 数据
    train_dl_global, valid_dl_global, test_dl_global = get_data(args)

    # CLIP
    clip_model, label_text_features, state_weight = build_clip_embeddings_for_voc(device)

    # 模型
    nets, _, _ = init_nets(
        args,
        is_global=True,
        state_weight=state_weight,
        label_weight=label_text_features
    )
    model = nets[0]

    ckpt_path = "/code/Fed/results/voc.3layer.bsz_16.adam0.0001.clip_embagg_avgcoarse_prompt_concat.lmt.unk_lossround_40.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)

    # ===== 遗忘前评估 =====
    print("==== Before unlearning ====")
    all_preds_before, all_targs_before, all_masks_before, all_ids_before, \
        test_loss_before, test_loss_unk_before = run_epoch(
            args,
            model,
            test_dl_global,
            optimizer=None,
            epoch=0,
            desc='Testing_before',
            train=False,
            warmup_scheduler=None,
            global_model=model,
            emb_feat=label_text_features,
            clip_model=clip_model
        )
    before_metrics = evaluate.compute_metrics(
        args,
        all_preds_before, all_targs_before, all_masks_before,
        test_loss_before, test_loss_unk_before,
        0, 1, verbose=False
    )
    print(f"mAP(before):   {before_metrics['mAP']:.3f}")
    print(f"O_mAP(before): {before_metrics['O_mAP']:.3f}")

    # ===== 选择要遗忘的类 & 统计 top-K 维 =====
    forget_cls = 14   # Person
    K = 64
    print(f"==== Collect top-{K} dims for class {forget_cls} ({VOC_CLASSES[forget_cls]}) ====")

    topk_idx, score = collect_topk_dims_for_class(
        model=model,
        dataloader=train_dl_global,
        forget_cls=forget_cls,
        K=K,
        device=device,
        args=args,
        emb_feat=label_text_features,
        clip_model=clip_model,
    )
    print("Top-K dims:", topk_idx.cpu().numpy())

    # ===== 真正执行遗忘 =====
    print("==== Start TOPK+LOGIT unlearning ====")
    model = unlearn_one_class_on_model_topk_logit(
        model=model,
        dataloader=train_dl_global,
        forget_cls=forget_cls,
        topk_idx=topk_idx,
        device=device,
        args=args,
        emb_feat=label_text_features,
        clip_model=clip_model,
        epochs=1,               # 可以先 1~2
        lambda_keep=1.0,
        lambda_forget_logit=20, # 这个可以调大一点
        lambda_forget_feat=1.0,
        lr=1e-4,
    )

    # ===== 遗忘后评估 =====
    print("==== After unlearning ====")
    all_preds_after, all_targs_after, all_masks_after, all_ids_after, \
        test_loss_after, test_loss_unk_after = run_epoch(
            args,
            model,
            test_dl_global,
            optimizer=None,
            epoch=0,
            desc='Testing_after',
            train=False,
            warmup_scheduler=None,
            global_model=model,
            emb_feat=label_text_features,
            clip_model=clip_model
        )
    after_metrics = evaluate.compute_metrics(
        args,
        all_preds_after, all_targs_after, all_masks_after,
        test_loss_after, test_loss_unk_after,
        0, 1, verbose=False
    )
    print(f"mAP(after):   {after_metrics['mAP']:.3f}")
    print(f"O_mAP(after): {after_metrics['O_mAP']:.3f}")

    # ===== per-class 报表 =====
    csv_path = f"./ulearn/voc_topklogit_unlearn_cls{forget_cls}_perclass_metrics.csv"
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
    ap_after  = per_after[c]['AP']
    p_before  = per_before[c]['P']
    p_after   = per_after[c]['P']
    r_before  = per_before[c]['R']
    r_after   = per_after[c]['R']
    f1_before = per_before[c]['F1']
    f1_after  = per_after[c]['F1']

    print(f"Class {c} ({name}):")
    print(f"  AP  {ap_before:.3f} -> {ap_after:.3f} (Δ={ap_after - ap_before:+.3f})")
    print(f"  P   {p_before:.3f}  -> {p_after:.3f}  (Δ={p_after - p_before:+.3f})")
    print(f"  R   {r_before:.3f}  -> {r_after:.3f}  (Δ={r_after - r_before:+.3f})")
    print(f"  F1  {f1_before:.3f} -> {f1_after:.3f} (Δ={f1_after - f1_before:+.3f})")
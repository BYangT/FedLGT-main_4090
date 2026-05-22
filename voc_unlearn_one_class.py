# voc_unlearn_one_class_vis.py  （建议新建这个文件）

import argparse
import torch
import numpy as np

from ulearn.unlearn_utils_projector import unlearn_one_class_on_model_vis_projector
from ulearn.unlearn_utils_up import (
    collect_topk_dims_for_class_up,
    unlearn_one_class_on_model_vis_up, FeatureProjector,
)
from config_args import get_args
from load_data import get_data
from fed_main import init_nets
import utils.evaluate as evaluate
from run_epoch import run_epoch
from ulearn.perclass_metrics_utils import summarize_before_after
import clip
import os


# ⭐ 新的工具函数：视觉特征版 topK + 遗忘

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

    # 一些视觉特征忘却相关的超参（如果 config 里没有，就在这里兜底）
    if not hasattr(args, "alpha_shrink_vis"):
        args.alpha_shrink_vis = 1.0      # 先纯 shrink 到 0
    if not hasattr(args, "beta_center_vis"):
        args.beta_center_vis = 0.0      # 暂时不用负样本中心
    if not hasattr(args, "lambda_vis"):
        args.lambda_vis = 1.0
    if not hasattr(args, "unlearn_epochs"):
        args.unlearn_epochs = 1
    if not hasattr(args, "unlearn_lr"):
        args.unlearn_lr = 1e-4

    # 2) 数据
    train_dl_global, valid_dl_global, test_dl_global = get_data(args)

    # 3) CLIP embedding（保持和 fed_main 一致）
    clip_model, label_text_features, state_weight = build_clip_embeddings_for_voc(device)

    # 4) 初始化一个“全局模型”，并加载你已经训练好的 VOC 模型
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

    from_dim = 512
    to_dim = 1024  # 可以先试 1024 / 2048，看 GPU 吃不吃得消
    projector = FeatureProjector(in_dim=from_dim, out_dim=to_dim).to(device)

    # 5) 遗忘前表现
    print("==== Before unlearning ====")
    all_preds_before, all_targs_before, all_masks_before, all_ids_before, test_loss_before, test_loss_unk_before = run_epoch(
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
        0, 1,
        verbose=False
    )
    print("mAP(before):   {:.3f}".format(before_metrics['mAP']))
    print("O_mAP(before): {:.3f}".format(before_metrics['O_mAP']))
    print("CF1(before):   {:.3f}".format(before_metrics['CF1']))
    print("OF1(before):   {:.3f}".format(before_metrics['OF1']))

    # ==== 6) 选择要遗忘的类 ====
    forget_cls = 14  # Person
    print("==== Start VIS-PROJECTOR unlearning ====")
    U_ckpt = torch.load("./ulearn/ulearn/U_vis_person_rank8.pt", map_location=device)
    U = U_ckpt["U"]  # (D, r)

    model = unlearn_one_class_on_model_vis_projector(
        model=model,
        dataloader=train_dl_global,
        forget_cls=forget_cls,
        device=device,
        args=args,
        emb_feat=label_text_features,
        clip_model=clip_model,
        U=U,
        epochs=1,  # 可以先 1~3 轮试
        lambda_keep=1.0,  # 保持其他类
        lambda_vis=70.0,  # 子空间抹除强度，先试 5~20
        lr=1e-4,
    )
    # 8) 遗忘后的表现
    print("==== After unlearning ====")
    all_preds_after, all_targs_after, all_masks_after, all_ids_after, test_loss_after, test_loss_unk_after = run_epoch(
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
        0, 1,
        verbose=False
    )
    print("mAP(after):   {:.3f}".format(after_metrics['mAP']))
    print("O_mAP(after): {:.3f}".format(after_metrics['O_mAP']))
    print("CF1(after):   {:.3f}".format(after_metrics['CF1']))
    print("OF1(after):   {:.3f}".format(after_metrics['OF1']))

    # ====== per-class before/after 指标 ======
    csv_path = f"./ulearn/voc_vis_unlearn_cls{forget_cls}_perclass_metrics.csv"

    per_before, per_after = summarize_before_after(
        all_preds_before, all_targs_before,
        all_preds_after,  all_targs_after,
        class_names=VOC_CLASSES,
        forget_cls=forget_cls,
        csv_path=csv_path,
        threshold=0.5,
    )
    print(f"[Per-class] metrics saved to: {csv_path}")

    # 顺便在终端打印一下被遗忘的那个类
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

    print(f"Class {c} ({name}):")
    print(f"  AP  {ap_before:.3f} -> {ap_after:.3f} (Δ={ap_after - ap_before:+.3f})")
    print(f"  P   {p_before:.3f}  -> {p_after:.3f}  (Δ={p_after - p_before:+.3f})")
    print(f"  R   {r_before:.3f}  -> {r_after:.3f}  (Δ={r_after - r_before:+.3f})")
    print(f"  F1  {f1_before:.3f} -> {f1_after:.3f} (Δ={f1_after - f1_before:+.3f})")
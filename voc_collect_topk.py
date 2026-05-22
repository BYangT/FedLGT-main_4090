# voc_collect_topk.py

import torch
import argparse
from config_args import get_args
from load_data import get_data
from models import CTranModel
from ulearn.unlearn_utils import collect_topk_dims_for_class
import clip
import os
import json
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_clip_embeddings_for_voc(args, device):
    # 跟 fed_main 里面一样的写法
    label_space = ['Aeroplane',
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
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    prompt = [f"The photo contains {x}." for x in label_space]
    with torch.no_grad():
        label_text = clip.tokenize(prompt).to(device)
        label_text_features = clip_model.encode_text(label_text)
        label_text_features = label_text_features / label_text_features.norm(dim=1, keepdim=True)

    # state embedding
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
    args.dataroot = '/code/Fed/data'   # 按你实际路径改
    args.learn_emb_type = 'clip'

    # 1) 数据
    train_dl_global, valid_dl_global, test_dl_global = get_data(args)

    # 2) CLIP embedding
    clip_model, label_text_features, state_weight = build_clip_embeddings_for_voc(args, device)

    # 3) 模型
    from fed_main import init_nets
    nets, _, _ = init_nets(args, is_global=True,
                           state_weight=state_weight,
                           label_weight=label_text_features)
    model = nets[0]
    # 加载你已有的 VOC 全局模型权重（举例）
    ckpt = torch.load("/code/Fed/results/voc.3layer.bsz_16.adam0.0001.clip_embagg_avgcoarse_prompt_concat.lmt.unk_lossround_40.pt", map_location=device)
    model.load_state_dict(ckpt['state_dict'])

    # 4) 选择要忘的类，比如 Person = 14（注意从 0 开始）
    forget_cls = 14
    K = 64

    # 只用一部分数据也行，可以先用 train_dl_global
    topk_idx, score = collect_topk_dims_for_class(
        model=model,
        dataloader=train_dl_global,
        forget_cls=forget_cls,
        K=K,
        device=device,
        args=args,
        emb_feat=label_text_features,
        clip_model=clip_model
    )

    print("Top-K dims for class", forget_cls, ":", topk_idx.cpu().numpy())
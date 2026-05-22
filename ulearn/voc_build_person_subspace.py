# voc_build_person_subspace.py

import argparse
import torch

from config_args import get_args
from load_data import get_data
from fed_main import init_nets
import clip
from ulearn.projector_subspace import build_vis_subspace_for_class

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    args.dataroot = '/code/Fed/data'
    args.learn_emb_type = 'clip'
    args.device = 'cuda:0'

    # 数据
    train_dl_global, valid_dl_global, test_dl_global = get_data(args)

    # CLIP embedding
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

    # Person 在你定义里是 index 14
    forget_cls = 14

    save_path = "./ulearn/U_vis_person_rank32.pt"
    U = build_vis_subspace_for_class(
        model=model,
        dataloader=train_dl_global,
        forget_cls=forget_cls,
        device=device,
        args=args,
        emb_feat=label_text_features,
        clip_model=clip_model,
        rank=32,               # 子空间维度，可以先试 8 或 16
        max_pos_images=4000,  # 正样本采多少张图
        save_path=save_path,
    )
import torch
import argparse
import numpy as np

from fed_unlearn_vis_projector import federated_unlearn_one_class_vis_projector
from load_data import get_data
from models import CTranModel
from config_args import get_args
import utils.evaluate as evaluate
import utils.logger as logger
from optim_schedule import WarmupLinearSchedule
from run_epoch import run_epoch
import logging
from tqdm import tqdm
import datetime
import os
import random
import clip
import json
import csv
import copy
import re

from scipy.special import softmax

from fed_unlearn_topklogit import (
    federated_oneshot_unlearn_one_class_subspace,
    federated_unlearn_one_class_topklogit,
)
from ulearn.per_class_report import save_voc_per_class_report_csv
from ulearn.recovery_train_simple import federated_recovery_simple
from ulearn.sae import SparseAutoEncoder
from ulearn.fed_sae_distill import federated_train_sae, federated_train_sae_with_distill


def init_nets(args, is_global=False, state_weight=None, label_weight=None):
    """根据参数初始化模型，返回的是模型，参数形状，参数模型"""
    if is_global:
        n_parties = 1
    else:
        n_parties = args.n_parties

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        model = CTranModel(
            args.num_labels,
            args.use_lmt,
            args.pos_emb,
            args.layers,
            args.heads,
            args.dropout,
            args.no_x_features,
            state_weight=state_weight,
            label_weight=label_weight
        )
        nets[net_i] = model

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def local_train_net(nets, args, u_id, test_dl=None, device="cpu", g_model=None, emb_feat=None, clip_model=None):
    """返回值：总样本数，每个客户端样本数，所有客户端的loss数"""
    data_pts = 0
    net_dataidx_map = {}
    loss_based_agg_list = []
    for net_id, net in nets.items():
        net.to(device)
        if args.dataset == 'coco' or args.dataset == 'voc':
            sub_dst = torch.utils.data.Subset(train_dl_global.dataset, partition_idx_map[net_id])
            train_dl_local = torch.utils.data.DataLoader(
                sub_dst,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=False
            )
            net_dataidx_map[net_id] = len(sub_dst)
            data_pts += len(sub_dst)
        else:
            train_dl_local, test_dl, _, train_dataset = get_data(args, curr_user=u_id[net_id])
            net_dataidx_map[net_id] = len(train_dataset)
            data_pts += len(train_dataset)

        n_epoch = args.epochs
        train_metrics, testacc = train_net(
            net_id, net, train_dl_local, test_dl, n_epoch, args,
            device=device, g_model=g_model, emb_feat=emb_feat, clip_model=clip_model
        )

        loss_based_agg_list.append(train_metrics['loss'])

    return data_pts, net_dataidx_map, loss_based_agg_list


def train_net(net_id, model, train_dataloader, valid_dataloader, epochs, args, device="cpu", g_model=None, emb_feat=None, clip_model=None):
    fl_logger.info('Training network %s' % str(net_id))
    loss_logger = logger.LossLogger(args.model_name)

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=1e-4
        )

    if args.warmup_scheduler:
        step_scheduler = None
        scheduler_warmup = WarmupLinearSchedule(optimizer, 1, 300000)
    else:
        scheduler_warmup = None
        if args.scheduler_type == 'plateau':
            step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5
            )
        elif args.scheduler_type == 'step':
            step_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.scheduler_step,
                gamma=args.scheduler_gamma
            )
        else:
            step_scheduler = None

    test_loader = None
    for epoch in range(epochs):
        desc = f"Client {net_id} Epoch {epoch}"
        all_preds, all_targs, all_masks, all_ids, train_loss, train_loss_unk = run_epoch(
            args, model, train_dataloader,
            optimizer, epoch,
            desc,
            train=True,
            warmup_scheduler=scheduler_warmup,
            global_model=g_model,
            emb_feat=emb_feat,
            clip_model=clip_model
        )

        train_metrics = evaluate.compute_metrics(
            args, all_preds, all_targs, all_masks,
            train_loss, train_loss_unk, 0,
            args.train_known_labels, verbose=False
        )
        loss_logger.log_losses('train.log', epoch, train_loss, train_metrics, train_loss_unk)

        if step_scheduler is not None:
            if args.scheduler_type == 'step':
                step_scheduler.step(epoch)
            elif args.scheduler_type == 'plateau':
                step_scheduler.step(train_loss_unk)

    fl_logger.info(f'{train_metrics["mAP"]}, {train_metrics["CF1"]}, {train_metrics["loss"]:.3f}')
    test_acc = 0
    fl_logger.info(' ** Training complete **')
    return train_metrics, test_acc


if __name__ == '__main__':
    metrics_log = []
    args = get_args(argparse.ArgumentParser())

    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f'Seed: {seed}')

    if args.dataset == 'coco' or args.dataset == 'voc':
        train_dl_global, valid_dl_global, test_dl_global = get_data(args)
    else:
        train_dl_global, valid_dl_global, test_dl_global, fed_hdf5 = get_data(args)
        id_list = list(fed_hdf5['train'].keys())
        sort_id_list = np.load('sorted_list.npy')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path = log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.results_dir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        level=logging.DEBUG,
        filemode='w'
    )

    fl_logger = logging.getLogger()
    fl_logger.setLevel(logging.INFO)

    device = torch.device(args.device)
    state_prompt = ['positive', 'negative']

    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    label_feats = []

    if args.dataset == 'coco':
        category_list = {
            1: u'person',
            2: u'bicycle',
            3: u'car',
            4: u'motorcycle',
            5: u'airplane',
            6: u'bus',
            7: u'train',
            8: u'truck',
            9: u'boat',
            10: u'traffic light',
            11: u'fire hydrant',
            12: u'stop sign',
            13: u'parking meter',
            14: u'bench',
            15: u'bird',
            16: u'cat',
            17: u'dog',
            18: u'horse',
            19: u'sheep',
            20: u'cow',
            21: u'elephant',
            22: u'bear',
            23: u'zebra',
            24: u'giraffe',
            25: u'backpack',
            26: u'umbrella',
            27: u'handbag',
            28: u'tie',
            29: u'suitcase',
            30: u'frisbee',
            31: u'skis',
            32: u'snowboard',
            33: u'sports ball',
            34: u'kite',
            35: u'baseball bat',
            36: u'baseball glove',
            37: u'skateboard',
            38: u'surfboard',
            39: u'tennis racket',
            40: u'bottle',
            41: u'wine glass',
            42: u'cup',
            43: u'fork',
            44: u'knife',
            45: u'spoon',
            46: u'bowl',
            47: u'banana',
            48: u'apple',
            49: u'sandwich',
            50: u'orange',
            51: u'broccoli',
            52: u'carrot',
            53: u'hot dog',
            54: u'pizza',
            55: u'donut',
            56: u'cake',
            57: u'chair',
            58: u'couch',
            59: u'potted plant',
            60: u'bed',
            61: u'dining table',
            62: u'toilet',
            63: u'tv',
            64: u'laptop',
            65: u'mouse',
            66: u'remote',
            67: u'keyboard',
            68: u'cell phone',
            69: u'microwave',
            70: u'oven',
            71: u'toaster',
            72: u'sink',
            73: u'refrigerator',
            74: u'book',
            75: u'clock',
            76: u'vase',
            77: u'scissors',
            78: u'teddy bear',
            79: u'hair drier',
            80: u'toothbrush'
        }
        label_space = list(category_list.values())
        prompt = [f'The photo contains {item}.' for item in label_space]
        with torch.no_grad():
            label_text = clip.tokenize(prompt).to(device)
            label_text_features = clip_model.encode_text(label_text)
            label_text_features = label_text_features / label_text_features.norm(dim=1, keepdim=True)

    elif args.dataset == 'voc':
        label_space = [
            'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
            'Bus', 'Car', 'Cat', 'Chair', 'Cow',
            'Diningtable', 'Dog', 'Horse', 'Motorbike', 'Person',
            'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor'
        ]
        prompt = [f'The photo contains {item}.' for item in label_space]
        with torch.no_grad():
            label_text = clip.tokenize(prompt).to(device)
            label_text_features = clip_model.encode_text(label_text)
            label_text_features = label_text_features / label_text_features.norm(dim=1, keepdim=True)

    elif args.dataset == 'flair_fed':
        if args.coarse_prompt_type == 'avg':
            with torch.no_grad():
                with open(os.path.join(args.dataroot, 'flair') + '/label_map_for_text.json') as f:
                    label_inp = json.load(f)
                    for k, v in label_inp.items():
                        pts = [f'The photo contains {text}' for text in v]
                        tokens = clip.tokenize(pts).to(device)
                        feats = clip_model.encode_text(tokens).cpu()
                        feats = torch.mean(feats, dim=0)
                        label_feats.append(feats.view(1, -1))
            label_text_features = torch.cat(label_feats, dim=0)
        elif args.coarse_prompt_type == 'concat':
            prompt = []
            if args.flair_fine:
                fg_label_space = np.load('fine_g.npy')
                for item in fg_label_space:
                    prompt.append(f'The photo contains {item}.')
            else:
                coarse_label_space = []
                with open(os.path.join(args.dataroot, 'flair') + '/label_map_for_text.json') as f:
                    label_inp = json.load(f)
                    for k, v in label_inp.items():
                        if len(v) >= 20:
                            tmp_v = v[:20]
                        else:
                            tmp_v = v
                        coarse_label_space.append(','.join(tmp_v))
                for item in coarse_label_space:
                    prompt.append(f'The photo contains {item}.')

            with torch.no_grad():
                label_text = clip.tokenize(prompt).to(device)
                label_text_features = clip_model.encode_text(label_text)
                label_text_features = label_text_features / label_text_features.norm(dim=1, keepdim=True)

    state_text = clip.tokenize(state_prompt).to(device)
    with torch.no_grad():
        weight = clip_model.encode_text(state_text)
        weight = weight / weight.norm(dim=1, keepdim=True)
        weight = torch.cat((torch.zeros(512).view(1, -1).to(device), weight), 0)

    if args.inference:
        test_id_list = list(fed_hdf5['test'].keys())
        tmp_model, _, _ = init_nets(args, is_global=True, state_weight=weight, label_weight=label_text_features)
        tmp_model = tmp_model[0]
        ckpt = torch.load(args.ckpt_path)
        tmp_model.load_state_dict(ckpt['state_dict'])
        tmp_model.to(device)
        result = []
        for i in tqdm(range(len(test_id_list))):
            test_dl_local, test_dl, _, test_dataset = get_data(args, curr_user=test_id_list[i])
            all_preds, all_targs, all_masks, all_ids, test_loss, test_loss_unk = run_epoch(
                args, tmp_model, test_dl_local, None, 1, 'Testing',
                global_model=tmp_model, emb_feat=label_text_features, clip_model=clip_model
            )
            test_metrics = evaluate.compute_metrics(
                args, all_preds, all_targs, all_masks, test_loss, test_loss_unk, 0, 1, verbose=False
            )
            save_metrics = {
                'C-AP': test_metrics['mAP'],
                'O-AP': test_metrics['O_mAP'],
                'CF1': test_metrics['CF1'],
                'OF1': test_metrics['OF1']
            }
            result.append(save_metrics)
        np.save('result_map.npy', np.array(result))
        print('Inference done!')
        exit()

    fl_logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(
        args, is_global=False, state_weight=weight, label_weight=label_text_features
    )
    global_models, global_model_meta_data, global_layer_type = init_nets(
        args, is_global=True, state_weight=weight, label_weight=label_text_features
    )
    global_model = global_models[0]
    global_para = global_model.state_dict()

    if args.is_same_initial:
        for net_id, net in nets.items():
            net.load_state_dict(global_para)

    n_train = len(train_dl_global.dataset)
    idxs = np.random.permutation(n_train)
    batch_idxs = np.array_split(idxs, args.n_parties)
    partition_idx_map = {i: batch_idxs[i] for i in range(args.n_parties)}

    for curr_round in tqdm(range(args.comm_round)):
        fl_logger.info("in comm round:" + str(curr_round))
        if args.dataset in ['coco', 'voc']:
            u_id = np.arange(args.n_parties)
        else:
            u_id = np.random.choice(sort_id_list, size=args.n_parties, replace=False)

        global_para = global_model.state_dict()
        for idx in range(len(u_id)):
            nets[idx].load_state_dict(global_para)

        global_model.to(device)

        total_data_points, net_dataidx_map, loss_based_agg_list = local_train_net(
            nets,
            args,
            u_id,
            test_dl=None,
            device=device,
            g_model=global_model,
            emb_feat=label_text_features,
            clip_model=clip_model
        )

        fed_avg_freqs = [net_dataidx_map[r] / total_data_points for r in range(len(u_id))]
        loss_based_agg_list_targ = [-1. * val for val in loss_based_agg_list]
        loss_based_freqs = softmax(loss_based_agg_list, axis=0)

        for idx in range(len(u_id)):
            net_para = nets[idx].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    if args.agg_type == 'fedavg':
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    elif args.agg_type == 'loss':
                        global_para[key] = net_para[key] * loss_based_freqs[idx]
            else:
                for key in net_para:
                    if args.agg_type == 'fedavg':
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
                    elif args.agg_type == 'loss':
                        global_para[key] += net_para[key] * loss_based_freqs[idx]

        global_model.load_state_dict(global_para)
        global_model.to(device)

        if (curr_round % 2 == 0) or (curr_round == args.comm_round - 1):
            all_preds, all_targs, all_masks, all_ids, test_loss, test_loss_unk = run_epoch(
                args, global_model, test_dl_global, None, 1, 'Testing',
                global_model=global_model, emb_feat=label_text_features, clip_model=clip_model
            )
            test_metrics = evaluate.compute_metrics(
                args, all_preds, all_targs, all_masks, test_loss, test_loss_unk, 0, 1
            )
            save_dict = {
                'state_dict': global_model.state_dict(),
                'test_mAP': test_metrics['mAP'],
                'test_O_mAP': test_metrics['O_mAP'],
            }
            fl_logger.info(
                f"[Round {curr_round}] "
                f"C-AP(mAP)={test_metrics['mAP']:.4f}, "
                f"O-AP={test_metrics['O_mAP']:.4f}, "
                f"CF1={test_metrics['CF1']:.4f}, "
                f"OF1={test_metrics['OF1']:.4f}"
            )
            metrics_log.append({
                "round": curr_round,
                "mAP": float(test_metrics["mAP"]),
                "O_mAP": float(test_metrics["O_mAP"]),
                "CF1": float(test_metrics["CF1"]),
                "OF1": float(test_metrics["OF1"]),
                "loss": float(test_loss),
                "unk_loss": float(test_loss_unk),
            })
            os.makedirs("ulearn_model", exist_ok=True)
            filename = os.path.join("ulearn_model", f"federated_coco_40.pt")
            torch.save(save_dict, filename)

    # ================= FedAvg 正常训练结束 =================
    print(" ** FedAvg training finished. **")
    for net_id, net in nets.items():
        net.to('cpu')
    torch.cuda.empty_cache()

    # ====== 训练 SAE（供 latent Top-K 使用） ======
    print("==== Start federated SAE warmup + distillation ====")

    sae_input_dim = 512
    sae_latent_dim = 1024
    sae_activation = "relu"
    sae_l1_lambda = 1e-4
    sae_warmup_rounds = getattr(args, "sae_warmup_rounds", 3)
    sae_warmup_local_epochs = getattr(args, "sae_warmup_local_epochs", args.epochs)
    sae_rounds = getattr(args, "sae_rounds", 3)
    sae_local_epochs = getattr(args, "sae_local_epochs", args.epochs)
    sae_lr = getattr(args, "sae_lr", 1e-3)
    sae_distill_lambda = getattr(args, "sae_distill_lambda", 1.0)
    sae_distill_type = getattr(args, "sae_distill_type", "cosine")
    sae_selective_lambda = getattr(args, "sae_selective_lambda", 0.05)
    sae_overlap_lambda = getattr(args, "sae_overlap_lambda", 0.1)
    sae_coupling_topm = getattr(args, "sae_coupling_topm", 5)
    topk_coupling_lambda = getattr(args, "topk_coupling_lambda", 0.5)
    use_sae_distill = not getattr(args, "disable_sae_distill", False)

    selective_tag = re.sub(
        r"[^0-9a-zA-Z]+",
        "p",
        f"sel{sae_selective_lambda:g}_ov{sae_overlap_lambda:g}_ct{sae_coupling_topm}_tk{topk_coupling_lambda:g}",
    )
    sae_warmup_path = os.path.join("ulearn", f"sae_512_to_1024_{selective_tag}.pth")
    sae_distill_path = os.path.join("ulearn_model", f"fed_sae_distill_{selective_tag}.pth")
    sae_warmup_log_path = os.path.join(args.results_new, f"fed_sae_warmup_round_logs_{selective_tag}.json")
    sae_round_log_path = os.path.join(args.results_new, f"fed_sae_distill_round_logs_{selective_tag}.json")

    if not os.path.exists(sae_warmup_path):
        print("==== Start federated SAE warmup ====")
        global_sae = SparseAutoEncoder(
            input_dim=sae_input_dim,
            latent_dim=sae_latent_dim,
            activation=sae_activation,
        )

        global_sae, sae_warmup_logs = federated_train_sae(
            args=args,
            global_model=global_model,
            global_sae=global_sae,
            train_dl_global=train_dl_global,
            partition_idx_map=partition_idx_map,
            device=device,
            emb_feat=label_text_features,
            clip_model=clip_model,
            sae_rounds=sae_warmup_rounds,
            sae_local_epochs=sae_warmup_local_epochs,
            sae_lr=sae_lr,
            l1_lambda=sae_l1_lambda,
            selective_lambda=sae_selective_lambda,
            overlap_lambda=sae_overlap_lambda,
            coupling_topm=sae_coupling_topm,
        )
        os.makedirs(os.path.dirname(sae_warmup_path), exist_ok=True)
        os.makedirs(args.results_new, exist_ok=True)
        torch.save(global_sae.cpu().state_dict(), sae_warmup_path)
        with open(sae_warmup_log_path, "w") as f:
            json.dump(sae_warmup_logs, f, indent=2)

        print(f"Federated SAE warmup saved to: {sae_warmup_path}")
        print(f"SAE warmup round logs saved to: {sae_warmup_log_path}")
    else:
        print(f"SAE warmup checkpoint already exists, skip warmup: {sae_warmup_path}")

    final_sae_path = sae_warmup_path
    if use_sae_distill and not os.path.exists(sae_distill_path):
        print("==== Start federated SAE distillation ====")
        global_sae = SparseAutoEncoder(
            input_dim=sae_input_dim,
            latent_dim=sae_latent_dim,
            activation=sae_activation,
        )
        global_sae.load_state_dict(torch.load(sae_warmup_path, map_location="cpu"))

        global_sae, sae_round_logs = federated_train_sae_with_distill(
            args=args,
            global_model=global_model,
            global_sae=global_sae,
            train_dl_global=train_dl_global,
            partition_idx_map=partition_idx_map,
            device=device,
            emb_feat=label_text_features,
            clip_model=clip_model,
            sae_rounds=sae_rounds,
            sae_local_epochs=sae_local_epochs,
            sae_lr=sae_lr,
            l1_lambda=sae_l1_lambda,
            distill_lambda=sae_distill_lambda,
            selective_lambda=sae_selective_lambda,
            overlap_lambda=sae_overlap_lambda,
            coupling_topm=sae_coupling_topm,
            distill_type=sae_distill_type,
        )

        os.makedirs(os.path.dirname(sae_distill_path), exist_ok=True)
        os.makedirs(args.results_new, exist_ok=True)
        torch.save(global_sae.cpu().state_dict(), sae_distill_path)
        with open(sae_round_log_path, "w") as f:
            json.dump(sae_round_logs, f, indent=2)

        print(f"Federated distilled SAE saved to: {sae_distill_path}")
        print(f"SAE distillation round logs saved to: {sae_round_log_path}")
        final_sae_path = sae_distill_path
    elif use_sae_distill:
        print(f"Federated distilled SAE checkpoint already exists, skip training: {sae_distill_path}")
        final_sae_path = sae_distill_path
    else:
        print("==== SAE distillation disabled, use warmup SAE directly ====")
        final_sae_path = sae_warmup_path

    args.sae_ckpt = final_sae_path
    args.sae_input_dim = sae_input_dim
    args.sae_latent_dim = sae_latent_dim
    args.sae_activation = sae_activation
    args.sae_selective_lambda = sae_selective_lambda
    args.sae_overlap_lambda = sae_overlap_lambda
    args.sae_coupling_topm = sae_coupling_topm
    args.topk_coupling_lambda = topk_coupling_lambda
    args.sae_use_layer_norm = True

    # ====== 联邦单类遗忘（完整版） + 按类别表格 ======
    do_unlearn = True
    if do_unlearn:
        dataset = args.dataset.lower()

        if dataset == "voc":
            CLASS_NAMES = [
                'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                'Bus', 'Car', 'Cat', 'Chair', 'Cow',
                'Diningtable', 'Dog', 'Horse', 'Motorbike', 'Person',
                'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor'
            ]
            default_forget_cls = 14
            csv_prefix = "voc_full_fed_unlearn"

        elif dataset == "coco":
            COCO_CLASSES = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            CLASS_NAMES = COCO_CLASSES
            default_forget_cls = 0
            csv_prefix = "coco_full_fed_unlearn"

        else:
            raise ValueError(f"Unlearn only implemented for VOC/COCO, got: {args.dataset}")

        forget_cls = getattr(args, "forget_cls", default_forget_cls)
        K = getattr(args, "k", 64)

        print(f"==== DATASET: {dataset}, forget_cls = {forget_cls} ({CLASS_NAMES[forget_cls]}) ====")

        print("==== Evaluate global model BEFORE unlearning ====")
        all_preds_before, all_targs_before, all_masks_b, all_ids_b, test_loss_b, test_loss_unk_b = run_epoch(
            args,
            global_model,
            test_dl_global,
            None,
            1,
            'Testing',
            global_model=global_model,
            emb_feat=label_text_features,
            clip_model=clip_model
        )
        test_metrics_before = evaluate.compute_metrics(
            args,
            all_preds_before, all_targs_before, all_masks_b,
            test_loss_b, test_loss_unk_b,
            0, 1
        )
        print(f"mAP(before):   {test_metrics_before['mAP']:.3f}")
        print(f"O_mAP(before): {test_metrics_before['O_mAP']:.3f}")
        print(f"CF1(before):   {test_metrics_before['CF1']:.3f}")
        print(f"OF1(before):   {test_metrics_before['OF1']:.3f}")

        if args.oneshot_subspace_nulling:
            print(f"==== Start ONE-SHOT subspace nulling for class {forget_cls} (top-{K}) ====")
            global_model = federated_oneshot_unlearn_one_class_subspace(
                args=args,
                global_model=global_model,
                nets=nets,
                train_dl_global=train_dl_global,
                partition_idx_map=partition_idx_map,
                device=device,
                emb_feat=label_text_features,
                clip_model=clip_model,
                forget_cls=forget_cls,
                K=K,
                min_pos=10,
            )
        else:
            print(f"==== Start FULL federated unlearning for class {forget_cls} (top-{K}) ====")
            global_model = federated_unlearn_one_class_topklogit(
                args=args,
                global_model=global_model,
                nets=nets,
                train_dl_global=train_dl_global,
                partition_idx_map=partition_idx_map,
                device=device,
                emb_feat=label_text_features,
                clip_model=clip_model,
                forget_cls=forget_cls,
                K=K,
                unlearn_rounds=3,
                client_frac=1.0,
                unlearn_epochs=4,
                unlearn_lr=1e-4,
                lambda_keep=2.0,
                lambda_forget_logit=0.0,
                lambda_forget_feat=4.0,
                min_pos=10,
                mode="feat_only",
            )

        print("==== Evaluate global model AFTER unlearning ====")
        all_preds_after, all_targs_after, all_masks_a, all_ids_a, test_loss_a, test_loss_unk_a = run_epoch(
            args,
            global_model,
            test_dl_global,
            None,
            1,
            'Testing',
            global_model=global_model,
            emb_feat=label_text_features,
            clip_model=clip_model
        )
        test_metrics_after = evaluate.compute_metrics(
            args,
            all_preds_after, all_targs_after, all_masks_a,
            test_loss_a, test_loss_unk_a,
            0, 1
        )
        print(f"mAP(after):   {test_metrics_after['mAP']:.3f}")
        print(f"O_mAP(after): {test_metrics_after['O_mAP']:.3f}")
        print(f"CF1(after):   {test_metrics_after['CF1']:.3f}")
        print(f"OF1(after):   {test_metrics_after['OF1']:.3f}")

        csv_path_unlearn = os.path.join(
            args.results_new,
            f"new_{csv_prefix}_cls{forget_cls}_per_class_report.csv"
        )
        save_voc_per_class_report_csv(
            all_targs_before=all_targs_before,
            all_preds_before=all_preds_before,
            all_targs_after=all_targs_after,
            all_preds_after=all_preds_after,
            class_names=CLASS_NAMES,
            csv_path=csv_path_unlearn,
            thr=0.5,
        )
        from ulearn.perclass_metrics_utils import summarize_forget_vs_others

        summary = summarize_forget_vs_others(
            all_preds_before, all_targs_before,
            all_preds_after, all_targs_after,
            forget_cls=forget_cls,
            class_names=CLASS_NAMES,
            csv_path=os.path.join(args.results_new, f"k_{K}new_forget_vs_others_cls{forget_cls}.csv"),
            threshold=0.5,
        )

        print("=== Forget class summary ===")
        print("目标类 before:", summary["forget_cls_before"])
        print("目标类 after :", summary["forget_cls_after"])
        print("非目标类均值 before:", summary["others_mean_before"])
        print("非目标类均值 after :", summary["others_mean_after"])

        if args.oneshot_subspace_nulling:
            print("==== Skip recovery for ONE-SHOT subspace nulling ====")
            all_preds_rec, all_targs_rec, all_masks_rec, all_ids_rec = all_preds_after, all_targs_after, all_masks_a, all_ids_a
            test_loss_rec, test_loss_unk_rec = test_loss_a, test_loss_unk_a
            test_metrics_rec = test_metrics_after
            print(f"mAP(after_recovery):   {test_metrics_rec['mAP']:.3f}")
            print(f"O_mAP(after_recovery): {test_metrics_rec['O_mAP']:.3f}")
            print(f"CF1(after_recovery):   {test_metrics_rec['CF1']:.3f}")
            print(f"OF1(after_recovery):   {test_metrics_rec['OF1']:.3f}")
        else:
            global_model = federated_recovery_simple(
                args=args,
                global_model=global_model,
                nets=nets,
                train_dl_global=train_dl_global,
                partition_idx_map=partition_idx_map,
                device=device,
                emb_feat=label_text_features,
                clip_model=clip_model,
                forget_cls=forget_cls,
                recovery_rounds=2,
            )

            print("==== Evaluate global model AFTER RECOVERY ====")
            all_preds_rec, all_targs_rec, all_masks_rec, all_ids_rec, test_loss_rec, test_loss_unk_rec = run_epoch(
                args,
                global_model,
                test_dl_global,
                None,
                1,
                'Testing-After-Recovery',
                global_model=global_model,
                emb_feat=label_text_features,
                clip_model=clip_model
            )
            test_metrics_rec = evaluate.compute_metrics(
                args,
                all_preds_rec, all_targs_rec, all_masks_rec,
                test_loss_rec, test_loss_unk_rec,
                0, 1
            )
            print(f"mAP(after_recovery):   {test_metrics_rec['mAP']:.3f}")
            print(f"O_mAP(after_recovery): {test_metrics_rec['O_mAP']:.3f}")
            print(f"CF1(after_recovery):   {test_metrics_rec['CF1']:.3f}")
            print(f"OF1(after_recovery):   {test_metrics_rec['OF1']:.3f}")

        csv_path_recovery = os.path.join(
            args.results_new,
            f"new_{csv_prefix}_cls{forget_cls}_recovery.csv"
        )
        save_voc_per_class_report_csv(
            all_targs_before=all_targs_before,
            all_preds_before=all_preds_before,
            all_targs_after=all_targs_rec,
            all_preds_after=all_preds_rec,
            class_names=CLASS_NAMES,
            csv_path=csv_path_recovery,
            thr=0.5,
        )

        print(f"按类别的 忘却前 vs 忘却后 指标保存在: {csv_path_unlearn}")
        print(f"按类别的 忘却前 vs 恢复后 指标保存在: {csv_path_recovery}")
        summary = summarize_forget_vs_others(
            all_preds_before, all_targs_before,
            all_preds_rec, all_targs_rec,
            forget_cls=forget_cls,
            class_names=CLASS_NAMES,
            csv_path=os.path.join(args.results_new, f"new_recovery_forget_vs_others_cls{forget_cls}.csv"),
            threshold=0.5,
        )

        print("=== Forget class summary ===")
        print("目标类 before:", summary["forget_cls_before"])
        print("目标类 after :", summary["forget_cls_after"])
        print("非目标类均值 before:", summary["others_mean_before"])
        print("非目标类均值 after :", summary["others_mean_after"])

    csv_path = os.path.join(args.results_new, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "mAP", "O_mAP", "CF1", "OF1", "loss", "unk_loss"]
        )
        writer.writeheader()
        writer.writerows(metrics_log)

    print(f"测试指标表格已保存到: {csv_path}")

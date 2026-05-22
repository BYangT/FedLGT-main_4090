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

from scipy.special import softmax

from fed_unlearn_full import federated_unlearn_one_class_full
from fed_unlearn_topklogit import federated_unlearn_one_class_topklogit
from ulearn.per_class_report import save_voc_per_class_report_csv
from ulearn.projector_subspace import build_vis_subspace_for_class
from ulearn.recovery_train import federated_recovery_no_target_samples
from ulearn.recovery_train_simple import federated_recovery_simple
from ulearn.unlearn_utils_topklogit import unlearn_one_class_on_model, collect_topk_dims_for_class
from retrain_no_target import federated_retrain_no_target_samples

def init_nets(args, is_global=False, state_weight=None, label_weight=None):
    """根据参数初始化模型，返回的是模型，参数形状，参数模型"""
    # 如果是全局模型的话，客户端数量自动设置为1
    if is_global:
        n_parties = 1
    else:
        n_parties = args.n_parties
    # 给对应的本地客户端模型初始化
    nets = {net_i: None for net_i in range(n_parties)}

    ### FLAIR
    # 给本地客户端建立模型
    for net_i in range(n_parties):
        model = CTranModel(args.num_labels, args.use_lmt, args.pos_emb, args.layers, args.heads, args.dropout,
                           args.no_x_features, state_weight=state_weight, label_weight=label_weight)
        nets[net_i] = model

    # model_meta_data参数的形状
    # layer_type参数的名字
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
        # TODO: for COCO-dataset, just use indexing of the original dataset to have new subset dataset
        # TODO: VOC dataset is similar
        if args.dataset == 'coco' or args.dataset == 'voc':
            # 拆分数据集
            # partition_idx_map这是一个字典
            sub_dst = torch.utils.data.Subset(train_dl_global.dataset, partition_idx_map[net_id])
            train_dl_local = torch.utils.data.DataLoader(
                sub_dst,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=False
            )
            # 记录每个客户端有多少个样本
            net_dataidx_map[net_id] = len(sub_dst)
            # 统计所有客户端一共有多少个样本
            data_pts += len(sub_dst)
        else:
            train_dl_local, test_dl, _, train_dataset = get_data(args, curr_user=u_id[net_id])
            # for fedavg
            net_dataidx_map[net_id] = len(train_dataset)
            data_pts += len(train_dataset)
        # epochs数
        n_epoch = args.epochs
        # 开始训练
        train_metrics, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args, device=device,
                                           g_model=g_model, emb_feat=emb_feat, clip_model=clip_model)

        # for loss-based agg.
        loss_based_agg_list.append(train_metrics['loss'])

    return data_pts, net_dataidx_map, loss_based_agg_list


def train_net(net_id, model, train_dataloader, valid_dataloader, epochs, args, device="cpu", g_model=None,
              emb_feat=None, clip_model=None):
    fl_logger.info('Training network %s' % str(net_id))
    # 日志
    loss_logger = logger.LossLogger(args.model_name)

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )  # , weight_decay=0.0004
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
            step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        elif args.scheduler_type == 'step':
            step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step,
                                                             gamma=args.scheduler_gamma)
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

        train_metrics = evaluate.compute_metrics(args, all_preds, all_targs, all_masks, train_loss, train_loss_unk, 0,
                                                 args.train_known_labels, verbose=False)
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
    metrics_log = []  # 用来存每次 Testing 的结果
    args = get_args(argparse.ArgumentParser())
    # 随机种子
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f'Seed: {seed}')
    if args.dataset == 'coco' or args.dataset == 'voc':
        # 所有数据集
        train_dl_global, valid_dl_global, test_dl_global = get_data(args)
    else:
        train_dl_global, valid_dl_global, test_dl_global, fed_hdf5 = get_data(args)
        id_list = list(fed_hdf5['train'].keys())
        sort_id_list = np.load('sorted_list.npy')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # logging.basicConfig()
    log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path = log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.results_dir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

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
            80: u'toothbrush'}
        label_space = list(category_list.values())
        prompt = []
        for item in label_space:
            prompt.append(f'The photo contains {item}.')
        with torch.no_grad():
            label_text = clip.tokenize(prompt).to(device)
            label_text_features = clip_model.encode_text(label_text)
            label_text_features = label_text_features / label_text_features.norm(dim=1, keepdim=True)
    elif args.dataset == 'voc':
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
        prompt = []
        for item in label_space:
            prompt.append(f'The photo contains {item}.')
        with torch.no_grad():
            label_text = clip.tokenize(prompt).to(device)
            label_text_features = clip_model.encode_text(label_text)
            label_text_features = label_text_features / label_text_features.norm(dim=1, keepdim=True)
    elif args.dataset == 'flair_fed':
        if args.coarse_prompt_type == 'avg':
            # TODO: pooling of the fg labels
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
                # for item in coarse_label_space:
                #     prompt.append(f'The photo contains {item}.')
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
    # state-embedding
    state_text = clip.tokenize(state_prompt).to(device)
    with torch.no_grad():
        weight = clip_model.encode_text(state_text)
        weight = weight / weight.norm(dim=1, keepdim=True)
        weight = torch.cat((torch.zeros(512).view(1, -1).to(device), weight), 0)
    if args.inference:
        test_id_list = list(fed_hdf5['test'].keys())
        # run inference
        tmp_model, _, _ = init_nets(args, is_global=True, state_weight=weight, label_weight=label_text_features)
        tmp_model = tmp_model[0]
        ckpt = torch.load(args.ckpt_path)
        tmp_model.load_state_dict(ckpt['state_dict'])
        tmp_model.to(device)
        result = []
        for i in tqdm(range(len(test_id_list))):
            test_dl_local, test_dl, _, test_dataset = get_data(args, curr_user=test_id_list[i])
            all_preds, all_targs, all_masks, all_ids, test_loss, test_loss_unk = run_epoch(args, tmp_model,
                                                                                           test_dl_local, None, 1,
                                                                                           'Testing',
                                                                                           global_model=tmp_model,
                                                                                           emb_feat=label_text_features,
                                                                                           clip_model=clip_model)
            test_metrics = evaluate.compute_metrics(args, all_preds, all_targs, all_masks, test_loss, test_loss_unk, 0,
                                                    1, verbose=False)
            save_metrics = {'C-AP': test_metrics['mAP'],
                            'O-AP': test_metrics['O_mAP'],
                            'CF1': test_metrics['CF1'],
                            'OF1': test_metrics['OF1']}
            result.append(save_metrics)
        np.save('result_map.npy', np.array(result))
        print('Inference done!')
        exit()

    # ---- fedavg algo. ---- #
    # init models
    # 初始化网络
    fl_logger.info("Initializing nets")
    # 返回的nets是初始化完事的，十个客户端就是十个网络
    # local_model_meta_data：参数形状
    # layer_type：参数类型
    nets, local_model_meta_data, layer_type = init_nets(args, is_global=False, state_weight=weight,
                                                        label_weight=label_text_features)
    global_models, global_model_meta_data, global_layer_type = init_nets(args, is_global=True, state_weight=weight,
                                                                         label_weight=label_text_features)
    global_model = global_models[0]
    # 拿到参数
    global_para = global_model.state_dict()
    init_global_state = copy.deepcopy(global_model.state_dict())  # ★ 保存初始权重，用于重训练 baseline
    if args.is_same_initial:
        for net_id, net in nets.items():
            net.load_state_dict(global_para)

    # TOTAL_LEN = 345879
    # TODO: COCO dataset, generate the partition map for use
    # Homo:
    # 总样本数
    n_train = len(train_dl_global.dataset)
    # 把 0 ~ n_train-1 这堆样本索引打乱顺序
    idxs = np.random.permutation(n_train)
    # 分割
    batch_idxs = np.array_split(
        idxs, args.n_parties
    )  # As many splits as n_nets = number of clients
    # 生成了一个字典
    partition_idx_map = {i: batch_idxs[i] for i in range(args.n_parties)}
    for curr_round in tqdm(range(args.comm_round)):
        fl_logger.info("in comm round:" + str(curr_round))
        if args.dataset in ['coco', 'voc']:
            # u_id = np.arange(10)
            # 结果: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            u_id = np.arange(args.n_parties)
        else:  # FLAIR dataset
            u_id = np.random.choice(sort_id_list, size=args.n_parties, replace=False)
        # print(f'Current select IDs: {u_id}')
        # 参数
        global_para = global_model.state_dict()
        # 加载全局参数
        for idx in range(len(u_id)):
            nets[idx].load_state_dict(global_para)
        # update global model
        global_model.to(device)

        # net_dataidx_map：每个客户端有多少个训练样本
        # total_data_points：所有客户端的样本总数
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
        # 开始聚合
        fed_avg_freqs = [net_dataidx_map[r] / total_data_points for r in range(len(u_id))]
        loss_based_agg_list_targ = [-1. * val for val in loss_based_agg_list]
        loss_based_freqs = softmax(loss_based_agg_list, axis=0)
        # global aggregation
        for idx in range(len(u_id)):
            ## --- Simulate that the client can perform on testing set --- ##
            # print(f'round {curr_round}: inference on net {idx}')
            # all_preds,all_targs,all_masks,all_ids,test_loss,test_loss_unk = run_epoch(args,nets[idx],test_dl_global,None,1,'Testing', global_model=global_model, emb_feat=label_text_features, clip_model=clip_model)
            # test_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,test_loss,test_loss_unk,0,1)
            ## --- ---##
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
            all_preds, all_targs, all_masks, all_ids, test_loss, test_loss_unk = run_epoch(args, global_model,
                                                                                           test_dl_global, None, 1,
                                                                                           'Testing',
                                                                                           global_model=global_model,
                                                                                           emb_feat=label_text_features,
                                                                                           clip_model=clip_model)
            test_metrics = evaluate.compute_metrics(args, all_preds, all_targs, all_masks, test_loss, test_loss_unk, 0,
                                                    1)
            # save global model
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
            os.makedirs("ulearn_model", exist_ok=True)  # 没有就创建
            filename = os.path.join("ulearn_model", f"federated_coco_40.pt")
            torch.save(save_dict, filename)
    # ================= FedAvg 正常训练结束 =================
    print(" ** FedAvg training finished. **")
    for net_id, net in nets.items():
        net.to('cpu')
    torch.cuda.empty_cache()

    # ====== 联邦单类遗忘（完整版） + 按类别表格 ======
    # ====== 联邦单类遗忘：改成“重训练 baseline”（只用非目标类数据训练），不做 recovery ======
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
            default_forget_cls = 0
            csv_prefix = "voc_retrain_no_target"

        elif dataset == "coco":
            CLASS_NAMES = [
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
            default_forget_cls = 2
            csv_prefix = "coco_retrain_no_target"
        else:
            raise ValueError(f"Retrain baseline only implemented for VOC/COCO, got: {args.dataset}")

        forget_cls = getattr(args, "forget_cls", default_forget_cls)
        print(f"==== DATASET: {dataset}, forget_cls = {forget_cls} ({CLASS_NAMES[forget_cls]}) ====")

        # 1) 重训练前：用“已训练好的 global_model”做 before 评估（作为对照）
        print("==== Evaluate global model BEFORE retrain-baseline ====")
        all_preds_before, all_targs_before, all_masks_b, all_ids_b, test_loss_b, test_loss_unk_b = run_epoch(
            args, global_model, test_dl_global, None, 1, 'Testing-Before-Retrain',
            global_model=global_model, emb_feat=label_text_features, clip_model=clip_model
        )
        test_metrics_before = evaluate.compute_metrics(
            args, all_preds_before, all_targs_before, all_masks_b, test_loss_b, test_loss_unk_b, 0, 1
        )
        print(f"mAP(before):   {test_metrics_before['mAP']:.3f}")
        print(f"O_mAP(before): {test_metrics_before['O_mAP']:.3f}")
        print(f"CF1(before):   {test_metrics_before['CF1']:.3f}")
        print(f"OF1(before):   {test_metrics_before['OF1']:.3f}")

        # 2) 重训练 baseline：从 init_global_state 重置，然后只用“非目标类样本”联邦训练
        print("==== Start Retrain-baseline (NO target-class samples) ====")
        global_model = federated_retrain_no_target_samples(
            args=args,
            global_model=global_model,
            nets=nets,
            train_dl_global=train_dl_global,
            partition_idx_map=partition_idx_map,
            device=device,
            emb_feat=label_text_features,
            clip_model=clip_model,
            forget_cls=forget_cls,
            retrain_rounds=getattr(args, "retrain_rounds", 5),  # 你也可以直接写死一个数
            client_frac=getattr(args, "client_frac", 1.0),
            local_epochs=getattr(args, "retrain_local_epochs", 5),  # 建议比原来 unlearn_epochs 稳一点
            retrain_lr=getattr(args, "retrain_lr", 1e-4),
            init_state_dict=init_global_state,  # ★ 关键：从头重训练 baseline
            min_keep_per_client=1,
        )

        # 3) 重训练后评估
        print("==== Evaluate global model AFTER retrain-baseline ====")
        all_preds_after, all_targs_after, all_masks_a, all_ids_a, test_loss_a, test_loss_unk_a = run_epoch(
            args, global_model, test_dl_global, None, 1, 'Testing-After-Retrain',
            global_model=global_model, emb_feat=label_text_features, clip_model=clip_model
        )
        test_metrics_after = evaluate.compute_metrics(
            args, all_preds_after, all_targs_after, all_masks_a, test_loss_a, test_loss_unk_a, 0, 1
        )
        print(f"mAP(after):   {test_metrics_after['mAP']:.3f}")
        print(f"O_mAP(after): {test_metrics_after['O_mAP']:.3f}")
        print(f"CF1(after):   {test_metrics_after['CF1']:.3f}")
        print(f"OF1(after):   {test_metrics_after['OF1']:.3f}")

        # 4) before vs after：按类别 CSV（保留你的原逻辑）
        csv_path_retrain = os.path.join(
            args.results_new,
            f"{args.n_parties}_{csv_prefix}_cls{forget_cls}_per_class_report.csv"
        )
        save_voc_per_class_report_csv(
            all_targs_before=all_targs_before,
            all_preds_before=all_preds_before,
            all_targs_after=all_targs_after,
            all_preds_after=all_preds_after,
            class_names=CLASS_NAMES,
            csv_path=csv_path_retrain,
            thr=0.5,
        )

        from ulearn.perclass_metrics_utils import summarize_forget_vs_others

        summary = summarize_forget_vs_others(
            all_preds_before, all_targs_before,
            all_preds_after, all_targs_after,
            forget_cls=forget_cls,
            class_names=CLASS_NAMES,
            csv_path=os.path.join(args.results_new, f"{args.n_parties}_forget_vs_other{forget_cls}.csv"),
            threshold=0.5,
        )
        print("=== Forget class summary (Retrain Baseline) ===")
        print("目标类 before:", summary["forget_cls_before"])
        print("目标类 after :", summary["forget_cls_after"])
        print("非目标类均值 before:", summary["others_mean_before"])
        print("非目标类均值 after :", summary["others_mean_after"])

        print(f"按类别的 before vs retrain-after 指标保存在: {csv_path_retrain}")

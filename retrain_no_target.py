# ulearn/retrain_no_target.py
import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from optim_schedule import WarmupLinearSchedule
from run_epoch import run_epoch
import utils.evaluate as evaluate
import utils.logger as logger

LOG = logging.getLogger(__name__)


def _extract_targets_from_sample(sample):
    """
    尽量兼容多种 dataset __getitem__ 返回格式，从 sample 里取出 multi-hot targets。
    期待 targets shape: (num_labels,) 或 (num_labels, 1) 等可 squeeze 的形式。
    """
    # 常见：tuple/list => (img, target, mask, id...) 或 (img, target)
    if isinstance(sample, (tuple, list)):
        # 直接猜第二个是 target
        if len(sample) >= 2:
            cand = sample[1]
            t = _to_tensor_1d(cand)
            if t is not None:
                return t
        # 否则遍历找一个像 label 向量的
        for x in sample:
            t = _to_tensor_1d(x)
            if t is not None:
                return t
        return None

    # dict => 常见 key
    if isinstance(sample, dict):
        for k in ["target", "targets", "label", "labels", "targ", "y"]:
            if k in sample:
                t = _to_tensor_1d(sample[k])
                if t is not None:
                    return t
        # 遍历 values
        for v in sample.values():
            t = _to_tensor_1d(v)
            if t is not None:
                return t
        return None

    return None


def _to_tensor_1d(x):
    """把可能的 numpy/list/torch 转成 1D torch.Tensor；如果不像 label 向量就返回 None"""
    if torch.is_tensor(x):
        t = x.detach()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        try:
            t = torch.tensor(x)
        except Exception:
            return None
    else:
        return None

    if t.numel() < 2:
        return None

    t = t.float().view(-1)  # squeeze to 1D
    return t


def filter_indices_no_target(
    dataset,
    indices: np.ndarray,
    forget_cls: int,
    min_keep: int = 1,
) -> List[int]:
    """
    过滤出“非目标类样本”：targets[forget_cls] == 0 的样本索引。
    """
    keep = []
    for idx in indices:
        sample = dataset[int(idx)]
        t = _extract_targets_from_sample(sample)
        if t is None:
            raise RuntimeError(
                "Cannot extract targets from dataset sample. "
                "Please adapt _extract_targets_from_sample to your dataset format."
            )
        # 目标类 absent 才保留
        val = float(t[forget_cls].item())
        if val <= 0.0:
            keep.append(int(idx))

    if len(keep) < min_keep:
        return []
    return keep


def _make_optimizer_and_scheduler(model, args, lr: float):
    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

    if getattr(args, "warmup_scheduler", False):
        scheduler_warmup = WarmupLinearSchedule(optimizer, 1, 300000)
        step_scheduler = None
    else:
        scheduler_warmup = None
        step_scheduler = None
        st = getattr(args, "scheduler_type", None)
        if st == "plateau":
            step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5
            )
        elif st == "step":
            step_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(args, "scheduler_step", 10),
                gamma=getattr(args, "scheduler_gamma", 0.1),
            )
    return optimizer, scheduler_warmup, step_scheduler


def _train_one_client_no_target(
    *,
    args,
    client_id: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    device,
    lr: float,
    local_epochs: int,
    emb_feat=None,
    clip_model=None,
    global_model=None,
):
    """
    单客户端训练：只在 train_loader（已过滤掉目标类样本）上训练。
    """
    model.to(device)
    loss_logger = logger.LossLogger(getattr(args, "model_name", "model"))

    optimizer, warmup_scheduler, step_scheduler = _make_optimizer_and_scheduler(model, args, lr)

    last_train_metrics = None
    last_train_loss = None
    last_train_loss_unk = None

    for ep in range(local_epochs):
        desc = f"[Retrain-NoTarget] Client {client_id} Epoch {ep}"
        all_preds, all_targs, all_masks, all_ids, train_loss, train_loss_unk = run_epoch(
            args,
            model,
            train_loader,
            optimizer,
            ep,
            desc,
            train=True,
            warmup_scheduler=warmup_scheduler,
            global_model=global_model,
            emb_feat=emb_feat,
            clip_model=clip_model,
        )
        train_metrics = evaluate.compute_metrics(
            args,
            all_preds,
            all_targs,
            all_masks,
            train_loss,
            train_loss_unk,
            0,
            getattr(args, "train_known_labels", 1),
            verbose=False,
        )
        loss_logger.log_losses("train_retrain_no_target.log", ep, train_loss, train_metrics, train_loss_unk)

        if step_scheduler is not None:
            st = getattr(args, "scheduler_type", None)
            if st == "step":
                step_scheduler.step()
            elif st == "plateau":
                step_scheduler.step(train_loss_unk)

        last_train_metrics = train_metrics
        last_train_loss = float(train_loss)
        last_train_loss_unk = float(train_loss_unk)

    return last_train_metrics, last_train_loss, last_train_loss_unk


@torch.no_grad()
def _aggregate_fedavg(global_state: Dict[str, torch.Tensor],
                      client_states: List[Dict[str, torch.Tensor]],
                      freqs: List[float]) -> Dict[str, torch.Tensor]:
    new_state = {}
    keys = global_state.keys()
    for k in keys:
        acc = None
        for i, cs in enumerate(client_states):
            w = freqs[i]
            t = cs[k].to("cpu") * w
            acc = t if acc is None else (acc + t)
        new_state[k] = acc
    return new_state


def federated_retrain_no_target_samples(
    *,
    args,
    global_model: torch.nn.Module,
    nets: Dict[int, torch.nn.Module],
    train_dl_global,
    partition_idx_map: Dict[int, np.ndarray],
    device,
    emb_feat=None,
    clip_model=None,
    forget_cls: int = 0,
    retrain_rounds: int = 10,
    client_frac: float = 1.0,
    local_epochs: int = 1,
    retrain_lr: Optional[float] = None,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    min_keep_per_client: int = 1,
):
    """
    联邦“重训练基线”：
    - 若提供 init_state_dict：先把 global_model/nets 重置到 init_state_dict（相当于从头训练）
    - 每个 round：每个 client 仅使用“非目标类样本”（targets[forget_cls]==0）训练
    - 用 FedAvg 聚合更新 global_model
    - 不包含任何 recovery 阶段

    返回：重训练后的 global_model
    """
    assert 0 < client_frac <= 1.0
    if retrain_lr is None:
        # 默认：优先用 args.unlearn_lr，否则 args.lr
        retrain_lr = float(getattr(args, "unlearn_lr", getattr(args, "lr", 1e-4)))

    # reset to init weights if provided
    if init_state_dict is not None:
        LOG.info("Resetting global_model and nets to init_state_dict for retraining baseline.")
        global_model.load_state_dict(copy.deepcopy(init_state_dict))
        for _, net in nets.items():
            net.load_state_dict(copy.deepcopy(init_state_dict))

    dataset = train_dl_global.dataset
    n_parties = getattr(args, "n_parties", len(nets))
    all_client_ids = list(range(n_parties))

    for r in range(retrain_rounds):
        LOG.info(f"[Retrain-NoTarget] Round {r}/{retrain_rounds-1}")

        # client sampling
        m = max(1, int(round(client_frac * n_parties)))
        selected = all_client_ids if m == n_parties else list(np.random.choice(all_client_ids, m, replace=False))

        # broadcast
        global_state = copy.deepcopy(global_model.state_dict())
        for cid in selected:
            nets[cid].load_state_dict(copy.deepcopy(global_state))

        client_states = []
        client_sizes = []

        # train locally
        for cid in selected:
            raw_indices = partition_idx_map[cid]
            keep_indices = filter_indices_no_target(
                dataset,
                raw_indices,
                forget_cls=forget_cls,
                min_keep=min_keep_per_client,
            )

            if len(keep_indices) == 0:
                LOG.warning(f"[Retrain-NoTarget] Client {cid}: no non-target samples, skipped.")
                continue

            sub_dst = Subset(dataset, keep_indices)
            train_loader = DataLoader(
                sub_dst,
                batch_size=getattr(args, "batch_size", 32),
                shuffle=True,
                num_workers=getattr(args, "workers", 4),
                drop_last=False,
            )

            _train_one_client_no_target(
                args=args,
                client_id=cid,
                model=nets[cid],
                train_loader=train_loader,
                device=device,
                lr=retrain_lr,
                local_epochs=local_epochs,
                emb_feat=emb_feat,
                clip_model=clip_model,
                global_model=global_model,
            )

            client_states.append(copy.deepcopy(nets[cid].cpu().state_dict()))
            client_sizes.append(len(sub_dst))

        if len(client_states) == 0:
            LOG.error("[Retrain-NoTarget] No clients produced updates this round. Stop.")
            break

        total = float(sum(client_sizes))
        freqs = [sz / total for sz in client_sizes]

        # aggregate
        new_global_state = _aggregate_fedavg(global_state, client_states, freqs)
        global_model.load_state_dict(new_global_state)
        global_model.to(device)

    return global_model

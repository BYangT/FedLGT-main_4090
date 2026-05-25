# ulearn/fed_sae_distill.py
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from .sae import (
        SparseAutoEncoder,
        sae_loss,
        latent_label_selective_loss,
        get_batch_coupled_label_stats,
        latent_coupling_overlap_loss,
    )
except ImportError:
    from sae import (
        SparseAutoEncoder,
        sae_loss,
        latent_label_selective_loss,
        get_batch_coupled_label_stats,
        latent_coupling_overlap_loss,
    )


def sae_distill_loss(
    x,
    x_hat_student,
    z_student,
    z_teacher,
    label_ids=None,
    num_labels=None,
    l1_lambda=1e-4,
    distill_lambda=1.0,
    selective_lambda=0.0,
    overlap_loss=None,
    overlap_lambda=0.0,
    distill_type="cosine",
):
    """
    student SAE 总损失：
      1) 重建损失
      2) 稀疏损失
      3) latent 蒸馏损失
    """
    total_sae_loss, recon_loss, sparse_loss = sae_loss(
        x, x_hat_student, z_student, l1_lambda=l1_lambda
    )
    selective_loss = latent_label_selective_loss(
        z_student,
        label_ids=label_ids,
        num_labels=num_labels,
    )

    if distill_type == "cosine":
        z_s = F.normalize(z_student, dim=1)
        z_t = F.normalize(z_teacher, dim=1)
        distill_loss = 1.0 - F.cosine_similarity(z_s, z_t, dim=1).mean()
    elif distill_type == "mse":
        distill_loss = F.mse_loss(z_student, z_teacher)
    else:
        raise ValueError(f"Unsupported distill_type: {distill_type}")

    if overlap_loss is None:
        overlap_loss = z_student.new_zeros(())

    total_loss = (
        total_sae_loss
        + distill_lambda * distill_loss
        + selective_lambda * selective_loss
        + overlap_lambda * overlap_loss
    )
    return total_loss, recon_loss, sparse_loss, distill_loss, selective_loss, overlap_loss


def local_train_sae(
    global_model,
    sae_model,
    dataloader,
    device,
    args,
    emb_feat,
    clip_model,
    epochs=1,
    lr=1e-3,
    l1_lambda=1e-4,
    selective_lambda=0.0,
    overlap_lambda=0.0,
    coupling_topm=5,
    use_layer_norm=True,
    client_id=None,
):
    """
    客户端本地训练普通 SAE（无蒸馏）
    """
    global_model.eval()
    global_model.to(device)
    for p in global_model.parameters():
        p.requires_grad = False

    sae_model.train()
    sae_model.to(device)

    optimizer = torch.optim.Adam(sae_model.parameters(), lr=lr)

    epoch_logs = []

    epoch_bar = tqdm(
        range(epochs),
        desc=f"[Client {client_id}] Warmup SAE Epoch",
        ncols=120,
        leave=False,
    )

    for ep in epoch_bar:
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        sparse_loss_sum = 0.0
        selective_loss_sum = 0.0
        overlap_loss_sum = 0.0
        total_num = 0

        batch_bar = tqdm(
            dataloader,
            desc=f"[Client {client_id}] warmup ep {ep+1}/{epochs}",
            ncols=120,
            leave=False,
        )

        for batch in batch_bar:
            images = batch["image"].float().to(device)
            labels = batch["labels"].float().to(device)
            mask = batch["mask"].float().to(device)

            with torch.no_grad():
                _, _, _, label_emb = global_model(
                    images,
                    mask.clone(),
                    args.learn_emb_type,
                    emb_feat,
                    clip_model,
                    return_label_emb=True,
                )

            label_ids = torch.arange(
                label_emb.shape[1],
                device=device
            ).unsqueeze(0).expand(label_emb.shape[0], -1).reshape(-1)
            feat = label_emb.reshape(-1, label_emb.shape[-1])
            if use_layer_norm:
                feat = F.layer_norm(feat, feat.shape[-1:])

            optimizer.zero_grad()
            x_hat, z = sae_model(feat)
            z_all = z.reshape(label_emb.shape[0], label_emb.shape[1], -1)
            sae_total_loss, recon_loss, sparse_loss = sae_loss(
                feat, x_hat, z, l1_lambda=l1_lambda
            )
            selective_loss = latent_label_selective_loss(
                z,
                label_ids=label_ids,
                num_labels=args.num_labels,
            )
            coupled_ids, coupled_weights = get_batch_coupled_label_stats(
                labels,
                forget_cls=args.forget_cls,
                top_m=coupling_topm,
            )
            overlap_loss = latent_coupling_overlap_loss(
                z_all,
                labels=labels,
                forget_cls=args.forget_cls,
                coupled_ids=coupled_ids,
                coupled_weights=coupled_weights,
            )
            loss = sae_total_loss + selective_lambda * selective_loss + overlap_lambda * overlap_loss
            loss.backward()
            optimizer.step()

            bs = feat.size(0)
            total_loss_sum += loss.item() * bs
            recon_loss_sum += recon_loss.item() * bs
            sparse_loss_sum += sparse_loss.item() * bs
            selective_loss_sum += selective_loss.item() * bs
            overlap_loss_sum += overlap_loss.item() * bs
            total_num += bs

            batch_bar.set_postfix({
                "featN": feat.size(0),
                "loss": f"{loss.item():.5f}",
                "recon": f"{recon_loss.item():.5f}",
                "sparse": f"{sparse_loss.item():.5f}",
                "selective": f"{selective_loss.item():.5f}",
                "overlap": f"{overlap_loss.item():.5f}",
            })

        epoch_log = {
            "total": total_loss_sum / max(total_num, 1),
            "recon": recon_loss_sum / max(total_num, 1),
            "sparse": sparse_loss_sum / max(total_num, 1),
            "selective": selective_loss_sum / max(total_num, 1),
            "overlap": overlap_loss_sum / max(total_num, 1),
        }
        epoch_logs.append(epoch_log)

        epoch_bar.set_postfix({
            "total": f"{epoch_log['total']:.5f}",
            "recon": f"{epoch_log['recon']:.5f}",
            "sparse": f"{epoch_log['sparse']:.5f}",
            "selective": f"{epoch_log['selective']:.5f}",
            "overlap": f"{epoch_log['overlap']:.5f}",
        })

    return sae_model, epoch_logs


def local_train_sae_with_distill(
    global_model,
    teacher_sae,
    student_sae,
    dataloader,
    device,
    args,
    emb_feat,
    clip_model,
    epochs=1,
    lr=1e-3,
    l1_lambda=1e-4,
    distill_lambda=1.0,
    selective_lambda=0.0,
    overlap_lambda=0.0,
    coupling_topm=5,
    distill_type="cosine",
    use_layer_norm=True,
    client_id=None,
):
    """
    客户端本地训练 student SAE
    """
    global_model.eval()
    global_model.to(device)
    for p in global_model.parameters():
        p.requires_grad = False

    teacher_sae.eval()
    teacher_sae.to(device)
    for p in teacher_sae.parameters():
        p.requires_grad = False

    student_sae.train()
    student_sae.to(device)

    optimizer = torch.optim.Adam(student_sae.parameters(), lr=lr)

    epoch_logs = []

    epoch_bar = tqdm(
        range(epochs),
        desc=f"[Client {client_id}] Local SAE Epoch",
        ncols=120,
        leave=False,
    )

    for ep in epoch_bar:
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        sparse_loss_sum = 0.0
        distill_loss_sum = 0.0
        selective_loss_sum = 0.0
        overlap_loss_sum = 0.0
        total_num = 0

        batch_bar = tqdm(
            dataloader,
            desc=f"[Client {client_id}] ep {ep+1}/{epochs}",
            ncols=120,
            leave=False,
        )

        for batch in batch_bar:
            images = batch["image"].float().to(device)
            labels = batch["labels"].float().to(device)
            mask   = batch["mask"].float().to(device)

            # 1) 冻结主模型提 label_emb
            with torch.no_grad():
                _, _, _, label_emb = global_model(
                    images,
                    mask.clone(),
                    args.learn_emb_type,
                    emb_feat,
                    clip_model,
                    return_label_emb=True,
                )

            label_ids = torch.arange(
                label_emb.shape[1],
                device=device
            ).unsqueeze(0).expand(label_emb.shape[0], -1).reshape(-1)
            # (B, L, D) -> (B*L, D)
            feat = label_emb.reshape(-1, label_emb.shape[-1])

            # 这里就是你说的“构建系数矩阵 / 特征矩阵”阶段
            # feat 相当于当前 batch 的 SAE 输入矩阵
            if use_layer_norm:
                feat = F.layer_norm(feat, feat.shape[-1:])

            # 2) teacher SAE 给 z_teacher
            with torch.no_grad():
                _, z_teacher = teacher_sae(feat)

            # 3) student SAE 前向
            optimizer.zero_grad()
            x_hat_student, z_student = student_sae(feat)
            z_all = z_student.reshape(label_emb.shape[0], label_emb.shape[1], -1)
            coupled_ids, coupled_weights = get_batch_coupled_label_stats(
                labels,
                forget_cls=args.forget_cls,
                top_m=coupling_topm,
            )
            overlap_loss = latent_coupling_overlap_loss(
                z_all,
                labels=labels,
                forget_cls=args.forget_cls,
                coupled_ids=coupled_ids,
                coupled_weights=coupled_weights,
            )

            # 4) 蒸馏版 SAE loss
            loss, recon_loss, sparse_loss, distill_loss, selective_loss, overlap_loss = sae_distill_loss(
                feat,
                x_hat_student,
                z_student,
                z_teacher,
                label_ids=label_ids,
                num_labels=args.num_labels,
                l1_lambda=l1_lambda,
                distill_lambda=distill_lambda,
                selective_lambda=selective_lambda,
                overlap_loss=overlap_loss,
                overlap_lambda=overlap_lambda,
                distill_type=distill_type,
            )

            loss.backward()
            optimizer.step()

            bs = feat.size(0)
            total_loss_sum += loss.item() * bs
            recon_loss_sum += recon_loss.item() * bs
            sparse_loss_sum += sparse_loss.item() * bs
            distill_loss_sum += distill_loss.item() * bs
            selective_loss_sum += selective_loss.item() * bs
            overlap_loss_sum += overlap_loss.item() * bs
            total_num += bs

            batch_bar.set_postfix({
                "featN": feat.size(0),
                "loss": f"{loss.item():.5f}",
                "recon": f"{recon_loss.item():.5f}",
                "sparse": f"{sparse_loss.item():.5f}",
                "distill": f"{distill_loss.item():.5f}",
                "selective": f"{selective_loss.item():.5f}",
                "overlap": f"{overlap_loss.item():.5f}",
            })

        epoch_log = {
            "total": total_loss_sum / max(total_num, 1),
            "recon": recon_loss_sum / max(total_num, 1),
            "sparse": sparse_loss_sum / max(total_num, 1),
            "distill": distill_loss_sum / max(total_num, 1),
            "selective": selective_loss_sum / max(total_num, 1),
            "overlap": overlap_loss_sum / max(total_num, 1),
        }
        epoch_logs.append(epoch_log)

        epoch_bar.set_postfix({
            "total": f"{epoch_log['total']:.5f}",
            "recon": f"{epoch_log['recon']:.5f}",
            "sparse": f"{epoch_log['sparse']:.5f}",
            "distill": f"{epoch_log['distill']:.5f}",
            "selective": f"{epoch_log['selective']:.5f}",
            "overlap": f"{epoch_log['overlap']:.5f}",
        })

    return student_sae, epoch_logs


def aggregate_sae_models(global_sae, local_saes, weights, verbose=True):
    """
    SAE 参数聚合（FedAvg）
    """
    global_dict = global_sae.state_dict()
    keys = list(global_dict.keys())

    agg_bar = tqdm(
        keys,
        desc="[Server] Aggregate SAE params",
        ncols=120,
        leave=False,
        disable=not verbose,
    )

    for k in agg_bar:
        agg = None
        for i, local_sae in enumerate(local_saes):
            local_tensor = local_sae.state_dict()[k].float()
            if agg is None:
                agg = weights[i] * local_tensor
            else:
                agg += weights[i] * local_tensor
        global_dict[k] = agg

    global_sae.load_state_dict(global_dict)
    return global_sae


def federated_train_sae(
    args,
    global_model,
    global_sae,
    train_dl_global,
    partition_idx_map,
    device,
    emb_feat,
    clip_model,
    sae_rounds=3,
    sae_local_epochs=1,
    sae_lr=1e-3,
    l1_lambda=1e-4,
    selective_lambda=0.0,
    overlap_lambda=0.0,
    coupling_topm=5,
):
    """
    联邦普通 SAE 训练（无蒸馏）
    """
    global_model.eval()
    global_model.to(device)
    for p in global_model.parameters():
        p.requires_grad = False

    global_sae.to(device)

    round_logs = []

    round_bar = tqdm(
        range(sae_rounds),
        desc="Federated SAE Warmup Round",
        ncols=120,
    )

    for r in round_bar:
        local_saes = []
        local_sizes = []
        client_logs = []

        client_bar = tqdm(
            range(args.n_parties),
            desc=f"[Warmup Round {r+1}/{sae_rounds}] Clients",
            ncols=120,
            leave=False,
        )

        for client_id in client_bar:
            # 取出当前客户端的样本并 dataloader
            sub_dst = Subset(train_dl_global.dataset, partition_idx_map[client_id])
            train_dl_local = DataLoader(
                sub_dst,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=False,
            )

            local_sae = copy.deepcopy(global_sae).to(device)

            local_sae, epoch_logs = local_train_sae(
                global_model=global_model,
                sae_model=local_sae,
                dataloader=train_dl_local,
                device=device,
                args=args,
                emb_feat=emb_feat,
                clip_model=clip_model,
                epochs=sae_local_epochs,
                lr=sae_lr,
                l1_lambda=l1_lambda,
                selective_lambda=selective_lambda,
                overlap_lambda=overlap_lambda,
                coupling_topm=coupling_topm,
                use_layer_norm=True,
                client_id=client_id,
            )

            local_saes.append(copy.deepcopy(local_sae).cpu())
            local_sizes.append(len(sub_dst))
            client_logs.append({
                "client_id": client_id,
                "num_samples": len(sub_dst),
                "epoch_logs": epoch_logs,
            })

            last_log = epoch_logs[-1]
            client_bar.set_postfix({
                "client": client_id,
                "samples": len(sub_dst),
                "total": f"{last_log['total']:.5f}",
                "recon": f"{last_log['recon']:.5f}",
                "selective": f"{last_log['selective']:.5f}",
                "overlap": f"{last_log['overlap']:.5f}",
            })

            del local_sae
            torch.cuda.empty_cache()

        total_size = sum(local_sizes)
        weights = [n / total_size for n in local_sizes]

        global_sae = aggregate_sae_models(global_sae.cpu(), local_saes, weights, verbose=True)
        global_sae = global_sae.to(device)

        mean_total = np_mean([c["epoch_logs"][-1]["total"] for c in client_logs])
        mean_recon = np_mean([c["epoch_logs"][-1]["recon"] for c in client_logs])
        mean_sparse = np_mean([c["epoch_logs"][-1]["sparse"] for c in client_logs])
        mean_selective = np_mean([c["epoch_logs"][-1]["selective"] for c in client_logs])
        mean_overlap = np_mean([c["epoch_logs"][-1]["overlap"] for c in client_logs])

        round_bar.set_postfix({
            "mean_total": f"{mean_total:.5f}",
            "mean_recon": f"{mean_recon:.5f}",
            "mean_sparse": f"{mean_sparse:.5f}",
            "mean_selective": f"{mean_selective:.5f}",
            "mean_overlap": f"{mean_overlap:.5f}",
        })

        print(
            f"\n[Warmup SAE Round {r+1}/{sae_rounds}] "
            f"mean_total={mean_total:.6f} "
            f"mean_recon={mean_recon:.6f} "
            f"mean_sparse={mean_sparse:.6f} "
            f"mean_selective={mean_selective:.6f} "
            f"mean_overlap={mean_overlap:.6f}"
        )

        round_logs.append({
            "round": r,
            "client_logs": client_logs,
            "round_mean_total": mean_total,
            "round_mean_recon": mean_recon,
            "round_mean_sparse": mean_sparse,
            "round_mean_selective": mean_selective,
            "round_mean_overlap": mean_overlap,
        })

    return global_sae, round_logs


def federated_train_sae_with_distill(
    args,
    global_model,
    global_sae,
    train_dl_global,
    partition_idx_map,
    device,
    emb_feat,
    clip_model,
    sae_rounds=3,
    sae_local_epochs=1,
    sae_lr=1e-3,
    l1_lambda=1e-4,
    distill_lambda=1.0,
    selective_lambda=0.0,
    overlap_lambda=0.0,
    coupling_topm=5,
    distill_type="cosine",
):
    """
    联邦 SAE + 蒸馏
    """
    global_model.eval()
    global_model.to(device)
    for p in global_model.parameters():
        p.requires_grad = False

    global_sae.to(device)

    round_logs = []

    round_bar = tqdm(
        range(sae_rounds),
        desc="Federated SAE Distill Round",
        ncols=120,
    )

    for r in round_bar:
        local_saes = []
        local_sizes = []
        client_logs = []

        client_bar = tqdm(
            range(args.n_parties),
            desc=f"[Round {r+1}/{sae_rounds}] Clients",
            ncols=120,
            leave=False,
        )

        for client_id in client_bar:
            sub_dst = Subset(train_dl_global.dataset, partition_idx_map[client_id])
            train_dl_local = DataLoader(
                sub_dst,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                drop_last=False,
            )

            # student 从当前 global_sae 初始化
            student_sae = copy.deepcopy(global_sae).to(device)

            # teacher 就是当前 global_sae
            teacher_sae = copy.deepcopy(global_sae).to(device)
            teacher_sae.eval()
            for p in teacher_sae.parameters():
                p.requires_grad = False

            student_sae, epoch_logs = local_train_sae_with_distill(
                global_model=global_model,
                teacher_sae=teacher_sae,
                student_sae=student_sae,
                dataloader=train_dl_local,
                device=device,
                args=args,
                emb_feat=emb_feat,
                clip_model=clip_model,
                epochs=sae_local_epochs,
                lr=sae_lr,
                l1_lambda=l1_lambda,
                distill_lambda=distill_lambda,
                selective_lambda=selective_lambda,
                overlap_lambda=overlap_lambda,
                coupling_topm=coupling_topm,
                distill_type=distill_type,
                use_layer_norm=True,
                client_id=client_id,
            )

            local_saes.append(copy.deepcopy(student_sae).cpu())
            local_sizes.append(len(sub_dst))
            client_logs.append({
                "client_id": client_id,
                "num_samples": len(sub_dst),
                "epoch_logs": epoch_logs,
            })

            last_log = epoch_logs[-1]
            client_bar.set_postfix({
                "client": client_id,
                "samples": len(sub_dst),
                "total": f"{last_log['total']:.5f}",
                "distill": f"{last_log['distill']:.5f}",
                "selective": f"{last_log['selective']:.5f}",
                "overlap": f"{last_log['overlap']:.5f}",
            })

            del teacher_sae
            del student_sae
            torch.cuda.empty_cache()

        total_size = sum(local_sizes)
        weights = [n / total_size for n in local_sizes]

        global_sae = aggregate_sae_models(global_sae.cpu(), local_saes, weights, verbose=True)
        global_sae = global_sae.to(device)

        # round摘要
        mean_total = np_mean([c["epoch_logs"][-1]["total"] for c in client_logs])
        mean_recon = np_mean([c["epoch_logs"][-1]["recon"] for c in client_logs])
        mean_sparse = np_mean([c["epoch_logs"][-1]["sparse"] for c in client_logs])
        mean_distill = np_mean([c["epoch_logs"][-1]["distill"] for c in client_logs])
        mean_selective = np_mean([c["epoch_logs"][-1]["selective"] for c in client_logs])
        mean_overlap = np_mean([c["epoch_logs"][-1]["overlap"] for c in client_logs])

        round_bar.set_postfix({
            "mean_total": f"{mean_total:.5f}",
            "mean_recon": f"{mean_recon:.5f}",
            "mean_sparse": f"{mean_sparse:.5f}",
            "mean_distill": f"{mean_distill:.5f}",
            "mean_selective": f"{mean_selective:.5f}",
            "mean_overlap": f"{mean_overlap:.5f}",
        })

        print(
            f"\n[SAE Round {r+1}/{sae_rounds}] "
            f"mean_total={mean_total:.6f} "
            f"mean_recon={mean_recon:.6f} "
            f"mean_sparse={mean_sparse:.6f} "
            f"mean_distill={mean_distill:.6f} "
            f"mean_selective={mean_selective:.6f} "
            f"mean_overlap={mean_overlap:.6f}"
        )

        round_logs.append({
            "round": r,
            "client_logs": client_logs,
            "round_mean_total": mean_total,
            "round_mean_recon": mean_recon,
            "round_mean_sparse": mean_sparse,
            "round_mean_distill": mean_distill,
            "round_mean_selective": mean_selective,
            "round_mean_overlap": mean_overlap,
        })

    return global_sae, round_logs


def np_mean(x):
    if len(x) == 0:
        return 0.0
    return sum(x) / len(x)

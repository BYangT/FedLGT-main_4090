import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .sae import SparseAutoEncoder
except ImportError:
    from sae import SparseAutoEncoder


_GLOBAL_SAE_MODEL = None
_GLOBAL_SAE_CKPT = None


def _get_sae_model(args, device):
    """
    懒加载 SAE，不改原函数签名。
    依赖 args 里的字段：
        args.sae_ckpt          : SAE 权重路径（必须）
        args.sae_input_dim     : 输入维度，默认 512
        args.sae_latent_dim    : latent 维度，默认 1024
        args.sae_activation    : "relu" / "softplus" / "identity"，默认 "relu"
        args.sae_use_layer_norm: 是否在送入 SAE 前做 layer norm，默认 True
    """
    global _GLOBAL_SAE_MODEL, _GLOBAL_SAE_CKPT

    sae_ckpt = getattr(args, "sae_ckpt", None)
    input_dim = getattr(args, "sae_input_dim", 512)
    latent_dim = getattr(args, "sae_latent_dim", 1024)
    activation = getattr(args, "sae_activation", "relu")

    if sae_ckpt is None:
        raise ValueError(
            "args.sae_ckpt is required. "
            "Please set for example: args.sae_ckpt = 'ulearn_model/fed_sae_distill.pth'"
        )

    # 如果缓存存在且 ckpt 路径没变，直接返回
    if _GLOBAL_SAE_MODEL is not None and _GLOBAL_SAE_CKPT == sae_ckpt:
        return _GLOBAL_SAE_MODEL

    sae_model = SparseAutoEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        activation=activation,
    )

    state_dict = torch.load(sae_ckpt, map_location=device)
    sae_model.load_state_dict(state_dict)
    sae_model.to(device)
    sae_model.eval()

    # 冻结 SAE 参数，但不能在后续 forward 时整体包 no_grad，
    # 否则 forget loss 不能通过 z 回传到主模型
    for p in sae_model.parameters():
        p.requires_grad = False

    _GLOBAL_SAE_MODEL = sae_model
    _GLOBAL_SAE_CKPT = sae_ckpt
    return _GLOBAL_SAE_MODEL


def _encode_label_emb_with_sae(label_emb, sae_model, args=None):
    """
    label_emb: (B, L, D_raw)
    return:
        z_all: (B, L, latent_dim)

    注意：
    联邦蒸馏训练 SAE 时，对输入 feat 做了 layer norm。
    这里必须保持一致，否则训练/使用分布不一致。
    """
    B, L, D_raw = label_emb.shape
    label_emb_flat = label_emb.reshape(B * L, D_raw)

    use_layer_norm = True if args is None else getattr(args, "sae_use_layer_norm", True)
    if use_layer_norm:
        label_emb_flat = F.layer_norm(label_emb_flat, label_emb_flat.shape[-1:])

    _, z_flat = sae_model(label_emb_flat)   # SAE forward -> (x_hat, z)
    z_all = z_flat.reshape(B, L, -1)
    return z_all


def collect_topk_dims_for_class(
    model,
    dataloader,
    forget_cls: int,
    K: int,
    device,
    args,
    emb_feat,
    clip_model,
):
    model.eval()
    model.to(device)

    sae_model = _get_sae_model(args, device)
    latent_dim = sae_model.latent_dim
    beta_neg = 0.5   # 你可以后面再调
    coupling_topm = getattr(args, "sae_coupling_topm", 5)
    coupling_penalty_lambda = getattr(args, "topk_coupling_lambda", 0.5)

    # ===== 目标类正/负样本统计 =====
    pos_sum = torch.zeros(latent_dim, device=device)
    pos_cnt = 0

    neg_sum = torch.zeros(latent_dim, device=device)
    neg_cnt = 0
    cooccur_counts = None

    # ===== 每个类单独统计正样本均值 =====
    num_labels = None
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].float().to(device)
            labels = batch['labels'].float().to(device)
            mask   = batch['mask'].float().to(device)

            _, _, _, label_emb = model(
                images,
                mask.clone(),
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )
            _, num_labels, _ = label_emb.shape
            break

    if num_labels is None:
        raise ValueError("Empty dataloader in collect_topk_dims_for_class")

    class_pos_sum = torch.zeros(num_labels, latent_dim, device=device)
    class_pos_cnt = torch.zeros(num_labels, device=device)
    cooccur_counts = torch.zeros(num_labels, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collect top-K stats (SAE latent)"):
            images = batch['image'].float().to(device)
            labels = batch['labels'].float().to(device)
            mask   = batch['mask'].float().to(device)
            mask_in = mask.clone()

            _, _, _, label_emb = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )
            # label_emb: (B, L, D_raw)
            B, L, D_raw = label_emb.shape

            # ===== 原始特征 -> SAE latent =====
            z_all = _encode_label_emb_with_sae(label_emb, sae_model, args)   # (B, L, latent_dim)

            # ===== 1) 目标类正/负样本统计 =====
            z_f = z_all[:, forget_cls, :]  # (B, latent_dim)

            pos_idx = (labels[:, forget_cls] == 1)
            neg_idx = (labels[:, forget_cls] == 0)

            if pos_idx.any():
                z_pos = z_f[pos_idx]
                pos_sum += z_pos.abs().sum(dim=0)
                pos_cnt += z_pos.size(0)
                cooccur_counts += labels[pos_idx].sum(dim=0)

            if neg_idx.any():
                z_neg = z_f[neg_idx]
                neg_sum += z_neg.abs().sum(dim=0)
                neg_cnt += z_neg.size(0)

            # ===== 2) 每个类单独统计正样本 =====
            for c in range(L):
                cls_idx = (labels[:, c] == 1)
                if cls_idx.any():
                    z_cls = z_all[cls_idx, c, :]  # (N_cls, latent_dim)
                    class_pos_sum[c] += z_cls.abs().sum(dim=0)
                    class_pos_cnt[c] += z_cls.size(0)

    # ===== 均值 =====
    pos_mean = pos_sum / (pos_cnt + 1e-6)      # (latent_dim,)
    neg_mean = neg_sum / (neg_cnt + 1e-6)      # (latent_dim,)

    class_mean = class_pos_sum / (class_pos_cnt.unsqueeze(1) + 1e-6)   # (L, latent_dim)
    cooccur_counts[forget_cls] = 0.0

    # ===== 最强竞争类 =====
    competitor_mean = class_mean.clone()
    competitor_mean[forget_cls] = -1e9

    max_comp_mean, max_comp_cls = competitor_mean.max(dim=0)   # (latent_dim,), (latent_dim,)

    # ===== 新版基础分数 =====
    # 目标类高于最强竞争类 + 目标类高于自己负样本
    base_score = (pos_mean - max_comp_mean) + beta_neg * (pos_mean - neg_mean)

    # ===== 分类头权重投影到 latent 空间 =====
    with torch.no_grad():
        W_c_raw = model.output_linear.weight[forget_cls].to(device)   # (D_raw,)
        W_enc = sae_model.encoder.weight.to(device)                   # (latent_dim, D_raw)
        W_c_latent = torch.matmul(W_enc, W_c_raw)                     # (latent_dim,)

    score = base_score * W_c_latent.abs()

    # ===== 高耦合类惩罚：从 top-K 打分里减去高耦合类响应 =====
    coupled_mask = cooccur_counts > 0
    coupled_penalty = torch.zeros(latent_dim, device=device)
    top_coupled_cls = torch.empty(0, dtype=torch.long, device=device)
    top_coupled_weights = torch.empty(0, device=device)
    if coupled_mask.any():
        top_m = min(int(coupling_topm), int(coupled_mask.sum().item()))
        top_coupled_vals, top_coupled_cls = torch.topk(cooccur_counts, k=top_m, largest=True)
        top_coupled_weights = top_coupled_vals / top_coupled_vals.sum().clamp_min(1e-6)
        coupled_penalty = (class_mean[top_coupled_cls] * top_coupled_weights.unsqueeze(1)).sum(dim=0)
        score = score - coupling_penalty_lambda * coupled_penalty * W_c_latent.abs()

    # ===== 公共特征过滤 =====
    pos_thr = pos_mean.mean() + 0.5 * pos_mean.std()

    valid_comp = max_comp_mean[max_comp_mean > -1e8]
    if valid_comp.numel() > 0:
        comp_thr = valid_comp.mean() + 0.5 * valid_comp.std()
    else:
        comp_thr = torch.tensor(0.0, device=device)

    public_mask = (pos_mean > pos_thr) & (max_comp_mean > comp_thr)

    score_thr = score.mean() + 0.5 * score.std()
    score_mask = score > score_thr

    mask_good = score_mask & (~public_mask)

    score_filtered = score.clone()
    score_filtered[~mask_good] = -1e9

    # ===== 取 Top-K latent 维度 =====
    topk_vals, topk_idx = torch.topk(score_filtered, k=K, largest=True)

    # ===== 额外输出：Top-K 维度里最相关的几个类 =====
    # 定义：Top-K 维度对应的 max competitor class 出现频次最高的类
    topk_comp_classes = max_comp_cls[topk_idx]   # (K,)

    related_counts = torch.bincount(topk_comp_classes, minlength=num_labels)
    related_counts[forget_cls] = 0  # 保险起见，去掉自己

    top_m = min(5, num_labels - 1)
    top_related_vals, top_related_cls = torch.topk(
        related_counts, k=top_m, largest=True
    )

    print(f"\n[forget_cls={forget_cls}] Top related classes by Top-K competitor frequency:")
    for cls_id, cnt in zip(top_related_cls.tolist(), top_related_vals.tolist()):
        if cnt > 0:
            print(f"  class {cls_id}: {cnt} times")

    if top_coupled_cls.numel() > 0:
        print(f"[forget_cls={forget_cls}] High-coupling classes used for score penalty:")
        for cls_id, weight in zip(top_coupled_cls.tolist(), top_coupled_weights.tolist()):
            print(f"  class {cls_id}: weight={weight:.4f}")

    return topk_idx, score


def collect_client_topk_stats(
    model,
    dataloader,
    forget_cls: int,
    device,
    args,
    emb_feat,
    clip_model,
):
    """
    客户端本地统计 top-K score 所需的一阶量。
    服务器后续只聚合这些统计量，不直接访问样本。
    """
    model.eval()
    model.to(device)

    sae_model = _get_sae_model(args, device)
    latent_dim = sae_model.latent_dim

    pos_sum = torch.zeros(latent_dim, device=device)
    neg_sum = torch.zeros(latent_dim, device=device)
    pos_cnt = 0
    neg_cnt = 0
    num_labels = None
    class_pos_sum = None
    class_pos_cnt = None
    cooccur_counts = None

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].float().to(device)
            labels = batch['labels'].float().to(device)
            mask = batch['mask'].float().to(device)

            _, _, _, label_emb = model(
                images,
                mask.clone(),
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )

            B, L, _ = label_emb.shape
            if num_labels is None:
                num_labels = L
                class_pos_sum = torch.zeros(num_labels, latent_dim, device=device)
                class_pos_cnt = torch.zeros(num_labels, device=device)
                cooccur_counts = torch.zeros(num_labels, device=device)

            z_all = _encode_label_emb_with_sae(label_emb, sae_model, args)
            z_f = z_all[:, forget_cls, :]

            pos_idx = labels[:, forget_cls] == 1
            neg_idx = labels[:, forget_cls] == 0

            if pos_idx.any():
                z_pos = z_f[pos_idx]
                pos_sum += z_pos.abs().sum(dim=0)
                pos_cnt += z_pos.size(0)
                cooccur_counts += labels[pos_idx].sum(dim=0)

            if neg_idx.any():
                z_neg = z_f[neg_idx]
                neg_sum += z_neg.abs().sum(dim=0)
                neg_cnt += z_neg.size(0)

            for c in range(L):
                cls_idx = labels[:, c] == 1
                if cls_idx.any():
                    z_cls = z_all[cls_idx, c, :]
                    class_pos_sum[c] += z_cls.abs().sum(dim=0)
                    class_pos_cnt[c] += z_cls.size(0)

    if num_labels is None:
        raise ValueError("Empty dataloader in collect_client_topk_stats")

    return {
        "pos_sum": pos_sum.detach().cpu(),
        "neg_sum": neg_sum.detach().cpu(),
        "pos_cnt": int(pos_cnt),
        "neg_cnt": int(neg_cnt),
        "class_pos_sum": class_pos_sum.detach().cpu(),
        "class_pos_cnt": class_pos_cnt.detach().cpu(),
        "cooccur_counts": cooccur_counts.detach().cpu(),
        "num_labels": int(num_labels),
    }


def aggregate_topk_score_from_client_stats(
    client_stats,
    model,
    forget_cls: int,
    device,
    args,
):
    """
    服务器只基于客户端上传的 top-K 统计量恢复全局 score。
    """
    sae_model = _get_sae_model(args, device)
    latent_dim = sae_model.latent_dim
    beta_neg = 0.5
    coupling_topm = getattr(args, "sae_coupling_topm", 5)
    coupling_penalty_lambda = getattr(args, "topk_coupling_lambda", 0.5)

    pos_sum = torch.zeros(latent_dim, device=device)
    neg_sum = torch.zeros(latent_dim, device=device)
    pos_cnt = 0
    neg_cnt = 0
    class_pos_sum = None
    class_pos_cnt = None
    cooccur_counts = None
    num_labels = None

    for s in client_stats:
        pos_sum += s["pos_sum"].to(device)
        neg_sum += s["neg_sum"].to(device)
        pos_cnt += int(s["pos_cnt"])
        neg_cnt += int(s["neg_cnt"])
        if class_pos_sum is None:
            num_labels = int(s["num_labels"])
            class_pos_sum = s["class_pos_sum"].to(device)
            class_pos_cnt = s["class_pos_cnt"].to(device)
            cooccur_counts = s["cooccur_counts"].to(device)
        else:
            class_pos_sum += s["class_pos_sum"].to(device)
            class_pos_cnt += s["class_pos_cnt"].to(device)
            cooccur_counts += s["cooccur_counts"].to(device)

    if num_labels is None:
        raise ValueError("Empty client_stats in aggregate_topk_score_from_client_stats")

    pos_mean = pos_sum / (pos_cnt + 1e-6)
    neg_mean = neg_sum / (neg_cnt + 1e-6)
    class_mean = class_pos_sum / (class_pos_cnt.unsqueeze(1) + 1e-6)
    cooccur_counts[forget_cls] = 0.0

    competitor_mean = class_mean.clone()
    competitor_mean[forget_cls] = -1e9
    max_comp_mean, max_comp_cls = competitor_mean.max(dim=0)

    base_score = (pos_mean - max_comp_mean) + beta_neg * (pos_mean - neg_mean)

    with torch.no_grad():
        W_c_raw = model.output_linear.weight[forget_cls].to(device)
        W_enc = sae_model.encoder.weight.to(device)
        W_c_latent = torch.matmul(W_enc, W_c_raw)

    score = base_score * W_c_latent.abs()

    top_coupled_cls = torch.empty(0, dtype=torch.long, device=device)
    top_coupled_weights = torch.empty(0, device=device)
    coupled_mask = cooccur_counts > 0
    if coupled_mask.any():
        top_m = min(int(coupling_topm), int(coupled_mask.sum().item()))
        top_coupled_vals, top_coupled_cls = torch.topk(cooccur_counts, k=top_m, largest=True)
        top_coupled_weights = top_coupled_vals / top_coupled_vals.sum().clamp_min(1e-6)
        coupled_penalty = (class_mean[top_coupled_cls] * top_coupled_weights.unsqueeze(1)).sum(dim=0)
        score = score - coupling_penalty_lambda * coupled_penalty * W_c_latent.abs()

    pos_thr = pos_mean.mean() + 0.5 * pos_mean.std()
    valid_comp = max_comp_mean[max_comp_mean > -1e8]
    if valid_comp.numel() > 0:
        comp_thr = valid_comp.mean() + 0.5 * valid_comp.std()
    else:
        comp_thr = torch.tensor(0.0, device=device)
    public_mask = (pos_mean > pos_thr) & (max_comp_mean > comp_thr)

    score_thr = score.mean() + 0.5 * score.std()
    score_mask = score > score_thr
    mask_good = score_mask & (~public_mask)
    score_filtered = score.clone()
    score_filtered[~mask_good] = -1e9

    topk_comp_classes = max_comp_cls[torch.topk(score_filtered, k=min(score_filtered.numel(),  min(64, score_filtered.numel())), largest=True).indices]
    related_counts = torch.bincount(topk_comp_classes, minlength=num_labels)
    related_counts[forget_cls] = 0
    top_m = min(5, num_labels - 1)
    top_related_vals, top_related_cls = torch.topk(related_counts, k=top_m, largest=True)

    print(f"\n[forget_cls={forget_cls}] Top related classes by Top-K competitor frequency:")
    for cls_id, cnt in zip(top_related_cls.tolist(), top_related_vals.tolist()):
        if cnt > 0:
            print(f"  class {cls_id}: {cnt} times")

    if top_coupled_cls.numel() > 0:
        print(f"[forget_cls={forget_cls}] High-coupling classes used for score penalty:")
        for cls_id, weight in zip(top_coupled_cls.tolist(), top_coupled_weights.tolist()):
            print(f"  class {cls_id}: weight={weight:.4f}")

    return score_filtered.detach().cpu()


def _minmax_normalize(x, eps=1e-8):
    if x.numel() == 0:
        return x
    x_min = x.min()
    x_max = x.max()
    if (x_max - x_min).abs() <= eps:
        return torch.ones_like(x)
    return (x - x_min) / (x_max - x_min + eps)


def rerank_topk_dims_with_gradient(
    model,
    dataloader,
    forget_cls: int,
    candidate_idx: torch.Tensor,
    candidate_score: torch.Tensor,
    final_k: int,
    device,
    args,
    emb_feat,
    clip_model,
    max_batches: int = 3,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.2,
):
    """
    低算力重排：
    1) 原始 score 先筛出 top-M 候选
    2) 只在少量目标类样本 batch 上，用一次反向传播拿候选维的梯度敏感度
    3) 再做去冗余贪心选择，得到最终 top-K 和对应权重
    """
    model.eval()
    model.to(device)

    sae_model = _get_sae_model(args, device)
    sae_model.eval()

    candidate_idx = candidate_idx.to(device)
    candidate_score = candidate_score.to(device)
    M = candidate_idx.numel()
    if M == 0:
        return candidate_idx, torch.empty(0, device=device)

    importance_sum = torch.zeros(M, device=device)
    sample_count = 0
    z_samples = []

    batch_seen = 0
    for batch in dataloader:
        if batch_seen >= max_batches:
            break

        images = batch["image"].float().to(device)
        labels = batch["labels"].float().to(device)
        mask = batch["mask"].float().to(device)

        pos_idx = labels[:, forget_cls] == 1
        if not pos_idx.any():
            continue

        model.zero_grad(set_to_none=True)

        _, _, _, label_emb = model(
            images,
            mask.clone(),
            args.learn_emb_type,
            emb_feat,
            clip_model,
            return_label_emb=True,
        )

        z_all = _encode_label_emb_with_sae(label_emb, sae_model, args)
        if not z_all.requires_grad:
            # 极端情况下如果上游前向把 label_emb 脱图了，就回退到原始候选分数。
            continue
        z_all.retain_grad()

        z_f = z_all[:, forget_cls, :]
        z_pos_full = z_f[pos_idx]
        z_pos = z_pos_full[:, candidate_idx]
        neg_idx = labels[:, forget_cls] == 0

        if neg_idx.any():
            z_neg = z_f[neg_idx][:, candidate_idx]
            mu_neg = z_neg.detach().mean(dim=0, keepdim=True)
            rank_loss = 0.8 * (z_pos ** 2).mean() + 0.2 * ((z_pos - mu_neg) ** 2).mean()
        else:
            rank_loss = (z_pos ** 2).mean()
        rank_loss.backward()

        g_pos = z_all.grad[pos_idx, forget_cls, :][:, candidate_idx]

        n_pos = z_pos.size(0)
        importance_sum += (z_pos.abs() * g_pos.abs()).sum(dim=0)
        sample_count += n_pos
        z_samples.append(z_pos.detach().cpu())
        batch_seen += 1

    if sample_count == 0:
        weight = torch.softmax(candidate_score[:min(final_k, M)], dim=0).detach()
        return candidate_idx[:min(final_k, M)], weight

    importance = importance_sum / float(sample_count)

    # 候选维去冗余：利用少量目标类样本上的候选激活相关性。
    z_cat = torch.cat(z_samples, dim=0)
    if z_cat.size(0) >= 2:
        z_centered = z_cat - z_cat.mean(dim=0, keepdim=True)
        denom = z_centered.pow(2).sum(dim=0).sqrt().clamp_min(1e-6)
        z_norm = z_centered / denom
        corr = (z_norm.t() @ z_norm) / max(z_norm.size(0) - 1, 1)
        corr = corr.abs().clamp(max=1.0).to(device)
        corr.fill_diagonal_(0.0)
    else:
        corr = torch.zeros((M, M), device=device)

    score_norm = _minmax_normalize(candidate_score)
    imp_norm = _minmax_normalize(importance)
    combined = alpha * score_norm + beta * imp_norm

    k_eff = min(final_k, M)
    selected = []
    selected_mask = torch.zeros(M, dtype=torch.bool, device=device)

    for _ in range(k_eff):
        if len(selected) == 0:
            cur_score = combined.clone()
        else:
            red = corr[:, selected].max(dim=1).values
            cur_score = combined - gamma * red
        cur_score[selected_mask] = -1e9
        pick = int(torch.argmax(cur_score).item())
        selected.append(pick)
        selected_mask[pick] = True

    selected_idx = torch.tensor(selected, device=device, dtype=torch.long)
    final_idx = candidate_idx[selected_idx]
    final_raw = combined[selected_idx].clamp_min(1e-6)
    final_weight = final_raw / final_raw.sum().clamp_min(1e-6)

    print(
        f"[Rerank] candidate_M={M}, final_K={k_eff}, "
        f"mean_base={candidate_score.mean().item():.4f}, "
        f"mean_importance={importance.mean().item():.4f}"
    )

    return final_idx.detach(), final_weight.detach()


def collect_client_rerank_stats(
    model,
    dataloader,
    forget_cls: int,
    candidate_idx: torch.Tensor,
    device,
    args,
    emb_feat,
    clip_model,
    max_batches: int = 3,
):
    """
    客户端本地统计：
      - candidate dims 的梯度敏感度累计
      - 目标类正样本在 candidate dims 上的一阶/二阶统计
    服务器只基于这些统计量恢复 importance 和相关性矩阵。
    """
    model.eval()
    model.to(device)

    sae_model = _get_sae_model(args, device)
    sae_model.eval()

    candidate_idx = candidate_idx.to(device)
    M = candidate_idx.numel()
    importance_sum = torch.zeros(M, device=device)
    sample_count = 0
    z_sum = torch.zeros(M, device=device)
    z_xxt = torch.zeros(M, M, device=device)

    batch_seen = 0
    for batch in dataloader:
        if batch_seen >= max_batches:
            break

        images = batch["image"].float().to(device)
        labels = batch["labels"].float().to(device)
        mask = batch["mask"].float().to(device)

        pos_idx = labels[:, forget_cls] == 1
        if not pos_idx.any():
            continue

        model.zero_grad(set_to_none=True)
        _, _, _, label_emb = model(
            images,
            mask.clone(),
            args.learn_emb_type,
            emb_feat,
            clip_model,
            return_label_emb=True,
        )

        z_all = _encode_label_emb_with_sae(label_emb, sae_model, args)
        if not z_all.requires_grad:
            continue
        z_all.retain_grad()

        z_f = z_all[:, forget_cls, :]
        z_pos_full = z_f[pos_idx]
        z_pos = z_pos_full[:, candidate_idx]
        neg_idx = labels[:, forget_cls] == 0

        if neg_idx.any():
            z_neg = z_f[neg_idx][:, candidate_idx]
            mu_neg = z_neg.detach().mean(dim=0, keepdim=True)
            rank_loss = 0.8 * (z_pos ** 2).mean() + 0.2 * ((z_pos - mu_neg) ** 2).mean()
        else:
            rank_loss = (z_pos ** 2).mean()
        rank_loss.backward()

        g_pos = z_all.grad[pos_idx, forget_cls, :][:, candidate_idx]
        n_pos = z_pos.size(0)
        importance_sum += (z_pos.abs() * g_pos.abs()).sum(dim=0)
        sample_count += n_pos
        z_sum += z_pos.detach().sum(dim=0)
        z_xxt += z_pos.detach().t() @ z_pos.detach()
        batch_seen += 1

    return {
        "importance_sum": importance_sum.detach().cpu(),
        "sample_count": int(sample_count),
        "z_sum": z_sum.detach().cpu(),
        "z_xxt": z_xxt.detach().cpu(),
    }


def aggregate_rerank_from_client_stats(
    client_stats,
    candidate_idx: torch.Tensor,
    candidate_score: torch.Tensor,
    final_k: int,
    device,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.2,
):
    """
    服务端只基于客户端上传的 rerank 统计量恢复：
      - importance
      - corr / redundancy
      - final top-K + weights
    """
    candidate_idx = candidate_idx.to(device)
    candidate_score = candidate_score.to(device)
    M = candidate_idx.numel()
    if M == 0:
        return candidate_idx, torch.empty(0, device=device)

    importance_sum = torch.zeros(M, device=device)
    z_sum = torch.zeros(M, device=device)
    z_xxt = torch.zeros(M, M, device=device)
    sample_count = 0

    for s in client_stats:
        importance_sum += s["importance_sum"].to(device)
        z_sum += s["z_sum"].to(device)
        z_xxt += s["z_xxt"].to(device)
        sample_count += int(s["sample_count"])

    if sample_count == 0:
        weight = torch.softmax(candidate_score[:min(final_k, M)], dim=0).detach()
        return candidate_idx[:min(final_k, M)], weight

    importance = importance_sum / float(sample_count)

    mean = z_sum / float(sample_count)
    exx = z_xxt / float(sample_count)
    cov = exx - torch.outer(mean, mean)
    cov = 0.5 * (cov + cov.t())
    var = torch.diag(cov).clamp_min(1e-6)
    denom = torch.sqrt(torch.outer(var, var))
    corr = (cov / denom).abs().clamp(max=1.0)
    corr.fill_diagonal_(0.0)

    score_norm = _minmax_normalize(candidate_score)
    imp_norm = _minmax_normalize(importance)
    combined = alpha * score_norm + beta * imp_norm

    k_eff = min(final_k, M)
    selected = []
    selected_mask = torch.zeros(M, dtype=torch.bool, device=device)
    for _ in range(k_eff):
        if len(selected) == 0:
            cur_score = combined.clone()
        else:
            red = corr[:, selected].max(dim=1).values
            cur_score = combined - gamma * red
        cur_score[selected_mask] = -1e9
        pick = int(torch.argmax(cur_score).item())
        selected.append(pick)
        selected_mask[pick] = True

    selected_idx = torch.tensor(selected, device=device, dtype=torch.long)
    final_idx = candidate_idx[selected_idx]
    final_raw = combined[selected_idx].clamp_min(1e-6)
    final_weight = final_raw / final_raw.sum().clamp_min(1e-6)

    print(
        f"[Rerank-Stats] candidate_M={M}, final_K={k_eff}, "
        f"mean_base={candidate_score.mean().item():.4f}, "
        f"mean_importance={importance.mean().item():.4f}"
    )

    return final_idx.detach(), final_weight.detach()


def estimate_target_subspace(
    model,
    dataloader,
    forget_cls: int,
    topk_idx: torch.Tensor,
    device,
    args,
    emb_feat,
    clip_model,
    topk_weights: torch.Tensor = None,
    max_batches: int = 3,
    subspace_rank: int = 16,
):
    """
    在少量目标类样本上估计 target-dominant latent subspace。
    子空间定义在当前 top-K latent 上，供后续 forget loss 压制投影能量。
    """
    model.eval()
    model.to(device)

    sae_model = _get_sae_model(args, device)
    sae_model.eval()

    topk_idx = topk_idx.to(device)
    if topk_weights is not None:
        topk_weights = topk_weights.detach().to(device).view(1, -1)
        metric_scale = topk_weights.sqrt()
    else:
        metric_scale = None

    pos_chunks = []
    neg_chunks = []
    batch_seen = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch_seen >= max_batches:
                break

            images = batch["image"].float().to(device)
            labels = batch["labels"].float().to(device)
            mask = batch["mask"].float().to(device)

            pos_idx = labels[:, forget_cls] == 1
            neg_idx = labels[:, forget_cls] == 0
            if not pos_idx.any():
                continue

            _, _, _, label_emb = model(
                images,
                mask.clone(),
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )

            z_all = _encode_label_emb_with_sae(label_emb, sae_model, args)
            z_f = z_all[:, forget_cls, :]

            z_pos = z_f[pos_idx][:, topk_idx]
            if metric_scale is not None:
                z_pos = z_pos * metric_scale
            pos_chunks.append(z_pos)

            if neg_idx.any():
                z_neg = z_f[neg_idx][:, topk_idx]
                if metric_scale is not None:
                    z_neg = z_neg * metric_scale
                neg_chunks.append(z_neg)

            batch_seen += 1

    if len(pos_chunks) == 0:
        return None, None

    z_pos_all = torch.cat(pos_chunks, dim=0)
    if len(neg_chunks) > 0:
        z_neg_all = torch.cat(neg_chunks, dim=0)
        neg_center = z_neg_all.mean(dim=0, keepdim=True)
        neg_count = z_neg_all.size(0)
    else:
        z_neg_all = None
        neg_center = torch.zeros((1, z_pos_all.size(1)), device=device, dtype=z_pos_all.dtype)
        neg_count = 0

    z_target = z_pos_all - neg_center
    rank_eff = max(1, min(int(subspace_rank), z_target.size(1), z_target.size(0)))

    if z_target.size(0) == 1:
        basis = F.normalize(z_target, dim=1).t()
    else:
        _, _, vh = torch.linalg.svd(z_target, full_matrices=False)
        basis = vh[:rank_eff].t().contiguous()

    print(
        f"[Subspace] forget_cls={forget_cls}, pos_samples={z_pos_all.size(0)}, "
        f"neg_samples={neg_count}, "
        f"rank={basis.size(1)}"
    )

    return basis.detach(), neg_center.detach()


def collect_client_target_subspace_stats(
    model,
    dataloader,
    forget_cls: int,
    topk_idx: torch.Tensor,
    device,
    args,
    emb_feat,
    clip_model,
    topk_weights: torch.Tensor = None,
    max_batches: int = 3,
):
    """
    客户端本地统计：
      - 目标类正样本一阶矩 sum(x)
      - 目标类正样本二阶矩 sum(x x^T)
      - 目标类负样本一阶矩 sum(x)
    服务器只基于这些统计量恢复全局 target subspace，不直接拼接样本。
    """
    model.eval()
    model.to(device)

    sae_model = _get_sae_model(args, device)
    sae_model.eval()

    topk_idx = topk_idx.to(device)
    if topk_weights is not None:
        topk_weights = topk_weights.detach().to(device).view(1, -1)
        metric_scale = topk_weights.sqrt()
    else:
        metric_scale = None

    K = topk_idx.numel()
    pos_count = 0
    neg_count = 0
    pos_sum = torch.zeros(K, device=device)
    pos_xxt = torch.zeros(K, K, device=device)
    neg_sum = torch.zeros(K, device=device)

    batch_seen = 0
    with torch.no_grad():
        for batch in dataloader:
            if batch_seen >= max_batches:
                break

            images = batch["image"].float().to(device)
            labels = batch["labels"].float().to(device)
            mask = batch["mask"].float().to(device)

            _, _, _, label_emb = model(
                images,
                mask.clone(),
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )

            z_all = _encode_label_emb_with_sae(label_emb, sae_model, args)
            z_f = z_all[:, forget_cls, :][:, topk_idx]
            if metric_scale is not None:
                z_f = z_f * metric_scale

            pos_idx = labels[:, forget_cls] == 1
            neg_idx = labels[:, forget_cls] == 0

            if pos_idx.any():
                z_pos = z_f[pos_idx]
                pos_sum += z_pos.sum(dim=0)
                pos_xxt += z_pos.t() @ z_pos
                pos_count += z_pos.size(0)

            if neg_idx.any():
                z_neg = z_f[neg_idx]
                neg_sum += z_neg.sum(dim=0)
                neg_count += z_neg.size(0)

            batch_seen += 1

    return {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_sum": pos_sum.detach().cpu(),
        "pos_xxt": pos_xxt.detach().cpu(),
        "neg_sum": neg_sum.detach().cpu(),
    }


def aggregate_target_subspace_from_stats(
    client_stats,
    subspace_rank: int,
    device,
):
    """
    服务端只基于客户端上传的一阶/二阶统计量恢复全局 target subspace。
    不需要访问任何原始样本或逐样本 latent。
    """
    total_pos = sum(int(s["pos_count"]) for s in client_stats)
    total_neg = sum(int(s["neg_count"]) for s in client_stats)
    if total_pos <= 0:
        return None, None

    pos_sum = None
    pos_xxt = None
    neg_sum = None
    for s in client_stats:
        if pos_sum is None:
            pos_sum = s["pos_sum"].to(device)
            pos_xxt = s["pos_xxt"].to(device)
            neg_sum = s["neg_sum"].to(device)
        else:
            pos_sum += s["pos_sum"].to(device)
            pos_xxt += s["pos_xxt"].to(device)
            neg_sum += s["neg_sum"].to(device)

    center = (
        neg_sum / float(total_neg)
        if total_neg > 0
        else torch.zeros_like(pos_sum)
    )

    mean_pos = pos_sum / float(total_pos)
    exx = pos_xxt / float(total_pos)
    cov = (
        exx
        - torch.outer(mean_pos, center)
        - torch.outer(center, mean_pos)
        + torch.outer(center, center)
    )
    cov = 0.5 * (cov + cov.t())

    K = cov.size(0)
    rank_eff = max(1, min(int(subspace_rank), K))

    evals, evecs = torch.linalg.eigh(cov)
    top_evals = evals[-rank_eff:]
    if torch.all(top_evals.abs() < 1e-8):
        direction = (mean_pos - center).view(1, -1)
        if direction.norm() <= 1e-8:
            basis = F.one_hot(torch.tensor(0, device=device), num_classes=K).float().view(K, 1)
        else:
            basis = F.normalize(direction, dim=1).t().contiguous()
    else:
        basis = evecs[:, -rank_eff:].contiguous()

    print(
        f"[Subspace-Stats] aggregated_pos={total_pos}, aggregated_neg={total_neg}, "
        f"rank={basis.size(1)}"
    )

    return basis.detach(), center.view(1, -1).detach()


def unlearn_one_class_on_model_topk_logit(
    model,
    dataloader,
    forget_cls: int,
    topk_idx: torch.Tensor,
    device,
    args,
    emb_feat,
    clip_model,
    epochs: int = 1,
    lambda_keep: float = 1.0,
    lambda_forget_logit: float = 0.0,
    lambda_forget_feat: float = 1.0,
    lr: float = 1e-4,
    topk_weights: torch.Tensor = None,
):
    """
    “top-K latent 维 + logits 惩罚” 单模型遗忘版本（SAE latent 版）

    - loss_keep:
        对【非目标类】做正常 BCE，保护其它 L-1 个类别；
    - loss_forget_logit:
        对目标类的 logit 强行拉向 0（目标 label=0）；
    - loss_forget_feat:
        对目标类正样本的 SAE latent 中 top-K 维做 L2 收缩。

    参数：
      topk_idx: collect_topk_dims_for_class 得到的 (K,) latent 维索引
    """

    model.train()
    model.to(device)

    sae_model = _get_sae_model(args, device)
    sae_model.eval()  # SAE 固定，不参与训练

    topk_idx = topk_idx.to(device)
    D = topk_idx.numel()
    if topk_weights is not None:
        topk_weights = topk_weights.detach().to(device).view(1, -1)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"[TopK+Logit] cls {forget_cls} | ep {ep}", ncols=100)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            images = batch["image"].float().to(device)
            labels = batch["labels"].float().to(device)
            mask   = batch["mask"].float().to(device)
            mask_in = mask.clone()

            optimizer.zero_grad()

            logits, _, _, label_emb = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )
            # logits: (B,L)
            # label_emb: (B,L,D_emb)
            B, L = logits.shape

            # ===== 映射到 SAE latent 空间 =====
            z_all = _encode_label_emb_with_sae(label_emb, sae_model, args)   # (B, L, latent_dim)

            # ===== 1) 保护其它类别的 BCE =====
            bce_all = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )   # (B,L)

            keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
            keep_mask[:, forget_cls] = False
            if keep_mask.sum() > 0:
                loss_keep = bce_all[keep_mask].mean()
            else:
                loss_keep = torch.tensor(0.0, device=device)

            # ===== 2) 目标类 logits 惩罚：强制该 logit ≈ 0 =====
            logits_f = logits[:, forget_cls]      # (B,)
            target_zero = torch.zeros_like(logits_f)
            loss_forget_logit = F.binary_cross_entropy_with_logits(
                logits_f, target_zero, reduction="mean"
            )

            # ===== 3) top-K latent 判别维度 L2 shrink（仅正样本） =====
            pos_idx = (labels[:, forget_cls] == 1)
            if pos_idx.any():
                z_f = z_all[:, forget_cls, :]       # (B, latent_dim)
                z_pos = z_f[pos_idx]                # (N_pos, latent_dim)
                z_pos_topk = z_pos[:, topk_idx]     # (N_pos, K)
                if topk_weights is None:
                    loss_forget_feat = (z_pos_topk ** 2).mean()
                else:
                    loss_forget_feat = ((z_pos_topk ** 2) * topk_weights).sum(dim=1).mean()
            else:
                loss_forget_feat = torch.tensor(0.0, device=device)

            # ===== 4) 总 loss =====
            loss = (
                lambda_keep * loss_keep +
                lambda_forget_logit * loss_forget_logit +
                lambda_forget_feat * loss_forget_feat
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / step

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "keep": f"{loss_keep.item():.3f}",
                "logit": f"{loss_forget_logit.item():.3f}",
                "feat": f"{loss_forget_feat.item():.3f}",
                "K": D,
            })

    return model


def unlearn_one_class_on_model(
    model,
    dataloader,
    forget_cls: int,
    topk_idx: torch.Tensor,
    device,
    args,
    emb_feat,
    clip_model,
    epochs: int = 1,
    lambda_keep: float = 1.0,
    lambda_forget_logit: float = 5.0,
    lambda_forget_feat: float = 1.0,
    lr: float = 1e-4,
    topk_weights: torch.Tensor = None,
    topk_subspace_basis: torch.Tensor = None,
    topk_subspace_center: torch.Tensor = None,
):
    """
    方案二（保护版，SAE latent 版）：
      - 冻结所有非目标类参数
      - 只更新：
          * output_linear 的第 forget_cls 行
          * label_lt 的第 forget_cls 行（可选）
          * ResNet50 最后一层：backbone.base_network.layer4
    """

    model.train()
    model.to(device)

    sae_model = _get_sae_model(args, device)
    sae_model.eval()  # SAE 固定，不参与训练

    topk_idx = topk_idx.to(device)
    D = topk_idx.numel()
    if topk_weights is not None:
        topk_weights = topk_weights.detach().to(device).view(1, -1)
    hardneg_topn = max(1, int(getattr(args, "hardneg_topn", 3)))
    shrink_w = float(getattr(args, "forget_shrink_weight", 0.3))
    pull_w = float(getattr(args, "forget_pull_weight", 0.7))
    subspace_w = float(getattr(args, "forget_subspace_weight", 0.4))
    if topk_subspace_basis is not None:
        topk_subspace_basis = topk_subspace_basis.detach().to(device)
    if topk_subspace_center is not None:
        topk_subspace_center = topk_subspace_center.detach().to(device)
    print("【保护版】只更新 head(Person) + backbone.layer4")

    # ===== ① 先把所有参数默认冻结 =====
    for p in model.parameters():
        p.requires_grad = False

    # ===== ② 解冻 output_linear，并只让第 forget_cls 行更新 =====
    W = model.output_linear.weight      # (num_labels, 512)
    b = model.output_linear.bias        # (num_labels,)

    W.requires_grad = True
    b.requires_grad = True

    mask_W = torch.zeros_like(W)
    mask_W[forget_cls] = 1.0

    def hook_W(grad):
        return grad * mask_W

    handle_W = W.register_hook(hook_W)

    mask_b = torch.zeros_like(b)
    mask_b[forget_cls] = 1.0

    def hook_b(grad):
        return grad * mask_b

    handle_b = b.register_hook(hook_b)

    # ===== ③ 可选：只更新 label_lt 里目标类那一行 =====
    if hasattr(model, "label_lt"):
        LE = model.label_lt
        LE.weight.requires_grad = True

        mask_LE = torch.zeros_like(LE.weight)
        mask_LE[forget_cls] = 1.0

        def hook_LE(grad):
            return grad * mask_LE

        handle_LE = LE.weight.register_hook(hook_LE)
    else:
        handle_LE = None

    # ===== ④ 解冻 ResNet50 的 layer4（高层视觉特征） =====
    for p in model.backbone.base_network.layer4.parameters():
        p.requires_grad = True

    # ===== ⑤ 优化器：只对 requires_grad=True 的参数更新 =====
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # ===== ⑥ loss 设计 =====
    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"[TopK+Logit] cls {forget_cls} | ep {ep}", ncols=100)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            images = batch["image"].float().to(device)
            labels = batch["labels"].float().to(device)
            mask   = batch["mask"].float().to(device)
            mask_in = mask.clone()

            optimizer.zero_grad()

            logits, _, _, label_emb = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )
            # logits: (B, L)
            # label_emb: (B, L, D_emb)

            # ===== 映射到 SAE latent 空间 =====
            z_all = _encode_label_emb_with_sae(label_emb, sae_model, args)   # (B, L, latent_dim)

            # 1) 保护其它类别：只在非目标类上算 BCE
            bce_all = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )   # (B,L)

            keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
            keep_mask[:, forget_cls] = False
            if keep_mask.sum() > 0:
                loss_keep = bce_all[keep_mask].mean()
            else:
                loss_keep = torch.tensor(0.0, device=device)

            # 2) 目标类 logit → 0
            logits_f = logits[:, forget_cls]      # (B,)
            target_zero = torch.zeros_like(logits_f)
            loss_forget_logit = F.binary_cross_entropy_with_logits(
                logits_f, target_zero, reduction="mean"
            )

            # 3) 目标类正样本 top-K latent 维收缩
            z_f = z_all[:, forget_cls, :]

            pos_idx = (labels[:, forget_cls] == 1)
            neg_idx = (labels[:, forget_cls] == 0)

            if pos_idx.any() and neg_idx.any():
                z_pos = z_f[pos_idx]
                z_neg = z_f[neg_idx]

                z_pos_topk = z_pos[:, topk_idx]
                z_neg_topk = z_neg[:, topk_idx]

                if topk_weights is None:
                    loss_shrink = (z_pos_topk ** 2).mean()
                    z_pos_metric = z_pos_topk
                    z_neg_metric = z_neg_topk
                else:
                    loss_shrink = ((z_pos_topk ** 2) * topk_weights).sum(dim=1).mean()
                    metric_scale = topk_weights.sqrt()
                    z_pos_metric = z_pos_topk * metric_scale
                    z_neg_metric = z_neg_topk * metric_scale

                # 对每个正样本，只拉向 batch 内最接近的几个负样本，而不是负样本均值。
                dist_mat = torch.cdist(z_pos_metric, z_neg_metric, p=2) ** 2   # (N_pos, N_neg)
                topn = min(hardneg_topn, dist_mat.size(1))
                hardneg_dist = torch.topk(dist_mat, k=topn, largest=False, dim=1).values
                loss_pull_neg = hardneg_dist.mean()

                if topk_subspace_basis is not None:
                    if topk_subspace_center is None:
                        z_centered = z_pos_metric
                    else:
                        z_centered = z_pos_metric - topk_subspace_center
                    proj = z_centered @ topk_subspace_basis
                    loss_subspace = (proj ** 2).sum(dim=1).mean()
                else:
                    loss_subspace = torch.tensor(0.0, device=device)

                norm = max(shrink_w + pull_w + subspace_w, 1e-6)
                loss_forget_feat = (
                    (shrink_w / norm) * loss_shrink +
                    (pull_w / norm) * loss_pull_neg +
                    (subspace_w / norm) * loss_subspace
                )
            else:
                loss_forget_feat = torch.tensor(0.0, device=device)

            loss = (
                lambda_keep * loss_keep +
                0 * loss_forget_logit +
                lambda_forget_feat * loss_forget_feat
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / step

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "keep": f"{loss_keep.item():.3f}",
                "logit": f"{loss_forget_logit.item():.3f}",
                "feat": f"{loss_forget_feat.item():.3f}",
                "K": D,
            })

    # ===== ⑦ 清理 hook =====
    handle_W.remove()
    handle_b.remove()
    if handle_LE is not None:
        handle_LE.remove()

    return model

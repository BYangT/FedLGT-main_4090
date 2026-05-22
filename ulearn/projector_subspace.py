# ulearn/projector_subspace.py
import torch
from torch import nn
from tqdm import tqdm
import os

@torch.no_grad()
def _collect_neg_mean_visfeat(
    model,
    dataloader,
    forget_cls: int,
    device,
    args,
    emb_feat,
    clip_model,
    max_images: int = 5000,
):
    """
    第一步：只算负样本的全局均值 μ_neg （按 image 平均掉 token）
    """
    model.eval()
    model.to(device)

    sum_neg = None
    cnt_neg = 0

    for batch in tqdm(dataloader, desc="[U-subspace] Pass1: neg mean"):
        images = batch["image"].float().to(device)
        labels = batch["labels"].float().to(device)
        mask   = batch["mask"].float().to(device)
        mask_in = mask.clone()

        logits, _, _, vis_feat = model(
            images,
            mask_in,
            args.learn_emb_type,
            emb_feat,
            clip_model,
            return_label_emb=True,  # 最后一个返回的是视觉 tokens
        )
        # vis_feat: (B, T, D)
        B, T, D = vis_feat.shape

        neg_idx = (labels[:, forget_cls] == 0)
        if not neg_idx.any():
            continue

        v_neg = vis_feat[neg_idx]        # (N_neg, T, D)
        # 对 token 做平均，得到 per-image 向量
        v_neg_img = v_neg.mean(dim=1)    # (N_neg, D)

        if sum_neg is None:
            sum_neg = v_neg_img.sum(dim=0)
        else:
            sum_neg += v_neg_img.sum(dim=0)
        cnt_neg += v_neg_img.size(0)

        if cnt_neg >= max_images:
            break

    if cnt_neg == 0:
        raise RuntimeError("没有采到任何负样本，检查 forget_cls 是否正确")

    mu_neg = sum_neg / cnt_neg          # (D,)
    return mu_neg


@torch.no_grad()
def build_vis_subspace_for_class(
    model,
    dataloader,
    forget_cls: int,
    device,
    args,
    emb_feat,
    clip_model,
    rank: int = 32,
    max_pos_images: int = 5000,
    save_path: str = "./ulearn/U_vis_cls.pt",
):
    """
    计算 “Person 视觉子空间”的基 U (D, r)，保存到 save_path
      1) 第一遍：算负样本均值 μ_neg
      2) 第二遍：采集正样本 per-image 向量 v_pos_img，并构造差分 X = v_pos_img - μ_neg
      3) 对 X 做 SVD → 取右奇异向量的前 r 个维度作为 basis U
    """
    model.eval()
    model.to(device)

    # ——— Step 1: 全局负样本均值 ———
    mu_neg = _collect_neg_mean_visfeat(
        model, dataloader, forget_cls, device, args, emb_feat, clip_model
    )   # (D,)
    D = mu_neg.numel()

    # ——— Step 2: 收集正样本差分向量 ———
    X_list = []
    cnt_pos = 0

    for batch in tqdm(dataloader, desc="[U-subspace] Pass2: collect X"):
        images = batch["image"].float().to(device)
        labels = batch["labels"].float().to(device)
        mask   = batch["mask"].float().to(device)
        mask_in = mask.clone()

        logits, _, _, vis_feat = model(
            images,
            mask_in,
            args.learn_emb_type,
            emb_feat,
            clip_model,
            return_label_emb=True,
        )
        # vis_feat: (B, T, D)
        pos_idx = (labels[:, forget_cls] == 1)
        if not pos_idx.any():
            continue

        v_pos = vis_feat[pos_idx]       # (N_pos, T, D)
        v_pos_img = v_pos.mean(dim=1)   # (N_pos, D)

        # 差分到 negative 均值
        diff = v_pos_img - mu_neg.unsqueeze(0)   # (N_pos, D)
        X_list.append(diff.cpu())
        cnt_pos += diff.size(0)

        if cnt_pos >= max_pos_images:
            break

    if cnt_pos == 0:
        raise RuntimeError("没有采到任何正样本，检查 forget_cls 是否正确")

    X = torch.cat(X_list, dim=0)   # (N_pos, D)
    # 中心化（可选）
    X = X - X.mean(dim=0, keepdim=True)

    # ——— Step 3: SVD 求子空间基 ———
    # 在 CPU 上做 SVD
    print(f"[U-subspace] Doing SVD on X: {X.shape}")
    # torch.linalg.svd: X = Ux * S * Vh
    # 我们要的是右奇异向量 Vh，形状 (D, D)，每一行一个方向
    Ux, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # 取前 rank 个奇异向量作为 basis
    r = min(rank, Vh.size(0))
    Vh_r = Vh[:r]              # (r, D)
    U_basis = Vh_r.T.clone()   # (D, r)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "U": U_basis,
            "forget_cls": forget_cls,
            "rank": r,
            "D": D,
        },
        save_path
    )
    print(f"[U-subspace] Saved U to {save_path}")
    return U_basis
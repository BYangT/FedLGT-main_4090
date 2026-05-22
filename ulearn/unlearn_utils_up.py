# ulearn/unlearn_utils_up.py

import torch
import torch.nn.functional as F
from tqdm import tqdm

class FeatureProjector(torch.nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 1024):
        super().__init__()
        # 和你 collect_topk_dims_for_class_up 里用的 projector.linear.out_features 对齐
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) 或 (B, T, D)
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)  # ✅ 替代 view
            h_flat = self.linear(x_flat)
            h = h_flat.reshape(B, T, -1)
            return h
        elif x.dim() == 2:
            return self.linear(x)              # (B, out_dim)
        else:
            raise ValueError(f"Unsupported shape for projector: {x.shape}")

def collect_topk_dims_for_class_up(
    model,
    dataloader,
    forget_cls: int,
    K: int,
    device,
    args,
    emb_feat,
    clip_model,
    projector,      # 升维模块
):
    model.eval().to(device)
    projector.eval().to(device)

    high_dim = projector.linear.out_features

    pos_sum = torch.zeros(high_dim, device=device)
    neg_sum = torch.zeros(high_dim, device=device)
    pos_cnt = 0
    neg_cnt = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collect top-K stats (up)"):
            images = batch['image'].float().to(device)
            labels = batch['labels'].float().to(device)
            mask   = batch['mask'].float().to(device)
            mask_in = mask.clone()

            logits, _, _, label_emb = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )
            # label_emb: (B, L, 512)
            z_f = label_emb[:, forget_cls, :]   # (B, 512)

            # 升维到高维空间： (B, D_high)
            h_f = projector(z_f)                # (B, high_dim)

            pos_idx = (labels[:, forget_cls] == 1)
            neg_idx = (labels[:, forget_cls] == 0)

            if pos_idx.any():
                h_pos = h_f[pos_idx]                     # (N_pos, D_high)
                pos_sum += h_pos.abs().sum(dim=0)
                pos_cnt += h_pos.size(0)

            if neg_idx.any():
                h_neg = h_f[neg_idx]                     # (N_neg, D_high)
                neg_sum += h_neg.abs().sum(dim=0)
                neg_cnt += h_neg.size(0)

    pos_mean = pos_sum / (pos_cnt + 1e-6)
    neg_mean = neg_sum / (neg_cnt + 1e-6)
    score = pos_mean - neg_mean

    mask_good = score > (score.mean() + 0.5 * score.std())
    score_filtered = score.clone()
    score_filtered[~mask_good] = -1e9

    K_use = min(K, high_dim)
    topk_vals, topk_idx = torch.topk(score_filtered, k=K_use, largest=True)

    return topk_idx, score


def unlearn_one_class_on_model_vis_up(
    model,
    dataloader,
    forget_cls,
    topk_idx,
    device,
    args,
    emb_feat,
    clip_model,
    projector,        # 升维模块
    epochs=1,
    lambda_keep=1.0,
    lambda_vis=1.0,
    lr=1e-4,
):
    """
    升维后的视觉特征方向抹除：
      - 先把视觉 token (B,T,512) 升到高维 (B,T,D_high)
      - 只在 D_high 的 top-K 子空间里估计判别方向并擦除
    """
    model.train().to(device)
    projector.train().to(device)

    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(projector.parameters()),
        lr=lr,
    )

    topk_idx = topk_idx.to(device)

    D_high = projector.linear.out_features
    topk_mask = torch.zeros(D_high, device=device)
    topk_mask[topk_idx] = 1.0

    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"[Vis-Forget(up,dir)] cls {forget_cls} | ep {ep}", ncols=100)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            images = batch['image'].float().to(device)
            labels = batch['labels'].float().to(device)
            mask   = batch['mask'].float().to(device)
            mask_in = mask.clone()

            optimizer.zero_grad()

            # 拿到视觉 token：原始 512 维
            logits, _, _, vis_feat = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )   # vis_feat: (B, T, 512)

            # 升维： (B,T,512) -> (B,T,D_high)
            h_feat = projector(vis_feat)   # projector 支持 3D

            # ===== 1) 保护其它类别 =====
            bce_all = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction='none'
            )
            keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
            keep_mask[:, forget_cls] = False
            if keep_mask.sum() > 0:
                loss_keep = bce_all[keep_mask].mean()
            else:
                loss_keep = torch.tensor(0.0, device=device)

            # ===== 2) 方向抹除（只对 Person 正样本） =====
            pos_idx = (labels[:, forget_cls] == 1)
            neg_idx = (labels[:, forget_cls] == 0)

            loss_vis = torch.tensor(0.0, device=device)

            if pos_idx.any():
                h_pos = h_feat[pos_idx]    # (N_pos, T, D_high)

                if neg_idx.any():
                    h_neg = h_feat[neg_idx]              # (N_neg, T, D_high)
                    mu_pos = h_pos.mean(dim=(0, 1))      # (D_high,)
                    mu_neg = h_neg.mean(dim=(0, 1))      # (D_high,)
                    u_raw = mu_pos - mu_neg
                else:
                    u_raw = h_pos.mean(dim=(0, 1))

                u_raw = u_raw * topk_mask

                if u_raw.abs().sum() > 1e-6:
                    u = F.normalize(u_raw, dim=0)        # (D_high,)

                    alpha = (h_pos * u).sum(dim=-1, keepdim=True)  # (N_pos, T, 1)
                    h_proj = alpha * u                             # (N_pos, T, D_high)

                    loss_vis = (h_proj ** 2).mean()

            loss = lambda_keep * loss_keep + lambda_vis * loss_vis
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / step
            pbar.set_postfix({
                "loss":    f"{avg_loss:.4f}",
                "keep":    f"{loss_keep.item():.3f}",
                "vis_dir": f"{loss_vis.item():.3f}",
            })

    return model, projector
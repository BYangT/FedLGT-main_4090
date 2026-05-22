# ulearn/unlearn_utils_projector.py

import torch
import torch.nn.functional as F
from tqdm import tqdm

def unlearn_one_class_on_model_vis_projector(
    model,
    dataloader,
    forget_cls: int,
    device,
    args,
    emb_feat,
    clip_model,
    U: torch.Tensor,
    epochs: int = 1,
    lambda_keep: float = 1.0,
    lambda_vis: float = 10.0,
    lr: float = 1e-4,
):
    """
    视觉特征子空间版 PROJECTOR：
      - 输入 U: (D, r) 是离线算好的 Person 子空间基（不参与梯度）
      - loss_keep: 非 Person 类的 BCE，保护其它 19 类
      - loss_vis: 对 Person 正样本视觉 token 的“子空间分量”做 L2 惩罚
    """

    model.train()
    model.to(device)
    U = U.to(device)
    U.requires_grad_(False)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"[Vis-Forget(PROJ)] cls {forget_cls} | ep {ep}", ncols=100)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            images = batch["image"].float().to(device)
            labels = batch["labels"].float().to(device)
            mask   = batch["mask"].float().to(device)
            mask_in = mask.clone()

            optimizer.zero_grad()

            logits, _, _, vis_feat = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )
            # vis_feat: (B, T, D)
            B, T, D = vis_feat.shape

            # ===== 1) 保护其它类别性能 =====
            bce_all = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )  # (B, L)

            keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
            keep_mask[:, forget_cls] = False
            if keep_mask.sum() > 0:
                loss_keep = bce_all[keep_mask].mean()
            else:
                loss_keep = torch.tensor(0.0, device=device)

            # ===== 2) 在 U 子空间上做投影惩罚（只对 Person 正样本） =====
            pos_idx = (labels[:, forget_cls] == 1)

            if pos_idx.any():
                v_pos = vis_feat[pos_idx]  # (N_pos, T, D)

                # 2.1 投到子空间： coords = v_pos @ U → (N_pos, T, r)
                # einsum 写法更清晰
                coords = torch.einsum("ntd,dr->ntr", v_pos, U)  # (N_pos, T, r)

                # 2.2 从子空间重建： v_sub = coords @ U^T → (N_pos, T, D)
                v_sub = torch.einsum("ntr,rd->ntd", coords, U.T)  # (N_pos, T, D)

                # 2.3 惩罚这部分子空间分量，让它尽量为 0
                loss_vis = (v_sub ** 2).mean()
            else:
                loss_vis = torch.tensor(0.0, device=device)

            # ===== 3) 总 loss =====
            loss = lambda_keep * loss_keep + lambda_vis * loss_vis

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / step
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "keep": f"{loss_keep.item():.3f}",
                "vis_proj": f"{loss_vis.item():.3f}",
            })

    return model
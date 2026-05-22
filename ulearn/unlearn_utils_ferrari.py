# ulearn/unlearn_utils_ferrari.py （可以放一起）
import torch
import torch.nn.functional as F
from tqdm import tqdm

def unlearn_one_class_on_model_ferrari(
    model,
    dataloader,
    forget_cls: int,
    device,
    args,
    emb_feat,
    clip_model,
    epochs: int = 1,
    lambda_keep: float = 1.0,
    lambda_sens: float = 5.0,   # Ferrari 风格的敏感度权重
    lr: float = 3e-4,
):
    """
    Ferrari 风格的特征忘却（单客户端版）：
      - 保留其他类别的 BCE（loss_keep）
      - 对 forget_cls 的特征做 Lipschitz 正则（loss_sens）

    注意：这里不再引入 logit BCE 到 0，只靠敏感度让记忆变“平”。
    """

    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    num_dirs = getattr(args, "num_noise_dirs", 3)        # 每个样本扰动方向数
    sigma    = getattr(args, "feat_noise_sigma", 0.1)    # 噪声尺度

    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"[Ferrari] Unlearn cls {forget_cls} | epoch {ep}", ncols=100)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            images = batch['image'].float().to(device)
            labels = batch['labels'].float().to(device)  # (B, L)
            mask   = batch['mask'].float().to(device)
            mask_in = mask.clone()

            optimizer.zero_grad()

            # 前向，拿到 label embedding
            logits, _, _, label_emb = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )

            # ===== 1) 保护其他类别的性能：对非 forget_cls 做 BCE =====
            bce_all = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction='none'
            )  # (B, L)

            keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
            keep_mask[:, forget_cls] = False
            if keep_mask.sum() > 0:
                loss_keep = bce_all[keep_mask].mean()
            else:
                loss_keep = torch.tensor(0.0, device=device)

            # ===== 2) Ferrari 特征敏感度损失：只在 forget_cls 上算 =====
            pos_idx = (labels[:, forget_cls] == 1)
            if pos_idx.any():
                # 取 Person 这个类在正样本上的特征： (N_pos, D)
                z_f = label_emb[pos_idx, forget_cls, :]  # (N_pos, D)

                # 输出层权重（近似 logit = W_c · z_f + b_c）
                W_c = model.output_linear.weight[forget_cls]  # (D,)
                b_c = model.output_linear.bias[forget_cls]    # ()

                sens_accum = 0.0
                for _ in range(num_dirs):
                    # 高斯噪声 δ~N(0, sigma^2 I)
                    noise = torch.randn_like(z_f) * sigma      # (N_pos, D)
                    z_pert = z_f + noise

                    logit_orig = (z_f     * W_c.unsqueeze(0)).sum(dim=1) + b_c  # (N_pos,)
                    logit_pert = (z_pert  * W_c.unsqueeze(0)).sum(dim=1) + b_c  # (N_pos,)

                    diff = (logit_orig - logit_pert).abs()                      # |Δlogit|
                    noise_norm = noise.view(noise.size(0), -1).norm(dim=1) + 1e-6

                    sens_accum += (diff / noise_norm).mean()

                loss_sens = sens_accum / float(num_dirs)
            else:
                loss_sens = torch.tensor(0.0, device=device)

            # ===== 3) 总 loss：Ferrari 纯血版不加 logit BCE_to_0，只靠敏感度 =====
            loss = lambda_keep * loss_keep + lambda_sens * loss_sens

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / step

            pbar.set_postfix({
                "loss":  f"{avg_loss:.4f}",
                "keep":  f"{loss_keep.item():.3f}",
                "sens":  f"{loss_sens.item():.3f}",
            })

    return model
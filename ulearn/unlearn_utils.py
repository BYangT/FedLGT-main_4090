# unlearn_utils.py

import torch
import torch.nn.functional as F
from tqdm import tqdm

import math

class FixedUpProjector(torch.nn.Module):
    def __init__(self, in_dim=512, out_dim=1024):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        # 用高斯随机初始化，然后冻结
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=1.0 / math.sqrt(in_dim))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: (..., in_dim)
        return self.linear(x)
# 单类
def collect_topk_dims_for_class(
    model,
    dataloader,
    forget_cls: int,        # 要遗忘的类别 index（0~L-1）
    K: int,                 # 选多少个维度，比如 64
    device,
    args,
    emb_feat,               # fed_main 里你已经算好的 label_text_features
    clip_model,             # fed_main 里加载的 CLIP 模型
):
    model.eval()
    model.to(device)

    # hidden 尺寸（一般是 512）
    hidden_dim = 512

    # 用来累计正样本在每个维度上的 |激活| 之和
    pos_sum = torch.zeros(hidden_dim, device=device)
    pos_cnt = 0

    # （可选）也统计负样本，用作对比；现在先留着接口
    neg_sum = torch.zeros(hidden_dim, device=device)
    neg_cnt = 0

    with torch.no_grad():
        # data_loader 的 batch 结构参考 run_epoch.py：
        # for batch_idx, (images, labels, mask, img_ids) in enumerate(data_loader):
        for batch in tqdm(dataloader, desc="Collect top-K stats"):
            images = batch['image'].float().to(device)  # (B, C, H, W)
            labels = batch['labels'].float().to(device)  # (B, L)
            mask = batch['mask'].float().to(device)  # (B, L)
            mask_in = mask.clone()

            # forward，拿到 label_embeddings
            logits, _, _, label_emb = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,  # 别忘了你在 CTranModel 里加了这个参数
            )
            # label_emb: (B, L, hidden_dim)

            # 取出目标类的 embedding： (B, hidden_dim)
            z_f = label_emb[:, forget_cls, :]  # 这一条就是“目标类别在这张图上的表示”

            # 正/负样本索引
            # tensor([True, False, True])
            pos_idx = (labels[:, forget_cls] == 1)
            neg_idx = (labels[:, forget_cls] == 0)

            if pos_idx.any():
                # 所有正样本的 embeddings，可以理解为每一行代表一个图，
                z_pos = z_f[pos_idx]  # (N_pos, hidden_dim)
                # 按列求和
                pos_sum += z_pos.abs().sum(dim=0)  # 对每个维度累加 |激活|
                # 累加正样本数量
                pos_cnt += z_pos.size(0)

            if neg_idx.any():
                z_neg = z_f[neg_idx]
                neg_sum += z_neg.abs().sum(dim=0)
                neg_cnt += z_neg.size(0)

    # ==== 统计结果 → 得到每个维度的“类专属程度” ====

    # 正样本平均绝对激活
    pos_mean = pos_sum / (pos_cnt + 1e-6)   # (hidden_dim,)

    # 负样本平均绝对激活（用来做对比；也可以不用）
    neg_mean = neg_sum / (neg_cnt + 1e-6)   # (hidden_dim,)

    # 评分：你可以先用最简单的一种
    #   score_d = pos_mean_d - neg_mean_d
    # 负样本的平均绝对激活很高的话，那么也就代表着这个维度在其他类别影响也很大，所以就有可能是背景什么的
    score = pos_mean - neg_mean   # (hidden_dim,)

    # 取出得分最大的 K 个维度
    # topk_vals, topk_idx = torch.topk(score, k=K, largest=True)

    mask_good = score > (score.mean() + 0.5 * score.std())
    score_filtered = score.clone()
    score_filtered[~mask_good] = -1e9  # 直接扔掉不明显的维度
    topk_vals, topk_idx = torch.topk(score_filtered, k=K, largest=True)

    # topk_idx: (K,) LongTensor，例如 tensor([  3,  17,  25, ...])
    return topk_idx, score

def collect_topk_dims_for_class_vis(
    model, dataloader, forget_cls, K, device, args, emb_feat, clip_model
):
    model.eval()
    model.to(device)
    hidden_dim = 512  # 比如 512

    pos_sum = torch.zeros(hidden_dim, device=device)
    neg_sum = torch.zeros(hidden_dim, device=device)
    pos_cnt = 0
    neg_cnt = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].float().to(device)
            labels = batch['labels'].float().to(device)
            mask   = batch['mask'].float().to(device)
            mask_in = mask.clone()

            # 这里要让 model 返回视觉特征
            # 你需要在 CTranModel 里加一个 return_vis_feat=True 的分支
            logits, _, _, vis_feat = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )   # vis_feat: (B, D)

            pos_idx = (labels[:, forget_cls] == 1)
            neg_idx = (labels[:, forget_cls] == 0)

            if pos_idx.any():
                v_pos = vis_feat[pos_idx]        # (N_pos, D)
                pos_sum += v_pos.abs().sum(dim=(0, 1))  # → (D,)
                pos_cnt += v_pos.size(0) * v_pos.size(1)  # 统计的是 “正样本 token 数”

            if neg_idx.any():
                v_neg = vis_feat[neg_idx]
                neg_sum += v_neg.abs().sum(dim=(0, 1))  # → (D,)
                neg_cnt += v_neg.size(0) * v_neg.size(1)

    pos_mean = pos_sum / (pos_cnt + 1e-6)
    neg_mean = neg_sum / (neg_cnt + 1e-6)

    score = pos_mean - neg_mean  # “这维更像 Person 而不像其他类”

    # 和你之前一样，可以过滤一遍再取 topK
    mask_good = score > (score.mean() + 0.5 * score.std())
    score_filtered = score.clone()
    score_filtered[~mask_good] = -1e9

    K_use = min(K, hidden_dim)
    topk_vals, topk_idx = torch.topk(score_filtered, k=K_use, largest=True)

    return topk_idx, score

def collect_topk_dims_for_class_up(
    model,
    dataloader,
    forget_cls: int,
    K: int,
    device,
    args,
    emb_feat,
    clip_model,
    projector,      # 新增：升维模块
):
    model.eval().to(device)
    projector.eval().to(device)

    high_dim = projector.linear.out_features  # 假设 projector 里面有 .linear

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

# def unlearn_one_class_on_model(
#     model,
#     dataloader,
#     forget_cls: int,
#     topk_idx: torch.Tensor,   # (K,)
#     device,
#     args,
#     emb_feat,
#     clip_model,
#     epochs: int = 1,
#     lambda_keep: float = 1.0,
#     lambda_forget_logit: float = 1.0,
#     lambda_forget_feat: float = 1.0,
#     lr: float = 1e-4,
#     mode: str = None,
# ):
#     """
#     在单客户端上，对指定类别 forget_cls 做“遗忘微调”。
#
#     - model: 已经训练好的 CTranModel（比如你当前的全局模型 / 某个客户端模型）
#     - dataloader: 这个客户端的数据 loader（可以用 train_dl_global 先试）
#     - forget_cls: 要遗忘的类别下标（0~L-1，VOC 里 person=14）
#     - topk_idx: collect_topk_dims_for_class 得到的 (K,) 维度索引
#     """
#
#     model.train()
#     model.to(device)
#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=lr,
#     )
#
#     for ep in range(epochs):
#         if mode is None:
#             mode = getattr(args, "forget_mode", "feat_only")
#         print(f"---------mode：{mode}--------------")
#         mode = mode.lower()
#         assert mode in ["both", "logit_only", "feat_only"], \
#             f"unknown forgset mode: {mode}"
#         pbar = tqdm(dataloader, desc=f"Unlearn cls {forget_cls} | epoch {ep}", ncols=100)
#         running_loss = 0.0
#
#         for step, batch in enumerate(pbar, start=1):
#             images = batch['image'].float().to(device)
#             labels = batch['labels'].float().to(device)  # (B, L)
#             mask = batch['mask'].float().to(device)
#             mask_in = mask.clone()
#
#             optimizer.zero_grad()
#             # 跑模型
#             logits, _, _, label_emb = model(
#                 images,
#                 mask_in,
#                 args.learn_emb_type,
#                 emb_feat,
#                 clip_model,
#                 return_label_emb=True,
#             )
#             # 求损失
#             bce_all = F.binary_cross_entropy_with_logits(
#                 logits,
#                 labels,
#                 reduction='none'
#             )  # (B, L)
#             # 构造一个保留哪些类的布偶掩码
#             # 做一个和 bce_all 一样形状的布尔矩阵，初始全是 True，形状 (B, L)
#             keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
#             # 把要遗忘的那个类别的整列改成 False
#             keep_mask[:, forget_cls] = False
#             # 在要保护的类别上求loss
#             if keep_mask.sum() > 0:
#                 loss_keep = bce_all[keep_mask].mean()
#             else:
#                 loss_keep = torch.tensor(0.0, device=device)
#             # 从 args里面读参数，是一个权重系数
#             lambda_neg_confuse = getattr(args, "lambda_neg_confuse", 0.1)
#             lambda_collapse = getattr(args, "lambda_collapse", 0.1)
#             # pos_idx = (labels[:, forget_cls] == 1)
#             # neg_idx = (labels[:, forget_cls] == 0)
#             # if neg_idx.any():
#             #     neg_logits = logits[neg_idx, forget_cls]
#             #     # 给负样本一点轻微的正强化（让它们概率不要太低）
#             #     loss_neg_confuse = (-neg_logits).mean() * 0.1
#             # if pos_idx.any():
#             #     # 拿到该类为 1 的图像”，模型对这个类的预测分数
#             #     forget_logits = logits[pos_idx, forget_cls]
#             #     target_zero = torch.zeros_like(forget_logits)
#             #     loss_forget_logit = F.binary_cross_entropy_with_logits(
#             #         forget_logits,
#             #         target_zero,
#             #         reduction='mean'
#             #     )
#             #     # ====== 特征 top-K 维度的 L2 惩罚（保持不变） ====== #
#             #     z_f = label_emb[pos_idx, forget_cls, :]  # (N_pos, hidden)
#             #     z_topk = z_f[:, topk_idx]  # (N_pos, K)
#             #     # 对这些关键维度做平方→越大惩罚越重（类似 L2 正则）
#             #     loss_forget_feat = (z_topk ** 2).mean()
#             # else:
#             #     loss_forget_logit = torch.tensor(0.0, device=device)
#             #     loss_forget_feat = torch.tensor(0.0, device=device)
#
#             pos_idx = (labels[:, forget_cls] == 1)
#             neg_idx = (labels[:, forget_cls] == 0)
#
#             # 先默认全部为 0，避免某一批次里没有正/负样本时报错
#             loss_forget_logit = torch.tensor(0.0, device=device)
#             loss_sens = torch.tensor(0.0, device=device)
#             loss_neg_confuse = torch.tensor(0.0, device=device)
#             loss_collapse = torch.tensor(0.0, device=device)
#             # ===== 新版：特征忘却为主 =====
#
#             # 这一列的 embedding：对 Person 来说，每张图都有一条 (B, D)
#             z_f_all = label_emb[:, forget_cls, :]
#             # ---- (1) top-K 维度收缩（所有样本都收缩） ----
#             z_topk_all = z_f_all[:, topk_idx]  # (B, K)
#             loss_shrink = (z_topk_all ** 2).mean()
#
#             # ---- (2) 向“负样本中心”塌陷 ----
#             loss_center = torch.tensor(0.0, device=device)
#
#             # ===== ① 正样本：继续做 logit + feature 遗忘约束 =====
#             if pos_idx.any():
#                 # 该类为 1 的图像上，这个类的 logit
#                 forget_logits = logits[pos_idx, forget_cls]  # (N_pos,)
#
#                 # ---- logit 压制（方案 A：带阈值的 hinge + L2 加强） ---- #
#                 tau = getattr(args, "forget_logit_tau", -3.0)  # 越低 → 越狠
#                 diff = forget_logits - tau  # 比 τ 高多少
#                 # 把 x 里所有 小于 0 的值截断为 0，大于 0 的保持不变。
#                 penalty = torch.clamp(diff, min=0.0)  # 只惩罚 logit > τ 的部分
#
#                 gamma = getattr(args, "forget_logit_gamma", 1.0)
#                 weight = torch.exp(gamma * penalty)  # penalty 大 → 权重大
#                 loss_forget_logit = (weight * penalty ** 2).mean()
#
#                 # ---- 特征 top-K 维度 L2 惩罚 ---- #
#                 # z_f = label_emb[pos_idx, forget_cls, :]  # (N_pos, hidden)
#                 # z_topk = z_f[:, topk_idx]  # (N_pos, K)
#                 # loss_forget_feat = (z_topk.abs().pow(1.5)).mean()
#
#                 # 特征惩罚第二版
#                 # z_f = label_emb[pos_idx, forget_cls, :]  # (N_pos, D)
#                 # loss_forget_feat = (z_f.abs().pow(1.5)).mean()
#
#                 # ---- 特征敏感度式忘却（Ferrari 风格简化版） ---- #
#                 # z_f: 当前类在这些正样本上的 label embedding
#                 z_f = label_emb[pos_idx, forget_cls, :]  # (N_pos, D)
#
#                 # 1) 只在 top-K 维度上加噪声（如果你想用全维就把这一段改一下）
#                 sigma = getattr(args, "feat_noise_sigma", 0.1)  # 噪声强度，先试 0.05~0.1
#                 if topk_idx is not None and topk_idx.numel() > 0:
#                     noise = torch.randn_like(z_f[:, topk_idx]) * sigma  # (N_pos, K)
#                     z_perturb = z_f.clone()
#                     z_perturb[:, topk_idx] = z_perturb[:, topk_idx] + noise
#                     # 展平噪声，用来算范数
#                     noise_flat = noise.view(noise.size(0), -1)  # (N_pos, K)
#                 else:
#                     # 没有 topk_idx 的情况，就全维加噪声
#                     noise_full = torch.randn_like(z_f) * sigma  # (N_pos, D)
#                     z_perturb = z_f + noise_full
#                     noise_flat = noise_full.view(noise_full.size(0), -1)  # (N_pos, D)
#
#                 # 2) 用输出层的权重近似算“这个类的 logit 变化”
#                 #    logit_c = W_c · z_c + b_c
#                 W_c = model.output_linear.weight[forget_cls]  # (D,)
#                 b_c = model.output_linear.bias[forget_cls]  # ()
#
#                 # 原始 logit
#                 logit_orig = (z_f * W_c.unsqueeze(0)).sum(dim=1) + b_c  # (N_pos,)
#                 # 加噪后的 logit
#                 logit_pert = (z_perturb * W_c.unsqueeze(0)).sum(dim=1) + b_c  # (N_pos,)
#
#                 # 3) 特征敏感度：输出变化 / 噪声范数
#                 diff = (logit_orig - logit_pert).pow(2).sqrt()  # |Δlogit|
#                 noise_norm = noise_flat.norm(dim=1) + 1e-6  # ‖δ‖
#                 sens = (diff / noise_norm).mean()
#
#                 # 4) 把敏感度作为特征忘却 loss
#                 loss_sens = sens
#
#             # # ===== ② 负样本：轻微“正向混淆”，让部分负样本 logit 不要太低 =====
#             # if neg_idx.any():
#             #     neg_logits = logits[neg_idx, forget_cls]  # (N_neg,)
#             #
#             #     # 我们希望：负样本的 logit 至少达到一个小 margin，让它们别全都躺在很负的区域
#             #     # m_neg 越大，负样本越容易被推高（AP 越容易掉）
#             #     m_neg = getattr(args, "neg_confuse_margin", 0.0)  # 0.0 对应概率约 0.5
#             #     neg_penalty = torch.clamp(m_neg - neg_logits, min=0.0)
#             #     # 当 neg_logits < m_neg 时，有正损失 → 反向推动 logit 往上长
#             #     loss_neg_confuse = (neg_penalty ** 2).mean()
#             if neg_idx.any():
#                 z_neg = z_f_all[neg_idx]  # (N_neg, D)
#                 mu_neg = z_neg.detach().mean(dim=0, keepdim=True)  # (1, D), 只当“目标点”，不反传
#                 # 所有样本往 mu_neg 靠（也可以只对正样本）
#                 loss_center = ((z_f_all - mu_neg) ** 2).mean()
#
#             # 组合成特征忘却损失
#             alpha_shrink = getattr(args, "alpha_shrink", 0.0)
#             beta_center = getattr(args, "beta_center", 0.0)
#             gamma_sens = getattr(args, "gamma_sens", 1.0)  # 新增超参
#
#             loss_forget_feat = (
#                     alpha_shrink * loss_shrink +
#                     beta_center * loss_center +
#                     gamma_sens * loss_sens
#             )
#
#             # ---- (3) logit 轻微往 0 压一压（可选，小权重） ----
#             z_person = logits[:, forget_cls]  # (B,)
#             p_person = torch.sigmoid(z_person)
#             # 想让 Person 不积极：鼓励 p_person → 0
#             loss_forget_logit = (p_person ** 2).mean()
#
#             # # ===== ③ 正负样本“特征塌陷”：正负 embedding 往一起拉 =====
#             # if pos_idx.any() and neg_idx.any():
#             #     z_pos = label_emb[pos_idx, forget_cls, :]  # (N_pos, D)
#             #     z_neg = label_emb[neg_idx, forget_cls, :]  # (N_neg, D)
#             #
#             #     # 简单做法：对比“正样本平均特征”和“负样本平均特征”
#             #     mean_pos = z_pos.mean(dim=0)
#             #     mean_neg = z_neg.mean(dim=0)
#             #     loss_collapse = F.mse_loss(mean_pos, mean_neg)
#
#             # loss = (
#             #         lambda_keep * loss_keep +
#             #         lambda_forget_logit * loss_forget_logit +
#             #         lambda_forget_feat * loss_forget_feat
#             # )
#             # ===== 根据 ablation 模式，关掉对应的遗忘项 =====
#             if mode == "logit_only":
#                 # 只保留 logit 惩罚：把特征项清零
#                 loss_forget_feat = torch.tensor(0.0, device=device)
#             elif mode == "feat_only":
#                 # 只保留特征惩罚：把 logit 惩罚清零
#                 loss_forget_logit = torch.tensor(0.0, device=device)
#
#             loss = (
#                     lambda_keep * loss_keep +
#                     lambda_forget_logit * loss_forget_logit +
#                     lambda_forget_feat * loss_forget_feat +
#                     lambda_neg_confuse * loss_neg_confuse +  # ⭐ 新增
#                     lambda_collapse * loss_collapse  # ⭐ 新增
#             )
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
#             optimizer.step()
#
#             running_loss += loss.item()
#             avg_loss = running_loss / step
#
#             pbar.set_postfix({
#                 "loss": f"{avg_loss:.4f}",
#                 "keep": f"{loss_keep.item():.3f}",
#                 "logit": f"{loss_forget_logit.item():.3f}",
#                 "feat": f"{loss_forget_feat.item():.3f}",
#             })
#
#     return model

def unlearn_one_class_on_model(
    model,
    dataloader,
    forget_cls: int,
    topk_idx: torch.Tensor,   # (K,)
    device,
    args,
    emb_feat,
    clip_model,
    epochs: int = 1,
    lambda_keep: float = 1.0,
    lambda_forget_logit: float = 0.0,   # 纯 Ferrari 可以设成 0
    lambda_forget_feat: float = 1.0,
    lr: float = 1e-4,
    mode: str = None,
):
    """
    在单客户端上，对指定类别 forget_cls 做“Ferrari 式特征遗忘”微调。

    - model: 已经训练好的 CTranModel（比如全局模型 / 某客户端模型）
    - dataloader: 该客户端的数据 loader
    - forget_cls: 要遗忘的类别下标（0~L-1，VOC 里 Person = 14）
    - topk_idx: collect_topk_dims_for_class 得到的 (K,) 维度索引，
                只在这些“对该类敏感”的维度上加噪声
    - lambda_keep: 非目标类 BCE 的权重
    - lambda_forget_feat: Ferrari 特征敏感度 loss 的权重（主角）
    - lambda_forget_logit: 可选的 logit 轻微压制权重（想纯 Ferrari 可设 0）
    """

    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for ep in range(epochs):
        if mode is None:
            mode = getattr(args, "forget_mode", "feat_only")
        print(f"[Unlearn] mode = {mode}")
        mode = mode.lower()
        assert mode in ["both", "logit_only", "feat_only"], \
            f"unknown forget mode: {mode}"

        pbar = tqdm(dataloader,
                    desc=f"Unlearn cls {forget_cls} | epoch {ep}",
                    ncols=100)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            images = batch['image'].float().to(device)
            labels = batch['labels'].float().to(device)  # (B, L)
            mask   = batch['mask'].float().to(device)
            mask_in = mask.clone()

            optimizer.zero_grad()

            # 前向：拿到 logits + label_emb
            logits, _, _, label_emb = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )

            # ========== 1) 非目标类 BCE（保护其它类） ==========
            bce_all = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction='none'
            )   # (B, L)

            keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
            keep_mask[:, forget_cls] = False   # 不对目标类做 BCE
            if keep_mask.sum() > 0:
                loss_keep = bce_all[keep_mask].mean()
            else:
                loss_keep = torch.tensor(0.0, device=device)

            # ========== 2) Ferrari 特征敏感度 loss ==========
            pos_idx = (labels[:, forget_cls] == 1)

            loss_forget_feat = torch.tensor(0.0, device=device)
            if pos_idx.any():
                # 正样本上，取出该类的 label embedding：z_f (N_pos, D)
                z_f = label_emb[pos_idx, forget_cls, :]   # (N_pos, D)

                # 输出层权重：logit_c = W_c · z_c + b_c
                W_c = model.output_linear.weight[forget_cls]  # (D,)
                b_c = model.output_linear.bias[forget_cls]    # ()

                # Ferrari：多次 MC 采样噪声，估计敏感度
                N_mc   = getattr(args, "sens_mc", 4)          # Monte-Carlo 次数
                sigma  = getattr(args, "feat_noise_sigma", 0.1)  # 噪声强度
                eps    = 1e-6

                sens_list = []

                for _ in range(N_mc):
                    # 在 top-K 维度上加噪声；若没有 topk_idx 就全维
                    if topk_idx is not None and topk_idx.numel() > 0:
                        noise = torch.randn_like(z_f[:, topk_idx]) * sigma   # (N_pos, K)
                        z_perturb = z_f.clone()
                        z_perturb[:, topk_idx] = z_perturb[:, topk_idx] + noise
                        noise_flat = noise.view(noise.size(0), -1)          # (N_pos, K)
                    else:
                        noise_full = torch.randn_like(z_f) * sigma          # (N_pos, D)
                        z_perturb = z_f + noise_full
                        noise_flat = noise_full.view(noise_full.size(0), -1)

                    # 原始 / 加噪后的 logit（用线性头近似）
                    logit_orig = (z_f * W_c.unsqueeze(0)).sum(dim=1) + b_c      # (N_pos,)
                    logit_pert = (z_perturb * W_c.unsqueeze(0)).sum(dim=1) + b_c

                    # Ferrari: |Δf| / ||δ||
                    delta_f   = (logit_orig - logit_pert).abs()                # (N_pos,)
                    delta_norm = noise_flat.norm(dim=1) + eps                   # (N_pos,)
                    sens = (delta_f / delta_norm).mean()
                    sens_list.append(sens)

                loss_forget_feat = torch.stack(sens_list).mean()

            # ========== 3) 可选：logit 轻微压到 0（整体不积极） ==========
            # 这一项不是 Ferrari 必须的，你想纯 Ferrari 就把 lambda_forget_logit 设 0
            z_person = logits[:, forget_cls]          # (B,)
            p_person = torch.sigmoid(z_person)        # 概率
            loss_forget_logit = (p_person ** 2).mean()  # 希望接近 0

            # ========== 4) ablation 模式开关 ==========
            if mode == "logit_only":
                loss_forget_feat = torch.tensor(0.0, device=device)
            elif mode == "feat_only":
                loss_forget_logit = torch.tensor(0.0, device=device)

            # ========== 5) 总 loss ==========
            loss = (
                lambda_keep        * loss_keep +
                lambda_forget_logit * loss_forget_logit +
                lambda_forget_feat  * loss_forget_feat
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / step

            pbar.set_postfix({
                "loss":  f"{avg_loss:.4f}",
                "keep":  f"{loss_keep.item():.3f}",
                "logit": f"{loss_forget_logit.item():.3f}",
                "sens":  f"{loss_forget_feat.item():.3f}",
            })

    return model

# # -------- 视觉特征忘却（按类） --------
# def unlearn_one_class_on_model_vis(
#     model,
#     dataloader,
#     forget_cls,
#     topk_idx,
#     device,
#     args,
#     emb_feat,
#     clip_model,
#     epochs=1,
#     lambda_keep=1.0,
#     lambda_vis=1.0,
#     lr=1e-4,
# ):
#     """
#     视觉特征按类忘却：
#       - loss_keep：保护所有“非 forget_cls”的 BCE
#       - loss_vis：只在目标类正样本上，把视觉 top-K 维度拉小 / 对齐负样本中心
#     """
#     model.train()
#     model.to(device)
#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=lr,
#     )
#
#     topk_idx = topk_idx.to(device).long()
#
#     for ep in range(epochs):
#         pbar = tqdm(dataloader, desc=f"[Vis-Forget] cls {forget_cls} | ep {ep}", ncols=100)
#         running_loss = 0.0
#
#         for step, batch in enumerate(pbar, start=1):
#             images = batch['image'].float().to(device)
#             labels = batch['labels'].float().to(device)
#             mask   = batch['mask'].float().to(device)
#             mask_in = mask.clone()
#
#             optimizer.zero_grad()
#
#             logits, _, _, vis_feat = model(
#                 images,
#                 mask_in,
#                 args.learn_emb_type,
#                 emb_feat,
#                 clip_model,
#                 return_label_emb=True,
#             )   # vis_feat: (B, T, D)
#
#             # ===== 1) 保留其他类别性能 =====
#             bce_all = F.binary_cross_entropy_with_logits(
#                 logits,
#                 labels,
#                 reduction='none'
#             )  # (B, L)
#
#             keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
#             keep_mask[:, forget_cls] = False
#             if keep_mask.sum() > 0:
#                 loss_keep = bce_all[keep_mask].mean()
#             else:
#                 loss_keep = torch.tensor(0.0, device=device)
#
#             # ===== 2) 视觉特征忘却：对所有样本的 top-K 维做 collapse =====
#             # vis_feat: (B, 49, 512) -> 先做全局 pooling
#             vis_feat_mean = vis_feat.mean(dim=1)  # (B, 512)
#             v_topk_all = vis_feat_mean[:, topk_idx]  # (B, K)
#
#             # 全局均值（detach，当目标点，不反传）
#             mu = v_topk_all.detach().mean(dim=0, keepdim=True)  # (1, K)
#
#             # 样本间 flatten + 推向 0
#             loss_flat = ((v_topk_all - mu) ** 2).mean() + (mu ** 2).mean()
#
#             lambda_feat = getattr(args, "lambda_feat_vis", 1.0)
#
#             loss_vis = lambda_feat * loss_flat
#             # ===== 3) 总损失 =====
#             loss = lambda_keep * loss_keep + lambda_vis * loss_vis
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
#             optimizer.step()
#
#             running_loss += loss.item()
#             avg_loss = running_loss / step
#
#             pbar.set_postfix({
#                 "loss": f"{avg_loss:.4f}",
#                 "keep": f"{loss_keep.item():.3f}",
#                 "vis":  f"{loss_vis.item():.3f}",
#             })
#
#     return model


def compute_vis_prototypes(
    model,
    dataloader,
    num_labels: int,
    device,
    args,
    emb_feat,
    clip_model,
):
    """
    统计每个类别 c 的视觉 prototype:
        proto[c] ≈ E_{x: y_c=1}[ mean_token(vis_feat(x)) ]

    返回:
        proto_vis: (num_labels, D) 的 tensor
    """
    model.eval()
    model.to(device)

    proto_sum = None   # (L, D)
    count = None       # (L,)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Proto] collect visual prototypes", ncols=100):
            images = batch["image"].float().to(device)
            labels = batch["labels"].float().to(device)   # (B, L)
            mask   = batch["mask"].float().to(device)
            mask_in = mask.clone()

            # logits, _, _, vis_feat: (B, T, D)
            logits, _, _, vis_feat = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )
            # 对每张图，先对 tokens 求平均 => (B, D)
            v_mean = vis_feat.mean(dim=1)   # token 平均

            B, L = labels.shape
            D = v_mean.size(1)
            if proto_sum is None:
                proto_sum = torch.zeros(L, D, device=device)
                count = torch.zeros(L, device=device)

            # 对每个类别 c，累加那些 y_c=1 样本的 v_mean
            for c in range(L):
                pos_idx = (labels[:, c] == 1)
                if pos_idx.any():
                    v_c = v_mean[pos_idx]         # (N_pos, D)
                    proto_sum[c] += v_c.sum(dim=0)
                    count[c]     += float(v_c.size(0))

    proto_vis = proto_sum / (count.unsqueeze(1) + 1e-6)   # (L, D)
    return proto_vis

def unlearn_one_class_on_model_vis(
    model,
    dataloader,
    forget_cls,
    topk_idx,
    device,
    args,
    emb_feat,
    clip_model,
    epochs=1,
    lambda_keep=1.0,   # 保护其它类
    lambda_vis=1.0,    # 方向抹除强度
    lr=1e-4,
):
    """
    视觉特征版“方向抹除”：
      - loss_keep：对非 forget_cls 的 BCE，保护其它 19 类
      - loss_vis：只对 forget_cls=Person 的视觉 token 做“判别方向擦除”

    vis_feat 形状： (B, T, D) = (B, 49, 512)
    topk_idx： collect_topk_dims_for_class_vis 得到的 (K,) 维度索引
    """

    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # 预先构建一个 top-K 维度的 mask，用在方向上
    D = 512
    topk_mask = torch.zeros(D, device=device)
    topk_mask[topk_idx] = 1.0  # 只有这些维度参与方向抹除

    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"[Vis-Forget(dir)] cls {forget_cls} | ep {ep}", ncols=100)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            images = batch['image'].float().to(device)
            labels = batch['labels'].float().to(device)
            mask   = batch['mask'].float().to(device)
            mask_in = mask.clone()

            optimizer.zero_grad()

            # 拿到视觉 token 特征
            logits, _, _, vis_feat = model(
                images,
                mask_in,
                args.learn_emb_type,
                emb_feat,
                clip_model,
                return_label_emb=True,
            )   # vis_feat: (B, T, D)

            # ===== 1) 保护其它类别性能（和之前一样） =====
            bce_all = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction='none'
            )  # (B, L)

            keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
            keep_mask[:, forget_cls] = False   # 目标类不做 BCE
            if keep_mask.sum() > 0:
                loss_keep = bce_all[keep_mask].mean()
            else:
                loss_keep = torch.tensor(0.0, device=device)

            # ===== 2) 方向抹除：只对 Person 正样本的 vis_feat 做“判别方向擦除” =====
            pos_idx = (labels[:, forget_cls] == 1)
            neg_idx = (labels[:, forget_cls] == 0)

            loss_vis = torch.tensor(0.0, device=device)

            # 至少要有正样本才有必要算方向
            if pos_idx.any():
                v_pos = vis_feat[pos_idx]  # (N_pos, T, D)

                # ---- 2.1 估一个“Person 判别方向” u_raw ----
                # 用当前 batch 的 pos/neg 平均特征差： μ_pos - μ_neg
                if neg_idx.any():
                    v_neg = vis_feat[neg_idx]  # (N_neg, T, D)
                    mu_pos = v_pos.mean(dim=(0, 1))   # (D,)
                    mu_neg = v_neg.mean(dim=(0, 1))   # (D,)
                    u_raw = mu_pos - mu_neg          # (D,)
                else:
                    # 没有负样本就用正样本均值凑合一下
                    u_raw = v_pos.mean(dim=(0, 1))    # (D,)

                # 只在 top-K 维度上做方向抹除
                u_raw = u_raw * topk_mask            # (D,)

                # 若方向几乎全 0，就直接跳过
                if u_raw.abs().sum() > 1e-6:
                    u = F.normalize(u_raw, dim=0)    # 单位向量 (D,)

                    # ---- 2.2 计算投影并惩罚：擦掉在 u 上的分量 ----
                    # v_pos: (N_pos, T, D)
                    # alpha: 每个 token 在 u 上的系数 (N_pos, T, 1)
                    alpha = (v_pos * u).sum(dim=-1, keepdim=True)
                    # 投影部分： alpha * u  （广播到 D 维）
                    v_proj = alpha * u  # (N_pos, T, D)

                    # 惩罚这一部分 → 相当于让 v_pos 在 u 方向上尽量为 0
                    loss_dir = (v_proj ** 2).mean()

                    loss_vis = loss_dir

            # ===== 3) 总 loss：保护其它类 + 方向抹除 =====
            loss = lambda_keep * loss_keep + lambda_vis * loss_vis

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / step

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "keep": f"{loss_keep.item():.3f}",
                "vis_dir": f"{loss_vis.item():.3f}",
            })

    return model

def unlearn_one_class_on_model_up(
    model,
    dataloader,
    forget_cls: int,
    topk_idx_high: torch.Tensor,  # (K,), 在高维空间里的索引
    device,
    args,
    emb_feat,
    clip_model,
    projector,
    epochs: int = 1,
    lambda_keep: float = 1.0,
    lambda_forget_feat: float = 1.0,
    lr: float = 1e-4,
):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"[Up-Feat] Unlearn cls {forget_cls} | ep {ep}", ncols=100)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            images = batch['image'].float().to(device)
            labels = batch['labels'].float().to(device)
            mask   = batch['mask'].float().to(device)
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

            # 1) 保留其他类别的 BCE
            bce_all = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction='none'
            )
            keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
            keep_mask[:, forget_cls] = False
            loss_keep = bce_all[keep_mask].mean() if keep_mask.sum() > 0 else torch.tensor(0.0, device=device)

            # 2) 高维空间里做特征收缩
            z_f = label_emb[:, forget_cls, :]            # (B, 512)
            h_f = projector(z_f)                         # (B, D_high)

            pos_idx = (labels[:, forget_cls] == 1)
            loss_forget_feat = torch.tensor(0.0, device=device)
            if pos_idx.any():
                h_pos = h_f[pos_idx]                     # (N_pos, D_high)
                h_pos_topk = h_pos[:, topk_idx_high]     # (N_pos, K)

                # 直接往 0 收缩
                loss_forget_feat = (h_pos_topk ** 2).mean()

            loss = lambda_keep * loss_keep + lambda_forget_feat * loss_forget_feat

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / step

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "keep": f"{loss_keep.item():.3f}",
                "feat": f"{loss_forget_feat.item():.3f}",
            })

    return model

def unlearn_one_class_on_model_vis_up(
    model,
    dataloader,
    forget_cls,
    topk_idx,
    device,
    args,
    emb_feat,
    clip_model,
    projector,        # 新增
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

    # 保证 topk_idx 在正确 device 上
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
            h_feat = projector(vis_feat)   # projector 要支持 (B,T,512) 输入，譬如 nn.Linear(512, D_high)

            # ===== 1) 保护其它类别 =====
            bce_all = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction='none'
            )
            keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
            keep_mask[:, forget_cls] = False
            loss_keep = bce_all[keep_mask].mean() if keep_mask.sum() > 0 \
                        else torch.tensor(0.0, device=device)

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
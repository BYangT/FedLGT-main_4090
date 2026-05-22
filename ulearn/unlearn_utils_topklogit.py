# # ulearn/unlearn_utils_topklogit.py
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
#
# def collect_topk_dims_for_class(
#     model,
#     dataloader,
#     forget_cls: int,        # 要遗忘的类别 index（0~L-1）
#     K: int,                 # 选多少个维度，比如 64
#     device,
#     args,
#     emb_feat,               # fed_main 里你已经算好的 label_text_features
#     clip_model,             # fed_main 里加载的 CLIP 模型
# ):
#     model.eval()
#     model.to(device)
#
#     # hidden 尺寸（一般是 512）
#     hidden_dim = 512
#
#     # 用来累计【目标类正样本】在每个维度上的 |激活| 之和
#     # pos_num 为一维，长度为512的全零向量
#     pos_sum = torch.zeros(hidden_dim, device=device)
#     pos_cnt = 0
#
#     # 统计【目标类负样本】（可以保留，方便之后扩展）
#     neg_sum = torch.zeros(hidden_dim, device=device)
#     neg_cnt = 0
#
#     # ⭐ 新增：统计【所有非目标类正样本】在每个维度上的 |激活| 之和
#     other_pos_sum = torch.zeros(hidden_dim, device=device)
#     other_pos_cnt = 0
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Collect top-K stats"):
#             images = batch['image'].float().to(device)   # (B, C, H, W)
#             labels = batch['labels'].float().to(device)  # (B, L)
#             mask   = batch['mask'].float().to(device)    # (B, L)
#             mask_in = mask.clone()
#
#             logits, _, _, label_emb = model(
#                 images,
#                 mask_in,
#                 args.learn_emb_type,
#                 emb_feat,
#                 clip_model,
#                 return_label_emb=True,
#             )
#             # label_emb: (B, L, hidden_dim)
#             B, L, D = label_emb.shape
#
#             # ====== 1) 目标类的 embedding 统计 ======
#             z_f = label_emb[:, forget_cls, :]  # (B, hidden_dim)
#
#             pos_idx = (labels[:, forget_cls] == 1)
#             neg_idx = (labels[:, forget_cls] == 0)
#
#             if pos_idx.any():
#                 # 目标类正样本上的512维特征
#                 z_pos = z_f[pos_idx]                      # (N_pos, hidden_dim)
#                 # 按列求和，求总激活值
#                 pos_sum += z_pos.abs().sum(dim=0)
#                 # 统计正样本数
#                 pos_cnt += z_pos.size(0)
#             # 这个是求目标类负样本的，流程和上面一样
#             if neg_idx.any():
#                 z_neg = z_f[neg_idx]
#                 neg_sum += z_neg.abs().sum(dim=0)
#                 neg_cnt += z_neg.size(0)
#
#             # ====== 2) ⭐ 非目标类正样本的 embedding 统计 ======
#             # 对所有 c != forget_cls，收集 labels[:, c] == 1 的样本对应的 label_emb[:, c, :]
#             for c in range(L):
#                 if c == forget_cls:
#                     continue
#                 other_idx = (labels[:, c] == 1)
#                 if other_idx.any():
#                     z_other = label_emb[other_idx, c, :]   # (N_other, hidden_dim)
#                     other_pos_sum += z_other.abs().sum(dim=0)
#                     other_pos_cnt += z_other.size(0)
#
#     # ==== 统计结果 → 得到各类的平均激活 ====
#
#     # 目标类正样本平均绝对激活
#     pos_mean = pos_sum/ (pos_cnt+ 1e-6)   # (hidden_dim,)
#
#     # 目标类负样本平均绝对激活（暂时不用）
#     neg_mean   = neg_sum/ (neg_cnt+ 1e-6)   # (hidden_dim,)
#
#     # ⭐ 非目标类正样本平均绝对激活（所有非目标类的“公共激活”）
#     other_mean = other_pos_sum / (other_pos_cnt + 1e-6)   # (hidden_dim,)
#
#     # ===== 1) 基础“类专属程度”：目标类高、其他类低更好 =====
#     base_score = pos_mean - other_mean      # (hidden_dim,)
#
#     # ===== 2) 融合分类头权重：看这个维度对 logit_c 的影响有多大 =====
#     # 假设你的分类头是 model.output_linear，形状 [num_classes, hidden_dim]
#     with torch.no_grad():
#         W_c = model.output_linear.weight[forget_cls].to(device)   # (hidden_dim,)
#
#     # 最终打分：维度既要“对这个类激活高且类专属”，又要“权重大、改一下就影响 logit_c”
#     score = base_score * W_c.abs()     # (hidden_dim,)
#
#     # ===== 3) 剔掉“公共特征”：目标类 & 非目标类都很高的维度 =====
#     pos_thr   = pos_mean.mean()   + 0.5 * pos_mean.std()
#     other_thr = other_mean.mean() + 0.5 * other_mean.std()
#
#     # public_mask: 目标类和非目标类都激活很高 → 公共特征，必须排除
#     public_mask = (pos_mean > pos_thr) & (other_mean > other_thr)
#
#     # 对 score 做显著性过滤
#     score_thr  = score.mean() + 0.5 * score.std()
#     score_mask = score > score_thr
#
#     # 综合过滤：既要 score 高，又不能是公共特征
#     mask_good = score_mask & (~public_mask)
#
#     score_filtered = score.clone()
#     score_filtered[~mask_good] = -1e9   # 直接打成极小值，防止进 top-K
#
#     # ===== 4) 取 top-K 维度 =====
#     topk_vals, topk_idx = torch.topk(score_filtered, k=K, largest=True)
#     return topk_idx, score   # 如果你本来就是这样返回的话，保持不变
#
#
# def unlearn_one_class_on_model_topk_logit(
#     model,
#     dataloader,
#     forget_cls: int,
#     topk_idx: torch.Tensor,
#     device,
#     args,
#     emb_feat,
#     clip_model,
#     epochs: int = 1,
#     lambda_keep: float = 1.0,
#     lambda_forget_logit: float = 0.0,
#     lambda_forget_feat: float = 1.0,
#     lr: float = 1e-4,
# ):
#     """
#     “top-K 维 + logits 惩罚” 单模型遗忘版本（label embedding 版）
#
#     - loss_keep:
#         对【非目标类】做正常 BCE，保护其它 L-1 个类别；
#     - loss_forget_logit:
#         对目标类的 logit 强行拉向 0（目标 label=0），
#         会直接压低 AP / P / R；
#     - loss_forget_feat:
#         对目标类正样本的 label embedding 中 top-K 维做 L2 收缩，
#         抹掉判别维度。
#
#     参数：
#       topk_idx: collect_topk_dims_for_class 得到的 (K,) 维索引
#     """
#
#     model.train()
#     model.to(device)
#
#     topk_idx = topk_idx.to(device)
#     D = topk_idx.numel()  # 实际 K，用于 log 打印而已
#
#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=lr,
#     )
#
#     for ep in range(epochs):
#         pbar = tqdm(dataloader, desc=f"[TopK+Logit] cls {forget_cls} | ep {ep}", ncols=100)
#         running_loss = 0.0
#
#         for step, batch in enumerate(pbar, start=1):
#             images = batch["image"].float().to(device)
#             labels = batch["labels"].float().to(device)
#             mask   = batch["mask"].float().to(device)
#             mask_in = mask.clone()
#
#             optimizer.zero_grad()
#
#             logits, _, _, label_emb = model(
#                 images,
#                 mask_in,
#                 args.learn_emb_type,
#                 emb_feat,
#                 clip_model,
#                 return_label_emb=True,
#             )
#             # logits: (B,L)
#             # label_emb: (B,L,D_emb)
#             B, L = logits.shape
#             _, _, D_emb = label_emb.shape
#
#             # ===== 1) 保护其它类别的 BCE =====
#             bce_all = F.binary_cross_entropy_with_logits(
#                 logits, labels, reduction="none"
#             )   # (B,L)
#
#             keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
#             keep_mask[:, forget_cls] = False
#             if keep_mask.sum() > 0:
#                 loss_keep = bce_all[keep_mask].mean()
#             else:
#                 loss_keep = torch.tensor(0.0, device=device)
#
#             # ===== 2) 目标类 logits 惩罚：强制该 logit ≈ 0 =====
#             logits_f = logits[:, forget_cls]      # (B,)
#             target_zero = torch.zeros_like(logits_f)
#             loss_forget_logit = F.binary_cross_entropy_with_logits(
#                 logits_f, target_zero, reduction="mean"
#             )
#
#             # ===== 3) top-K 判别维度 L2 shrink（仅正样本） =====
#             pos_idx = (labels[:, forget_cls] == 1)
#             if pos_idx.any():
#                 z_f = label_emb[:, forget_cls, :]       # (B,D_emb)
#                 z_pos = z_f[pos_idx]                    # (N_pos, D_emb)
#                 # 只取 top-K 维度
#                 z_pos_topk = z_pos[:, topk_idx]         # (N_pos, K)
#                 loss_forget_feat = (z_pos_topk ** 2).mean()
#             else:
#                 loss_forget_feat = torch.tensor(0.0, device=device)
#
#             # ===== 4) 总 loss =====
#             loss = (
#                 lambda_keep * loss_keep +
#                 lambda_forget_logit * loss_forget_logit +
#                 lambda_forget_feat * loss_forget_feat
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
#                 "K": D,
#             })
#
#     return model
#
#
# def unlearn_one_class_on_model(
#     model,
#     dataloader,
#     forget_cls: int,
#     topk_idx: torch.Tensor,
#     device,
#     args,
#     emb_feat,
#     clip_model,
#     epochs: int = 1,
#     lambda_keep: float = 1.0,
#     lambda_forget_logit: float = 5.0,
#     lambda_forget_feat: float = 1.0,
#     lr: float = 1e-4,
# ):
#     """
#     方案二（结合你当前 CTranModel）：
#       - 冻结所有非目标类参数
#       - 只更新：
#           * output_linear 的第 forget_cls 行
#           * label_lt 的第 forget_cls 行（可选）
#           * ResNet50 最后一层：backbone.base_network.layer4
#     """
#
#     import torch.nn.functional as F
#     from tqdm import tqdm
#
#     model.train()
#     model.to(device)
#
#     topk_idx = topk_idx.to(device)
#     D = topk_idx.numel()
#     print("【保护版】只更新 head(Person) + backbone.layer4")
#
#     # ===== ① 先把所有参数默认冻结 =====
#     for p in model.parameters():
#         p.requires_grad = False
#
#     # ===== ② 解冻 output_linear，并只让第 forget_cls 行更新 =====
#     W = model.output_linear.weight      # (num_labels, 512)
#     b = model.output_linear.bias        # (num_labels,)
#
#     W.requires_grad = True
#     b.requires_grad = True
#
#     # 只让第 forget_cls 行的梯度通过
#     mask_W = torch.zeros_like(W)
#     mask_W[forget_cls] = 1.0
#
#     def hook_W(grad):
#         # grad: (num_labels, 512)
#         return grad * mask_W
#
#     handle_W = W.register_hook(hook_W)
#
#     mask_b = torch.zeros_like(b)
#     mask_b[forget_cls] = 1.0
#
#     def hook_b(grad):
#         return grad * mask_b
#
#     handle_b = b.register_hook(hook_b)
#
#     # ===== ③ 可选：只更新 label_lt 里目标类那一行 =====
#     if hasattr(model, "label_lt"):
#         LE = model.label_lt                      # nn.Embedding(num_labels, 512)
#         LE.weight.requires_grad = True
#
#         mask_LE = torch.zeros_like(LE.weight)
#         mask_LE[forget_cls] = 1.0               # 只给这一行梯度
#
#         def hook_LE(grad):
#             return grad * mask_LE
#
#         handle_LE = LE.weight.register_hook(hook_LE)
#     else:
#         handle_LE = None
#
#     # ===== ④ 解冻 ResNet50 的 layer4（高层视觉特征） =====
#     # 你给的 Backbone 里：self.base_network = models.resnet50(...)
#     # 所以这里直接：
#     for p in model.backbone.base_network.layer4.parameters():
#         p.requires_grad = True
#
#     # ===== ⑤ 优化器：只对 requires_grad=True 的参数更新 =====
#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=lr,
#     )
#
#     # ===== ⑥ 下面保持原来的 loss 设计不变 =====
#     for ep in range(epochs):
#         pbar = tqdm(dataloader, desc=f"[TopK+Logit] cls {forget_cls} | ep {ep}", ncols=100)
#         running_loss = 0.0
#
#         for step, batch in enumerate(pbar, start=1):
#             images = batch["image"].float().to(device)
#             labels = batch["labels"].float().to(device)
#             mask   = batch["mask"].float().to(device)
#             mask_in = mask.clone()
#
#             optimizer.zero_grad()
#
#             logits, _, _, label_emb = model(
#                 images,
#                 mask_in,
#                 args.learn_emb_type,
#                 emb_feat,
#                 clip_model,
#                 return_label_emb=True,
#             )
#             # logits: (B, L)
#             # label_emb: (B, L, D_emb)
#
#             # 1) 保护其它类别：只在非目标类上算 BCE
#             bce_all = F.binary_cross_entropy_with_logits(
#                 logits, labels, reduction="none"
#             )   # (B,L)
#
#             keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
#             keep_mask[:, forget_cls] = False
#             if keep_mask.sum() > 0:
#                 loss_keep = bce_all[keep_mask].mean()
#             else:
#                 loss_keep = torch.tensor(0.0, device=device)
#
#             # 2) 目标类 logit → 0（或接近 0）
#             logits_f = logits[:, forget_cls]      # (B,)
#             target_zero = torch.zeros_like(logits_f)
#             loss_forget_logit = F.binary_cross_entropy_with_logits(
#                 logits_f, target_zero, reduction="mean"
#             )
#
#             # 3) 目标类正样本 top-K 维收缩
#             pos_idx = (labels[:, forget_cls] == 1)
#             if pos_idx.any():
#                 z_f = label_emb[:, forget_cls, :]       # (B, D_emb)
#                 z_pos = z_f[pos_idx]                    # (N_pos, D_emb)
#                 z_pos_topk = z_pos[:, topk_idx]         # (N_pos, K)
#                 loss_forget_feat = (z_pos_topk ** 2).mean()
#             else:
#                 loss_forget_feat = torch.tensor(0.0, device=device)
#
#             loss = (
#                 lambda_keep * loss_keep +
#                 lambda_forget_logit * loss_forget_logit +
#                 lambda_forget_feat * loss_forget_feat
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
#                 "K": D,
#             })
#
#     # ===== ⑦ 清理 hook，避免影响后续别的训练阶段 =====
#     handle_W.remove()
#     handle_b.remove()
#     if handle_LE is not None:
#         handle_LE.remove()
#
#     return model
###111
# ulearn/unlearn_utils_topklogit.py
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
#
# try:
#     from .sae import SparseAutoEncoder
# except ImportError:
#     from sae import SparseAutoEncoder
#
#
# _GLOBAL_SAE_MODEL = None
#
#
# def _get_sae_model(args, device):
#     """
#     懒加载 SAE，不改原函数签名。
#     依赖 args 里的字段：
#         args.sae_ckpt       : SAE 权重路径（必须）
#         args.sae_input_dim  : 输入维度，默认 512
#         args.sae_latent_dim : latent 维度，默认 1024
#         args.sae_activation : "relu" / "softplus" / "identity"，默认 "relu"
#     """
#     global _GLOBAL_SAE_MODEL
#
#     if _GLOBAL_SAE_MODEL is not None:
#         return _GLOBAL_SAE_MODEL
#
#     sae_ckpt = getattr(args, "sae_ckpt", None)
#     input_dim = getattr(args, "sae_input_dim", 512)
#     latent_dim = getattr(args, "sae_latent_dim", 1024)
#     activation = getattr(args, "sae_activation", "relu")
#
#     if sae_ckpt is None:
#         raise ValueError(
#             "args.sae_ckpt is required. "
#             "Please set for example: args.sae_ckpt = 'ulearn/sae_512_to_1024.pth'"
#         )
#
#     sae_model = SparseAutoEncoder(
#         input_dim=input_dim,
#         latent_dim=latent_dim,
#         activation=activation,
#     )
#
#     state_dict = torch.load(sae_ckpt, map_location=device)
#     sae_model.load_state_dict(state_dict)
#     sae_model.to(device)
#     sae_model.eval()
#
#     # 冻结 SAE 参数，但不能用 no_grad 包住 forward；
#     # 否则 loss_forget_feat 无法回传到主模型
#     for p in sae_model.parameters():
#         p.requires_grad = False
#
#     _GLOBAL_SAE_MODEL = sae_model
#     return _GLOBAL_SAE_MODEL
#
#
# def _encode_label_emb_with_sae(label_emb, sae_model):
#     """
#     label_emb: (B, L, D_raw)
#     return:
#         z_all: (B, L, latent_dim)
#     """
#     B, L, D_raw = label_emb.shape
#     label_emb_flat = label_emb.reshape(B * L, D_raw)
#     _, z_flat = sae_model(label_emb_flat)   # SAE forward -> (x_hat, z)
#     z_all = z_flat.reshape(B, L, -1)
#     return z_all
#
#
# def collect_topk_dims_for_class(
#     model,
#     dataloader,
#     forget_cls: int,
#     K: int,
#     device,
#     args,
#     emb_feat,
#     clip_model,
# ):
#     model.eval()
#     model.to(device)
#
#     sae_model = _get_sae_model(args, device)
#     latent_dim = sae_model.latent_dim
#     beta_neg = 0.5   # 你可以后面再调
#
#     # ===== 目标类正/负样本统计 =====
#     pos_sum = torch.zeros(latent_dim, device=device)
#     pos_cnt = 0
#
#     neg_sum = torch.zeros(latent_dim, device=device)
#     neg_cnt = 0
#
#     # ===== 每个类单独统计正样本均值 =====
#     num_labels = None
#     with torch.no_grad():
#         for batch in dataloader:
#             images = batch['image'].float().to(device)
#             labels = batch['labels'].float().to(device)
#             mask   = batch['mask'].float().to(device)
#
#             _, _, _, label_emb = model(
#                 images,
#                 mask.clone(),
#                 args.learn_emb_type,
#                 emb_feat,
#                 clip_model,
#                 return_label_emb=True,
#             )
#             _, num_labels, _ = label_emb.shape
#             break
#
#     if num_labels is None:
#         raise ValueError("Empty dataloader in collect_topk_dims_for_class")
#
#     class_pos_sum = torch.zeros(num_labels, latent_dim, device=device)
#     class_pos_cnt = torch.zeros(num_labels, device=device)
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Collect top-K stats (SAE latent)"):
#             images = batch['image'].float().to(device)
#             labels = batch['labels'].float().to(device)
#             mask   = batch['mask'].float().to(device)
#             mask_in = mask.clone()
#
#             _, _, _, label_emb = model(
#                 images,
#                 mask_in,
#                 args.learn_emb_type,
#                 emb_feat,
#                 clip_model,
#                 return_label_emb=True,
#             )
#             # label_emb: (B, L, D_raw)
#             B, L, D_raw = label_emb.shape
#
#             # ===== 原始特征 -> SAE latent =====
#             z_all = _encode_label_emb_with_sae(label_emb, sae_model)   # (B, L, latent_dim)
#
#             # ===== 1) 目标类正/负样本统计 =====
#             z_f = z_all[:, forget_cls, :]  # (B, latent_dim)
#
#             pos_idx = (labels[:, forget_cls] == 1)
#             neg_idx = (labels[:, forget_cls] == 0)
#
#             if pos_idx.any():
#                 z_pos = z_f[pos_idx]
#                 pos_sum += z_pos.abs().sum(dim=0)
#                 pos_cnt += z_pos.size(0)
#
#             if neg_idx.any():
#                 z_neg = z_f[neg_idx]
#                 neg_sum += z_neg.abs().sum(dim=0)
#                 neg_cnt += z_neg.size(0)
#
#             # ===== 2) 每个类单独统计正样本 =====
#             for c in range(L):
#                 cls_idx = (labels[:, c] == 1)
#                 if cls_idx.any():
#                     z_cls = z_all[cls_idx, c, :]  # (N_cls, latent_dim)
#                     class_pos_sum[c] += z_cls.abs().sum(dim=0)
#                     class_pos_cnt[c] += z_cls.size(0)
#
#     # ===== 均值 =====
#     pos_mean = pos_sum / (pos_cnt + 1e-6)      # (latent_dim,)
#     neg_mean = neg_sum / (neg_cnt + 1e-6)      # (latent_dim,)
#
#     class_mean = class_pos_sum / (class_pos_cnt.unsqueeze(1) + 1e-6)   # (L, latent_dim)
#
#     # ===== 最强竞争类 =====
#     competitor_mean = class_mean.clone()
#     competitor_mean[forget_cls] = -1e9
#
#     max_comp_mean, max_comp_cls = competitor_mean.max(dim=0)   # (latent_dim,), (latent_dim,)
#
#     # ===== 新版基础分数 =====
#     # 目标类高于最强竞争类 + 目标类高于自己负样本
#     base_score = (pos_mean - max_comp_mean) + beta_neg * (pos_mean - neg_mean)
#
#     # ===== 分类头权重投影到 latent 空间 =====
#     with torch.no_grad():
#         W_c_raw = model.output_linear.weight[forget_cls].to(device)   # (D_raw,)
#         W_enc = sae_model.encoder.weight.to(device)                   # (latent_dim, D_raw)
#         W_c_latent = torch.matmul(W_enc, W_c_raw)                     # (latent_dim,)
#
#     score = base_score * W_c_latent.abs()
#
#     # ===== 公共特征过滤 =====
#     pos_thr = pos_mean.mean() + 0.5 * pos_mean.std()
#
#     valid_comp = max_comp_mean[max_comp_mean > -1e8]
#     if valid_comp.numel() > 0:
#         comp_thr = valid_comp.mean() + 0.5 * valid_comp.std()
#     else:
#         comp_thr = torch.tensor(0.0, device=device)
#
#     public_mask = (pos_mean > pos_thr) & (max_comp_mean > comp_thr)
#
#     score_thr = score.mean() + 0.5 * score.std()
#     score_mask = score > score_thr
#
#     mask_good = score_mask & (~public_mask)
#
#     score_filtered = score.clone()
#     score_filtered[~mask_good] = -1e9
#
#     # ===== 取 Top-K latent 维度 =====
#     topk_vals, topk_idx = torch.topk(score_filtered, k=K, largest=True)
#
#     # ===== 额外输出：Top-K 维度里最相关的几个类 =====
#     # 定义：Top-K 维度对应的 max competitor class 出现频次最高的类
#     topk_comp_classes = max_comp_cls[topk_idx]   # (K,)
#
#     related_counts = torch.bincount(topk_comp_classes, minlength=num_labels)
#     related_counts[forget_cls] = 0  # 保险起见，去掉自己
#
#     top_m = min(5, num_labels - 1)
#     top_related_vals, top_related_cls = torch.topk(
#         related_counts, k=top_m, largest=True
#     )
#
#     print(f"\n[forget_cls={forget_cls}] Top related classes by Top-K competitor frequency:")
#     for cls_id, cnt in zip(top_related_cls.tolist(), top_related_vals.tolist()):
#         if cnt > 0:
#             print(f"  class {cls_id}: {cnt} times")
#
#     # 也可以顺便打印每个 Top-K 维度对应的竞争类（调试用）
#     # print("Top-K competitor classes:", topk_comp_classes.tolist())
#
#     return topk_idx, score
#
#
# def unlearn_one_class_on_model_topk_logit(
#     model,
#     dataloader,
#     forget_cls: int,
#     topk_idx: torch.Tensor,
#     device,
#     args,
#     emb_feat,
#     clip_model,
#     epochs: int = 1,
#     lambda_keep: float = 1.0,
#     lambda_forget_logit: float = 0.0,
#     lambda_forget_feat: float = 1.0,
#     lr: float = 1e-4,
# ):
#     """
#     “top-K latent 维 + logits 惩罚” 单模型遗忘版本（SAE latent 版）
#
#     - loss_keep:
#         对【非目标类】做正常 BCE，保护其它 L-1 个类别；
#     - loss_forget_logit:
#         对目标类的 logit 强行拉向 0（目标 label=0）；
#     - loss_forget_feat:
#         对目标类正样本的 SAE latent 中 top-K 维做 L2 收缩。
#
#     参数：
#       topk_idx: collect_topk_dims_for_class 得到的 (K,) latent 维索引
#     """
#
#     model.train()
#     model.to(device)
#
#     sae_model = _get_sae_model(args, device)
#     sae_model.eval()  # SAE 固定，不参与训练
#
#     topk_idx = topk_idx.to(device)
#     D = topk_idx.numel()
#
#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=lr,
#     )
#
#     for ep in range(epochs):
#         pbar = tqdm(dataloader, desc=f"[TopK+Logit] cls {forget_cls} | ep {ep}", ncols=100)
#         running_loss = 0.0
#
#         for step, batch in enumerate(pbar, start=1):
#             images = batch["image"].float().to(device)
#             labels = batch["labels"].float().to(device)
#             mask   = batch["mask"].float().to(device)
#             mask_in = mask.clone()
#
#             optimizer.zero_grad()
#
#             logits, _, _, label_emb = model(
#                 images,
#                 mask_in,
#                 args.learn_emb_type,
#                 emb_feat,
#                 clip_model,
#                 return_label_emb=True,
#             )
#             # logits: (B,L)
#             # label_emb: (B,L,D_emb)
#             B, L = logits.shape
#
#             # ===== 映射到 SAE latent 空间 =====
#             # 注意：这里不能用 no_grad，否则 loss_forget_feat 无法回传到主模型
#             z_all = _encode_label_emb_with_sae(label_emb, sae_model)   # (B, L, latent_dim)
#
#             # ===== 1) 保护其它类别的 BCE =====
#             bce_all = F.binary_cross_entropy_with_logits(
#                 logits, labels, reduction="none"
#             )   # (B,L)
#
#             keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
#             keep_mask[:, forget_cls] = False
#             if keep_mask.sum() > 0:
#                 loss_keep = bce_all[keep_mask].mean()
#             else:
#                 loss_keep = torch.tensor(0.0, device=device)
#
#             # ===== 2) 目标类 logits 惩罚：强制该 logit ≈ 0 =====
#             logits_f = logits[:, forget_cls]      # (B,)
#             target_zero = torch.zeros_like(logits_f)
#             loss_forget_logit = F.binary_cross_entropy_with_logits(
#                 logits_f, target_zero, reduction="mean"
#             )
#
#             # ===== 3) top-K latent 判别维度 L2 shrink（仅正样本） =====
#             pos_idx = (labels[:, forget_cls] == 1)
#             if pos_idx.any():
#                 z_f = z_all[:, forget_cls, :]       # (B, latent_dim)
#                 z_pos = z_f[pos_idx]                # (N_pos, latent_dim)
#                 z_pos_topk = z_pos[:, topk_idx]     # (N_pos, K)
#                 loss_forget_feat = (z_pos_topk ** 2).mean()
#             else:
#                 loss_forget_feat = torch.tensor(0.0, device=device)
#
#             # ===== 4) 总 loss =====
#             loss = (
#                 lambda_keep * loss_keep +
#                 lambda_forget_logit * loss_forget_logit +
#                 lambda_forget_feat * loss_forget_feat
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
#                 "K": D,
#             })
#
#     return model
#
#
# def unlearn_one_class_on_model(
#     model,
#     dataloader,
#     forget_cls: int,
#     topk_idx: torch.Tensor,
#     device,
#     args,
#     emb_feat,
#     clip_model,
#     epochs: int = 1,
#     lambda_keep: float = 1.0,
#     lambda_forget_logit: float = 5.0,
#     lambda_forget_feat: float = 1.0,
#     lr: float = 1e-4,
# ):
#     """
#     方案二（保护版，SAE latent 版）：
#       - 冻结所有非目标类参数
#       - 只更新：
#           * output_linear 的第 forget_cls 行
#           * label_lt 的第 forget_cls 行（可选）
#           * ResNet50 最后一层：backbone.base_network.layer4
#     """
#
#     model.train()
#     model.to(device)
#
#     sae_model = _get_sae_model(args, device)
#     sae_model.eval()  # SAE 固定，不参与训练
#
#     topk_idx = topk_idx.to(device)
#     D = topk_idx.numel()
#     print("【保护版】只更新 head(Person) + backbone.layer4")
#
#     # ===== ① 先把所有参数默认冻结 =====
#     for p in model.parameters():
#         p.requires_grad = False
#
#     # ===== ② 解冻 output_linear，并只让第 forget_cls 行更新 =====
#     W = model.output_linear.weight      # (num_labels, 512)
#     b = model.output_linear.bias        # (num_labels,)
#
#     W.requires_grad = True
#     b.requires_grad = True
#
#     mask_W = torch.zeros_like(W)
#     mask_W[forget_cls] = 1.0
#
#     def hook_W(grad):
#         return grad * mask_W
#
#     handle_W = W.register_hook(hook_W)
#
#     mask_b = torch.zeros_like(b)
#     mask_b[forget_cls] = 1.0
#
#     def hook_b(grad):
#         return grad * mask_b
#
#     handle_b = b.register_hook(hook_b)
#
#     # ===== ③ 可选：只更新 label_lt 里目标类那一行 =====
#     if hasattr(model, "label_lt"):
#         LE = model.label_lt
#         LE.weight.requires_grad = True
#
#         mask_LE = torch.zeros_like(LE.weight)
#         mask_LE[forget_cls] = 1.0
#
#         def hook_LE(grad):
#             return grad * mask_LE
#
#         handle_LE = LE.weight.register_hook(hook_LE)
#     else:
#         handle_LE = None
#
#     # ===== ④ 解冻 ResNet50 的 layer4（高层视觉特征） =====
#     for p in model.backbone.base_network.layer4.parameters():
#         p.requires_grad = True
#
#     # ===== ⑤ 优化器：只对 requires_grad=True 的参数更新 =====
#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=lr,
#     )
#
#     # ===== ⑥ loss 设计 =====
#     for ep in range(epochs):
#         pbar = tqdm(dataloader, desc=f"[TopK+Logit] cls {forget_cls} | ep {ep}", ncols=100)
#         running_loss = 0.0
#
#         for step, batch in enumerate(pbar, start=1):
#             images = batch["image"].float().to(device)
#             labels = batch["labels"].float().to(device)
#             mask   = batch["mask"].float().to(device)
#             mask_in = mask.clone()
#
#             optimizer.zero_grad()
#
#             logits, _, _, label_emb = model(
#                 images,
#                 mask_in,
#                 args.learn_emb_type,
#                 emb_feat,
#                 clip_model,
#                 return_label_emb=True,
#             )
#             # logits: (B, L)
#             # label_emb: (B, L, D_emb)
#
#             # ===== 映射到 SAE latent 空间 =====
#             # 注意：这里不能用 no_grad，否则 loss_forget_feat 无法回传到主模型
#             z_all = _encode_label_emb_with_sae(label_emb, sae_model)   # (B, L, latent_dim)
#
#             # 1) 保护其它类别：只在非目标类上算 BCE
#             bce_all = F.binary_cross_entropy_with_logits(
#                 logits, labels, reduction="none"
#             )   # (B,L)
#
#             keep_mask = torch.ones_like(bce_all, dtype=torch.bool, device=device)
#             keep_mask[:, forget_cls] = False
#             if keep_mask.sum() > 0:
#                 loss_keep = bce_all[keep_mask].mean()
#             else:
#                 loss_keep = torch.tensor(0.0, device=device)
#
#             # 2) 目标类 logit → 0
#             logits_f = logits[:, forget_cls]      # (B,)
#             target_zero = torch.zeros_like(logits_f)
#             loss_forget_logit = F.binary_cross_entropy_with_logits(
#                 logits_f, target_zero, reduction="mean"
#             )
#
#             # 3) 目标类正样本 top-K latent 维收缩
#             z_f = z_all[:, forget_cls, :]
#
#             pos_idx = (labels[:, forget_cls] == 1)
#             neg_idx = (labels[:, forget_cls] == 0)
#
#             if pos_idx.any() and neg_idx.any():
#                 z_pos = z_f[pos_idx]
#                 z_neg = z_f[neg_idx]
#
#                 z_pos_topk = z_pos[:, topk_idx]
#                 z_neg_topk = z_neg[:, topk_idx]
#
#                 mu_neg_topk = z_neg_topk.mean(dim=0, keepdim=True)
#
#                 loss_shrink = (z_pos_topk ** 2).mean()
#                 loss_pull_neg = ((z_pos_topk - mu_neg_topk) ** 2).mean()
#
#                 loss_forget_feat = 0.8 * loss_shrink + 0.2 * loss_pull_neg
#             else:
#                 loss_forget_feat = torch.tensor(0.0, device=device)
#
#             loss = (
#                 lambda_keep * loss_keep +
#                 0 * loss_forget_logit +
#                 lambda_forget_feat * loss_forget_feat
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
#                 "K": D,
#             })
#
#     # ===== ⑦ 清理 hook =====
#     handle_W.remove()
#     handle_b.remove()
#     if handle_LE is not None:
#         handle_LE.remove()
#
#     return model

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

    # ===== 目标类正/负样本统计 =====
    pos_sum = torch.zeros(latent_dim, device=device)
    pos_cnt = 0

    neg_sum = torch.zeros(latent_dim, device=device)
    neg_cnt = 0

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

    return topk_idx, score


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
                loss_forget_feat = (z_pos_topk ** 2).mean()
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

                mu_neg_topk = z_neg_topk.mean(dim=0, keepdim=True)

                loss_shrink = (z_pos_topk ** 2).mean()
                loss_pull_neg = ((z_pos_topk - mu_neg_topk) ** 2).mean()

                loss_forget_feat = 0.8 * loss_shrink + 0.2 * loss_pull_neg
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
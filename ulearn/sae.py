import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseAutoEncoder(nn.Module):
    """
    最小可用版 SAE
    - 输入:  (N, input_dim)
    - 输出:
        x_hat: (N, input_dim)   重建后的特征
        z:     (N, latent_dim)  稀疏 latent 表示
    """
    def __init__(self, input_dim=512, latent_dim=1024, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation

        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        z_pre = self.encoder(x)

        if self.activation == "relu":
            z = F.relu(z_pre)
        elif self.activation == "softplus":
            z = F.softplus(z_pre)
        elif self.activation == "identity":
            z = z_pre
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def sae_loss(x, x_hat, z, l1_lambda=1e-4):
    """
    SAE 损失:
    1. 重建损失
    2. 稀疏正则（L1 on z）
    """
    recon_loss = F.mse_loss(x_hat, x)
    sparse_loss = z.abs().mean()
    total_loss = recon_loss + l1_lambda * sparse_loss
    return total_loss, recon_loss, sparse_loss


def latent_label_selective_loss(z, label_ids, num_labels, eps=1e-8):
    """
    维度标签熵正则：
    - 每个 latent 维统计“被哪些 label token 在使用”
    - 惩罚被很多标签平均共享的维度
    - 鼓励每个 latent 维更偏向少数标签

    参数:
    - z: (N, latent_dim)
    - label_ids: (N,) 每一行 latent 对应的标签索引
    - num_labels: 标签总数
    """
    if label_ids is None or num_labels is None or num_labels <= 1:
        return z.new_zeros(())

    if label_ids.numel() == 0:
        return z.new_zeros(())

    act = z.abs()
    latent_dim = act.size(1)
    label_ids = label_ids.long()

    # 先统计每个标签在每个 latent 维上的激活均值，避免被 batch 内样本数轻易主导。
    label_sum = act.new_zeros((num_labels, latent_dim))
    label_sum.index_add_(0, label_ids, act)

    label_cnt = act.new_zeros((num_labels,))
    label_cnt.index_add_(0, label_ids, act.new_ones((label_ids.size(0),)))
    label_mean = label_sum / label_cnt.clamp_min(1.0).unsqueeze(1)

    dim_mass = label_mean.sum(dim=0)
    active_mask = dim_mass > eps
    if not active_mask.any():
        return z.new_zeros(())

    probs = label_mean[:, active_mask] / dim_mass[active_mask].clamp_min(eps).unsqueeze(0)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=0)
    entropy = entropy / max(math.log(num_labels), 1.0)
    return entropy.mean()


def get_batch_coupled_label_stats(labels, forget_cls, top_m=5, eps=1e-8):
    """
    根据 batch 内目标类正样本，统计与 forget_cls 高耦合的标签。
    返回:
    - coupled_ids: (M,)
    - coupled_weights: (M,)
    """
    if labels is None or labels.numel() == 0:
        device = labels.device if labels is not None else "cpu"
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, device=device)

    pos_idx = labels[:, forget_cls] > 0.5
    if not pos_idx.any():
        return torch.empty(0, dtype=torch.long, device=labels.device), torch.empty(0, device=labels.device)

    co_counts = labels[pos_idx].sum(dim=0)
    co_counts = co_counts.clone()
    co_counts[forget_cls] = 0.0
    valid_mask = co_counts > 0
    if not valid_mask.any():
        return torch.empty(0, dtype=torch.long, device=labels.device), torch.empty(0, device=labels.device)

    top_m = min(int(top_m), int(valid_mask.sum().item()))
    vals, idx = torch.topk(co_counts, k=top_m, largest=True)
    weights = vals / vals.sum().clamp_min(eps)
    return idx.long(), weights


def latent_coupling_overlap_loss(z_all, labels, forget_cls, coupled_ids, coupled_weights, eps=1e-8):
    """
    目标类与高耦合类重叠惩罚：
    - 对 forget_cls 的 latent 原型，与高耦合标签的 latent 原型做 cosine
    - 希望它们不要太像

    参数:
    - z_all: (B, L, latent_dim)
    - labels: (B, L)
    """
    if coupled_ids is None or coupled_ids.numel() == 0:
        return z_all.new_zeros(())

    target_mask = labels[:, forget_cls] > 0.5
    if not target_mask.any():
        return z_all.new_zeros(())

    target_proto = z_all[target_mask, forget_cls, :].abs().mean(dim=0, keepdim=True)
    if target_proto.abs().sum() <= eps:
        return z_all.new_zeros(())

    total_loss = z_all.new_zeros(())
    total_weight = z_all.new_zeros(())

    for cls_id, cls_weight in zip(coupled_ids.tolist(), coupled_weights):
        cls_mask = labels[:, cls_id] > 0.5
        if not cls_mask.any():
            continue
        cls_proto = z_all[cls_mask, cls_id, :].abs().mean(dim=0, keepdim=True)
        if cls_proto.abs().sum() <= eps:
            continue
        overlap = F.cosine_similarity(target_proto, cls_proto, dim=1).mean()
        total_loss = total_loss + cls_weight * overlap
        total_weight = total_weight + cls_weight

    if total_weight.abs() <= eps:
        return z_all.new_zeros(())
    return total_loss / total_weight.clamp_min(eps)


@torch.no_grad()
def get_sae_sparsity(z, eps=1e-8):
    """
    统计稀疏程度
    返回:
    - avg_l1: 平均 |z|
    - active_ratio: 大于 eps 的比例
    """
    avg_l1 = z.abs().mean().item()
    active_ratio = (z.abs() > eps).float().mean().item()
    return avg_l1, active_ratio

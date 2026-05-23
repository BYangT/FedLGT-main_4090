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

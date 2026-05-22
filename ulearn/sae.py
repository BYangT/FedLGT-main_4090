import torch
import torch.nn as nn
import torch.nn.functional as F


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
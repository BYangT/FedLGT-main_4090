import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from .sae import SparseAutoEncoder, sae_loss
except ImportError:
    from sae import SparseAutoEncoder, sae_loss


class FeatureDataset(Dataset):
    def __init__(self, feature_tensor):
        self.x = feature_tensor.float()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx]


@torch.no_grad()
def collect_label_emb_features(
    model,
    dataloader,
    device,
    args,
    emb_feat,
    clip_model,
    max_batches=None,
):
    """
    收集 label_emb[:, c, :] -> (N_total, D)
    """
    model.eval()
    model.to(device)

    all_features = []

    pbar = tqdm(dataloader, desc="Collect label_emb for SAE")
    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        images = batch["image"].float().to(device)
        mask   = batch["mask"].float().to(device)

        _, _, _, label_emb = model(
            images,
            mask.clone(),
            args.learn_emb_type,
            emb_feat,
            clip_model,
            return_label_emb=True,
        )
        # (B, L, D) -> (B*L, D)
        feat = label_emb.reshape(-1, label_emb.shape[-1]).detach().cpu()
        all_features.append(feat)

    feature_tensor = torch.cat(all_features, dim=0)
    return feature_tensor


def train_sae_on_features(
    feature_tensor,
    save_path,
    input_dim=512,
    latent_dim=1024,
    activation="relu",
    batch_size=256,
    epochs=20,
    lr=1e-3,
    l1_lambda=1e-4,
    device="cuda",
):
    dataset = FeatureDataset(feature_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )

    sae = SparseAutoEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        activation=activation,
    ).to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    for ep in range(epochs):
        sae.train()

        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        sparse_loss_sum = 0.0
        total_num = 0

        pbar = tqdm(loader, desc=f"Train SAE | ep {ep+1}/{epochs}")
        for x in pbar:
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, z = sae(x)
            loss, recon_loss, sparse_loss = sae_loss(
                x, x_hat, z, l1_lambda=l1_lambda
            )

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss_sum += loss.item() * bs
            recon_loss_sum += recon_loss.item() * bs
            sparse_loss_sum += sparse_loss.item() * bs
            total_num += bs

            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "recon": f"{recon_loss.item():.6f}",
                "sparse": f"{sparse_loss.item():.6f}",
            })

        print(
            f"[Epoch {ep+1}/{epochs}] "
            f"total={total_loss_sum / total_num:.6f} "
            f"recon={recon_loss_sum / total_num:.6f} "
            f"sparse={sparse_loss_sum / total_num:.6f}"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(sae.state_dict(), save_path)
    print(f"SAE saved to: {save_path}")

    return sae
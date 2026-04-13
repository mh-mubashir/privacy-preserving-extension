"""
VQ-VAE wrapper for the ARL pipeline.

Malia's models/vqvae.py has two issues that prevent ARL use:
  1. VectorQuantizer uses a global `device` variable — crashes on import
  2. VQVAE.forward() returns (embedding_loss, x_hat, perplexity) not recon

This wrapper reimplements the same architecture cleanly, fixing both issues,
so the VQ-VAE can be used as a drop-in encoder in the ARL pipeline alongside
VanillaVAE, BetaVAE, CVAE, FactorVAE, ResidualVAE.

Note: VQ-VAE has no mu/logvar (discrete codebook, not Gaussian posterior).
When return_aux=True, mu and logvar are returned as None.
evaluate.py handles this — KL metrics are skipped for vq_vae.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, 3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, 1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self.res_block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super().__init__()
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)


class VQEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, 3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

    def forward(self, x):
        return self.net(x)


class VQDecoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, 3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 3, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.e_dim)

        d = (z_flat.pow(2).sum(1, keepdim=True)
             + self.embedding.weight.pow(2).sum(1)
             - 2 * z_flat @ self.embedding.weight.t())

        idx = torch.argmin(d, dim=1).unsqueeze(1)
        one_hot = torch.zeros(idx.shape[0], self.n_e, device=z.device)
        one_hot.scatter_(1, idx, 1)

        z_q = (one_hot @ self.embedding.weight).view(z.shape)
        loss = (z_q.detach() - z).pow(2).mean() + self.beta * (z_q - z.detach()).pow(2).mean()
        z_q = z + (z_q - z).detach()   # straight-through estimator

        e_mean = one_hot.mean(0)
        perplexity = torch.exp(-(e_mean * (e_mean + 1e-10).log()).sum())
        return loss, z_q.permute(0, 3, 1, 2).contiguous(), perplexity


class VQVAEWrapper(nn.Module):
    """
    ARL-compatible VQ-VAE.  Same architecture as Malia's implementation,
    with the device bug fixed and sigmoid added to decoder output.

    Default hyperparameters match Malia's script defaults.

    forward(x)                  -> recon in [0, 1]
    forward(x, return_aux=True) -> (recon, None, None, z_q)
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        h_dim=128,
        res_h_dim=32,
        n_res_layers=2,
        n_embeddings=512,
        embedding_dim=64,
        beta=0.25,
        img_size=224,
    ):
        super().__init__()
        self.encoder       = VQEncoder(in_channels, h_dim, n_res_layers, res_h_dim)
        self.pre_quant     = nn.Conv2d(h_dim, embedding_dim, 1)
        self.vq            = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder       = VQDecoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x, return_aux=False):
        z_e = self.pre_quant(self.encoder(x))
        _, z_q, _ = self.vq(z_e)
        recon = torch.sigmoid(self.decoder(z_q))
        if return_aux:
            return recon, None, None, z_q
        return recon

    def get_latent(self, x):
        z_e = self.pre_quant(self.encoder(x))
        _, z_q, _ = self.vq(z_e)
        return z_q

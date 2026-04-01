"""
Beta-VAE for the ARL pipeline (Member 1).

Same architecture as VanillaVAE, but with a stronger KL penalty (beta > 1).
Based on Higgins et al. 2017 — the idea is that a higher beta pushes the
posterior toward a more factorised prior, which can help separate out
privacy-sensitive factors (e.g. gender) from utility factors (e.g. smile).

Default beta=4.0, consistent with the original paper's CelebA experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vanilla_vae import VanillaVAEEncoder, VanillaVAEDecoder


class BetaVAE(nn.Module):
    """
    Beta-VAE drop-in encoder for the ARL loop.

    Architecturally identical to VanillaVAE — only the loss weighting differs.
    beta=1 recovers the standard VAE; beta=4 is Higgins et al.'s default.

    forward(x)                  -> recon in [0,1]
    forward(x, return_aux=True) -> (recon, mu, logvar, z)
    """

    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, img_size=224, beta=4.0):
        super().__init__()
        self.encoder = VanillaVAEEncoder(in_channels, latent_dim)
        self.decoder = VanillaVAEDecoder(latent_dim, out_channels)
        self.latent_dim = latent_dim
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-4, 4))
        return mu + torch.randn_like(std) * std

    def forward(self, x, return_aux=False):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        if return_aux:
            return recon, mu, logvar, z
        return recon

    def get_latent(self, x):
        """Mean embedding without sampling — for evaluation/visualisation."""
        mu, _ = self.encoder(x)
        return mu


def beta_vae_loss(recon, x, mu, logvar, beta=4.0):
    """
    Beta-VAE loss: MSE recon + beta * KL.

    Higher beta = stronger disentanglement pressure, lower recon quality.
    beta=1 is equivalent to vanilla_vae_loss.
    """
    B = x.size(0)
    recon_loss = F.mse_loss(recon, x, reduction='sum') / B
    lv = logvar.clamp(-4, 4)
    mu_c = mu.clamp(-10, 10)
    kl_loss = -0.5 * torch.sum(1 + lv - mu_c.pow(2) - lv.exp()) / B
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

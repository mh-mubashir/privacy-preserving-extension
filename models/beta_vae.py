"""
Beta-VAE for Privacy-Preserving Edge Vision (Member 1).

Extends the standard VAE (Higgins et al., 2017, "beta-VAE: Learning Basic
Visual Concepts with a Constrained Variational Framework") by up-weighting
the KL divergence term with a factor beta > 1.

The higher beta encourages stronger disentanglement of latent factors at the
cost of reconstruction fidelity — useful for learning representations where
privacy-sensitive factors (e.g. gender) can be isolated and suppressed.

Architecture is identical to VanillaVAE; the only difference is beta > 1 in
the training loss, making it a controlled ablation of the KL penalty strength.

Output contract (same as all ARL encoders):
  forward(x)                  -> recon (B, 3, 224, 224) in [0, 1]
  forward(x, return_aux=True) -> (recon, mu, logvar, z)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vanilla_vae import VanillaVAEEncoder, VanillaVAEDecoder


# ---------------------------------------------------------------------------
# Beta-VAE (same architecture as VanillaVAE, configurable beta)
# ---------------------------------------------------------------------------

class BetaVAE(nn.Module):
    """
    Beta-VAE drop-in encoder for the ARL pipeline.

    Identical architecture to VanillaVAE; the beta > 1 multiplier on the KL
    term encourages disentangled representations by penalising factorial
    deviations from the prior more strongly.

    Typical values: beta=4 (Higgins et al.) or beta=6–10 for CelebA faces.

    ARL interface:
        forward(x)                  -> recon (B, 3, 224, 224) in [0, 1]
        forward(x, return_aux=True) -> (recon, mu, logvar, z)
    """

    def __init__(
        self,
        in_channels:  int   = 3,
        out_channels: int   = 3,
        latent_dim:   int   = 256,
        img_size:     int   = 224,
        beta:         float = 4.0,
    ):
        super().__init__()
        self.encoder    = VanillaVAEEncoder(in_channels, latent_dim)
        self.decoder    = VanillaVAEDecoder(latent_dim, out_channels)
        self.latent_dim = latent_dim
        self.beta       = beta          # stored for reference; loss uses vae_beta arg

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar.clamp(-4, 4))
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        if return_aux:
            return recon, mu, logvar, z
        return recon

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic (mean) forward for evaluation."""
        mu, _ = self.encoder(x)
        return self.decoder(mu)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def beta_vae_loss(recon, x, mu, logvar, beta=4.0):
    """
    Beta-VAE loss: MSE reconstruction + beta * KL divergence.

    With beta=1 this is identical to the standard VAE ELBO.
    With beta>1 (e.g. 4.0) it penalises the KL term more strongly,
    pushing the posterior toward a factorised prior and encouraging
    disentangled latent representations.

    Args:
        recon:   (B, C, H, W) reconstruction
        x:       (B, C, H, W) original
        mu:      (B, D) latent mean
        logvar:  (B, D) latent log-variance
        beta:    KL weight (default 4.0 following Higgins et al.)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    B          = x.size(0)
    recon_loss = F.mse_loss(recon, x, reduction='sum') / B
    lv         = logvar.clamp(-4, 4)
    mu_c       = mu.clamp(-10, 10)
    kl_loss    = -0.5 * torch.sum(1 + lv - mu_c.pow(2) - lv.exp()) / B
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

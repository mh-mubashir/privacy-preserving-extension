"""
Vanilla VAE for Privacy-Preserving Edge Vision (Member 1).

Standard Variational Autoencoder (Kingma & Welling, 2013).
Serves as the primary VAE baseline in the ARL framework.

Architecture:
  Encoder: 3->32->64->128->256->512 (stride-2 convs) -> FC to mu/logvar
  Decoder: FC -> 512*7*7 -> ConvTranspose2d back to 224x224
  Channel schedule matches CVAE for fair comparison.

Output contract (same as all ARL encoders):
  forward(x)                -> recon (B, 3, 224, 224) in [0, 1]
  forward(x, return_aux=True) -> (recon, mu, logvar, z)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (shared with CVAE style)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d -> BN -> ReLU (inplace=False for ARL retain_graph compatibility)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    """ConvTranspose2d -> BN -> ReLU (2x upsample)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VanillaVAEEncoder(nn.Module):
    """
    Encoder: (B, 3, 224, 224) -> mu (B, D), logvar (B, D).
    Channel schedule: 3->32->64->128->256->512, spatial 224->7.
    """

    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 32,  stride=2)   # 224 -> 112
        self.conv2 = ConvBlock(32,  64,  stride=2)            # 112 -> 56
        self.conv3 = ConvBlock(64,  128, stride=2)            # 56  -> 28
        self.conv4 = ConvBlock(128, 256, stride=2)            # 28  -> 14
        self.conv5 = ConvBlock(256, 512, stride=2)            # 14  -> 7

        self.fc_mu     = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x):
        B = x.size(0)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = h.view(B, -1)
        return self.fc_mu(h), self.fc_logvar(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class VanillaVAEDecoder(nn.Module):
    """
    Decoder: (B, D) -> (B, 3, 224, 224).
    Mirrors encoder: FC -> 7x7 -> ConvTranspose2d x5 -> 224x224.
    """

    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.fc       = nn.Linear(latent_dim, 512 * 7 * 7)
        self.deconv1  = DeconvBlock(512, 256)   # 7  -> 14
        self.deconv2  = DeconvBlock(256, 128)   # 14 -> 28
        self.deconv3  = DeconvBlock(128, 64)    # 28 -> 56
        self.deconv4  = DeconvBlock(64,  32)    # 56 -> 112
        self.deconv5  = nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1)  # 112 -> 224

    def forward(self, z):
        B = z.size(0)
        h = F.relu(self.fc(z), inplace=False).view(B, 512, 7, 7)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)
        h = self.deconv4(h)
        return torch.sigmoid(self.deconv5(h))


# ---------------------------------------------------------------------------
# Vanilla VAE
# ---------------------------------------------------------------------------

class VanillaVAE(nn.Module):
    """
    Vanilla VAE drop-in encoder for the ARL pipeline.

    Standard VAE (Kingma & Welling 2013) with no conditioning.
    Beta = 1 (standard ELBO). For stronger KL regularisation use BetaVAE.

    ARL interface:
        forward(x)                  -> recon (B, 3, 224, 224) in [0, 1]
        forward(x, return_aux=True) -> (recon, mu, logvar, z)
    """

    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, img_size=224):
        super().__init__()
        self.encoder    = VanillaVAEEncoder(in_channels, latent_dim)
        self.decoder    = VanillaVAEDecoder(latent_dim, out_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-4, 4))
        return mu + torch.randn_like(std) * std

    def forward(self, x, return_aux=False):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        if return_aux:
            return recon, mu, logvar, z
        return recon

    def encode_decode(self, x):
        """Deterministic (mean) forward for evaluation."""
        mu, _ = self.encoder(x)
        return self.decoder(mu)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def vanilla_vae_loss(recon, x, mu, logvar, beta=1.0):
    """
    Standard VAE loss: MSE reconstruction + beta * KL divergence.

    Args:
        recon:   (B, C, H, W) reconstruction
        x:       (B, C, H, W) original
        mu:      (B, D) latent mean
        logvar:  (B, D) latent log-variance
        beta:    KL weight (1.0 for standard VAE)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    B = x.size(0)
    recon_loss = F.mse_loss(recon, x, reduction='sum') / B
    lv         = logvar.clamp(-4, 4)
    mu_c       = mu.clamp(-10, 10)
    kl_loss    = -0.5 * torch.sum(1 + lv - mu_c.pow(2) - lv.exp()) / B
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

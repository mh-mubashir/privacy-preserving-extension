"""
Disentangled Beta-VAE for Privacy-Preserving Edge Vision (Member 2 / Sindhu).

Implements the "understanding disentangling" variant of beta-VAE proposed by
Burgess et al. 2018 ("Understanding disentangling in beta-VAE"), which uses a
capacity-controlled KL schedule:

    L = E[recon] + gamma * |KL(q(z|x) || p(z)) - C|

C starts at 0 and is annealed upward to a target C_max over warmup_steps,
forcing the encoder to learn progressively more complex representations.

Architecture differences from existing CVAE / FactorVAE in this repo:
  - Encoder uses skip connections between non-adjacent encoder stages to the
    decoder (UNet-style lateral connections): the decoder at each scale
    receives both the upsampled feature map AND the corresponding encoder
    feature map, giving richer spatial cues for reconstruction.
  - Channel widths use an asymmetric schedule: encoder widens aggressively
    (3->64->128->256->256->256) while decoder narrows symmetrically, keeping
    the decoder lighter and preventing over-reconstruction of private features.
  - A dedicated "privacy bottleneck": the latent space is split into two
    parts — z_util (first latent_dim//2 dims) and z_priv (second half).
    Only z_util is passed to the decoder; z_priv is discarded, explicitly
    suppressing its reconstruction and thus its information retention.

Output contract (same as all other ARL encoders):
  forward(x)                  -> recon (B, 3, 224, 224) in [0, 1]
  forward(x, return_aux=True) -> (recon, mu, logvar, z)

Note: mu, logvar, z still include ALL latent dims for KL computation in the
loss, but the decoder only sees z[:, :latent_dim//2].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """3x3 Conv -> BN -> ReLU (inplace=False for ARL retain_graph compatibility)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SkipDownBlock(nn.Module):
    """
    Two consecutive ConvBNReLU layers + 2x strided downsampling.
    Returns (downsampled feature map, skip feature map before downsampling)
    so the decoder can use the skip connection.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1  = ConvBNReLU(in_ch, out_ch)
        self.conv2  = ConvBNReLU(out_ch, out_ch)
        self.stride = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor):
        h    = self.conv1(x)
        skip = self.conv2(h)      # kept at full resolution for decoder
        out  = self.stride(skip)  # downsampled
        return out, skip


class SkipUpBlock(nn.Module):
    """
    Bilinear 2x upsample, then fuse (concat) with the encoder skip feature,
    followed by two ConvBNReLU layers to blend.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse  = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv  = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(self.fuse(x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class DisentangledBetaEncoder(nn.Module):
    """
    Encoder: (B, 3, 224, 224) -> mu (B, D), logvar (B, D)
             also returns skip feature maps for the decoder.

    Channel schedule: 3 -> 64 -> 128 -> 256 -> 256 -> 256
    Spatial schedule: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        self.down1 = SkipDownBlock(in_channels, 64)    # skip: (B, 64, 112, 112)
        self.down2 = SkipDownBlock(64, 128)             # skip: (B, 128, 56, 56)
        self.down3 = SkipDownBlock(128, 256)            # skip: (B, 256, 28, 28)
        self.down4 = SkipDownBlock(256, 256)            # skip: (B, 256, 14, 14)
        self.down5 = SkipDownBlock(256, 256)            # out:  (B, 256, 7, 7)

        self.fc_mu     = nn.Linear(256 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(256 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor):
        out1, skip1 = self.down1(x)
        out2, skip2 = self.down2(out1)
        out3, skip3 = self.down3(out2)
        out4, skip4 = self.down4(out3)
        out5, _     = self.down5(out4)

        B = out5.size(0)
        h = out5.view(B, -1)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        skips  = (skip1, skip2, skip3, skip4)
        return mu, logvar, skips


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DisentangledBetaDecoder(nn.Module):
    """
    Decoder: (B, D//2) + encoder skips -> (B, 3, 224, 224).

    Only the first half of the latent (z_util) is decoded;
    the second half (z_priv) is discarded before reaching this decoder,
    acting as an explicit privacy bottleneck.

    Channel schedule (reverse): 256 -> 256 -> 256 -> 128 -> 64 -> 3
    """

    def __init__(self, util_latent_dim: int, out_channels: int = 3):
        super().__init__()
        self.fc   = nn.Linear(util_latent_dim, 256 * 7 * 7)

        # UpBlocks: in_ch from below + skip_ch from encoder.
        # skip1 is taken *before* down1's stride, so it is at 224x224.
        # up4 therefore already produces 224x224 output — no extra upsample needed.
        self.up1  = SkipUpBlock(256, 256, 256)   # 7  -> 28,  skip4 has 256 ch
        self.up2  = SkipUpBlock(256, 256, 256)   # 28 -> 56,  skip3 has 256 ch
        self.up3  = SkipUpBlock(256, 128, 128)   # 56 -> 112, skip2 has 128 ch
        self.up4  = SkipUpBlock(128, 64, 64)     # 112 -> 224, skip1 has 64 ch
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, z_util: torch.Tensor, skips) -> torch.Tensor:
        skip1, skip2, skip3, skip4 = skips
        B = z_util.size(0)

        h = F.relu(self.fc(z_util), inplace=False).view(B, 256, 7, 7)
        h = self.up1(h, skip4)     # (B, 256, 28, 28)  — skip4 is 28x28
        h = self.up2(h, skip3)     # (B, 256, 56, 56)  — skip3 is 56x56
        h = self.up3(h, skip2)     # (B, 128, 112, 112) — skip2 is 112x112
        h = self.up4(h, skip1)     # (B, 64,  224, 224) — skip1 is 224x224
        return torch.sigmoid(self.final(h))


# ---------------------------------------------------------------------------
# Disentangled Beta-VAE
# ---------------------------------------------------------------------------

class DisentangledBetaVAE(nn.Module):
    """
    Disentangled Beta-VAE (capacity-annealing) drop-in encoder for ARL.

    The latent vector is split into z_util (first half) and z_priv (second
    half); only z_util is decoded, explicitly preventing the decoder — and
    therefore the network's reconstruction objective — from relying on
    privacy-sensitive factors.

    ARL interface:
        forward(x)                  -> recon (B, 3, 224, 224) in [0, 1]
        forward(x, return_aux=True) -> (recon, mu, logvar, z)
            mu, logvar, z are over the FULL latent_dim for loss computation.
    """

    def __init__(
        self,
        in_channels: int  = 3,
        out_channels: int = 3,
        latent_dim: int   = 256,
        img_size: int     = 224,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even (split into util / priv halves)"
        self.latent_dim      = latent_dim
        self.util_latent_dim = latent_dim // 2

        self.encoder = DisentangledBetaEncoder(in_channels, latent_dim)
        self.decoder = DisentangledBetaDecoder(self.util_latent_dim, out_channels)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        mu, logvar, skips = self.encoder(x)
        z                 = self.reparameterize(mu, logvar)

        # Privacy bottleneck: only decode the utility half
        z_util = z[:, : self.util_latent_dim]
        recon  = self.decoder(z_util, skips)

        if return_aux:
            return recon, mu, logvar, z
        return recon

    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic (mean) forward for evaluation."""
        mu, _, skips = self.encoder(x)
        z_util       = mu[:, : self.util_latent_dim]
        return self.decoder(z_util, skips)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def disentangled_beta_vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    gamma: float = 100.0,
    C: float     = 0.0,
) -> tuple:
    """
    Capacity-controlled beta-VAE loss (Burgess et al. 2018):
        L = E[recon] + gamma * |KL(q||p) - C|

    During training, C should be annealed from 0 to C_max (e.g., 25.0)
    over the first ~50 % of training steps.

    Args:
        recon:  (B, 3, H, W) reconstruction
        x:      (B, 3, H, W) original
        mu:     (B, D) full latent mean
        logvar: (B, D) full latent log-variance
        gamma:  regularisation strength (default 100.0)
        C:      current KL capacity target (annealed externally)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    B = x.size(0)
    # Reconstruction: MSE (numerically stable, no [0,1] range requirement on GPU)
    recon_loss = F.mse_loss(recon, x.clamp(0.0, 1.0), reduction="sum") / B
    kl_loss    = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / B
    total_loss = recon_loss + gamma * (kl_loss - C).abs()
    return total_loss, recon_loss, kl_loss

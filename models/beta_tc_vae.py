"""
Beta-TC VAE for Privacy-Preserving Edge Vision (Member 2 / Sindhu).

Extends the standard beta-VAE by decomposing the KL term and explicitly
penalising the Total Correlation (TC) component, following Chen et al. 2018
"Isolating Sources of Disentanglement in VAEs" (beta-TC VAE).

Architecture differences from the existing CVAE / FactorVAE in this repo:
  - Encoder uses residual blocks (two conv layers + identity shortcut) instead
    of plain ConvBlocks, giving the network capacity to learn richer features
    at each spatial scale without vanishing gradients.
  - Channel widths are wider (64-128-256-512-512) vs the 32-64-128-256-512
    progression used in cvae.py, increasing representational capacity.
  - Decoder mirrors the encoder width schedule and uses bilinear upsampling
    followed by a 3x3 conv (instead of a transposed conv) to avoid the
    checkerboard artefacts common in ConvTranspose2d decoders.
  - TC penalty is estimated analytically via a minibatch-weighted approach
    (no extra discriminator network required), keeping the model simpler and
    faster than FactorVAE while still penalising factorial deviation.

Output contract (same as all other ARL encoders):
  forward(x) -> (B, 3, 224, 224) in [0, 1]
  forward(x, return_aux=True) -> (recon, mu, logvar, z)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual block: two 3x3 convolutions with a learned shortcut.
    Preserves spatial dimensions (no stride); strided downsampling is done
    separately via a 2x2 average-pool step after the block.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=False)


class DownBlock(nn.Module):
    """
    Strided 2x convolution for downsampling followed by a residual refinement
    block (when channel width stays the same) or a plain conv otherwise.
    in_ch -> out_ch, spatial /2.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )
        # Residual refinement block only when channels match (avoids orphaned params)
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """
    Bilinear 2x upsample followed by 3x3 conv + BN + ReLU.
    Avoids checkerboard artefacts from ConvTranspose2d.
    out_ch, spatial *2.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class BetaTCEncoder(nn.Module):
    """
    Encoder: (B, 3, 224, 224) -> mu (B, D), logvar (B, D).

    Five DownBlocks progressively halve spatial resolution.
    Channel schedule: 3 -> 64 -> 128 -> 256 -> 512 -> 512
    Output spatial size: 224 / 2^5 = 7.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        self.down1 = DownBlock(in_channels, 64)   # -> (B, 64, 112, 112)
        self.down2 = DownBlock(64, 128)            # -> (B, 128, 56, 56)
        self.down3 = DownBlock(128, 256)           # -> (B, 256, 28, 28)
        self.down4 = DownBlock(256, 512)           # -> (B, 512, 14, 14)
        self.down5 = DownBlock(512, 512)           # -> (B, 512, 7, 7)

        self.fc_mu     = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        h = self.down1(x)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.down5(h)
        h = h.view(B, -1)
        return self.fc_mu(h), self.fc_logvar(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class BetaTCDecoder(nn.Module):
    """
    Decoder: (B, D) -> (B, 3, 224, 224).

    Mirrors encoder: FC projects latent to 7x7 feature map, then five
    UpBlocks double spatial resolution back to 224x224.
    Channel schedule (reverse): 512 -> 512 -> 256 -> 128 -> 64 -> 3
    """

    def __init__(self, latent_dim: int = 256, out_channels: int = 3):
        super().__init__()
        self.fc    = nn.Linear(latent_dim, 512 * 7 * 7)
        self.up1   = UpBlock(512, 512)   # -> (B, 512, 14, 14)
        self.up2   = UpBlock(512, 256)   # -> (B, 256, 28, 28)
        self.up3   = UpBlock(256, 128)   # -> (B, 128, 56, 56)
        self.up4   = UpBlock(128, 64)    # -> (B, 64, 112, 112)
        self.up5   = UpBlock(64, 64)     # -> (B, 64, 224, 224)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = F.relu(self.fc(z), inplace=False).view(B, 512, 7, 7)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        return torch.sigmoid(self.final(h))


# ---------------------------------------------------------------------------
# Beta-TC VAE
# ---------------------------------------------------------------------------

class BetaTCVAE(nn.Module):
    """
    Beta-TC VAE drop-in encoder for the ARL pipeline.

    Penalises the Total Correlation component of the KL term (alpha * MI +
    beta * TC + gamma * dim-wise KL decomposition), following Chen et al. 2018.

    The TC is estimated analytically using the minibatch-stratified sampling
    trick (Eq. 2 in the paper): no extra discriminator network needed.

    ARL interface:
        forward(x)                -> recon (B, 3, 224, 224) in [0, 1]
        forward(x, return_aux=True) -> (recon, mu, logvar, z)
    """

    def __init__(
        self,
        in_channels: int  = 3,
        out_channels: int = 3,
        latent_dim: int   = 256,
        img_size: int     = 224,
    ):
        super().__init__()
        self.encoder    = BetaTCEncoder(in_channels, latent_dim)
        self.decoder    = BetaTCDecoder(latent_dim, out_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
# Loss functions
# ---------------------------------------------------------------------------

def _log_density_gaussian(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Log-density of a diagonal Gaussian: log q(z|x).
    z, mu, logvar: (B, D)
    Returns: (B, D)
    """
    norm   = -0.5 * math.log(2 * math.pi)
    inv_var = torch.exp(-logvar)
    return norm - 0.5 * logvar - 0.5 * (z - mu) ** 2 * inv_var


def beta_tc_vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    alpha: float = 1.0,
    beta: float  = 4.0,
    gamma: float = 1.0,
    dataset_size: int = 162770,
) -> tuple:
    """
    Beta-TC VAE loss decomposition (Chen et al. 2018):
        L = E[recon] + alpha * MI + beta * TC + gamma * dim-KL

    Args:
        recon:        (B, 3, H, W) reconstruction
        x:            (B, 3, H, W) original
        mu:           (B, D) encoder mean
        logvar:       (B, D) encoder log-variance
        z:            (B, D) sampled latent
        alpha:        mutual information weight (default 1.0)
        beta:         total correlation weight  (default 4.0, higher -> stronger disentanglement)
        gamma:        dim-wise KL weight         (default 1.0)
        dataset_size: total training samples for minibatch TC estimate

    Returns:
        total_loss, recon_loss, mi_loss, tc_loss, dw_kl_loss
    """
    B, D = z.size()

    # Reconstruction: MSE (numerically stable, no [0,1] range requirement on GPU)
    recon_loss = F.mse_loss(recon, x.clamp(0.0, 1.0), reduction="sum") / B

    # log q(z|x):  (B, D) -> (B,)
    log_q_zx = _log_density_gaussian(z, mu, logvar).sum(dim=1)

    # log p(z): standard normal (B, D) -> (B,)
    zeros = torch.zeros_like(z)
    log_p_z = _log_density_gaussian(z, zeros, zeros).sum(dim=1)

    # Minibatch estimate of log q(z): broadcast z (B,1,D) against mu/logvar (1,B,D)
    z_expand   = z.unsqueeze(1)         # (B, 1, D)
    mu_expand  = mu.unsqueeze(0)        # (1, B, D)
    lv_expand  = logvar.unsqueeze(0)    # (1, B, D)

    # (B, B, D)
    log_q_z_matrix = _log_density_gaussian(z_expand, mu_expand, lv_expand)
    # Stratified importance-weighted estimate: log (1 / (B * N)) + logsumexp over batch dim
    log_prod_q_z = (
        torch.logsumexp(log_q_z_matrix.sum(dim=2), dim=1)
        - math.log(B * dataset_size)
    )                                   # (B,)
    log_q_z_product = (
        torch.logsumexp(log_q_z_matrix, dim=1)
        - math.log(B * dataset_size)
    ).sum(dim=1)                        # (B,)

    # Mutual information: E[log q(z|x) - log q(z)]
    mi_loss    = (log_q_zx - log_prod_q_z).mean()
    # Total correlation: E[log q(z) - log prod q(z_i)]
    tc_loss    = (log_prod_q_z - log_q_z_product).mean()
    # Dim-wise KL: E[log prod q(z_i) - log p(z)]
    dw_kl_loss = (log_q_z_product - log_p_z).mean()

    total_loss = recon_loss + alpha * mi_loss + beta * tc_loss + gamma * dw_kl_loss
    return total_loss, recon_loss, mi_loss, tc_loss, dw_kl_loss

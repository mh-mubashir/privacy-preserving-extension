"""
ResidualVAE for the ARL pipeline.

Same channel schedule and latent dim as VanillaVAE for a fair comparison.
The encoder replaces each plain ConvBlock with a ResBlock — two convolutions
with a skip connection. This gives richer gradient flow and better feature reuse.

Decoder is unchanged from VanillaVAE, keeping the comparison clean:
  VanillaVAE  -> feedforward encoder, beta=1
  BetaVAE     -> feedforward encoder, beta=4
  ResidualVAE -> residual encoder,    beta=1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vanilla_vae import VanillaVAEDecoder


class ResBlock(nn.Module):
    """
    Two-layer residual block with optional projection on the skip path.
    Stride-2 downsampling is done in the first conv when stride > 1.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ResidualVAEEncoder(nn.Module):
    # 224 -> 112 -> 56 -> 28 -> 14 -> 7  (same spatial schedule as VanillaVAE)
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.block1 = ResBlock(in_channels, 32,  stride=2)
        self.block2 = ResBlock(32,  64,  stride=2)
        self.block3 = ResBlock(64,  128, stride=2)
        self.block4 = ResBlock(128, 256, stride=2)
        self.block5 = ResBlock(256, 512, stride=2)
        self.fc_mu     = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.block5(self.block4(self.block3(self.block2(self.block1(x)))))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class ResidualVAE(nn.Module):
    """
    Drop-in residual VAE encoder for the ARL pipeline.

    Each encoder stage uses a two-layer ResBlock with a skip connection.
    Decoder is identical to VanillaVAE for a clean architectural comparison.

    forward(x)                  -> recon in [0, 1]
    forward(x, return_aux=True) -> (recon, mu, logvar, z)
    """

    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, img_size=224):
        super().__init__()
        self.encoder = ResidualVAEEncoder(in_channels, latent_dim)
        self.decoder = VanillaVAEDecoder(latent_dim, out_channels)
        self.latent_dim = latent_dim

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
        mu, _ = self.encoder(x)
        return mu

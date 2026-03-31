"""
Vanilla VAE implementation for the ARL pipeline (Member 1).

Based on Kingma & Welling (2013). Encoder maps a 224x224 image down to
a latent mean and log-variance; decoder maps back up. Reconstruction
loss is MSE, KL weight is 1 (standard ELBO).

Used as the simplest VAE baseline — no conditioning, no extra penalties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class VanillaVAEEncoder(nn.Module):
    # 224 -> 112 -> 56 -> 28 -> 14 -> 7, then flatten to latent
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 32,  stride=2)
        self.conv2 = ConvBlock(32,  64,  stride=2)
        self.conv3 = ConvBlock(64,  128, stride=2)
        self.conv4 = ConvBlock(128, 256, stride=2)
        self.conv5 = ConvBlock(256, 512, stride=2)
        self.fc_mu     = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class VanillaVAEDecoder(nn.Module):
    # 7 -> 14 -> 28 -> 56 -> 112 -> 224
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.fc      = nn.Linear(latent_dim, 512 * 7 * 7)
        self.deconv1 = DeconvBlock(512, 256)
        self.deconv2 = DeconvBlock(256, 128)
        self.deconv3 = DeconvBlock(128, 64)
        self.deconv4 = DeconvBlock(64,  32)
        self.deconv5 = nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.fc(z)).view(-1, 512, 7, 7)
        h = self.deconv4(self.deconv3(self.deconv2(self.deconv1(h))))
        return torch.sigmoid(self.deconv5(h))


class VanillaVAE(nn.Module):
    """
    Drop-in VAE encoder for the ARL training loop.
    Satisfies the same interface as UNet, CVAE, FactorVAE etc.

    forward(x)                  -> recon in [0,1]
    forward(x, return_aux=True) -> (recon, mu, logvar, z)
    """

    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, img_size=224):
        super().__init__()
        self.encoder = VanillaVAEEncoder(in_channels, latent_dim)
        self.decoder = VanillaVAEDecoder(latent_dim, out_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        # clamp logvar to keep std in a reasonable range during early training
        std = torch.exp(0.5 * logvar.clamp(-4, 4))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_aux=False):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        if return_aux:
            return recon, mu, logvar, z
        return recon

    def get_latent(self, x):
        """Return mean embedding (no sampling) — useful for visualisation."""
        mu, _ = self.encoder(x)
        return mu


def vanilla_vae_loss(recon, x, mu, logvar, beta=1.0):
    """MSE reconstruction + KL. Returns (total, recon_loss, kl_loss)."""
    B = x.size(0)
    recon_loss = F.mse_loss(recon, x, reduction='sum') / B
    lv = logvar.clamp(-4, 4)
    mu_c = mu.clamp(-10, 10)
    kl_loss = -0.5 * torch.sum(1 + lv - mu_c.pow(2) - lv.exp()) / B
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

"""
Factor VAE for Privacy-Preserving Edge Vision.

Encourages disentanglement through a total correlation penalty term via a
discriminator on permuted latents. Serves as a drop-in encoder replacement
in the ARL framework. Outputs (B, 3, 224, 224) images in [0, 1] for classifiers.

Reference: Kim & Mnih, "Disentangling by Factorising" (Factor VAE), ICML 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def permute_dims(z):
    """
    Permute each dimension of z across the batch to sample from product of marginals.
    Used for Factor VAE total correlation estimation.
    """
    B, D = z.size()
    z_perm = z
    for d in range(D):
        perm = torch.randperm(B, device=z.device)
        z_perm = z_perm.clone()
        z_perm[:, d] = z[perm, d]
    return z_perm


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """ConvTranspose2d -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))


class FactorVAEEncoder(nn.Module):
    """Encoder: x -> mu, logvar. No conditioning."""

    def __init__(self, in_channels=3, latent_dim=256, img_size=224):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.conv1 = ConvBlock(in_channels, 32, stride=2, padding=1)    # 224 -> 112
        self.conv2 = ConvBlock(32, 64, stride=2, padding=1)               # 112 -> 56
        self.conv3 = ConvBlock(64, 128, stride=2, padding=1)              # 56 -> 28
        self.conv4 = ConvBlock(128, 256, stride=2, padding=1)            # 28 -> 14
        self.conv5 = ConvBlock(256, 512, stride=2, padding=1)            # 14 -> 7

        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x):
        B = x.size(0)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = h.view(B, -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class FactorVAEDecoder(nn.Module):
    """Decoder: z -> x_recon"""

    def __init__(self, latent_dim=256, out_channels=3, img_size=224):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.deconv1 = DeconvBlock(512, 256)
        self.deconv2 = DeconvBlock(256, 128)
        self.deconv3 = DeconvBlock(128, 64)
        self.deconv4 = DeconvBlock(64, 32)
        self.deconv5 = nn.ConvTranspose2d(32, out_channels, 4, 2, 1)

    def forward(self, z):
        B = z.size(0)
        h = self.fc(z)
        h = h.view(B, 512, 7, 7)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)
        h = self.deconv4(h)
        h = self.deconv5(h)
        return torch.sigmoid(h)


class LatentDiscriminator(nn.Module):
    """
    Discriminator to estimate total correlation.
    Input: z (B, latent_dim). Output: logit for "z from encoder" vs "z from product of marginals".
    """

    def __init__(self, latent_dim=256, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class FactorVAE(nn.Module):
    """
    Factor VAE for ARL encoder.

    Forward returns reconstructed image (B, 3, 224, 224) in [0, 1].
    Also returns mu, logvar, z for loss computation.
    Uses a discriminator for total correlation penalty.

    Interface compatible with ARL: call forward(x) to get encoded image.
    """

    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, img_size=224):
        super().__init__()
        self.encoder = FactorVAEEncoder(in_channels, latent_dim, img_size)
        self.decoder = FactorVAEDecoder(latent_dim, out_channels, img_size)
        self.discriminator = LatentDiscriminator(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)
        return mu + eps * std

    def forward(self, x, return_aux=False):
        """
        Args:
            x: (B, 3, 224, 224) input images
            return_aux: if True, also return mu, logvar, z for loss computation

        Returns:
            recon: (B, 3, 224, 224) reconstructed image in [0, 1]
            If return_aux: (recon, mu, logvar, z)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        if return_aux:
            return recon, mu, logvar, z
        return recon

    def encode_decode(self, x):
        """Deterministic forward (use mean, no sampling). For eval."""
        mu, logvar = self.encoder(x)
        recon = self.decoder(mu)
        return recon


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """VAE loss: reconstruction (BCE) + beta * KL."""
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def discriminator_loss(z, z_perm, discriminator):
    """
    Discriminator loss: classify z (label 1) vs z_perm (label 0).
    """
    d_z = discriminator(z)
    d_z_perm = discriminator(z_perm)
    loss = 0.5 * (F.binary_cross_entropy_with_logits(d_z, torch.ones_like(d_z, device=z.device)) +
                  F.binary_cross_entropy_with_logits(d_z_perm, torch.zeros_like(d_z_perm, device=z.device)))
    return loss


def factor_vae_tc_loss(z, z_perm, discriminator, gamma=10.0):
    """
    Total correlation term for encoder: encourage z to be indistinguishable from z_perm.
    Encoder minimizes TC by producing z that the discriminator classifies as "from product of marginals"
    (i.e., we want D(z) small). We add gamma * E[log(D(z))] so minimizing pushes D(z) -> 0.
    """
    d_z = discriminator(z)
    tc_loss = gamma * F.logsigmoid(d_z).mean()
    return tc_loss


def factor_vae_encoder_loss(recon, x, mu, logvar, z, z_perm, discriminator, beta=1.0, gamma=10.0):
    """
    Full Factor VAE encoder loss: recon + beta*KL + gamma*TC_term.
    """
    vae_l, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta)
    tc_l = factor_vae_tc_loss(z, z_perm, discriminator, gamma)
    return vae_l + tc_l, recon_l, kl_l, tc_l

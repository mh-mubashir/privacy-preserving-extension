"""
Conditional VAE (CVAE) for Privacy-Preserving Edge Vision.

Conditions the latent space on the utility attribute (e.g., smile) to enable
more controlled representation learning. Serves as a drop-in encoder replacement
in the ARL framework. Outputs (B, 3, 224, 224) images in [0, 1] for classifiers.

Reference: CVAE - conditioning on auxiliary labels for controlled representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)  # required when using retain_graph + second backward

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """ConvTranspose2d -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)  # required when using retain_graph + second backward

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))


class CVAEEncoder(nn.Module):
    """Encoder: (x, y) -> mu, logvar. Conditions on binary label y (e.g., smile)."""

    def __init__(self, in_channels=3, latent_dim=256, img_size=224, num_conditions=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_conditions = num_conditions

        # Condition embedding: (B,) -> (B, embed_dim) for broadcast
        self.cond_embed = nn.Embedding(num_conditions + 1, 64)  # +1 for safety, use 2 for binary

        # Encoder: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        # Input: x (B,3,224,224) + cond channel (B,1,224,224) -> (B,4,224,224)
        self.conv1 = ConvBlock(4, 32, stride=2, padding=1)    # 224 -> 112
        self.conv2 = ConvBlock(32, 64, stride=2, padding=1)    # 112 -> 56
        self.conv3 = ConvBlock(64, 128, stride=2, padding=1)   # 56 -> 28
        self.conv4 = ConvBlock(128, 256, stride=2, padding=1)   # 28 -> 14
        self.conv5 = ConvBlock(256, 512, stride=2, padding=1)  # 14 -> 7

        # 7*7*512 = 25088
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x, y_cond):
        """
        Args:
            x: (B, 3, 224, 224) input images
            y_cond: (B,) or (B, 1) binary condition (utility label, e.g., smile).
                    Values in {0, 1}. For continuous, clamp/round to 0 or 1.
        """
        B = x.size(0)
        if y_cond.dim() == 1:
            y_cond = y_cond.long().clamp(0, 1)
        else:
            y_cond = (y_cond > 0.5).long().squeeze(-1).clamp(0, 1)

        # Embed and broadcast: (B, 64) -> (B, 64, 1, 1) -> (B, 1, 224, 224)
        cond_emb = self.cond_embed(y_cond)  # (B, 64)
        cond_spatial = cond_emb.view(B, 64, 1, 1).expand(B, 64, self.img_size, self.img_size)
        # Use first channel only for simplicity: (B, 1, 224, 224)
        cond_ch = cond_emb[:, :1].view(B, 1, 1, 1).expand(B, 1, self.img_size, self.img_size)
        x_cond = torch.cat([x, cond_ch], dim=1)  # (B, 4, 224, 224)

        h = self.conv1(x_cond)   # (B, 32, 112, 112)
        h = self.conv2(h)       # (B, 64, 56, 56)
        h = self.conv3(h)       # (B, 128, 28, 28)
        h = self.conv4(h)       # (B, 256, 14, 14)
        h = self.conv5(h)       # (B, 512, 7, 7)

        h = h.view(B, -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class CVAEDecoder(nn.Module):
    """Decoder: (z, y) -> x_recon"""

    def __init__(self, latent_dim=256, out_channels=3, img_size=224, num_conditions=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.cond_embed = nn.Embedding(num_conditions + 1, 64)

        # z: (B, latent_dim), cond: (B, 64) -> concat (B, latent_dim+64)
        self.fc = nn.Linear(latent_dim + 64, 512 * 7 * 7)

        self.deconv1 = DeconvBlock(512, 256)   # 7 -> 14
        self.deconv2 = DeconvBlock(256, 128)   # 14 -> 28
        self.deconv3 = DeconvBlock(128, 64)    # 28 -> 56
        self.deconv4 = DeconvBlock(64, 32)     # 56 -> 112
        self.deconv5 = nn.ConvTranspose2d(32, out_channels, 4, 2, 1)  # 112 -> 224

    def forward(self, z, y_cond):
        B = z.size(0)
        if y_cond.dim() == 1:
            y_cond = y_cond.long().clamp(0, 1)
        else:
            y_cond = (y_cond > 0.5).long().squeeze(-1).clamp(0, 1)

        cond_emb = self.cond_embed(y_cond)  # (B, 64)
        h = torch.cat([z, cond_emb], dim=1)
        h = self.fc(h)
        h = h.view(B, 512, 7, 7)

        h = self.deconv1(h)   # (B, 256, 14, 14)
        h = self.deconv2(h)   # (B, 128, 28, 28)
        h = self.deconv3(h)   # (B, 64, 56, 56)
        h = self.deconv4(h)   # (B, 32, 112, 112)
        h = self.deconv5(h)   # (B, 3, 224, 224)
        return torch.sigmoid(h)


class CVAE(nn.Module):
    """
    Conditional VAE for ARL encoder.

    Forward returns reconstructed image (B, 3, 224, 224) in [0, 1].
    Also returns mu, logvar, z for loss computation.

    Interface compatible with ARL: call forward(x, y_util) to get encoded image
    for utility classifier and adversary.
    """

    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, img_size=224):
        super().__init__()
        self.encoder = CVAEEncoder(in_channels, latent_dim, img_size)
        self.decoder = CVAEDecoder(latent_dim, out_channels, img_size)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)
        return mu + eps * std

    def forward(self, x, y_cond, return_aux=False):
        """
        Args:
            x: (B, 3, 224, 224) input images
            y_cond: (B,) utility attribute (e.g., smile 0/1) for conditioning
            return_aux: if True, also return mu, logvar, z for loss computation

        Returns:
            recon: (B, 3, 224, 224) reconstructed image in [0, 1]
            If return_aux: (recon, mu, logvar, z)
        """
        mu, logvar = self.encoder(x, y_cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, y_cond)

        if return_aux:
            return recon, mu, logvar, z
        return recon

    def encode_decode(self, x, y_cond):
        """Deterministic forward (use mean, no sampling). For eval."""
        mu, logvar = self.encoder(x, y_cond)
        recon = self.decoder(mu, y_cond)
        return recon


def cvae_loss(recon, x, mu, logvar, beta=1.0):
    """VAE loss: reconstruction (BCE) + beta * KL."""
    # Clamp recon to avoid numeric issues in BCE on GPU (input must be in [0, 1])
    recon_clamped = recon.clamp(1e-6, 1 - 1e-6)
    recon_loss = F.binary_cross_entropy(recon_clamped, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

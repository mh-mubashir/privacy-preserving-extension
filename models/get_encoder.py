"""
Encoder factory for ARL (Adversarial Representation Learning).

Provides a unified interface to instantiate encoder architectures as drop-in
replacements in the privacy-preserving vision pipeline. All encoders produce
(B, 3, 224, 224) output in [0, 1] for downstream ResNet classifiers.

Encoders:
- unet: Baseline UNet (deterministic)
- cvae: Conditional VAE (conditions on utility attribute)
- factor_vae: Factor VAE (disentanglement via total correlation)
"""

import torch.nn as nn

from .unet import UNet
from .cvae import CVAE
from .factor_vae import FactorVAE


def get_encoder(encoder_name, img_size=224, **kwargs):
    """
    Create an encoder model for the ARL pipeline.

    Args:
        encoder_name: One of 'unet', 'cvae', 'factor_vae'
        img_size: Spatial size (default 224)
        **kwargs: Additional arguments for the encoder (e.g., unet_size, latent_dim)

    Returns:
        nn.Module: Encoder that takes (B, 3, H, W) and returns (B, 3, H, W) in [0, 1].
                  CVAE and Factor VAE have additional forward signatures for loss computation.
    """
    encoder_name = encoder_name.lower()

    if encoder_name == "unet":
        size = kwargs.get("unet_size", "tiny")
        return UNet(3, 3, size=size)

    elif encoder_name == "cvae":
        latent_dim = kwargs.get("latent_dim", 256)
        return CVAE(
            in_channels=3,
            out_channels=3,
            latent_dim=latent_dim,
            img_size=img_size,
        )

    elif encoder_name == "factor_vae":
        latent_dim = kwargs.get("latent_dim", 256)
        return FactorVAE(
            in_channels=3,
            out_channels=3,
            latent_dim=latent_dim,
            img_size=img_size,
        )

    else:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. "
            f"Available: unet, cvae, factor_vae"
        )

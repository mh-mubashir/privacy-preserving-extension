"""
Encoder factory for ARL (Adversarial Representation Learning).

All encoders produce (B, 3, 224, 224) output in [0, 1] for downstream
ResNet classifiers. VAE-family encoders also support return_aux=True
to return (recon, mu, logvar, z).

Encoders:
- unet:                  Baseline UNet (deterministic)
- vanilla_vae:           Standard VAE — Kingma & Welling 2013, beta=1 ELBO
- beta_vae:              Beta-VAE — Higgins et al. 2017, beta>1 disentanglement
- residual_vae:          VAE with residual encoder blocks
- cvae:                  Conditional VAE — conditions on utility label
- factor_vae:            Factor VAE — total correlation penalty via discriminator
- beta_tc_vae:           Beta-TC VAE — explicit total correlation penalty
- disentangled_beta_vae: Beta-VAE with skip connections and privacy bottleneck
- vq_vae:                VQ-VAE — discrete codebook latent space
"""

import torch.nn as nn

from .unet import UNet
from .vanilla_vae import VanillaVAE
from .beta_vae import BetaVAE
from .residual_vae import ResidualVAE
from .cvae import CVAE
from .factor_vae import FactorVAE
from .beta_tc_vae import BetaTCVAE
from .disentangled_beta_vae import DisentangledBetaVAE
from .vqvae_wrapper import VQVAEWrapper


def get_encoder(encoder_name, img_size=224, **kwargs):
    """
    Return an encoder model for the ARL pipeline.

    Args:
        encoder_name : one of unet / vanilla_vae / beta_vae / residual_vae /
                       cvae / factor_vae / beta_tc_vae / disentangled_beta_vae / vq_vae
        img_size     : spatial resolution (default 224)
        **kwargs     : latent_dim, beta, unet_size, etc.

    Returns:
        nn.Module with forward(x) -> (B,3,H,W) in [0,1]
    """
    name = encoder_name.lower()

    if name == "unet":
        return UNet(3, 3, size=kwargs.get("unet_size", "tiny"))

    elif name == "vanilla_vae":
        return VanillaVAE(
            in_channels=3, out_channels=3,
            latent_dim=kwargs.get("latent_dim", 256),
            img_size=img_size,
        )

    elif name == "beta_vae":
        return BetaVAE(
            in_channels=3, out_channels=3,
            latent_dim=kwargs.get("latent_dim", 256),
            img_size=img_size,
            beta=kwargs.get("beta", 4.0),
        )

    elif name == "residual_vae":
        return ResidualVAE(
            in_channels=3, out_channels=3,
            latent_dim=kwargs.get("latent_dim", 256),
            img_size=img_size,
        )

    elif name == "cvae":
        return CVAE(
            in_channels=3, out_channels=3,
            latent_dim=kwargs.get("latent_dim", 256),
            img_size=img_size,
        )

    elif name == "factor_vae":
        return FactorVAE(
            in_channels=3, out_channels=3,
            latent_dim=kwargs.get("latent_dim", 256),
            img_size=img_size,
        )

    elif name == "beta_tc_vae":
        return BetaTCVAE(
            in_channels=3, out_channels=3,
            latent_dim=kwargs.get("latent_dim", 256),
            img_size=img_size,
        )

    elif name == "disentangled_beta_vae":
        return DisentangledBetaVAE(
            in_channels=3, out_channels=3,
            latent_dim=kwargs.get("latent_dim", 256),
            img_size=img_size,
        )

    elif name == "vq_vae":
        return VQVAEWrapper(
            in_channels=3, out_channels=3,
            h_dim=kwargs.get("h_dim", 128),
            res_h_dim=kwargs.get("res_h_dim", 32),
            n_res_layers=kwargs.get("n_res_layers", 2),
            n_embeddings=kwargs.get("n_embeddings", 256),
            embedding_dim=kwargs.get("embedding_dim", 64),
            beta=kwargs.get("beta", 0.5),
            img_size=img_size,
        )

    else:
        raise ValueError(
            f"Unknown encoder '{encoder_name}'. "
            f"Choose from: unet, vanilla_vae, beta_vae, residual_vae, "
            f"cvae, factor_vae, beta_tc_vae, disentangled_beta_vae, vq_vae"
        )

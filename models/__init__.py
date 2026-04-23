"""
models — encoder architectures for the ARL privacy-preserving pipeline.

All encoders expose a common interface:
    forward(x)                     -> (B, 3, H, W) reconstruction in [0, 1]
    forward(x, return_aux=True)    -> (recon, mu, logvar, z)  [VAE variants]

Use get_encoder(name, img_size, **kwargs) as the single entry point.
"""

from .get_encoder import get_encoder
from .vanilla_vae import VanillaVAE
from .beta_vae import BetaVAE
from .residual_vae import ResidualVAE
from .cvae import CVAE
from .factor_vae import FactorVAE
from .beta_tc_vae import BetaTCVAE
from .disentangled_beta_vae import DisentangledBetaVAE
from .vqvae_wrapper import VQVAEWrapper

__all__ = [
    "get_encoder",
    "VanillaVAE",
    "BetaVAE",
    "ResidualVAE",
    "CVAE",
    "FactorVAE",
    "BetaTCVAE",
    "DisentangledBetaVAE",
    "VQVAEWrapper",
]

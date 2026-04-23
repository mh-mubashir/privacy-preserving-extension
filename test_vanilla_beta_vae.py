"""
test_vanilla_beta_vae.py -- Smoke tests for VanillaVAE and BetaVAE in the ARL pipeline.

Runs a full forward + backward pass on random CPU tensors.
No CelebA download required.

Checks:
  - recon shape matches (B, 3, 224, 224)
  - reconstruction values are in [0, 1]
  - mu / logvar shapes are (B, latent_dim)
  - ARL encoder loss is finite after a full forward + backward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.get_encoder import get_encoder
from models.cifar_like.resnet import ResNet18


def _build_arl_models(encoder_name: str):
    enc = get_encoder(encoder_name, img_size=224)
    clf = ResNet18()
    clf.linear = nn.Linear(512, 1)
    adv = ResNet18()
    adv.linear = nn.Linear(512, 1)
    return enc, clf, adv


def _run_arl_step(encoder_name: str) -> None:
    """One forward + backward pass through the ARL objective."""
    B = 4
    enc, clf, adv = _build_arl_models(encoder_name)
    enc.train(); clf.train(); adv.train()

    x = torch.rand(B, 3, 224, 224)
    y_util = torch.randint(0, 2, (B,)).float()
    y_priv = torch.randint(0, 2, (B,)).float()

    recon, mu, logvar, z = enc(x, return_aux=True)

    assert recon.shape == (B, 3, 224, 224), f"bad recon shape: {recon.shape}"
    assert mu.shape == (B, 256), f"bad mu shape: {mu.shape}"
    assert logvar.shape == mu.shape, "logvar shape mismatch"
    assert recon.min() >= 0.0 and recon.max() <= 1.0, (
        f"recon outside [0,1]: [{recon.min():.3f}, {recon.max():.3f}]"
    )

    crit = nn.BCEWithLogitsLoss()
    loss_clf = crit(clf(recon).flatten(), y_util)
    loss_adv = crit(adv(recon).flatten(), y_priv)

    lv = logvar.clamp(-4, 4)
    mu_c = mu.clamp(-10, 10)
    recon_loss = F.mse_loss(recon, x, reduction='sum') / B
    kl_loss = -0.5 * torch.sum(1 + lv - mu_c.pow(2) - lv.exp()) / B
    vae_loss = recon_loss + kl_loss

    enc_loss = (loss_clf - loss_adv) + 0.1 * vae_loss
    assert not torch.isnan(enc_loss), f"enc_loss is NaN for {encoder_name}"

    enc_loss.backward()

    print(f"  [{encoder_name}] enc_loss={enc_loss.item():.4f} "
          f"recon={recon_loss.item():.2f} kl={kl_loss.item():.2f}")


def test_vanilla_vae():
    print("Testing VanillaVAE...")
    _run_arl_step("vanilla_vae")
    print("  PASSED\n")


def test_beta_vae():
    print("Testing BetaVAE (beta=4.0)...")
    _run_arl_step("beta_vae")
    print("  PASSED\n")


def test_encoder_factory_all_vae_keys():
    """Verify get_encoder instantiates all VAE variants without error."""
    keys = ["vanilla_vae", "beta_vae", "residual_vae",
            "cvae", "factor_vae", "beta_tc_vae", "disentangled_beta_vae"]
    for key in keys:
        enc = get_encoder(key, img_size=64)
        assert enc is not None, f"get_encoder returned None for {key}"
    print(f"  encoder factory: all {len(keys)} keys instantiated OK")


if __name__ == "__main__":
    test_vanilla_vae()
    test_beta_vae()
    test_encoder_factory_all_vae_keys()
    print("All tests passed.")

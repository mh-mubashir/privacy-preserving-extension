"""
Quick validation of VanillaVAE and BetaVAE in the ARL pipeline.
No CelebA download required — runs on random tensors (CPU).

Mirrors test_factor_vae.py for the Member 1 encoder implementations.

Tests:
  - VanillaVAE: forward shapes, [0,1] output, backward pass
  - BetaVAE:    forward shapes, [0,1] output, backward pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.get_encoder import get_encoder
from models.cifar_like.resnet import ResNet18


def _arl_step(encoder_name):
    device = torch.device("cpu")
    B, C, H, W = 4, 3, 224, 224

    encoder = get_encoder(encoder_name, img_size=224).to(device)
    clf = ResNet18()
    clf.linear = nn.Linear(512, 1)
    clf = clf.to(device)
    adv = ResNet18()
    adv.linear = nn.Linear(512, 1)
    adv = adv.to(device)

    inputs     = torch.rand(B, C, H, W, device=device)
    targets_u  = torch.randint(0, 2, (B,), dtype=torch.float32, device=device)
    targets_adv = torch.randint(0, 2, (B,), dtype=torch.float32, device=device)

    encoder.train(); clf.train(); adv.train()

    recon, mu, logvar, z = encoder(inputs, return_aux=True)

    assert recon.shape == (B, C, H, W), f"{encoder_name} recon shape: {recon.shape}"
    assert mu.shape    == (B, 256),     f"{encoder_name} mu shape: {mu.shape}"
    assert logvar.shape == mu.shape,    f"{encoder_name} logvar shape mismatch"
    assert recon.min() >= 0.0 and recon.max() <= 1.0, (
        f"{encoder_name} recon out of [0,1]: min={recon.min():.3f} max={recon.max():.3f}"
    )

    criterion = nn.BCEWithLogitsLoss()
    loss_clf  = criterion(clf(recon).flatten(), targets_u)
    loss_adv  = criterion(adv(recon).flatten(), targets_adv)

    lv      = logvar.clamp(-4, 4)
    mu_c    = mu.clamp(-10, 10)
    recon_l = F.mse_loss(recon, inputs, reduction='sum') / B
    kl_l    = -0.5 * torch.sum(1 + lv - mu_c.pow(2) - lv.exp()) / B
    vae_l   = recon_l + 1.0 * kl_l

    enc_loss = (loss_clf - loss_adv) + 0.1 * vae_l

    assert not torch.isnan(enc_loss), f"{encoder_name} enc_loss is NaN"
    enc_loss.backward()

    print(f"  enc_loss={enc_loss.item():.4f}  "
          f"recon_l={recon_l.item():.4f}  kl_l={kl_l.item():.4f}")


def test_vanilla_vae():
    print("=== VanillaVAE ===")
    _arl_step("vanilla_vae")
    print("  PASSED: forward, shape checks, NaN guard, backward all succeeded.\n")


def test_beta_vae():
    print("=== BetaVAE (beta=4.0) ===")
    _arl_step("beta_vae")
    print("  PASSED: forward, shape checks, NaN guard, backward all succeeded.\n")


if __name__ == "__main__":
    test_vanilla_vae()
    test_beta_vae()
    print("All Member 1 encoder tests PASSED.")

"""
Quick validation of Member 2 custom encoders in the ARL pipeline.
No CelebA download required — runs on random tensors (CPU).

Tests:
  - BetaTCVAE:          forward, shapes, backward
  - DisentangledBetaVAE: forward, shapes, backward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.get_encoder import get_encoder
from models.cifar_like.resnet import ResNet18


def _make_networks(encoder_name, device):
    encoder = get_encoder(encoder_name, img_size=224).to(device)
    clf = ResNet18()
    clf.linear = nn.Linear(512, 1)
    clf = clf.to(device)
    adv = ResNet18()
    adv.linear = nn.Linear(512, 1)
    adv = adv.to(device)
    return encoder, clf, adv


def _run_arl_step(encoder, clf, adv, encoder_name, device):
    B, C, H, W = 4, 3, 224, 224
    inputs = torch.rand(B, C, H, W, device=device)
    targets_u   = torch.randint(0, 2, (B,), dtype=torch.float32, device=device)
    targets_adv = torch.randint(0, 2, (B,), dtype=torch.float32, device=device)

    encoder.train()
    clf.train()
    adv.train()

    # Encoder forward with aux outputs
    recon, mu, logvar, z = encoder(inputs, return_aux=True)

    # Shape checks
    assert recon.shape == (B, C, H, W), f"{encoder_name} recon shape wrong: {recon.shape}"
    assert mu.shape[0] == B,            f"{encoder_name} mu batch dim wrong: {mu.shape}"
    assert logvar.shape == mu.shape,    f"{encoder_name} logvar shape mismatch"
    assert z.shape == mu.shape,         f"{encoder_name} z shape mismatch"

    # Value check — sigmoid output must be in [0,1]
    assert recon.min() >= 0.0 and recon.max() <= 1.0, (
        f"{encoder_name} recon out of [0,1]: min={recon.min():.4f} max={recon.max():.4f}"
    )

    # Classifier + adversary forward
    criterion = nn.BCEWithLogitsLoss()
    u_logits  = clf(recon).flatten()
    p_logits  = adv(recon).flatten()
    loss_clf  = criterion(u_logits, targets_u)
    loss_adv  = criterion(p_logits, targets_adv)

    # Stable VAE loss (same as ARL loop)
    lv      = logvar.clamp(-4, 4)
    mu_c    = mu.clamp(-10, 10)
    recon_l = F.mse_loss(recon, inputs, reduction='sum') / B
    kl_l    = -0.5 * torch.sum(1 + lv - mu_c.pow(2) - lv.exp()) / B
    vae_l   = recon_l + 1.0 * kl_l

    # ARL objective + VAE regulariser
    enc_loss = (loss_clf - loss_adv) + 0.1 * vae_l

    # NaN check before backward
    assert not torch.isnan(enc_loss), (
        f"{encoder_name} enc_loss is NaN  "
        f"(recon_l={recon_l:.4f} kl_l={kl_l:.4f} loss_clf={loss_clf:.4f})"
    )

    enc_loss.backward()
    print(f"  enc_loss={enc_loss.item():.4f}  recon_l={recon_l.item():.4f}  kl_l={kl_l.item():.4f}")


def test_beta_tc_vae():
    print("=== BetaTCVAE ===")
    device = torch.device("cpu")
    encoder, clf, adv = _make_networks("beta_tc_vae", device)
    _run_arl_step(encoder, clf, adv, "beta_tc_vae", device)
    print("  PASSED: forward, shape checks, NaN guard, backward all succeeded.\n")


def test_disentangled_beta_vae():
    print("=== DisentangledBetaVAE ===")
    device = torch.device("cpu")
    encoder, clf, adv = _make_networks("disentangled_beta_vae", device)
    _run_arl_step(encoder, clf, adv, "disentangled_beta_vae", device)
    print("  PASSED: forward, shape checks, NaN guard, backward all succeeded.\n")


if __name__ == "__main__":
    test_beta_tc_vae()
    test_disentangled_beta_vae()
    print("All Member 2 encoder tests PASSED.")

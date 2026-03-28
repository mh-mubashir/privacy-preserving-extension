"""
Smoke test for VanillaVAE and BetaVAE in the ARL pipeline.
Runs on random tensors on CPU so no CelebA download needed.

Checks:
- output shape matches input (B, 3, 224, 224)
- reconstruction values stay in [0, 1] (sigmoid output)
- latent mu/logvar shapes are correct
- enc_loss is not NaN after a full ARL forward+backward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.get_encoder import get_encoder
from models.cifar_like.resnet import ResNet18


def build_arl_models(encoder_name):
    enc = get_encoder(encoder_name, img_size=224)
    clf = ResNet18()
    clf.linear = nn.Linear(512, 1)
    adv = ResNet18()
    adv.linear = nn.Linear(512, 1)
    return enc, clf, adv


def run_arl_step(encoder_name):
    B = 4
    enc, clf, adv = build_arl_models(encoder_name)
    enc.train(); clf.train(); adv.train()

    x = torch.rand(B, 3, 224, 224)
    y_util = torch.randint(0, 2, (B,)).float()
    y_priv = torch.randint(0, 2, (B,)).float()

    recon, mu, logvar, z = enc(x, return_aux=True)

    assert recon.shape == (B, 3, 224, 224), f"bad recon shape: {recon.shape}"
    assert mu.shape == (B, 256), f"bad mu shape: {mu.shape}"
    assert logvar.shape == mu.shape
    assert recon.min() >= 0.0 and recon.max() <= 1.0, \
        f"recon out of range: [{recon.min():.3f}, {recon.max():.3f}]"

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
    run_arl_step("vanilla_vae")
    print("  OK\n")


def test_beta_vae():
    print("Testing BetaVAE (beta=4.0)...")
    run_arl_step("beta_vae")
    print("  OK\n")


if __name__ == "__main__":
    test_vanilla_vae()
    test_beta_vae()
    print("All tests passed.")

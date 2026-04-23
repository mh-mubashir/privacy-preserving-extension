"""
test_factor_vae.py -- Smoke test for FactorVAE in the ARL pipeline.

Runs a full forward + backward pass on random CPU tensors.
No CelebA download required.

Checks:
  - recon shape matches (B, 3, 224, 224)
  - latent z shape matches (B, latent_dim)
  - discriminator loss and encoder loss are finite
  - backward pass completes without error
"""

import torch
import torch.nn as nn

from models.get_encoder import get_encoder
from models.factor_vae import discriminator_loss, factor_vae_encoder_loss, permute_dims
from models.cifar_like.resnet import ResNet18


def test_factor_vae_shapes_and_backward():
    device = torch.device("cpu")
    B, C, H, W = 4, 3, 224, 224

    encoder = get_encoder("factor_vae", img_size=224).to(device)
    clf = ResNet18()
    clf.linear = nn.Linear(512, 1)
    clf = clf.to(device)
    adv = ResNet18()
    adv.linear = nn.Linear(512, 1)
    adv = adv.to(device)

    inputs = torch.rand(B, C, H, W, device=device)
    targets_u = torch.randint(0, 2, (B,), device=device, dtype=torch.float32)
    targets_adv = torch.randint(0, 2, (B,), device=device, dtype=torch.float32)

    encoder.train()
    recon, mu, logvar, z = encoder(inputs, return_aux=True)
    z_perm = permute_dims(z)
    disc = encoder.discriminator

    assert recon.shape == (B, 3, H, W), f"unexpected recon shape: {recon.shape}"
    assert z.shape == (B, 256), f"unexpected z shape: {z.shape}"
    assert recon.min() >= 0.0 and recon.max() <= 1.0, (
        f"recon outside [0,1]: [{recon.min():.3f}, {recon.max():.3f}]"
    )

    criterion = nn.BCEWithLogitsLoss()
    loss_clf = criterion(clf(recon).flatten(), targets_u)
    loss_adv = criterion(adv(recon).flatten(), targets_adv)

    vae_enc_loss, _, _, _ = factor_vae_encoder_loss(
        recon, inputs, mu, logvar, z, z_perm, disc, beta=1.0, gamma=10.0
    )
    loss_disc = discriminator_loss(z.detach(), z_perm.detach(), disc)

    enc_loss = (loss_clf - 1.0 * loss_adv) + 0.1 * vae_enc_loss
    assert not torch.isnan(enc_loss), "enc_loss is NaN"
    assert not torch.isnan(loss_disc), "discriminator loss is NaN"

    enc_loss.backward()
    loss_disc.backward()

    print(f"  enc_loss={enc_loss.item():.4f}  disc_loss={loss_disc.item():.4f}")
    print("test_factor_vae PASSED")


if __name__ == "__main__":
    test_factor_vae_shapes_and_backward()

"""Quick test of Factor VAE in ARL pipeline - no CelebA required."""
import torch
import torch.nn as nn
from models.get_encoder import get_encoder
from models.factor_vae import (
    permute_dims,
    discriminator_loss,
    factor_vae_encoder_loss,
)
from models.cifar_like.resnet import ResNet18

def test_factor_vae():
    device = torch.device("cpu")
    B, C, H, W = 4, 3, 224, 224  # Small batch

    # Create models
    encoder = get_encoder("factor_vae", 224)
    encoder = encoder.to(device)
    clf = ResNet18()
    clf.linear = nn.Linear(512, 1)
    clf = clf.to(device)
    adv = ResNet18()
    adv.linear = nn.Linear(512, 1)
    adv = adv.to(device)

    # Dummy data
    inputs = torch.rand(B, C, H, W, device=device)
    targets_u = torch.randint(0, 2, (B,), device=device, dtype=torch.float32)
    targets_adv = torch.randint(0, 2, (B,), device=device, dtype=torch.float32)

    # Forward
    encoder.train()
    recon, mu, logvar, z = encoder(inputs, return_aux=True)
    z_perm = permute_dims(z)
    disc = encoder.discriminator

    assert recon.shape == (B, 3, 224, 224), f"recon shape {recon.shape}"
    assert z.shape == (B, 256), f"z shape {z.shape}"

    # Classifier forward
    u_logits = clf(recon).flatten()
    p_logits = adv(recon).flatten()
    criterion = nn.BCEWithLogitsLoss()
    loss_clf = criterion(u_logits, targets_u)
    loss_adv = criterion(p_logits, targets_adv)

    # Factor VAE losses
    vae_enc_loss, _, _, _ = factor_vae_encoder_loss(
        recon, inputs, mu, logvar, z, z_perm, disc, beta=1.0, gamma=10.0
    )
    loss_disc = discriminator_loss(z.detach(), z_perm.detach(), disc)

    # Backward
    arl_loss = loss_clf - 1.0 * loss_adv
    enc_loss = arl_loss + 0.1 * vae_enc_loss
    enc_loss.backward()
    loss_disc.backward()

    print("Factor VAE test PASSED: forward, losses, and backward succeeded.")

if __name__ == "__main__":
    test_factor_vae()

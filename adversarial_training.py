import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from models.get_encoder import get_encoder
from models.cvae import cvae_loss
from models.factor_vae import (
    permute_dims,
    discriminator_loss,
    factor_vae_encoder_loss,
)
from models.cifar_like.resnet import ResNet18
from models.vqvae_wrapper import GradientReversalLayer
import argparse
try:
    import wandb
except ImportError:
    wandb = None
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


CELEBA_ATTR_ORDER = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
    'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
    'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
    'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young',
]


class CelebAHFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("--data_source huggingface requires: pip install datasets")
        self.data = load_dataset("flwrlabs/celeba", split=hf_split)
        self.transform = transform
        self.attr_order = CELEBA_ATTR_ORDER

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
        img = row['image']
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
        img = self.transform(img)
        attrs = torch.tensor(
            [1.0 if row.get(k, False) else 0.0 for k in self.attr_order],
            dtype=torch.float32,
        )
        return img, attrs


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (use 64 for VQ-VAE to reduce memory)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate_enc', type=float, default=0.001)
    parser.add_argument('--learning_rate_clf', type=float, default=0.001)
    parser.add_argument('--learning_rate_adv', type=float, default=0.001)
    parser.add_argument('--encoder', type=str, default='unet',
                        choices=['unet', 'vanilla_vae', 'beta_vae', 'residual_vae',
                                 'cvae', 'factor_vae', 'vq_vae'])
    parser.add_argument('--vae_weight', type=float, default=0.1)
    parser.add_argument('--vae_beta', type=float, default=1.0)
    parser.add_argument('--vae_gamma', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_source', type=str, default='torchvision',
                        choices=['torchvision', 'huggingface'])
    parser.add_argument('--hf_cache_dir', type=str, default=None)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--lambda_clf', type=float, default=1.0)
    parser.add_argument('--lambda_warmup_epochs', type=int, default=0,
                        help='Linearly ramp lambda_clf from 0 to target over this many epochs.')
    parser.add_argument('--use_grl', action='store_true',
                        help='Use gradient reversal layer instead of manual ARL loss. '
                             'More stable for VQ-VAE.')
    parser.add_argument('--exp_name', type=str, default='celeb')
    parser.add_argument('--max_train_samples', type=int, default=60000)
    parser.add_argument('--max_val_samples', type=int, default=10000)
    parser.add_argument('--max_test_samples', type=int, default=10000)
    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size  = args.batch_size
    num_epochs  = args.num_epochs
    lr_enc      = args.learning_rate_enc
    lr_clf      = args.learning_rate_clf
    lr_adv      = args.learning_rate_adv
    device      = torch.device(args.device)
    data_dir    = args.data_dir
    lambda_clf  = args.lambda_clf
    img_size    = args.img_size
    unet_size   = 'tiny'
    encoder_name = args.encoder
    vae_weight  = args.vae_weight
    vae_beta    = args.vae_beta
    vae_gamma   = args.vae_gamma
    p_task      = 20  # gender
    u_task      = 31  # smile

    transforms_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    hf_cache = args.hf_cache_dir
    if hf_cache:
        os.environ['HF_HOME'] = hf_cache
        os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(hf_cache, 'hub')
        print(f"HF cache: {hf_cache}", flush=True)

    if args.data_source == 'huggingface':
        print("Loading CelebA from Hugging Face (flwrlabs/celeba)...", flush=True)
        trainset = CelebAHFDataset("train", transforms_train)
        valset   = CelebAHFDataset("valid", transforms_test)
        testset  = CelebAHFDataset("test",  transforms_test)
    else:
        trainset = torchvision.datasets.CelebA(
            root=data_dir, split='train', target_type=['attr'],
            transform=transforms_train, download=args.download)
        valset = torchvision.datasets.CelebA(
            root=data_dir, split='valid', target_type=['attr'],
            transform=transforms_test, download=args.download)
        testset = torchvision.datasets.CelebA(
            root=data_dir, split='test', target_type=['attr'],
            transform=transforms_test, download=args.download)

    train_set = Subset(trainset, range(min(args.max_train_samples, len(trainset))))
    val_set   = Subset(valset,   range(min(args.max_val_samples,   len(valset))))
    test_set  = Subset(testset,  range(min(args.max_test_samples,  len(testset))))

    nw = args.num_workers
    print(f"Data loaded: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}, "
          f"batch_size={batch_size}, num_workers={nw}", flush=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=nw)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=nw)

    encoder_model = get_encoder(encoder_name, img_size, unet_size=unet_size)
    if torch.cuda.device_count() > 1:
        encoder_model = nn.DataParallel(encoder_model)
    encoder_model = encoder_model.to(device)

    clf_model = ResNet18()
    clf_model.linear = nn.Linear(512, 1)
    if torch.cuda.device_count() > 1:
        clf_model = nn.DataParallel(clf_model)
    clf_model = clf_model.to(device)

    adv_model = ResNet18()
    adv_model.linear = nn.Linear(512, 1)
    if torch.cuda.device_count() > 1:
        adv_model = nn.DataParallel(adv_model)
    adv_model = adv_model.to(device)

    # GRL instance — lambda updated each epoch during warmup
    grl = GradientReversalLayer(lambda_=0.0).to(device)

    optimizer_enc = optim.Adam(encoder_model.parameters(), lr=lr_enc)
    optimizer_clf = optim.Adam(clf_model.parameters(), lr=lr_clf)
    optimizer_adv = optim.Adam(adv_model.parameters(), lr=lr_adv)
    optimizers = {'enc': optimizer_enc, 'clf': optimizer_clf, 'adv': optimizer_adv}

    if encoder_name == 'factor_vae':
        disc = encoder_model.module.discriminator if hasattr(encoder_model, 'module') \
               else encoder_model.discriminator
        optimizer_disc = optim.Adam(disc.parameters(), lr=lr_enc)
        optimizers['disc'] = optimizer_disc
        scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, T_max=num_epochs)

    criterion    = nn.BCEWithLogitsLoss()
    scheduler_enc = optim.lr_scheduler.CosineAnnealingLR(optimizer_enc, T_max=num_epochs)
    scheduler_clf = optim.lr_scheduler.CosineAnnealingLR(optimizer_clf, T_max=num_epochs)
    scheduler_adv = optim.lr_scheduler.CosineAnnealingLR(optimizer_adv, T_max=num_epochs)

    if args.use_wandb:
        wandb.init(project='privacy-preserving', name=args.exp_name)
        wandb.config.update(args)
        wandb.config.update({'lambda_clf': lambda_clf, 'unet_size': unet_size,
                              'p_task': p_task, 'u_task': u_task})

    num_batches = len(train_loader)
    for epoch in range(num_epochs):

        # Lambda warmup: ramp from 0 -> lambda_clf over warmup epochs
        if args.lambda_warmup_epochs > 0 and epoch < args.lambda_warmup_epochs:
            effective_lambda = lambda_clf * (epoch / args.lambda_warmup_epochs)
        else:
            effective_lambda = lambda_clf

        # Update GRL lambda each epoch
        grl.set_lambda(effective_lambda)

        encoder_model.train(); adv_model.train(); clf_model.train()
        running_loss_clf = 0.0
        running_loss_adv = 0.0
        print(f"[Epoch {epoch + 1}/{num_epochs}] lambda={effective_lambda:.3f} "
              f"use_grl={args.use_grl} ({num_batches} batches)...", flush=True)

        for i, (inputs, targets) in enumerate(train_loader):
            if i == 0:
                print("  First batch loaded, running forward/backward...", flush=True)
            inputs      = inputs.to(device)
            targets_u   = targets[:, u_task].float().to(device)
            targets_adv = targets[:, p_task].float().to(device)

            # ── Encoder forward ───────────────────────────────────────────────
            if encoder_name == 'cvae':
                recon, mu, logvar, _ = encoder_model(inputs, targets_u, return_aux=True)
                blurred = recon
            elif encoder_name == 'factor_vae':
                recon, mu, logvar, z = encoder_model(inputs, return_aux=True)
                blurred = recon
                z_perm = permute_dims(z)
                disc = encoder_model.module.discriminator if hasattr(encoder_model, 'module') \
                       else encoder_model.discriminator
            elif encoder_name in ('vanilla_vae', 'beta_vae', 'residual_vae'):
                recon, mu, logvar, z = encoder_model(inputs, return_aux=True)
                blurred = recon
            elif encoder_name == 'vq_vae':
                recon, _, _, z_q = encoder_model(inputs, return_aux=True)
                blurred = recon
            else:
                blurred = encoder_model(inputs)
            vis_imgs = blurred

            # ── Utility classifier loss ───────────────────────────────────────
            u_logits = clf_model(blurred).flatten()
            loss_clf = criterion(u_logits, targets_u)

            if args.use_grl:
                # ── GRL path: single backward updates encoder + clf + adversary
                # GRL reverses gradients flowing into encoder from adversary loss,
                # so encoder learns to fool adversary without a separate forward pass
                reversed_features = grl(blurred)
                adv_logits_grl    = adv_model(reversed_features).flatten()
                loss_adv          = criterion(adv_logits_grl, targets_adv)

                # VAE reconstruction terms
                if encoder_name == 'vq_vae':
                    recon_l    = F.mse_loss(recon, inputs)
                    codebook_l = encoder_model.last_codebook_loss or 0.0
                    enc_loss   = loss_clf + loss_adv + vae_weight * (recon_l + codebook_l)
                elif encoder_name == 'cvae':
                    vae_l, _, _ = cvae_loss(recon, inputs, mu, logvar, beta=vae_beta)
                    enc_loss = loss_clf + loss_adv + vae_weight * vae_l
                elif encoder_name == 'factor_vae':
                    vae_enc_loss, _, _, _ = factor_vae_encoder_loss(
                        recon, inputs, mu, logvar, z, z_perm, disc,
                        beta=vae_beta, gamma=vae_gamma)
                    enc_loss = loss_clf + loss_adv + vae_weight * vae_enc_loss
                elif encoder_name in ('vanilla_vae', 'beta_vae', 'residual_vae'):
                    B_enc   = recon.size(0)
                    lv      = logvar.clamp(-4, 4)
                    mu_c    = mu.clamp(-10, 10)
                    recon_l = F.mse_loss(recon, inputs, reduction='sum') / B_enc
                    kl_l    = -0.5 * torch.sum(1 + lv - mu_c.pow(2) - lv.exp()) / B_enc
                    enc_loss = loss_clf + loss_adv + vae_weight * (recon_l + vae_beta * kl_l)
                else:
                    enc_loss = loss_clf + loss_adv

                # Single backward updates encoder, clf, and adversary together
                optimizer_enc.zero_grad()
                optimizer_clf.zero_grad()
                optimizer_adv.zero_grad()
                enc_loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(clf_model.parameters(),     max_norm=1.0)
                if not torch.isnan(enc_loss):
                    optimizer_enc.step()
                    optimizer_clf.step()
                    optimizer_adv.step()
                else:
                    print(f"  WARNING: NaN enc_loss at step {i+1}, skipping", flush=True)
                    optimizer_enc.zero_grad()
                    optimizer_clf.zero_grad()
                    optimizer_adv.zero_grad()

            else:
                # ── Original manual ARL path (non-GRL) ───────────────────────
                with torch.no_grad():
                    adv_logits_enc = adv_model(blurred).flatten()
                    loss_adv_enc   = criterion(adv_logits_enc, targets_adv)

                blurred_detached = blurred.detach()
                p_logits_adv = adv_model(blurred_detached).flatten()
                loss_adv     = criterion(p_logits_adv, targets_adv)
                optimizer_adv.zero_grad()
                loss_adv.backward()
                optimizer_adv.step()

                arl_loss = loss_clf - effective_lambda * loss_adv_enc

                if encoder_name == 'cvae':
                    vae_l, _, _ = cvae_loss(recon, inputs, mu, logvar, beta=vae_beta)
                    enc_loss = arl_loss + vae_weight * vae_l
                elif encoder_name == 'factor_vae':
                    vae_enc_loss, _, _, _ = factor_vae_encoder_loss(
                        recon, inputs, mu, logvar, z, z_perm, disc,
                        beta=vae_beta, gamma=vae_gamma)
                    enc_loss = arl_loss + vae_weight * vae_enc_loss
                elif encoder_name in ('vanilla_vae', 'beta_vae', 'residual_vae'):
                    B_enc   = recon.size(0)
                    lv      = logvar.clamp(-4, 4)
                    mu_c    = mu.clamp(-10, 10)
                    recon_l = F.mse_loss(recon, inputs, reduction='sum') / B_enc
                    kl_l    = -0.5 * torch.sum(1 + lv - mu_c.pow(2) - lv.exp()) / B_enc
                    enc_loss = arl_loss + vae_weight * (recon_l + vae_beta * kl_l)
                elif encoder_name == 'vq_vae':
                    recon_l    = F.mse_loss(recon, inputs)
                    codebook_l = encoder_model.last_codebook_loss or 0.0
                    enc_loss   = arl_loss + vae_weight * (recon_l + codebook_l)
                else:
                    enc_loss = arl_loss

                optimizer_enc.zero_grad()
                optimizer_clf.zero_grad()
                enc_loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(clf_model.parameters(),     max_norm=1.0)
                if torch.isnan(enc_loss):
                    print(f"  WARNING: NaN enc_loss at step {i+1}, skipping", flush=True)
                    optimizer_enc.zero_grad()
                    optimizer_clf.zero_grad()
                else:
                    optimizer_enc.step()
                    optimizer_clf.step()

            if encoder_name == 'factor_vae':
                optimizer_disc = optimizers['disc']
                optimizer_disc.zero_grad()
                loss_disc = discriminator_loss(z.detach(), z_perm.detach(), disc)
                loss_disc.backward()
                optimizer_disc.step()

            running_loss_clf += loss_clf.item()
            running_loss_adv += loss_adv.item()

            step         = i + 1
            log_interval = 10 if num_batches <= 500 else 100
            if step == 1:
                print(f'  Step [1/{num_batches}] loss_clf: {running_loss_clf:.4f} '
                      f'loss_adv: {running_loss_adv:.4f}', flush=True)
            elif step % log_interval == 0:
                print(f'  Step [{step}/{num_batches}] '
                      f'loss_clf: {running_loss_clf / log_interval:.4f} '
                      f'loss_adv: {running_loss_adv / log_interval:.4f}', flush=True)
                running_loss_clf = 0.0
                running_loss_adv = 0.0

        scheduler_enc.step(); scheduler_clf.step(); scheduler_adv.step()
        if encoder_name == 'factor_vae':
            scheduler_disc.step()

        # ── Validation ────────────────────────────────────────────────────────
        encoder_model.eval(); adv_model.eval(); clf_model.eval()
        val_correct = val_correct_adv = 0
        val_loss_sum = val_loss_adv_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs      = inputs.to(device)
                targets_u   = targets[:, u_task].float().to(device)
                targets_adv = targets[:, p_task].float().to(device)
                B = inputs.size(0)

                blurred    = encoder_model(inputs, targets_u) if encoder_name == 'cvae' \
                             else encoder_model(inputs)
                logits_u   = clf_model(blurred).flatten()
                adv_logits = adv_model(blurred).flatten()

                val_loss_sum     += criterion(logits_u,   targets_u).item()   * B
                val_loss_adv_sum += criterion(adv_logits, targets_adv).item() * B
                val_n += B
                val_correct     += ((torch.sigmoid(logits_u)   > 0.5).float() == targets_u).sum().item()
                val_correct_adv += ((torch.sigmoid(adv_logits) > 0.5).float() == targets_adv).sum().item()

        val_acc     = 100.0 * val_correct     / val_n
        val_acc_adv = 100.0 * val_correct_adv / val_n
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Acc: {val_acc:.2f}, '
              f'Val Acc Adv: {val_acc_adv:.2f}, Val Loss: {val_loss_sum / val_n:.4f}, '
              f'Val Loss Adv: {val_loss_adv_sum / val_n:.4f}')

        # ── Reconstruction grid ───────────────────────────────────────────────
        try:
            vis_dir = os.path.join(os.getcwd(), 'visuals')
            os.makedirs(vis_dir, exist_ok=True)
            with torch.no_grad():
                sample_in  = inputs[:8].cpu()
                sample_out = encoder_model(inputs[:8], targets_u[:8]).cpu() \
                             if encoder_name == 'cvae' else encoder_model(inputs[:8]).cpu()
            grid = torchvision.utils.make_grid(
                torch.cat([sample_in, sample_out], dim=0), nrow=8,
                normalize=True, value_range=(0, 1))
            torchvision.utils.save_image(
                grid, os.path.join(vis_dir, f'{args.exp_name}_epoch{epoch+1:02d}.png'))
            print(f'  Saved reconstruction grid -> visuals/{args.exp_name}_epoch{epoch+1:02d}.png',
                  flush=True)
        except Exception as e:
            print(f'  Warning: could not save visual grid: {e}', flush=True)

        if args.use_wandb:
            wandb.log({
                'train_imgs':       wandb.Image(torchvision.utils.make_grid(inputs[:8].detach().cpu())),
                'train_blurs':      wandb.Image(torchvision.utils.make_grid(vis_imgs[:8].detach().cpu())),
                'val_loss':         val_loss_sum / val_n,
                'val_loss_adv':     val_loss_adv_sum / val_n,
                'val_acc':          val_acc,
                'val_acc_adv':      val_acc_adv,
                'effective_lambda': effective_lambda,
            })

    # ── Test ──────────────────────────────────────────────────────────────────
    encoder_model.eval(); adv_model.eval(); clf_model.eval()
    test_correct = test_correct_adv = test_n = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs      = inputs.to(device)
            targets_u   = targets[:, u_task].float().to(device)
            targets_adv = targets[:, p_task].float().to(device)

            blurred    = encoder_model(inputs, targets_u) if encoder_name == 'cvae' \
                         else encoder_model(inputs)
            logits_u   = clf_model(blurred).flatten()
            adv_logits = adv_model(blurred).flatten()

            test_correct     += ((torch.sigmoid(logits_u)   > 0.5).float() == targets_u).sum().item()
            test_correct_adv += ((torch.sigmoid(adv_logits) > 0.5).float() == targets_adv).sum().item()
            test_n += inputs.size(0)

    print(f'Test Acc: {100.0 * test_correct / test_n:.2f}, '
          f'Test Acc Adv: {100.0 * test_correct_adv / test_n:.2f}')

    torch.save(encoder_model.state_dict(), f'encoder_model_{args.exp_name}.pt')
    torch.save(clf_model.state_dict(),     f'clf_model_{args.exp_name}.pt')
    torch.save(adv_model.state_dict(),     f'adv_model_{args.exp_name}.pt')
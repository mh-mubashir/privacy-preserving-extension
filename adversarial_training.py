import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from models.unet import UNet
from models.cifar_like.resnet import ResNet18
import wandb
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate_enc', type=float, default=0.001)
    parser.add_argument('--learning_rate_adv', type=float, default=0.001)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--data_dir', type=str, default='/projects/xz-group/datasets/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--lambda_clf', type=float, default=1.0, help='Weight for utility loss (encoder minimizes loss_clf - lambda*loss_adv)')
    parser.add_argument('--exp_name', type=str, default='celeb')
    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr_enc = args.learning_rate_enc
    lr_clf = args.learning_rate_clf
    lr_adv = args.learning_rate_adv
    device = torch.device(args.device)
    data_dir = args.data_dir
    lambda_clf = args.lambda_clf
    img_size = 224
    unet_size = 'tiny'

    # utility task (e.g. smile) vs private task (e.g. gender) for adversary
    p_task = 20 # gender
    u_task = 31 # smile

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

    trainset = torchvision.datasets.CelebA(
        root=data_dir, split='train', target_type=['attr'],
        transform=transforms_train,
    )
    valset = torchvision.datasets.CelebA(
        root=data_dir, split='valid', target_type=['attr'],
        transform=transforms_test,
    )
    testset = torchvision.datasets.CelebA(
        root=data_dir, split='test', target_type=['attr'],
        transform=transforms_test,
    )

    train_set = Subset(trainset, range(60000))
    val_set = Subset(valset, range(10000))
    test_set = Subset(testset, range(10000))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Encoder (UNet: 3-channel RGB in -> 3-channel out for ResNet)
    encoder_model = UNet(3, 3, size=unet_size)
    if torch.cuda.device_count() > 1:
        encoder_model = nn.DataParallel(encoder_model)
    encoder_model = encoder_model.to(device)

    # Utility classifier (trainable)
    clf_model = ResNet18()
    clf_model.linear = nn.Linear(512, 1)
    if torch.cuda.device_count() > 1:
        clf_model = nn.DataParallel(clf_model)
    clf_model = clf_model.to(device)

    # Adversary (predicts private attribute)
    adv_model = ResNet18()
    adv_model.linear = nn.Linear(512, 1)
    if torch.cuda.device_count() > 1:
        adv_model = nn.DataParallel(adv_model)
    adv_model = adv_model.to(device)

    optimizer_enc = optim.Adam(encoder_model.parameters(), lr=lr_enc)
    optimizer_clf = optim.Adam(clf_model.parameters(), lr=lr_clf)
    optimizer_adv = optim.Adam(adv_model.parameters(), lr=lr_adv)
    criterion = nn.BCEWithLogitsLoss()

    scheduler_enc = optim.lr_scheduler.CosineAnnealingLR(optimizer_enc, T_max=num_epochs)
    scheduler_clf = optim.lr_scheduler.CosineAnnealingLR(optimizer_clf, T_max=num_epochs)
    scheduler_adv = optim.lr_scheduler.CosineAnnealingLR(optimizer_adv, T_max=num_epochs)

    # init wandb
    if args.use_wandb:
        wandb.init(project='privacy-preserving', name=args.exp_name)
        wandb.config.update(args)
        wandb.config.update({
            'lambda_clf': lambda_clf,
            'unet_size': unet_size,
            'p_task': p_task,
            'u_task': u_task,
        })

    for epoch in range(num_epochs):
        encoder_model.train()
        adv_model.train()
        clf_model.train()
        running_loss_clf = 0.0
        running_loss_adv = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets_u = targets[:, u_task].float().to(device)
            targets_adv = targets[:, p_task].float().to(device)
            B = inputs.size(0)

            # Encode (inputs 3ch -> encoder 3ch)
            blurred = encoder_model(inputs)
            vis_imgs = blurred

            u_logits = clf_model(blurred).flatten()
            p_logits = adv_model(blurred).flatten()

            loss_clf = criterion(u_logits, targets_u)
            loss_adv = criterion(adv_logits, targets_adv)

            # Update adversary: minimize loss_adv
            optimizer_adv.zero_grad()
            loss_adv.backward(retain_graph=True)
            optimizer_adv.step()

            # Update encoder and utility classifier: minimize loss_clf (utility) - lambda * loss_adv (privacy)
            optimizer_enc.zero_grad()
            optimizer_clf.zero_grad()
            (loss_clf - lambda_clf * loss_adv).backward()
            optimizer_enc.step()
            optimizer_clf.step()

            running_loss_clf += loss_clf.item()
            running_loss_adv += loss_adv.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss_clf: {running_loss_clf / 100:.4f}, Loss_adv: {running_loss_adv / 100:.4f}')
                running_loss_clf = 0.0
                running_loss_adv = 0.0

        scheduler_enc.step()
        scheduler_clf.step()
        scheduler_adv.step()

        # Validation
        encoder_model.eval()
        adv_model.eval()
        clf_model.eval()
        val_correct, val_correct_adv = 0, 0
        val_loss_sum, val_loss_adv_sum = 0.0, 0.0
        val_n = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets_u = targets[:, u_task].float().to(device)
                targets_adv = targets[:, p_task].float().to(device)
                B = inputs.size(0)

                blurred = encoder_model(inputs)
                logits_u = clf_model(blurred).flatten()
                adv_logits = adv_model(blurred).flatten()

                loss_clf = criterion(logits_u, targets_u)
                loss_adv = criterion(adv_logits, targets_adv)
                val_loss_sum += loss_clf.item() * B
                val_loss_adv_sum += loss_adv.item() * B
                val_n += B

                pred_u = (torch.sigmoid(logits_u) > 0.5).float()
                pred_adv = (torch.sigmoid(adv_logits) > 0.5).float()
                val_correct += (pred_u == targets_u).sum().item()
                val_correct_adv += (pred_adv == targets_adv).sum().item()

        val_acc = 100.0 * val_correct / val_n
        val_acc_adv = 100.0 * val_correct_adv / val_n
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Acc: {val_acc:.2f}, Val Acc Adv: {val_acc_adv:.2f}, Val Loss: {val_loss_sum / val_n:.4f}, Val Loss Adv: {val_loss_adv_sum / val_n:.4f}')

        if args.use_wandb:
            imgs = torchvision.utils.make_grid(inputs[:8].detach().cpu())
            blrs = torchvision.utils.make_grid(vis_imgs[:8].detach().cpu())
            wandb.log({
                'train_imgs': wandb.Image(imgs),
                'train_blurs': wandb.Image(blrs),
                'val_loss': val_loss_sum / val_n,
                'val_loss_adv': val_loss_adv_sum / val_n,
                'val_acc': val_acc,
                'val_acc_adv': val_acc_adv,
            })

    # Test
    encoder_model.eval()
    adv_model.eval()
    clf_model.eval()
    test_correct, test_correct_adv = 0, 0
    test_n = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets_u = targets[:, u_task].float().to(device)
            targets_adv = targets[:, p_task].float().to(device)
            B = inputs.size(0)

            blurred = encoder_model(inputs)
            logits_u = clf_model(blurred).flatten()
            adv_logits = adv_model(blurred).flatten()

            pred_u = (torch.sigmoid(logits_u) > 0.5).float()
            pred_adv = (torch.sigmoid(adv_logits) > 0.5).float()
            test_correct += (pred_u == targets_u).sum().item()
            test_correct_adv += (pred_adv == targets_adv).sum().item()
            test_n += B

    test_acc = 100.0 * test_correct / test_n
    test_acc_adv = 100.0 * test_correct_adv / test_n
    print(f'Test Acc: {test_acc:.2f}, Test Acc Adv: {test_acc_adv:.2f}')

    torch.save(encoder_model.state_dict(), f'encoder_model_{args.exp_name}.pt')
    torch.save(clf_model.state_dict(), f'clf_model_{args.exp_name}.pt')
    torch.save(adv_model.state_dict(), f'adv_model_{args.exp_name}.pt')

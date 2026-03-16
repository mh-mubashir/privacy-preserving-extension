import argparse
from pathlib import Path

# Force CPU before importing torch so pythae trainer sees no GPU (e.g. RTX 50xx / sm_120)
_parser = argparse.ArgumentParser()
_parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
_args, _ = _parser.parse_known_args()
if _args.device == "cpu":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from pythae.models import (
    DisentangledBetaVAE,
    DisentangledBetaVAEConfig,
    BetaTCVAE,
    BetaTCVAEConfig,
    FactorVAE,
    FactorVAEConfig,
)
from pythae.pipelines import TrainingPipeline
from pythae.trainers import AdversarialTrainerConfig, BaseTrainerConfig


def build_celeba_dataloaders_torchvision(
    data_dir: str,
    img_size: int,
    max_train: int,
    max_val: int,
    batch_size: int,
    num_workers: int,
    download: bool,
):
    transforms_train = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transforms_test = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    trainset = torchvision.datasets.CelebA(
        root=data_dir,
        split="train",
        target_type=["attr"],
        transform=transforms_train,
        download=download,
    )
    valset = torchvision.datasets.CelebA(
        root=data_dir,
        split="valid",
        target_type=["attr"],
        transform=transforms_test,
        download=download,
    )

    train_set = Subset(trainset, range(min(max_train, len(trainset))))
    val_set = Subset(valset, range(min(max_val, len(valset))))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


class CelebAHFDataset(torch.utils.data.Dataset):
    """Hugging Face CelebA wrapper that returns only images for pythae."""

    def __init__(self, hf_split, img_size: int):
        if load_dataset is None:
            raise ImportError(
                "datasets is required for --data_source huggingface. "
                "Install with: pip install datasets"
            )
        self.data = load_dataset("flwrlabs/celeba", split=hf_split)
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
        img = row["image"]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        img = self.transform(img)
        # pythae default encoder expects (C, H, W) with C=3; ensure 3 channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img


def build_celeba_dataloaders_hf(
    img_size: int,
    max_train: int,
    max_val: int,
    batch_size: int,
    num_workers: int,
):
    trainset = CelebAHFDataset("train", img_size=img_size)
    valset = CelebAHFDataset("valid", img_size=img_size)

    train_set = Subset(trainset, range(min(max_train, len(trainset))))
    val_set = Subset(valset, range(min(max_val, len(valset))))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def build_pythae_model(
    variant: str,
    img_size: int,
    latent_dim: int,
    beta: float,
    gamma: float,
):
    input_dim = (3, img_size, img_size)

    if variant == "disentangled_betavae":
        config = DisentangledBetaVAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=beta,
        )
        model = DisentangledBetaVAE(model_config=config)
    elif variant == "betatcvae":
        config = BetaTCVAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=beta,
            gamma=gamma,
        )
        model = BetaTCVAE(model_config=config)
    elif variant == "factorvae":
        config = FactorVAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            gamma=gamma,
        )
        model = FactorVAE(model_config=config)
    else:
        raise ValueError(f"Unknown pythae variant: {variant}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train DisentangledBetaVAE, BetaTCVAE, or FactorVAE on CelebA using pythae"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="disentangled_betavae",
        choices=["disentangled_betavae", "betatcvae", "factorvae"],
        help="Which pythae model to train",
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./pythae_runs")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=10.0,
        help="Total correlation weight (BetaTCVAE / FactorVAE)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=100000,
        help="Subset of CelebA train split (for faster runs)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=10000,
        help="Subset of CelebA val split",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CelebA via torchvision (may hit Google Drive quota)",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="torchvision",
        choices=["torchvision", "huggingface"],
        help="CelebA source: torchvision files under --data_dir or huggingface dataset",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="[huggingface] Cache dir for HF downloads (e.g. ./hf_cache)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on (use cpu to avoid GPU capability mismatch, e.g. RTX 50xx)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Training {args.variant} with pythae "
        f"on device={device}, img_size={args.img_size}, latent_dim={args.latent_dim}"
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.data_source == "huggingface":
        if args.hf_cache_dir is not None:
            import os

            os.environ["HF_HOME"] = args.hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(
                Path(args.hf_cache_dir) / "hub"
            )
        train_loader, val_loader = build_celeba_dataloaders_hf(
            img_size=args.img_size,
            max_train=args.max_train_samples,
            max_val=args.max_val_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        train_loader, val_loader = build_celeba_dataloaders_torchvision(
            data_dir=args.data_dir,
            img_size=args.img_size,
            max_train=args.max_train_samples,
            max_val=args.max_val_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.download,
        )

    model = build_pythae_model(
        variant=args.variant,
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        beta=args.beta,
        gamma=args.gamma,
    )

    training_config = BaseTrainerConfig(
        output_dir=str(Path(args.output_dir) / args.variant),
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
    )

    # FactorVAE is trained with the adversarial trainer (autoencoder + discriminator)
    if args.variant == "factorvae":
        training_config = AdversarialTrainerConfig(
            output_dir=str(Path(args.output_dir) / args.variant),
            num_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            autoencoder_learning_rate=args.learning_rate,
            discriminator_learning_rate=args.learning_rate,
        )

    pipeline = TrainingPipeline(
        training_config=training_config,
        model=model,
    )

    # pythae pipelines take tensors or arrays; extract image tensors from the
    # dataloaders. For torchvision CelebA, each batch is (images, attrs);
    # for the HF wrapper, each batch is just a tensor of images.
    def _get_images(batch):
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    train_batches = [_get_images(batch) for batch in train_loader]
    val_batches = [_get_images(batch) for batch in val_loader]

    train_data = torch.cat(train_batches, dim=0)
    val_data = torch.cat(val_batches, dim=0)

    pipeline(
        train_data=train_data,
        eval_data=val_data,
    )


if __name__ == "__main__":
    main()


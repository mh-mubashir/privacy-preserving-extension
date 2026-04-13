import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import time
import os
import numpy as np

try:
    from datasets.block import BlockDataset, LatentBlockDataset
except ImportError:
    BlockDataset = None
    LatentBlockDataset = None


def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_celeba(data_dir, img_size=64, max_train=60000, max_val=10000):
    transforms_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transforms_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CelebA(
        root=data_dir, split="train", target_type="attr",
        transform=transforms_train, download=False
    )
    valset = datasets.CelebA(
        root=data_dir, split="valid", target_type="attr",
        transform=transforms_val, download=False
    )

    train = Subset(trainset, range(min(max_train, len(trainset))))
    val   = Subset(valset,   range(min(max_val,   len(valset))))
    return train, val


def load_block():
    if BlockDataset is None:
        raise ImportError("BlockDataset not available.")
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))
    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val


def load_latent_block():
    if LatentBlockDataset is None:
        raise ImportError("LatentBlockDataset not available.")
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,  transform=None)
    val   = LatentBlockDataset(data_file_path, train=False, transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size, data_dir=None, img_size=64,
                                max_train=60000, max_val=10000):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'CELEBA':
        if data_dir is None:
            raise ValueError("data_dir must be provided for CelebA dataset.")
        training_data, validation_data = load_celeba(data_dir, img_size, max_train, max_val)
        training_loader, validation_loader = data_loaders(training_data, validation_data, batch_size)
        # estimate variance from a sample batch
        sample_loader = DataLoader(training_data, batch_size=256, shuffle=True)
        sample_imgs, _ = next(iter(sample_loader))
        x_train_var = np.var(sample_imgs.numpy())

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data)

    else:
        raise ValueError('Invalid dataset: choose from CIFAR10, CELEBA, BLOCK, LATENT_BLOCK.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save, SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
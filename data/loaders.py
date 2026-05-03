import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from data.transforms import get_transforms
from data.contract import VAL_SPLIT_RATIO, RANDOM_SEED
from data.stats import get_stats
import os
import logging

logger = logging.getLogger(__name__)

DATASET_NAME = 'cifar10'


def get_loaders(batch_size: int):
    train_transform, test_transform = get_transforms()
    pin_memory = torch.cuda.is_available()
    num_workers = min(2, os.cpu_count() or 1)

    # Load the training set ONCE and split into train/val via indices
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    total = len(full_train)
    train_size = int((1 - VAL_SPLIT_RATIO) * total)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(total, generator=generator)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train, train_indices)
    val_base = datasets.CIFAR10(root='./data', train=True, download=False, transform=test_transform)
    val_dataset = Subset(val_base, val_indices)

    # Logging
    logger.info("=" * 50)
    logger.info("CIFAR-10 Dataset Info")
    logger.info("=" * 50)
    logger.info(f"  Total training samples : {total}")
    logger.info(f"  Train split            : {train_size} ({(1 - VAL_SPLIT_RATIO) * 100:.0f}%)")
    logger.info(f"  Validation split       : {val_size} ({VAL_SPLIT_RATIO * 100:.0f}%)")
    logger.info(f"  Test samples           : {len(test_set)}")
    logger.info(f"  Batch size             : {batch_size}")
    logger.info(f"  Num workers            : {num_workers}")
    logger.info(f"  Pin memory             : {pin_memory}")

    stats = get_stats()
    logger.info(f"  Normalization mean     : {stats['mean']}")
    logger.info(f"  Normalization std      : {stats['std']}")
    logger.info("=" * 50)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

STATS_CACHE_PATH = Path('./data/dataset_stats.json')
CACHE_KEY = 'cifar10'
N_CHANNELS = 3


def compute_stats() -> dict:
    """Compute exact per-channel mean and std over the entire CIFAR-10 dataset."""
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=2)

    # Accumulate sum and sum-of-squares across ALL pixels for exact stats
    pixel_sum = torch.zeros(N_CHANNELS)
    pixel_sq_sum = torch.zeros(N_CHANNELS)
    pixel_count = 0

    for imgs, _ in loader:
        # imgs shape: [B, C, H, W]
        b, c, h, w = imgs.shape
        n_pixels = b * h * w
        pixel_count += n_pixels
        for ch in range(c):
            channel = imgs[:, ch, :, :]
            pixel_sum[ch] += channel.sum()
            pixel_sq_sum[ch] += (channel ** 2).sum()

    mean = pixel_sum / pixel_count
    std = torch.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    logger.info(f"Computed global CIFAR-10 stats — mean: {mean.tolist()}, std: {std.tolist()}")

    return {'mean': mean.tolist(), 'std': std.tolist()}


def get_stats() -> dict:
    if STATS_CACHE_PATH.exists():
        with open(STATS_CACHE_PATH) as f:
            all_stats = json.load(f)
        if CACHE_KEY in all_stats:
            logger.debug("Loaded cached stats for CIFAR-10")
            return all_stats[CACHE_KEY]

    logger.info("No cached stats found for CIFAR-10, computing…")
    stats = compute_stats()

    all_stats = {}
    if STATS_CACHE_PATH.exists():
        with open(STATS_CACHE_PATH) as f:
            all_stats = json.load(f)

    all_stats[CACHE_KEY] = stats
    STATS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_CACHE_PATH, 'w') as f:
        json.dump(all_stats, f, indent=2)

    return stats

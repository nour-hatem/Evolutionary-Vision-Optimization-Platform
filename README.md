# ☁️ Cloud Computing Project — Data Pipeline

> CIFAR-10 data loading pipeline for distributed deep learning.  
> FCAI Capital University (formerly Helwan) — Spring 2026

---

## What This Does

A clean, modular data package that handles everything you need to start training on CIFAR-10:

- ✅ Downloads the dataset automatically
- ✅ Splits train into **80% train / 20% validation** (seeded, reproducible)
- ✅ Applies proper augmentation (random crop + flip) for training
- ✅ Normalizes images using exact per-channel statistics
- ✅ Returns ready-to-use PyTorch `DataLoader`s

---

## Usage

```python
from data import get_loaders

train_loader, val_loader, test_loader = get_loaders(batch_size=64)

for images, labels in train_loader:
    # images: [64, 3, 32, 32]
    # labels: [64]
    output = model(images)
    ...
```

---

## Branch Structure

```
data/
├── __init__.py           # Exports get_loaders()
├── loaders.py            # Main entry — builds train/val/test loaders
├── transforms.py         # Augmentation pipelines (train vs test)
├── stats.py              # Computes & caches mean/std for normalization
├── contract.py           # Shared constants (split ratio, seed, shapes)
├── dataset_stats.json    # Cached normalization values
└── tests/
    └── test_loaders.py   # Tests for data loaders
```

---

## Module Descriptions

### `loaders.py`
The main file. Downloads CIFAR-10, splits the training set into train/val using index-based splitting, and returns three `DataLoader`s ready for training. Auto-detects GPU for `pin_memory` and picks a reasonable `num_workers`.

### `transforms.py`
Defines two transform pipelines — **train** (with random crop + horizontal flip for augmentation) and **test** (normalization only). Uses the cached stats from `stats.py`.

### `stats.py`
Computes the exact per-channel mean and std across all 50k training images. Runs once, then saves results to `dataset_stats.json` so it never recomputes.

### `contract.py`
Single source of truth for constants: validation split ratio (`0.2`), random seed (`42`), and expected tensor shapes. Keeps magic numbers out of the other files.

### `dataset_stats.json`
The cached normalization values. Generated automatically by `stats.py` on first run.

### `tests/test_loaders.py`

A simple sanity check for the data pipeline. Loads a batch from the training loader and verifies tensor shapes match CIFAR-10 expectations `(64, 3, 32, 32)` for images and `(64,)` for labels. Helps catch broken transforms, wrong splits, or shape mismatches early.

---

## Data Contract

```python
get_loaders(batch_size: int) -> (train_loader, val_loader, test_loader)
```

| Property | Value |
|---|---|
| Image shape | `[B, 3, 32, 32]` |
| Labels | `0 – 9` (10 classes) |
| Train size | 40,000 (augmented) |
| Val size | 10,000 (clean) |
| Test size | 10,000 (clean) |
| Seed | `42` — same split every time |

---

## Normalization Stats

Computed once across all 50k training images, then cached:

| Channel | Mean | Std |
|---|---|---|
| R | 0.4914 | 0.2470 |
| G | 0.4822 | 0.2435 |
| B | 0.4465 | 0.2616 |

---

## Augmentation

| Split | Transforms |
|---|---|
| **Train** | RandomCrop(32, pad=4) → HFlip → Normalize |
| **Val / Test** | Normalize only |

---

## Run Tests

```bash
python -m tests.test_loaders
```

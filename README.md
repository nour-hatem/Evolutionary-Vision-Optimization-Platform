# ☁️ Vision Optimization Platform

> Genetic Algorithm + CNN + Data Pipeline for CIFAR-10  
> FCAI — Spring 2026

---

# 🧬 Genetic Algorithm Optimizer (Ahmad)

This module runs a Genetic Algorithm (GA) to find the best hyperparameter configuration for the CNN model.

## Results

- Best validation accuracy: **73.38%**
- Convergence from **70.02% → 73.38%**

## Best Configuration

| Parameter     | Value  |
| ------------- | ------ |
| Learning rate | 0.0003 |
| Filters       | 96     |
| Layers        | 4      |
| Dropout       | 0.3    |
| Batch size    | 32     |

## Output Files

- `best_config.json` → sent to model training
- `ga_log.csv` → used for dashboard
- `ga_checkpoint.json` → resume runs

---

# 📦 Data Pipeline (Nour)

Handles CIFAR-10 data loading and preprocessing.

## Features

- Automatic dataset download
- Train/validation split (80/20)
- Data augmentation
- Normalization
- Ready-to-use PyTorch DataLoaders

## Usage

```python
from data import get_loaders

train_loader, val_loader, test_loader = get_loaders(batch_size=64)
```

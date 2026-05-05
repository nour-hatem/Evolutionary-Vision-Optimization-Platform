"""
model/train.py
--------------
Core training function consumed by:
  - model/run_baseline.py  (Yassin, full training)
  - optimizer/fitness.py   (Ahmad, fast_mode=True during GA search)
  - scripts/retrain_best.py (integration, full training on best config)

train(config, fast_mode) → metrics dict
  fast_mode=True  : 3 epochs, used by GA (speed matters)
  fast_mode=False : full epochs from config, used for real results
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from model.cnn import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds.argmax(dim=1) == labels).float().mean().item()


def _f1_score(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> float:
    """Macro F1 — computed manually to avoid extra library deps."""
    pred_classes = preds.argmax(dim=1)
    f1_scores = []
    for c in range(num_classes):
        tp = ((pred_classes == c) & (labels == c)).sum().item()
        fp = ((pred_classes == c) & (labels != c)).sum().item()
        fn = ((pred_classes != c) & (labels == c)).sum().item()
        precision = tp / (tp + fp + 1e-8) #Quality. When the model claims an image is a "Cat," how often is it right?
        recall    = tp / (tp + fn + 1e-8) #Quantity. Out of all the actual "Cats" in the dataset, how many did the model manage to find?
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    return round(sum(f1_scores) / num_classes, 4)


def _run_epoch_train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _run_epoch_val(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images)
            total_loss += criterion(out, labels).item()
            all_preds.append(out)
            all_labels.append(labels)
    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return total_loss / len(loader), preds, labels


# ── public API ────────────────────────────────────────────────────────────────

def train(
    config: dict,
    fast_mode: bool = False,
    save_checkpoint: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Train a CNN with the given config and return a metrics dict.

    Parameters
    ----------
    config : dict
        Keys:
          lr           float  learning rate          default 0.001
          num_filters  int    base conv filters       default 32
          num_layers   int    conv blocks             default 2
          dropout      float  dropout rate            default 0.5
          batch_size   int    training batch size     default 64
          epochs       int    training epochs         default 10
          num_classes  int    output classes          default 10
          weight_decay float  L2 regularisation       default 1e-4

    fast_mode : bool
        If True, caps training at 3 epochs (for GA fitness evaluation).

    save_checkpoint : str or None
        If given, saves model weights to checkpoints/<save_checkpoint>.pth

    verbose : bool
        Print epoch-by-epoch progress.

    Returns
    -------
    dict with keys:
        accuracy, f1, train_loss (list), val_loss (list),
        epochs_run, training_time_seconds, config
    """

    # ── config defaults ────────────────────────────────────────────────────────
    lr           = float(config.get("lr",           0.001))
    batch_size   = int(config.get("batch_size",     64))
    epochs       = 3 if fast_mode else int(config.get("epochs", 10))
    weight_decay = float(config.get("weight_decay", 1e-4))
    num_classes  = int(config.get("num_classes",    10))

    # ── data ───────────────────────────────────────────────────────────────────
    # Import here so this file can be imported without torchvision installed
    try:
        from data.loaders import get_loaders
        dataset = config.get("dataset", "cifar10")
        train_loader, val_loader, _ = get_loaders(dataset, batch_size)
    except ImportError:
        raise ImportError(
            "data/loaders.py not found. "
            "Make sure Nour's data module is in the repo before running training."
        )

    # ── model, loss, optimiser ─────────────────────────────────────────────────
    model     = build_model({**config, "num_classes": num_classes , "input_channels": int(config.get("input_channels", 3))}).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #weight_decay = (L2 Regularization)It penalizes the model for having excessively large weights, which forces the model to stay simple and prevents it from "memorizing" the noise in the training data (Overfitting).
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*55}")
        print(f"  Device      : {DEVICE}")
        print(f"  Config      : filters={config.get('num_filters',32)}, "
              f"layers={config.get('num_layers',2)}, lr={lr}")
        print(f"  Parameters  : {total_params:,}")
        print(f"  Epochs      : {epochs}  {'(fast mode)' if fast_mode else ''}")
        print(f"{'='*55}")

    # ── training loop ──────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_acc = 0.0
    best_state   = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss = _run_epoch_train(model, train_loader, optimizer, criterion)
        vl_loss, preds, labels = _run_epoch_val(model, val_loader, criterion)
        scheduler.step()

        acc = _accuracy(preds, labels)
        train_losses.append(round(tr_loss, 4))
        val_losses.append(round(vl_loss, 4))

        if acc > best_val_acc:
            best_val_acc = acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"  Epoch {epoch:2d}/{epochs}  "
                  f"train_loss={tr_loss:.4f}  "
                  f"val_loss={vl_loss:.4f}  "
                  f"val_acc={acc:.4f}")

    elapsed = round(time.time() - t0, 1)

    # ── final metrics on best weights ──────────────────────────────────────────
    model.load_state_dict(best_state)
    _, preds, labels = _run_epoch_val(model, val_loader, criterion)
    final_acc = _accuracy(preds, labels)
    final_f1  = _f1_score(preds, labels, num_classes)

    if verbose:
        print(f"\n  Best val_acc : {final_acc:.4f}")
        print(f"  F1 (macro)   : {final_f1:.4f}")
        print(f"  Time         : {elapsed}s")
        print(f"{'='*55}\n")

    # ── save checkpoint ────────────────────────────────────────────────────────
    if save_checkpoint:
        ckpt_path = CHECKPOINT_DIR / f"{save_checkpoint}.pth"
        torch.save({
            "model_state_dict": best_state,
            "config":           config,
            "metrics": {
                "accuracy": final_acc,
                "f1":       final_f1,
            },
        }, ckpt_path)
        if verbose:
            print(f"  Checkpoint saved → {ckpt_path}")

    # ── return metrics dict ────────────────────────────────────────────────────
    return {
        "accuracy":               round(final_acc, 4),
        "f1":                     final_f1,
        "train_loss":             train_losses,
        "val_loss":               val_losses,
        "epochs_run":             epochs,
        "training_time_seconds":  elapsed,
        "config":                 config,
    }
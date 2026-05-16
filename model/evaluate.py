"""
model/evaluate.py
-----------------
Load a saved checkpoint and evaluate it on the test set.

Usage:
    python -m model.evaluate --checkpoint baseline_model
    python -m model.evaluate --checkpoint optimized_model
"""

import argparse
import json
from pathlib import Path

import torch

from model.cnn import build_model
from model.train import _accuracy, _f1_score, DEVICE

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def evaluate(checkpoint_name: str, dataset: str = "cifar10", batch_size: int = 64) -> dict:
    """
    Load checkpoint and return test-set metrics.

    Parameters
    ----------
    checkpoint_name : str   filename without .pth
    dataset         : str   'cifar10' or 'mnist'
    batch_size      : int

    Returns
    -------
    dict: accuracy, f1, num_samples, checkpoint
    """
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt   = torch.load(ckpt_path, map_location=DEVICE)
    config = ckpt["config"]
    model  = build_model(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from data.loaders import get_loaders
    _, _, test_loader = get_loaders(dataset, batch_size)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            out    = model(images)
            all_preds.append(out)
            all_labels.append(labels.to(DEVICE))

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    acc = _accuracy(preds, labels)
    f1  = _f1_score(preds, labels, num_classes=config.get("num_classes", 10))

    results = {
        "checkpoint":  checkpoint_name,
        "dataset":     dataset,
        "accuracy":    round(acc, 4),
        "f1":          round(f1, 4),
        "num_samples": len(labels),
    }

    print(f"\n{'='*45}")
    print(f"  Checkpoint  : {checkpoint_name}")
    print(f"  Dataset     : {dataset} (test set)")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 (macro)  : {f1:.4f}")
    print(f"  Samples     : {len(labels):,}")
    print(f"{'='*45}\n")

    return results


def predict_single(image_tensor: torch.Tensor, checkpoint_name: str) -> dict:
    """
    Run inference on a single pre-processed image tensor.

    Parameters
    ----------
    image_tensor    : torch.Tensor  shape (1, C, H, W), normalised
    checkpoint_name : str           e.g. 'optimized_model' or 'baseline_model'

    Returns
    -------
    dict: predicted_class (int), confidence (float), class_name (str)
    """
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pth"
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)
    model     = build_model(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        out   = model(image_tensor.to(DEVICE))
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(dim=1)

    return {
        "predicted_class": int(pred.item()),
        "class_name":      CIFAR10_CLASSES[pred.item()],
        "confidence":      round(conf.item(), 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the test set.")
    parser.add_argument("--checkpoint", default="baseline_model",
                        help="Checkpoint name without .pth (default: baseline_model)")
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "mnist"],
                        help="Dataset to evaluate on (default: cifar10)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_json", default=None,
                        help="Optional path to save results as JSON")
    args = parser.parse_args()

    results = evaluate(args.checkpoint, args.dataset, args.batch_size)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save_json}")

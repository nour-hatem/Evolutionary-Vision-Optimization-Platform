"""
scripts/retrain_best.py
-----------------------
Re-train with Ahmad's best GA config and update metrics.json.

Run after updating best_config.json with Ahmad's values:
    python -m scripts.retrain_best

Saves:
  - model/checkpoints/optimized_model.pth
  - metrics.json  (updated with both baseline AND optimized results)

Owner: Yassin
"""

import json
from pathlib import Path

from model.train import train

ROOT             = Path(__file__).parent.parent
BEST_CONFIG_PATH = ROOT / "best_config.json"       # at project root
METRICS_PATH     = ROOT / "metrics.json"


if __name__ == "__main__":

    # ── load Ahmad's best config ───────────────────────────────────────────────
    if not BEST_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"best_config.json not found at {BEST_CONFIG_PATH}\n"
            "Make sure best_config.json is in the root of the project."
        )

    with open(BEST_CONFIG_PATH) as f:
        raw = json.load(f)

    # handle both flat config and nested {"best_config": {...}} formats
    best_config = raw.get("best_config", raw)

    # make sure required fields are present
    best_config.setdefault("dataset",        "cifar10")
    best_config.setdefault("num_classes",    10)
    best_config.setdefault("input_channels", 3)
    best_config.setdefault("weight_decay",   1e-4)
    best_config["epochs"] = 20     # full training — more than GA search epochs

    print("Re-training with Ahmad's best config:")
    print(json.dumps(best_config, indent=2))
    print()

    # ── train optimized model ──────────────────────────────────────────────────
    optimized_metrics = train(
        config=best_config,
        fast_mode=False,
        save_checkpoint="optimized_model",
        verbose=True,
    )

    # ── load existing baseline metrics ─────────────────────────────────────────
    combined = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            combined = json.load(f)

        # if metrics.json still has flat keys from run_baseline, rename them
        if "accuracy" in combined:
            combined["baseline_accuracy"]   = combined.pop("accuracy")
            combined["baseline_f1"]         = combined.pop("f1",          None)
            combined["baseline_train_loss"] = combined.pop("train_loss",  None)
            combined["baseline_val_loss"]   = combined.pop("val_loss",    None)

    # ── add optimized results ──────────────────────────────────────────────────
    combined["optimized_accuracy"]   = optimized_metrics["accuracy"]
    combined["optimized_f1"]         = optimized_metrics["f1"]
    combined["optimized_train_loss"] = optimized_metrics["train_loss"]
    combined["optimized_val_loss"]   = optimized_metrics["val_loss"]
    combined["optimized_config"]     = best_config
    combined["optimized_time_s"]     = optimized_metrics["training_time_seconds"]

    with open(METRICS_PATH, "w") as f:
        json.dump(combined, f, indent=2)

    # ── print comparison ───────────────────────────────────────────────────────
    baseline_acc  = combined.get("baseline_accuracy", "N/A")
    optimized_acc = combined["optimized_accuracy"]
    print(f"\nmetrics.json updated → {METRICS_PATH}")
    print(f"\n{'='*45}")
    print(f"  Baseline  accuracy : {baseline_acc}")
    print(f"  Optimized accuracy : {optimized_acc}")
    if isinstance(baseline_acc, float):
        delta = round(optimized_acc - baseline_acc, 4)
        sign  = "+" if delta > 0 else ""
        print(f"  Improvement        : {sign}{delta}")
    print(f"{'='*45}")
    print(f"\nFiles saved:")
    print(f"  model/checkpoints/optimized_model.pth")
    print(f"  metrics.json")
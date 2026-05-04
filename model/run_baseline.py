import json
from pathlib import Path

from model.train import train

# ── baseline configuration ────────────────────────────────────────────────────
# These are the fixed defaults — NOT optimised by GA
BASELINE_CONFIG = {
    "dataset":        "cifar10",  
    "lr":             0.0005,
    "num_filters":    64,
    "num_layers":     3,
    "dropout":        0.5,
    "batch_size":     64,
    "epochs":         20,
    "num_classes":    10,
    "input_channels": 3,       
    "weight_decay":   1e-4,
}

if __name__ == "__main__":
    print("Training baseline CNN ...")
    metrics = train(
        config=BASELINE_CONFIG,
        fast_mode=False,
        save_checkpoint="baseline_model",
        verbose=True,
    )

    # save metrics.json at repo root for Nour + Mahmoud
    metrics_path = Path(__file__).parent.parent / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"metrics.json saved → {metrics_path}")
    print(f"\nBaseline results:")
    print(f"  Accuracy : {metrics['accuracy']}")
    print(f"  F1       : {metrics['f1']}")
    print(f"  Time     : {metrics['training_time_seconds']}s")

    if metrics["accuracy"] < 0.60:
        print("\n[WARNING] Accuracy below 60%. Try increasing num_filters to 64 "
              "or num_layers to 3 in BASELINE_CONFIG.")
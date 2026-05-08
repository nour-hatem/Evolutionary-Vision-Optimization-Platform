"""
metrics.py — Metric computation functions for CNN evaluation.
"""
import time
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_accuracy(labels, preds):
    return round(accuracy_score(labels, preds), 4)

def compute_f1(labels, preds, average="macro"):
    return round(f1_score(labels, preds, average=average), 4)

def compute_confusion_matrix(labels, preds):
    return confusion_matrix(labels, preds)

def compute_per_class_accuracy(cm):
    return (cm.diagonal() / cm.sum(axis=1)).round(4)

def evaluate(model, loader, device, name, pth_path):
    model.eval()
    preds, labels = [], []
    t0 = time.perf_counter()
    with __import__("torch").no_grad():
        for imgs, lbls in loader:
            preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())
    elapsed_ms = (time.perf_counter() - t0) / len(loader) * 1000

    preds, labels = np.array(preds), np.array(labels)
    cm      = compute_confusion_matrix(labels, preds)
    per_cls = compute_per_class_accuracy(cm)

    metrics = {
        "model":                  name,
        "accuracy":               compute_accuracy(labels, preds),
        "f1_macro":               compute_f1(labels, preds, average="macro"),
        "f1_weighted":            compute_f1(labels, preds, average="weighted"),
        "params":                 sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_ms_per_batch": round(elapsed_ms, 2),
        "file_size_mb":           round(os.path.getsize(pth_path) / 1e6, 2),
        "num_filters":            model.num_filters,
        "num_layers":             model.num_layers,
    }
    return metrics, preds, labels, cm, per_cls

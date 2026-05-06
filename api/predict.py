"""
api/predict.py
--------------
Model loading and single-image inference for the CIFAR-10 CNN.

Model search order (project root, then checkpoints directory):
  1. optimized_model.pth
  2. best_model.pth
  3. Most recently modified *.pth inside model/checkpoints/

Checkpoint format written by model/train.py:
  {
      "model_state_dict": <OrderedDict>,
      "config":           <dict>,   # contains num_filters, num_layers, dropout …
      "metrics":          <dict>,
  }
Raw state-dicts (no wrapper dict) are also accepted as a fallback.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# ── project-level imports ──────────────────────────────────────────────────────
# Resolve the project root so imports work regardless of where uvicorn is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.cnn import build_model  # noqa: E402

logger = logging.getLogger(__name__)

# ── CIFAR-10 constants ─────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Exact per-channel statistics computed over the full CIFAR-10 training set
# (values sourced from data/dataset_stats.json).
_CIFAR10_MEAN = [0.4913996160030365, 0.482158362865448,  0.44653090834617615]
_CIFAR10_STD  = [0.2470323145389557, 0.24348513782024384, 0.26158788800239563]

# ── preprocessing pipeline (mirrors data/transforms.py → test() branch) ───────
_INFERENCE_TRANSFORM = T.Compose([
    T.Resize((32, 32)),     # normalise spatial dimensions to CIFAR-10 format
    T.ToTensor(),           # [0,255] uint8  →  [0.0,1.0] float32, shape (C,H,W)
    T.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
])

# ── model singleton ────────────────────────────────────────────────────────────
_model: torch.nn.Module | None = None
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_checkpoint() -> Path:
    """Return the path of the first available model checkpoint."""
    candidates = [
        PROJECT_ROOT / "optimized_model.pth",
        PROJECT_ROOT / "best_model.pth",
    ]

    checkpoints_dir = PROJECT_ROOT / "model" / "checkpoints"
    if checkpoints_dir.is_dir():
        pth_files = sorted(
            checkpoints_dir.glob("*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(pth_files)

    for path in candidates:
        if path.exists():
            logger.info("Found model checkpoint: %s", path)
            return path

    raise FileNotFoundError(
        "No model checkpoint found. Expected one of: optimized_model.pth, "
        "best_model.pth, or any *.pth inside model/checkpoints/. "
        "Train the model first (e.g. python -m model.run_baseline)."
    )


def _load_best_config() -> dict:
    """Load architecture params from best_config.json as a fallback."""
    config_path = PROJECT_ROOT / "best_config.json"
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        return data.get("best_config", {})
    return {}


def load_model() -> torch.nn.Module:
    """
    Load the trained CNN into memory (called once at startup).

    Returns the model in eval mode on the appropriate device.
    """
    global _model

    if _model is not None:
        return _model

    ckpt_path = _find_checkpoint()
    payload = torch.load(ckpt_path, map_location=_device)

    # Determine whether the file is a wrapped checkpoint or a raw state-dict.
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        config = payload.get("config", {})
        logger.info("Loaded wrapped checkpoint (config: %s)", config)
    else:
        # Assume raw state-dict saved with torch.save(model.state_dict(), …)
        state_dict = payload
        config = {}
        logger.warning(
            "Checkpoint has no 'model_state_dict' key; treating as raw state-dict. "
            "Architecture will be inferred from best_config.json."
        )

    # Fall back to best_config.json for any missing architecture params.
    fallback = _load_best_config()
    merged_config = {**fallback, **config}

    model = build_model({
        "num_filters":   int(merged_config.get("num_filters",   32)),
        "num_layers":    int(merged_config.get("num_layers",    2)),
        "num_classes":   int(merged_config.get("num_classes",   10)),
        "dropout":       float(merged_config.get("dropout",     0.5)),
        "input_channels": int(merged_config.get("input_channels", 3)),
    })

    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()

    _model = model
    logger.info(
        "Model loaded on %s — filters=%s layers=%s classes=%s",
        _device,
        merged_config.get("num_filters"),
        merged_config.get("num_layers"),
        merged_config.get("num_classes"),
    )
    return _model


# ── public inference API ───────────────────────────────────────────────────────

def predict_single(image: Union[Image.Image, bytes]) -> dict:
    """
    Run inference on a single image.

    Parameters
    ----------
    image : PIL.Image.Image or bytes
        The input image in any format supported by Pillow.

    Returns
    -------
    dict
        {
            "prediction": str,   # CIFAR-10 class label
            "confidence": float, # softmax probability of the top class (0-1)
        }
    """
    # Accept raw bytes or a PIL Image
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Preprocess: resize → tensor → normalise → add batch dim → (1, 3, 32, 32)
    tensor: torch.Tensor = _INFERENCE_TRANSFORM(image)
    tensor = tensor.unsqueeze(0).to(_device)          # (1, C, H, W)

    model = load_model()

    with torch.no_grad():
        logits = model(tensor)                         # (1, num_classes)
        probs  = F.softmax(logits, dim=1)              # (1, num_classes)

    confidence, class_idx = probs.max(dim=1)
    class_idx  = class_idx.item()
    confidence = round(float(confidence.item()), 4)

    label = (
        CIFAR10_CLASSES[class_idx]
        if class_idx < len(CIFAR10_CLASSES)
        else str(class_idx)
    )

    return {"prediction": label, "confidence": confidence}

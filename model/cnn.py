"""
model/cnn.py
------------
SimpleCNN: a configurable convolutional neural network for image classification.

Config parameters (all injectable so Ahmad's GA can tune them):
  - num_filters  : int  — number of filters in the first conv block (default 32)
  - num_layers   : int  — number of conv+pool blocks (default 2, max recommended 3)
  - num_classes  : int  — output classes (10 for CIFAR-10 / MNIST)
  - dropout      : float — dropout rate before classifier (default 0.5)

Owner: Yassin
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_filters: int = 32,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.5,
        input_channels: int = 3,
    ):
        super().__init__()

        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_classes = num_classes

        # ── convolutional feature extractor ───────────────────────────────────
        layers = []
        in_ch = input_channels
        for i in range(num_layers):
            out_ch = num_filters * (2 ** i)   # 32 → 64 → 128 ...
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=0.1),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        # ── compute flattened size automatically ──────────────────────────────
        self._flat_size = self._compute_flat_size(input_channels)

        # ── fully connected classifier ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self._flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        # ── weight initialisation ─────────────────────────────────────────────
        self._init_weights()

    def _compute_flat_size(self, input_channels: int) -> int:
        """Run a dummy forward pass to get the flattened feature size."""
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 32, 32)
            out = self.features(dummy)
            return int(out.view(1, -1).shape[1])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(config: dict) -> nn.Module:
    """
    Build a SimpleCNN from a config dict.
    Called by model/train.py and Ahmad's fitness function.
    """
    return SimpleCNN(
        num_filters=int(config.get("num_filters", 32)),
        num_layers=int(config.get("num_layers", 2)),
        num_classes=int(config.get("num_classes", 10)),
        dropout=float(config.get("dropout", 0.5)),
        input_channels=int(config.get("input_channels", 3)),
    )
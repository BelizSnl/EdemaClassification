from __future__ import annotations
import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, in_features: int, n_classes: int, hidden: tuple[int, int] = (256, 128), p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden[0]),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden[1], n_classes) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

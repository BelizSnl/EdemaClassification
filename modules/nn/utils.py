#imports
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import random

#setze zufallsseed für reproduzierbarkeit
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(False)

#gerät bestimmen
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

#gerät information
def device_info(dev: torch.device) -> str:
    if dev.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "NVIDIA GPU"
        return f"cuda ({name})"
    if dev.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"

#erstelle dataloader aus numpy arrays
def make_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    if device is None:
        device = get_device()
    pin_memory = (device.type == "cuda")

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test,  dtype=torch.float32)
    yte = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
    )
    test_loader  = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )
    return train_loader, test_loader
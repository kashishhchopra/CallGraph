"""Torch device selection."""

from __future__ import annotations

import torch


def pick_device(pref: str = "auto") -> torch.device:
    """'auto' -> cuda, then mps, then cpu."""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

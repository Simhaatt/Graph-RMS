"""Tiny shared helper -- its own module so rms.py doesn't need to import
from a training module that no longer exists."""

from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

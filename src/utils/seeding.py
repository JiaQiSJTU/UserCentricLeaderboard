# encoding = "utf-8"

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # noqa: D401
    torch = None  # type: ignore

__all__ = ["set_global_seed"]


def set_global_seed(seed: int = 42, *, deterministic: bool = False, warn: bool = True):
    """Set random seeds for Python, NumPy, and Torch (if available).

    Args:
        seed: Integer random seed.
        deterministic: If True, enable deterministic algorithms for torch (may slow down performance).
        warn: Whether to print a warning if torch is not available and deterministic=True.
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        # multiple GPUs
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    elif deterministic and warn:
        print("[utils.seeding] Torch not available, cannot set deterministic algorithms.") 
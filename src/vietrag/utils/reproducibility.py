"""Reproducibility and run-environment metadata."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import sys
from typing import Any


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def collect_run_metadata(seed: int) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("PyYAML", "numpy", "sentence-transformers", "faiss-cpu", "streamlit"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "packages": packages,
    }

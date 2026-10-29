from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path.resolve())
    return config


def resolve_path(config: dict[str, Any], value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_dir = Path(config["_config_path"]).parent
    return (config_dir / path).resolve()


def set_reproducibility_seed(seed: int, framework: str | None = None) -> None:
    """Set deterministic seeds without initializing an unused GPU framework."""
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    selected = None if framework is None else framework.lower()

    if selected in {None, "tensorflow"}:
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except ImportError:
            pass

    if selected in {None, "torch"}:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

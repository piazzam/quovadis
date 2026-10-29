from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExplanationArtifact:
    heatmaps: np.ndarray
    metadata: dict[str, Any]


def _to_spatial_heatmaps(array: np.ndarray) -> np.ndarray:
    """Convert common saved layouts to ``(N, H, W)``."""
    array = np.asarray(array)
    if array.ndim == 3:
        return array.astype(np.float32, copy=False)
    if array.ndim == 4:
        if array.shape[1] == 1:
            return array[:, 0].astype(np.float32, copy=False)
        if array.shape[-1] == 1:
            return array[..., 0].astype(np.float32, copy=False)
        # Legacy output: N,K,H,W. The paper explains the first saved target.
        return array[:, 0].astype(np.float32, copy=False)
    if array.ndim == 5:
        # Legacy output: N,K,H,W,C. Select first target and first channel.
        return array[:, 0, :, :, 0].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported heatmap shape: {array.shape}")


def load_explanations(path: str | Path) -> ExplanationArtifact:
    """Load explanations from a method/shape directory or a ``.npy`` file.

    The expected native layout is::

        5_row/
            explanations.npy

    All saved explanations must follow the dataset order. Therefore no sample
    index file is needed: ``explanations[i]`` is evaluated on ``dataset[i]``.
    """
    requested = Path(path)
    if requested.is_file():
        if requested.suffix.lower() != ".npy":
            raise ValueError(f"Explanation file must be a .npy file: {requested}")
        root = requested.parent
        raw_heatmaps = np.load(requested, allow_pickle=False)
    else:
        root = requested
        native_path = root / "explanations.npy"
        alternate_path = root / "heatmaps.npy"
        if native_path.exists():
            raw_heatmaps = np.load(native_path, allow_pickle=False)
        elif alternate_path.exists():
            raw_heatmaps = np.load(alternate_path, allow_pickle=False)
        else:
            raise FileNotFoundError(
                f"Expected {native_path.name} (or {alternate_path.name}) in {root}"
            )

    metadata: dict[str, Any] = {}
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    return ExplanationArtifact(
        heatmaps=_to_spatial_heatmaps(raw_heatmaps),
        metadata=metadata,
    )


def save_explanations(
    path: str | Path,
    heatmaps: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Save one complete explanation set in dataset order."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)

    heatmaps = _to_spatial_heatmaps(np.asarray(heatmaps))
    np.save(root / "explanations.npy", heatmaps.astype(np.float32, copy=False))
    with (root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

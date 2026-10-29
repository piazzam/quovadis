from __future__ import annotations

import numpy as np


def contamination_aware(
    template_heatmaps: np.ndarray,
    partitions: np.ndarray,
) -> np.ndarray:
    """Configuration used in the experiments (Appendix D).

    template_heatmaps: (R,N,H,W)
    partitions: (R,H,W)
    """
    maps = np.asarray(template_heatmaps, dtype=np.float32)
    partitions = np.asarray(partitions)
    if maps.ndim != 4 or partitions.shape != (maps.shape[0], maps.shape[2], maps.shape[3]):
        raise ValueError("Expected heatmaps (R,N,H,W) and partitions (R,H,W)")

    corrected = np.empty_like(maps)
    for r in range(maps.shape[0]):
        others = np.delete(maps, r, axis=0)
        preliminary = np.median(others, axis=0)
        current = maps[r]
        current_corrected = current.copy()
        for patch_id in np.unique(partitions[r]):
            patch_mask = partitions[r] == patch_id
            patch_size = int(patch_mask.sum())
            if patch_size <= 1:
                continue
            patch_values = preliminary[:, patch_mask]
            patch_sums = patch_values.sum(axis=1, keepdims=True)
            neighbour_mean = (patch_sums - patch_values) / float(patch_size - 1)
            contamination = np.maximum(neighbour_mean, 0.0)
            original_values = current[:, patch_mask]
            corrected_positive = np.maximum(original_values - contamination, 0.0)
            current_corrected[:, patch_mask] = np.where(
                original_values > 0.0,
                corrected_positive,
                original_values,
            )
        corrected[r] = current_corrected

    return np.maximum(np.median(corrected, axis=0), 0.0)


def aggregate_template_heatmaps(
    template_heatmaps: np.ndarray,
    strategy: str,
    partitions: np.ndarray | None = None,
) -> np.ndarray:
    maps = np.asarray(template_heatmaps, dtype=np.float32)
    if maps.ndim != 4:
        raise ValueError(f"Expected template heatmaps with shape (R,N,H,W), got {maps.shape}")
    strategy = strategy.lower()
    if strategy == "median":
        return np.median(maps, axis=0).astype(np.float32)
    if strategy in {"q25", "25th_quantile", "percentile_25"}:
        return np.percentile(maps, 25, axis=0).astype(np.float32)
    if strategy in {"contamination", "contamination_aware"}:
        if partitions is None:
            raise ValueError("partitions are required for contamination-aware aggregation")
        return contamination_aware(maps, partitions).astype(np.float32)
    raise ValueError(f"Unknown aggregation strategy: {strategy}")

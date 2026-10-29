"""Minimal SHAP-BPT integration used by the thesis experiments.

This module intentionally contains only the adapter code required by this
repository. It does not depend on the private ``xai_mind`` project or on the
legacy ``qg_paper`` source tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .adapters import ModelAdapter
from .masking import to_channel_last


def _ensure_numpy_bool_alias() -> None:
    # shap-bpt 1.0.1 still references np.bool, removed in NumPy 1.24.
    if "bool" not in np.__dict__:
        np.bool = bool  # type: ignore[attr-defined]


def sample_to_uint8_rgb(sample: np.ndarray, channel_first: bool) -> np.ndarray:
    image = to_channel_last(np.asarray(sample), channel_first)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[-1] not in (1, 3):
        raise ValueError(f"SHAP-BPT expects an image with 1 or 3 channels, got {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    if np.issubdtype(image.dtype, np.integer):
        return np.clip(image, 0, 255).astype(np.uint8)

    image_float = image.astype(np.float32)
    finite = image_float[np.isfinite(image_float)]
    if finite.size == 0:
        raise ValueError("Cannot build a SHAP-BPT partition tree from a non-finite image")
    lo, hi = float(finite.min()), float(finite.max())
    if lo >= 0.0 and hi <= 1.0:
        scaled = image_float * 255.0
    else:
        # The tree only needs an RGB image representation. Model predictions
        # below still use the original preprocessed tensor values.
        scaled = image_float - lo
        if hi > lo:
            scaled = scaled / (hi - lo)
        scaled = scaled * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _mask_to_spatial(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    masks = np.asarray(masks).astype(bool)
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[-1] in (1, 3):
        masks = masks[..., 0]
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError(
            f"Unsupported SHAP-BPT mask shape {masks.shape}; expected (*,{height},{width})"
        )
    return masks


def make_fixed_target_predict_fn(
    adapter: ModelAdapter,
    sample: np.ndarray,
    target_class: int,
    baseline_value: float = 0.0,
):
    sample_np = np.asarray(sample, dtype=np.float32)
    sample_hwc = to_channel_last(sample_np, adapter.channel_first)
    if sample_hwc.ndim == 2:
        sample_hwc = sample_hwc[..., None]
    height, width = sample_hwc.shape[:2]

    def predict_fn(masks: np.ndarray) -> np.ndarray:
        spatial_masks = _mask_to_spatial(masks, height, width)
        batch_hwc = np.repeat(sample_hwc[None, ...], len(spatial_masks), axis=0)
        batch_hwc = np.where(
            spatial_masks[..., None],
            batch_hwc,
            np.asarray(baseline_value, dtype=np.float32),
        ).astype(np.float32)
        if adapter.channel_first:
            batch = np.transpose(batch_hwc, (0, 3, 1, 2))
        else:
            batch = batch_hwc
        probabilities = adapter.predict_proba(batch)
        return probabilities[:, [int(target_class)]]

    return predict_fn


def output_to_map(values: Any) -> np.ndarray:
    if hasattr(values, "values"):
        values = values.values
    elif isinstance(values, dict):
        for key in ("values", "shap_values", "owen_values"):
            if key in values:
                values = values[key]
                break
    array = np.asarray(values)
    array = np.squeeze(array)
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    if array.ndim == 3:
        if array.shape[0] == 1:
            return array[0].astype(np.float32, copy=False)
        if array.shape[-1] == 1:
            return array[..., 0].astype(np.float32, copy=False)
        if array.shape[-1] in (1, 3):
            return np.mean(array, axis=-1).astype(np.float32, copy=False)
    if array.ndim == 4 and array.shape[0] == 1:
        return output_to_map(array[0])
    raise ValueError(f"Cannot convert SHAP-BPT output to one spatial map: {array.shape}")


@dataclass
class ShapBPTExplainer:
    adapter: ModelAdapter
    settings: dict[str, Any]

    def explain(self, images: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        _ensure_numpy_bool_alias()
        try:
            import shap_bpt
        except ImportError as exc:
            raise ImportError(
                "SHAP-BPT is not installed. Install the repository with the "
                "'shap-bpt' extra or run `pip install shap-bpt==1.0.1`."
            ) from exc

        max_evals = int(self.settings.get("max_evals", 254))
        baseline = float(self.settings.get("baseline", 0.0))
        method = str(self.settings.get("method", "BPT"))
        batch_size = int(self.settings.get("batch_size", 64))
        balance_area = bool(self.settings.get("balance_area", False))
        min_area = int(self.settings.get("min_area", 1))
        max_weight = self.settings.get("max_weight")
        verbose = bool(self.settings.get("verbose", False))

        maps: list[np.ndarray] = []
        for sample, target in zip(images, target_classes):
            tree_image = sample_to_uint8_rgb(sample, self.adapter.channel_first)
            predict_fn = make_fixed_target_predict_fn(
                self.adapter,
                sample,
                int(target),
                baseline_value=baseline,
            )
            try:
                explainer = shap_bpt.Explainer(
                    predict_fn,
                    tree_image,
                    num_explained_classes=1,
                    balance_area=balance_area,
                    verbose=verbose,
                )
            except TypeError:
                explainer = shap_bpt.Explainer(
                    predict_fn,
                    tree_image,
                    num_explained_classes=1,
                )
            try:
                values = explainer.explain_instance(
                    max_evals=max_evals,
                    method=method,
                    batch_size=batch_size,
                    min_area=min_area,
                    max_weight=max_weight,
                )
            except TypeError:
                # Compatibility with the public 1.0.1 API.
                try:
                    values = explainer.explain_instance(
                        max_evals=max_evals,
                        method=method,
                        batch_size=batch_size,
                    )
                except TypeError:
                    values = explainer.explain_instance(max_evals=max_evals)
            maps.append(output_to_map(values))
        return np.stack(maps, axis=0).astype(np.float32, copy=False)

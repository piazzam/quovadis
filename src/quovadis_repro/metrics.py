from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .adapters import ModelAdapter
from .masking import from_channel_last, to_channel_last


def _rank_map(heatmap: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(heatmap).reshape(-1))[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks.reshape(heatmap.shape)


def _selected_mask(heatmap: np.ndarray, fraction: float, direction: str) -> np.ndarray:
    flat = np.asarray(heatmap).reshape(-1)
    count = max(1, int(np.ceil(fraction * len(flat))))
    order = np.argsort(flat)
    direction = direction.lower()
    if direction not in {"morf", "lerf"}:
        raise ValueError(f"ROAD direction must be 'morf' or 'lerf', got {direction!r}")
    selected = order[-count:] if direction == "morf" else order[:count]
    mask = np.zeros(len(flat), dtype=bool)
    mask[selected] = True
    return mask.reshape(heatmap.shape)


def road_noisy_linear_imputation(
    image: np.ndarray,
    mask: np.ndarray,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[..., None]
        squeeze = True
    else:
        squeeze = False
    height, width, channels = image.shape
    target_y, target_x = np.where(mask)
    count = len(target_y)
    if count == 0:
        return image[..., 0] if squeeze else image.copy()

    pixel_to_index = -np.ones((height, width), dtype=np.int64)
    pixel_to_index[target_y, target_x] = np.arange(count)
    output = image.copy()
    direct, diagonal = 1.0 / 6.0, 1.0 / 12.0
    neighbours = (
        (-1, 0, direct), (1, 0, direct), (0, -1, direct), (0, 1, direct),
        (-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal),
    )

    for channel in range(channels):
        matrix = lil_matrix((count, count))
        rhs = np.zeros(count, dtype=np.float64)
        for row, (y, x) in enumerate(zip(target_y, target_x)):
            matrix[row, row] = 1.0
            for dy, dx, weight in neighbours:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                neighbour_index = pixel_to_index[ny, nx]
                if neighbour_index >= 0:
                    matrix[row, neighbour_index] -= weight
                else:
                    rhs[row] += weight * image[ny, nx, channel]
        values = spsolve(matrix.tocsr(), rhs)
        values += rng.normal(0.0, noise_sigma, size=count)
        output[target_y, target_x, channel] = np.clip(values, image.min(), image.max())

    return output[..., 0] if squeeze else output


def road(
    adapter: ModelAdapter,
    images: np.ndarray,
    labels: np.ndarray,
    heatmaps: np.ndarray,
    percentages: list[float],
    direction: str = "morf",
    noise_sigma: float = 0.01,
    seed: int = 42,
    perturbation_batch_size: int | None = None,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    batch_size = int(perturbation_batch_size or adapter.batch_size)
    accuracies: list[float] = []
    correctness: list[np.ndarray] = []
    labels = np.asarray(labels).reshape(-1)
    for percentage in percentages:
        prediction_parts: list[np.ndarray] = []
        for start_index in range(0, len(images), batch_size):
            end_index = min(start_index + batch_size, len(images))
            perturbed = []
            for image, heatmap in zip(
                images[start_index:end_index],
                heatmaps[start_index:end_index],
            ):
                image_cl = to_channel_last(image, adapter.channel_first)
                mask = _selected_mask(heatmap, percentage, direction)
                modified = road_noisy_linear_imputation(image_cl, mask, noise_sigma, rng)
                perturbed.append(from_channel_last(modified, adapter.channel_first))
            probabilities = adapter.predict_proba(np.asarray(perturbed, dtype=np.float32))
            prediction_parts.append(np.argmax(probabilities, axis=1))
        predictions = np.concatenate(prediction_parts)
        correct = predictions == labels
        correctness.append(correct)
        accuracies.append(accuracy_score(labels, predictions))
    x = np.asarray(percentages, dtype=np.float64)
    y = np.asarray(accuracies, dtype=np.float64)
    auc = float(np.trapz(y, x))
    return {
        "percentages": x,
        "accuracies": y,
        "correctness": np.asarray(correctness),
        "auc": auc,
    }


def saco(
    adapter: ModelAdapter,
    images: np.ndarray,
    heatmaps: np.ndarray,
    target_classes: np.ndarray,
    groups: int = 10,
) -> np.ndarray:
    original_probabilities = adapter.predict_proba(images)
    scores = np.zeros(len(images), dtype=np.float64)
    for sample_index in tqdm(range(len(images)), desc="SaCo", unit="img"):
        image = images[sample_index]
        image_cl = to_channel_last(image, adapter.channel_first)
        heatmap = heatmaps[sample_index]
        target = int(target_classes[sample_index])
        rank = _rank_map(heatmap)
        pixels = heatmap.size
        group_scores: list[float] = []
        perturbed: list[np.ndarray] = []
        for group in range(groups):
            lower = group * pixels / groups
            upper = (group + 1) * pixels / groups
            mask = (rank >= lower) & (rank < upper)
            group_scores.append(float(heatmap[mask].sum()))
            modified = image_cl.copy()
            if modified.ndim == 2:
                modified[mask] = modified.mean()
            else:
                modified[mask] = modified.mean(axis=(0, 1))
            perturbed.append(from_channel_last(modified, adapter.channel_first))
        perturbed_probabilities = adapter.predict_proba(np.asarray(perturbed))[:, target]
        drops = original_probabilities[sample_index, target] - perturbed_probabilities
        numerator = 0.0
        denominator = 0.0
        for first in range(groups - 1):
            for second in range(first + 1, groups):
                saliency_difference = group_scores[first] - group_scores[second]
                numerator += saliency_difference if drops[first] >= drops[second] else -saliency_difference
                denominator += abs(saliency_difference)
        scores[sample_index] = numerator / denominator if denominator else 0.0
    return scores


def _all_square_masks(height: int, width: int, sizes: list[int]) -> np.ndarray:
    masks = []
    for size in sizes:
        size = min(int(size), height, width)
        for top in range(height - size + 1):
            for left in range(width - size + 1):
                mask = np.zeros((height, width), dtype=bool)
                mask[top : top + size, left : left + size] = True
                masks.append(mask)
    return np.asarray(masks)


def _sample_square_masks(
    height: int,
    width: int,
    sizes: list[int],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    normalized_sizes = [min(int(size), height, width) for size in sizes]
    counts = np.asarray(
        [(height - size + 1) * (width - size + 1) for size in normalized_sizes],
        dtype=np.int64,
    )
    cumulative = np.cumsum(counts)
    draws = rng.integers(0, int(cumulative[-1]), size=count)
    masks = np.zeros((count, height, width), dtype=bool)
    for index, draw in enumerate(draws):
        size_index = int(np.searchsorted(cumulative, draw, side="right"))
        previous = 0 if size_index == 0 else int(cumulative[size_index - 1])
        local = int(draw) - previous
        size = normalized_sizes[size_index]
        positions_per_row = width - size + 1
        top, left = divmod(local, positions_per_row)
        masks[index, top : top + size, left : left + size] = True
    return masks


def infidelity(
    adapter: ModelAdapter,
    images: np.ndarray,
    heatmaps: np.ndarray,
    target_classes: np.ndarray,
    patch_sizes: list[int],
    num_samples: int = 1000,
    all_patches: bool = False,
    apply_beta: bool = True,
    seed: int = 42,
    perturbation_batch_size: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = np.zeros(len(images), dtype=np.float64)
    batch_size = int(perturbation_batch_size or adapter.batch_size)
    for sample_index in tqdm(range(len(images)), desc="Infidelity", unit="img"):
        image_cl = to_channel_last(images[sample_index], adapter.channel_first)
        height, width = image_cl.shape[:2]
        masks = (
            _all_square_masks(height, width, patch_sizes)
            if all_patches
            else _sample_square_masks(height, width, patch_sizes, num_samples, rng)
        )
        target = int(target_classes[sample_index])
        original_probability = adapter.predict_proba(images[sample_index : sample_index + 1])[0, target]
        if image_cl.ndim == 3:
            removed_signal = image_cl.sum(axis=-1)
        else:
            removed_signal = image_cl
        weighted_signal = removed_signal * heatmaps[sample_index]

        delta_parts: list[np.ndarray] = []
        approximation_parts: list[np.ndarray] = []
        for start_index in range(0, len(masks), batch_size):
            mask_batch = masks[start_index : start_index + batch_size]
            if image_cl.ndim == 3:
                modified = np.where(mask_batch[..., None], 0.0, image_cl[None, ...])
            else:
                modified = np.where(mask_batch, 0.0, image_cl[None, ...])
            if adapter.channel_first and modified.ndim == 4:
                model_inputs = np.transpose(modified, (0, 3, 1, 2))
            else:
                model_inputs = modified
            perturbed_probabilities = adapter.predict_proba(model_inputs)[:, target]
            delta_parts.append(original_probability - perturbed_probabilities)
            approximation_parts.append(
                np.sum(mask_batch * weighted_signal[None, ...], axis=(1, 2))
            )

        delta_output = np.concatenate(delta_parts).astype(np.float64, copy=False)
        approximation = np.concatenate(approximation_parts).astype(np.float64, copy=False)
        if apply_beta:
            beta = np.mean(delta_output * approximation) / (np.mean(approximation**2) + 1e-12)
            approximation = beta * approximation
        scores[sample_index] = np.mean((approximation - delta_output) ** 2)
    return scores


def sensitivity(
    images: np.ndarray,
    heatmaps: np.ndarray,
    target_classes: np.ndarray,
    explain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    epsilon: float = 0.1,
    n_perturbations: int = 50,
    clip_values: tuple[float, float] | None = (0.0, 1.0),
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = np.zeros(len(images), dtype=np.float64)
    for index in tqdm(range(len(images)), desc="Sensitivity", unit="img"):
        original = heatmaps[index].astype(np.float64)
        denominator = np.linalg.norm(original.reshape(-1), ord=2)
        noise = rng.uniform(
            -epsilon,
            epsilon,
            size=(n_perturbations,) + images[index].shape,
        ).astype(images.dtype)
        perturbed = images[index][None, ...] + noise
        if clip_values is not None:
            perturbed = np.clip(perturbed, clip_values[0], clip_values[1])
        targets = np.full(n_perturbations, int(target_classes[index]), dtype=np.int64)
        perturbed_maps = np.asarray(explain_fn(perturbed, targets), dtype=np.float64)
        differences = perturbed_maps.reshape(n_perturbations, -1) - original.reshape(1, -1)
        relative = np.linalg.norm(differences, axis=1) / (denominator + 1e-12)
        scores[index] = relative.max()
    return scores


def complexity(heatmaps: np.ndarray) -> np.ndarray:
    maps = np.abs(np.asarray(heatmaps, dtype=np.float64)).reshape(len(heatmaps), -1)
    denominator = maps.sum(axis=1, keepdims=True)
    probabilities = np.divide(maps, denominator, out=np.zeros_like(maps), where=denominator > 0)
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -np.sum(terms, axis=1)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": int(len(values)),
    }

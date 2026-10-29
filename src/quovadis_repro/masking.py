from __future__ import annotations

import numpy as np


REMOVAL_ALIASES = {
    "average_blur": "average_blur",
    "avg": "average_blur",
    "avg_blur": "average_blur",
    "average": "average_blur",
    "gaussian_blur": "gaussian_blur",
    "gaussian": "gaussian_blur",
    "black": "black",
    "constant_black": "black",
}


def canonical_removal_strategy(name: str) -> str:
    key = str(name).strip().lower()
    if key not in REMOVAL_ALIASES:
        raise ValueError(
            f"Unknown feature-removal strategy '{name}'. "
            f"Supported values: {sorted(set(REMOVAL_ALIASES.values()))}"
        )
    return REMOVAL_ALIASES[key]


def to_channel_last(image: np.ndarray, channel_first: bool) -> np.ndarray:
    image = np.asarray(image)
    if channel_first and image.ndim == 3:
        return np.transpose(image, (1, 2, 0))
    return image


def from_channel_last(image: np.ndarray, channel_first: bool) -> np.ndarray:
    image = np.asarray(image)
    if channel_first and image.ndim == 3:
        return np.transpose(image, (2, 0, 1))
    return image


def average_blur(image: np.ndarray, kernel_size: tuple[int, int] = (11, 11)) -> np.ndarray:
    """Average blur used in the QUO VADIS experiments."""
    import cv2

    image = np.asarray(image, dtype=np.float32)
    blurred = cv2.blur(image, kernel_size)
    if image.ndim == 3 and blurred.ndim == 2:
        blurred = blurred[..., None]
    return blurred.astype(np.float32, copy=False)


def gaussian_blur(image: np.ndarray, kernel_size: tuple[int, int] = (11, 11)) -> np.ndarray:
    """Gaussian blur used in the thesis ablation (OpenCV sigma=0 auto selection)."""
    import cv2

    image = np.asarray(image, dtype=np.float32)
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    if image.ndim == 3 and blurred.ndim == 2:
        blurred = blurred[..., None]
    return blurred.astype(np.float32, copy=False)


def replacement_image(
    image: np.ndarray,
    strategy: str,
    kernel_size: tuple[int, int] = (11, 11),
    black_value: float = 0.0,
) -> np.ndarray:
    strategy = canonical_removal_strategy(strategy)
    if strategy == "average_blur":
        return average_blur(image, kernel_size)
    if strategy == "gaussian_blur":
        return gaussian_blur(image, kernel_size)
    if strategy == "black":
        return np.full_like(np.asarray(image, dtype=np.float32), float(black_value), dtype=np.float32)
    raise AssertionError(strategy)


def apply_coalition_masks(
    image: np.ndarray,
    coalition_masks: np.ndarray,
    partition: np.ndarray,
    kernel_size: tuple[int, int] = (11, 11),
    channel_first: bool = False,
    strategy: str = "average_blur",
    black_value: float = 0.0,
) -> np.ndarray:
    """Create all masked inputs for one quotient game.

    Parameters
    ----------
    coalition_masks:
        Array with shape ``(n_coalitions, n_players)``. A one means that the
        corresponding patch is retained.
    strategy:
        ``average_blur``, ``gaussian_blur`` or ``black``. These are the three
        feature-removal operators evaluated in the thesis chapter.
    """
    image_cl = to_channel_last(image, channel_first).astype(np.float32, copy=False)
    replacement = replacement_image(
        image_cl,
        strategy=strategy,
        kernel_size=kernel_size,
        black_value=black_value,
    )
    pixel_masks = np.asarray(coalition_masks, dtype=np.float32)[:, partition]
    if image_cl.ndim == 3:
        pixel_masks = pixel_masks[..., None]
    masked = pixel_masks * image_cl[None, ...] + (1.0 - pixel_masks) * replacement[None, ...]
    if channel_first and masked.ndim == 4:
        masked = np.transpose(masked, (0, 3, 1, 2))
    return masked.astype(np.float32, copy=False)

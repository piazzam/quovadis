from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path


@dataclass(frozen=True)
class DatasetBundle:
    images: np.ndarray
    labels: np.ndarray


def _truncate(bundle: DatasetBundle, limit: int | None) -> DatasetBundle:
    if limit is None:
        return bundle
    if limit < 1:
        raise ValueError("limit must be positive")
    if limit > len(bundle.images):
        raise ValueError(
            f"Requested {limit} samples, but the dataset contains only {len(bundle.images)}"
        )
    return DatasetBundle(
        images=np.asarray(bundle.images[:limit]),
        labels=np.asarray(bundle.labels[:limit]),
    )


def _load_from_numpy(config: dict[str, Any], limit: int | None) -> DatasetBundle:
    data_cfg = config["data"]
    images_path = resolve_path(config, data_cfg["images_npy"])
    labels_path = resolve_path(config, data_cfg["labels_npy"])
    assert images_path is not None and labels_path is not None
    images = np.load(images_path, mmap_mode=data_cfg.get("mmap_mode", "r"))
    labels = np.load(labels_path, allow_pickle=False).reshape(-1)
    return _truncate(DatasetBundle(images=images, labels=labels), limit)


def _load_keras_dataset(name: str, limit: int | None) -> DatasetBundle:
    import tensorflow as tf

    if name == "mnist":
        (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
        images = images[..., None]
    elif name == "fashion_mnist":
        (_, _), (images, labels) = tf.keras.datasets.fashion_mnist.load_data()
        images = images[..., None]
    elif name == "cifar10":
        (_, _), (images, labels) = tf.keras.datasets.cifar10.load_data()
        labels = labels.reshape(-1)
    else:
        raise ValueError(name)
    images = images.astype(np.float32) / 255.0
    bundle = DatasetBundle(images=images, labels=labels.astype(np.int64))
    return _truncate(bundle, limit)


def _timm_transform(model_name: str, num_classes: int, requires_detectors: bool):
    import timm

    if requires_detectors:
        import detectors  # noqa: F401
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    try:
        data_config = timm.data.resolve_model_data_config(model)
    except AttributeError:
        data_config = timm.data.resolve_data_config({}, model=model)
    return timm.data.create_transform(**data_config, is_training=False)


def _load_huggingface(config: dict[str, Any], limit: int | None) -> DatasetBundle:
    from datasets import load_dataset
    from PIL import Image

    data_cfg = config["data"]
    model_cfg = config["model"]
    transform = _timm_transform(
        model_cfg["name"],
        int(model_cfg["num_classes"]),
        bool(model_cfg.get("requires_detectors", False)),
    )

    if data_cfg.get("parquet"):
        parquet = resolve_path(config, data_cfg["parquet"])
        assert parquet is not None
        if not parquet.is_file():
            dataset_name = str(data_cfg.get("name", "")).lower()
            if dataset_name == "imagenet":
                raise FileNotFoundError(
                    f"ImageNet evaluation Parquet not found: {parquet}. "
                    "Reconstruct it first with "
                    "`python scripts/create_imagenet_sample.py`."
                )
            raise FileNotFoundError(f"Dataset Parquet not found: {parquet}")
        dataset = load_dataset(
            "parquet",
            data_files=str(parquet),
            split=data_cfg.get("split", "train"),
        )
    else:
        dataset = load_dataset(
            data_cfg["dataset_name"],
            split=data_cfg.get("split", "test"),
        )

    count = len(dataset) if limit is None else int(limit)
    if count > len(dataset):
        raise ValueError(
            f"Requested {count} samples, but the dataset contains only {len(dataset)}"
        )

    images: list[np.ndarray] = []
    labels: list[int] = []
    for idx in range(count):
        example = dataset[idx]
        image = example.get("image", example.get("img"))
        label = example.get("label", example.get("fine_label"))
        if image is None or label is None:
            raise KeyError(f"Unsupported dataset fields: {list(example.keys())}")
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        tensor = transform(image.convert("RGB"))
        images.append(tensor.numpy().copy())
        labels.append(int(label))

    return DatasetBundle(
        images=np.stack(images, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
    )


def load_dataset_bundle(
    config: dict[str, Any],
    limit: int | None = None,
) -> DatasetBundle:
    """Load the evaluation dataset in its original order.

    ``limit`` selects the first N samples and is used for smoke tests and for
    evaluating a shortened ``explanations.npy`` file.
    """
    data_cfg = config["data"]
    if data_cfg.get("images_npy"):
        return _load_from_numpy(config, limit)

    name = data_cfg["name"].lower()
    if name in {"mnist", "fashion_mnist", "cifar10"}:
        return _load_keras_dataset(name, limit)
    if name in {"cifar100", "imagenet"}:
        return _load_huggingface(config, limit)
    raise ValueError(f"Unsupported dataset: {name}")

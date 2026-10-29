from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import ModelAdapter
from .architectures import build_keras_model
from .config import resolve_path


def load_model_adapter(config: dict[str, Any]) -> ModelAdapter:
    model_cfg = config["model"]
    framework = model_cfg["framework"].lower()
    batch_size = int(model_cfg.get("batch_size", 128))

    if framework == "tensorflow":
        import tensorflow as tf

        checkpoint = resolve_path(config, model_cfg.get("checkpoint"))
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(
                "Add the trained Keras checkpoint and set model.checkpoint in the config. "
                f"Current value: {checkpoint}"
            )
        checkpoint_type = str(model_cfg.get("checkpoint_type", "auto")).lower()
        model = None
        load_error: Exception | None = None
        if checkpoint_type in {"auto", "full_model"}:
            try:
                model = tf.keras.models.load_model(checkpoint)
            except Exception as exc:
                load_error = exc
                if checkpoint_type == "full_model":
                    raise
        if model is None:
            architecture = model_cfg.get("architecture")
            if architecture is None:
                raise ValueError(
                    "A weights-only TensorFlow checkpoint requires model.architecture in the config"
                ) from load_error
            model = build_keras_model(architecture, int(model_cfg["num_classes"]))
            model.load_weights(checkpoint)
        return ModelAdapter(model=model, framework="tensorflow", channel_first=False, batch_size=batch_size)

    if framework == "torch":
        import torch
        import timm

        if model_cfg.get("requires_detectors", False):
            try:
                import detectors  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "The CIFAR-100 checkpoint requires the 'detectors' package."
                ) from exc

        model_name = model_cfg["name"]
        checkpoint = resolve_path(config, model_cfg.get("checkpoint"))
        pretrained = bool(model_cfg.get("pretrained", checkpoint is None))
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=int(model_cfg["num_classes"]),
        )
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {key.removeprefix("module."): value for key, value in state.items()}
            model.load_state_dict(state)

        device_name = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)
        model.eval().to(device)
        return ModelAdapter(model=model, framework="torch", channel_first=True, batch_size=batch_size)

    raise ValueError(f"Unsupported framework: {framework}")

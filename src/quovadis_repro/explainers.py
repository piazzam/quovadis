from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .adapters import ModelAdapter
from .masking import from_channel_last, to_channel_last
from .quovadis import QuovadisExplainer
from .shap_bpt import ShapBPTExplainer


class Explainer(Protocol):
    def explain(self, images: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        """Return one spatial heatmap per image with shape (N, H, W)."""


def _spatial_maps(values: Any, channel_first: bool) -> np.ndarray:
    """Reduce common attribution layouts to (N,H,W).

    The original experiment pipeline retained the first attribution channel when
    a method returned one score per input channel. We keep the same convention.
    """
    array = np.asarray(values)
    if array.ndim == 2:
        return array[None, ...].astype(np.float32, copy=False)
    if array.ndim == 3:
        return array.astype(np.float32, copy=False)
    if array.ndim != 4:
        raise ValueError(f"Unsupported attribution shape: {array.shape}")
    if channel_first:
        return array[:, 0].astype(np.float32, copy=False)
    return array[..., 0].astype(np.float32, copy=False)


def _one_hot(target_classes: np.ndarray, num_classes: int) -> np.ndarray:
    targets = np.zeros((len(target_classes), num_classes), dtype=np.float32)
    targets[np.arange(len(target_classes)), target_classes.astype(np.int64)] = 1.0
    return targets


@dataclass
class _QuovadisWrapper:
    implementation: QuovadisExplainer

    def explain(self, images: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        return np.asarray(self.implementation.explain(images, target_classes), dtype=np.float32)


@dataclass
class _XpliqueExplainer:
    adapter: ModelAdapter
    method: str
    num_classes: int
    settings: dict[str, Any]

    def _build(self):
        from xplique.attributions import GradCAM, GradCAMPP, IntegratedGradients, Occlusion

        if self.method == "integrated_gradients":
            return IntegratedGradients(
                self.adapter.model,
                batch_size=int(self.settings.get("batch_size", 1)),
                steps=int(self.settings.get("steps", 50)),
                baseline_value=float(self.settings.get("baseline", 0.0)),
            )
        if self.method == "gradcam":
            return GradCAM(
                self.adapter.model,
                output_layer=None,
                batch_size=int(self.settings.get("batch_size", 1)),
            )
        if self.method == "gradcampp":
            return GradCAMPP(
                self.adapter.model,
                output_layer=None,
                batch_size=int(self.settings.get("batch_size", 1)),
            )
        if self.method == "occlusion":
            window = tuple(int(v) for v in self.settings.get("window", [16, 16]))
            stride = tuple(int(v) for v in self.settings.get("stride", [4, 4]))
            return Occlusion(
                self.adapter.model,
                patch_size=window,
                patch_stride=stride,
                occlusion_value=float(self.settings.get("baseline", 0.0)),
                batch_size=int(self.settings.get("batch_size", 64)),
            )
        raise ValueError(self.method)

    def explain(self, images: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        if self.adapter.framework != "tensorflow":
            raise ValueError("Xplique wrapper is used only for TensorFlow models")
        explainer = self._build()
        targets = _one_hot(np.asarray(target_classes), self.num_classes)
        values = explainer.explain(np.asarray(images, dtype=np.float32), targets)
        if hasattr(values, "numpy"):
            values = values.numpy()
        return _spatial_maps(values, channel_first=False)


@dataclass
class _TorchIntegratedGradients:
    adapter: ModelAdapter
    settings: dict[str, Any]

    def explain(self, images: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        import torch
        from captum.attr import IntegratedGradients

        model = self.adapter.model
        device = next(model.parameters()).device
        tensor = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(device)
        targets = torch.as_tensor(target_classes, dtype=torch.long, device=device)
        method = IntegratedGradients(model)
        values = method.attribute(
            tensor,
            baselines=torch.zeros_like(tensor) + float(self.settings.get("baseline", 0.0)),
            target=targets,
            n_steps=int(self.settings.get("steps", 50)),
            internal_batch_size=self.settings.get("internal_batch_size"),
        )
        return _spatial_maps(values.detach().cpu().numpy(), channel_first=True)


@dataclass
class _TorchOcclusion:
    adapter: ModelAdapter
    settings: dict[str, Any]

    def explain(self, images: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        import torch
        from captum.attr import Occlusion

        model = self.adapter.model
        device = next(model.parameters()).device
        tensor = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(device)
        targets = torch.as_tensor(target_classes, dtype=torch.long, device=device)
        channels = int(tensor.shape[1])
        window_h, window_w = (int(v) for v in self.settings.get("window", [16, 16]))
        stride_h, stride_w = (int(v) for v in self.settings.get("stride", [4, 4]))
        method = Occlusion(model)
        values = method.attribute(
            tensor,
            sliding_window_shapes=(channels, window_h, window_w),
            strides=(channels, stride_h, stride_w),
            baselines=float(self.settings.get("baseline", 0.0)),
            target=targets,
            perturbations_per_eval=int(self.settings.get("batch_size", 64)),
        )
        return _spatial_maps(values.detach().cpu().numpy(), channel_first=True)


def _last_conv_layer(model: Any):
    import torch.nn as nn

    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("No torch.nn.Conv2d layer found for Grad-CAM")
    return last


def _vit_reshape_transform(tensor):
    import math

    tokens = tensor[:, 1:, :]
    grid = int(math.sqrt(tokens.shape[1]))
    if grid * grid != tokens.shape[1]:
        raise ValueError(f"Cannot reshape {tokens.shape[1]} ViT tokens into a square grid")
    return tokens.reshape(tokens.shape[0], grid, grid, tokens.shape[2]).permute(0, 3, 1, 2)


@dataclass
class _TorchCAM:
    adapter: ModelAdapter
    method: str
    settings: dict[str, Any]

    def explain(self, images: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        import torch
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        model = self.adapter.model
        device = next(model.parameters()).device
        is_vit = self.method == "gradcam_vit"
        if is_vit:
            if not hasattr(model, "blocks"):
                raise ValueError("Grad-CAM-ViT expects a timm-style model.blocks attribute")
            target_layer = model.blocks[-1].norm1
            cam_class = GradCAM
            reshape_transform = _vit_reshape_transform
        else:
            target_layer = _last_conv_layer(model)
            cam_class = GradCAMPlusPlus if self.method == "gradcampp" else GradCAM
            reshape_transform = None

        cam = cam_class(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
        )
        maps: list[np.ndarray] = []
        for image, target in zip(images, target_classes):
            input_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None].to(device)
            value = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(int(target))],
            )[0]
            maps.append(np.asarray(value, dtype=np.float32))
        return np.stack(maps, axis=0)


@dataclass
class _LimeExplainer:
    adapter: ModelAdapter
    settings: dict[str, Any]

    def _classifier(self, batch_hwc: np.ndarray, original_channels: int) -> np.ndarray:
        batch = np.asarray(batch_hwc, dtype=np.float32)
        if original_channels == 1 and batch.shape[-1] == 3:
            batch = batch[..., :1]
        if self.adapter.channel_first:
            batch = np.transpose(batch, (0, 3, 1, 2))
        return self.adapter.predict_proba(batch)

    def explain(self, images: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        from lime import lime_image

        method = lime_image.LimeImageExplainer(random_state=int(self.settings.get("seed", 42)))
        maps: list[np.ndarray] = []
        for image, target in zip(images, target_classes):
            image_hwc = to_channel_last(image, self.adapter.channel_first)
            if image_hwc.ndim == 2:
                image_hwc = image_hwc[..., None]
            original_channels = image_hwc.shape[-1]
            lime_input = image_hwc
            if original_channels == 1:
                lime_input = np.repeat(image_hwc, 3, axis=-1)

            explanation = method.explain_instance(
                np.asarray(lime_input, dtype=np.float64),
                lambda batch: self._classifier(batch, original_channels),
                labels=(int(target),),
                hide_color=float(self.settings.get("baseline", 0.0)),
                num_samples=int(self.settings.get("num_samples", 1000)),
                batch_size=int(self.settings.get("batch_size", 10)),
                random_seed=int(self.settings.get("seed", 42)),
            )
            heatmap = np.zeros(explanation.segments.shape, dtype=np.float32)
            for segment_id, weight in explanation.local_exp[int(target)]:
                heatmap[explanation.segments == segment_id] = float(weight)
            maps.append(heatmap)
        return np.stack(maps, axis=0)


@dataclass
class _PartitionShapExplainer:
    adapter: ModelAdapter
    settings: dict[str, Any]

    def explain(self, images: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        import shap

        maps: list[np.ndarray] = []
        blur = self.settings.get("blur", [16, 16])
        max_evals = int(self.settings.get("max_evals", 500))
        batch_size = int(self.settings.get("batch_size", 64))

        for image, target in zip(images, target_classes):
            image_hwc = to_channel_last(image, self.adapter.channel_first)
            if image_hwc.ndim == 2:
                image_hwc = image_hwc[..., None]

            def predict(batch_hwc: np.ndarray) -> np.ndarray:
                batch = np.asarray(batch_hwc, dtype=np.float32)
                if self.adapter.channel_first:
                    batch = np.transpose(batch, (0, 3, 1, 2))
                return self.adapter.predict_proba(batch)

            masker = shap.maskers.Image(f"blur({int(blur[0])},{int(blur[1])})", image_hwc.shape)
            method = shap.Explainer(predict, masker, algorithm="partition")
            explanation = method(
                image_hwc[None, ...],
                max_evals=max_evals,
                batch_size=batch_size,
                outputs=[int(target)],
            )
            values = np.asarray(explanation.values)
            # SHAP image values are normally (N,H,W,C,K).
            if values.ndim == 5:
                heatmap = values[0, :, :, 0, 0]
            elif values.ndim == 4:
                heatmap = values[0, :, :, 0]
            elif values.ndim == 3:
                heatmap = values[0]
            else:
                raise ValueError(f"Unexpected PartitionSHAP output shape: {values.shape}")
            maps.append(np.asarray(heatmap, dtype=np.float32))
        return np.stack(maps, axis=0)


_ALIASES = {
    "qg": "quovadis",
    "quovadis": "quovadis",
    "integrated_gradients": "integrated_gradients",
    "integratedgradients": "integrated_gradients",
    "intgrad": "integrated_gradients",
    "ig": "integrated_gradients",
    "gradcam": "gradcam",
    "grad_cam": "gradcam",
    "gradcampp": "gradcampp",
    "gradcam++": "gradcampp",
    "grad_cam_pp": "gradcampp",
    "gradcam_vit": "gradcam_vit",
    "grad-cam-vit": "gradcam_vit",
    "lime": "lime",
    "partitionshap": "partition_shap",
    "partition_shap": "partition_shap",
    "shap": "partition_shap",
    "occlusion": "occlusion",
    "shap_bpt": "shap_bpt",
    "shap-bpt": "shap_bpt",
    "shapbpt": "shap_bpt",
}


def canonical_method_name(name: str) -> str:
    key = name.strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"Unsupported explanation method: {name}")
    return _ALIASES[key]


def build_explainer(
    method_name: str,
    adapter: ModelAdapter,
    config: dict[str, Any],
) -> Explainer:
    method = canonical_method_name(method_name)
    settings = dict(config.get("explainers", {}).get(method, {}))
    num_classes = int(config["model"]["num_classes"])

    if method == "quovadis":
        implementation = QuovadisExplainer(
            adapter=adapter,
            solution=str(settings.get("solution", "banzhaf")),
            aggregation=str(settings.get("aggregation", "median")),
            configuration=str(settings.get("configuration", "A")),
            blur_kernel=tuple(int(v) for v in settings.get("blur_kernel", [11, 11])),
            removal_strategy=str(settings.get("removal_strategy", "average_blur")),
            black_value=float(settings.get("black_value", 0.0)),
            template_names=(
                tuple(str(v) for v in settings["template_names"])
                if settings.get("template_names") is not None
                else None
            ),
        )
        return _QuovadisWrapper(implementation)

    if method == "lime":
        return _LimeExplainer(adapter, settings)
    if method == "partition_shap":
        return _PartitionShapExplainer(adapter, settings)
    if method == "shap_bpt":
        return ShapBPTExplainer(adapter, settings)

    if adapter.framework == "tensorflow":
        if method == "gradcam_vit":
            raise ValueError("Grad-CAM-ViT is supported for the PyTorch ViT used in the paper")
        return _XpliqueExplainer(adapter, method, num_classes, settings)

    if method == "integrated_gradients":
        return _TorchIntegratedGradients(adapter, settings)
    if method == "occlusion":
        return _TorchOcclusion(adapter, settings)
    if method in {"gradcam", "gradcampp", "gradcam_vit"}:
        return _TorchCAM(adapter, method, settings)
    raise ValueError(f"Unsupported method/framework combination: {method}/{adapter.framework}")

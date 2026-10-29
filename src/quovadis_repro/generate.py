from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, set_reproducibility_seed
from .data import load_dataset_bundle
from .explainers import build_explainer, canonical_method_name
from .io import save_explanations
from .models import load_model_adapter
from .templates import CONFIGURATIONS


def _chunks(length: int, size: int):
    for start in range(0, length, size):
        yield start, min(start + size, length)


def generate_explanations(
    config: dict[str, Any],
    method_name: str,
    output: str | Path,
    limit: int | None = None,
    chunk_size: int = 32,
) -> Path:
    """Generate explanations in the original dataset order.

    The i-th saved explanation always corresponds to the i-th dataset sample.
    Target classes are the model top-1 predictions and are recomputed during
    evaluation, so no separate sample-index or target-class files are needed.
    """
    set_reproducibility_seed(int(config.get("seed", 42)), config["model"]["framework"])
    method = canonical_method_name(method_name)
    dataset = load_dataset_bundle(config, limit=limit)

    adapter = load_model_adapter(config)
    target_classes = np.argmax(
        adapter.predict_proba(dataset.images), axis=1
    ).astype(np.int64)

    explainer = build_explainer(method, adapter, config)
    aggregate_parts: list[np.ndarray] = []

    for start, end in _chunks(len(dataset.images), max(1, int(chunk_size))):
        images = np.asarray(dataset.images[start:end])
        targets = target_classes[start:end]
        aggregate_parts.append(
            np.asarray(explainer.explain(images, targets), dtype=np.float32)
        )

    heatmaps = np.concatenate(aggregate_parts, axis=0)
    output_path = Path(output)
    metadata: dict[str, Any] = {
        "dataset": config["data"]["name"],
        "method": method,
        "target": "predicted_class",
        "sample_order": "dataset_order",
        "seed": int(config.get("seed", 42)),
        "n_samples": int(len(heatmaps)),
        "heatmap_shape": list(heatmaps.shape[1:]),
        "method_configuration": config.get("explainers", {}).get(method, {}),
        "model": {
            key: value
            for key, value in config["model"].items()
            if key not in {"checkpoint", "device"}
        },
    }
    if method == "quovadis":
        qg_settings = config.get("explainers", {}).get("quovadis", {})
        configuration_name = str(qg_settings.get("configuration", "A")).upper()
        explicit_templates = qg_settings.get("template_names")
        metadata["template_names"] = (
            [str(name) for name in explicit_templates]
            if explicit_templates is not None
            else [spec.name for spec in CONFIGURATIONS[configuration_name]]
        )
        metadata["feature_removal"] = str(qg_settings.get("removal_strategy", "average_blur"))

    save_explanations(output_path, heatmaps=heatmaps, metadata=metadata)
    with (output_path / "generation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_samples": int(len(heatmaps)),
                "method": method,
                "dataset": config["data"]["name"],
                "sample_order": "dataset_order",
                "output": str(output_path),
            },
            handle,
            indent=2,
        )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate explanations used in the QUOVADIS experiments"
    )
    parser.add_argument("--config", required=True, help="Dataset JSON configuration")
    parser.add_argument(
        "--solution",
        choices=["banzhaf", "shapley"],
        help="Override QUOVADIS solution concept",
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "q25", "contamination_aware"],
        help="Override QUOVADIS aggregation",
    )
    parser.add_argument(
        "--template-configuration",
        choices=["A", "B", "C"],
        help="Override QUOVADIS template set",
    )
    parser.add_argument(
        "--removal-strategy",
        choices=["average_blur", "gaussian_blur", "black"],
        help="Override QUO VADIS feature-removal operator",
    )
    parser.add_argument(
        "--black-value",
        type=float,
        help="Constant used by the black feature-removal operator",
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        help="Optional explicit QUO VADIS template names, e.g. 5_row 5_column",
    )
    parser.add_argument("--method", required=True, help="Explanation method")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate explanations only for the first N dataset samples",
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    qg_settings = config.setdefault("explainers", {}).setdefault("quovadis", {})
    if args.solution is not None:
        qg_settings["solution"] = args.solution
    if args.aggregation is not None:
        qg_settings["aggregation"] = args.aggregation
    if args.template_configuration is not None:
        qg_settings["configuration"] = args.template_configuration
    if args.removal_strategy is not None:
        qg_settings["removal_strategy"] = args.removal_strategy
    if args.black_value is not None:
        qg_settings["black_value"] = args.black_value
    if args.templates is not None:
        qg_settings["template_names"] = args.templates
    path = generate_explanations(
        config=config,
        method_name=args.method,
        output=args.output,
        limit=args.limit,
        chunk_size=args.chunk_size,
    )
    print(f"Saved explanations to {path}")


if __name__ == "__main__":
    main()

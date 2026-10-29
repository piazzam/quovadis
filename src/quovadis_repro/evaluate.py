from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, set_reproducibility_seed
from .data import load_dataset_bundle
from .explainers import build_explainer, canonical_method_name
from .io import load_explanations
from .metrics import complexity, infidelity, road, saco, sensitivity, summarize
from .models import load_model_adapter


SUPPORTED_METRICS = ("road", "road_morf", "road_lerf", "saco", "infidelity", "sensitivity", "complexity")


def _parse_metrics(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip().lower() for part in value.split(",") if part.strip())
    invalid = sorted(set(parsed) - set(SUPPORTED_METRICS))
    if invalid:
        raise ValueError(f"Unsupported metrics: {invalid}")
    return list(dict.fromkeys(parsed))


def _save_scores(output: Path, name: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    np.save(output / f"{name}_scores.npy", values)
    return summarize(values)


def evaluate_explanations(
    config: dict[str, Any],
    method_name: str,
    explanations_path: str | Path,
    output: str | Path,
    metric_names: list[str],
) -> Path:
    """Evaluate saved explanations using the original dataset order.

    ``explanations[i]`` is always matched with ``dataset[i]``. Explained target
    classes are recomputed from the same model and preprocessing used by the
    experiment.
    """
    set_reproducibility_seed(int(config.get("seed", 42)), config["model"]["framework"])
    method = canonical_method_name(method_name)
    artifact = load_explanations(explanations_path)

    dataset = load_dataset_bundle(config, limit=len(artifact.heatmaps))
    if len(dataset.images) != len(artifact.heatmaps):
        raise ValueError("Loaded images and heatmaps have different lengths")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": config["data"]["name"],
        "method": method,
        "explanations": str(explanations_path),
        "n_samples": int(len(dataset.images)),
        "sample_order": "dataset_order",
        "metrics": {},
    }

    needs_model = any(name != "complexity" for name in metric_names)
    needs_targets = any(
        name in {"saco", "infidelity", "sensitivity"} for name in metric_names
    )
    adapter = load_model_adapter(config) if needs_model else None
    target_classes: np.ndarray | None = None

    if needs_targets:
        assert adapter is not None
        target_classes = np.argmax(
            adapter.predict_proba(dataset.images), axis=1
        ).astype(np.int64)
        summary["target_class_source"] = "recomputed_model_predictions"
    else:
        summary["target_class_source"] = None

    metric_config = config.get("metrics", {})
    for name in metric_names:
        settings = dict(metric_config.get(name, {}))
        if name == "complexity":
            values = complexity(artifact.heatmaps)
            summary["metrics"][name] = _save_scores(output_path, name, values)

        elif name in {"road", "road_morf", "road_lerf"}:
            assert adapter is not None
            # ``road`` is kept as a backwards-compatible alias whose direction
            # is read from config. The explicit variants make it possible to
            # evaluate MoRF and LeRF in the same run and save separate outputs.
            road_settings = dict(metric_config.get(name, metric_config.get("road", {})))
            if name == "road_morf":
                direction = "morf"
            elif name == "road_lerf":
                direction = "lerf"
            else:
                direction = str(road_settings.get("direction", "morf")).lower()
            result = road(
                adapter=adapter,
                images=dataset.images,
                labels=dataset.labels,
                heatmaps=artifact.heatmaps,
                percentages=[
                    float(v)
                    for v in road_settings.get(
                        "percentages", [0.1, 0.3, 0.5, 0.7, 0.9]
                    )
                ],
                direction=direction,
                noise_sigma=float(road_settings.get("noise_sigma", 0.01)),
                seed=int(road_settings.get("seed", config.get("seed", 42))),
                perturbation_batch_size=int(
                    road_settings.get("batch_size", adapter.batch_size)
                ),
            )
            result_name = name if name != "road" else f"road_{direction}"
            np.savez_compressed(
                output_path / f"{result_name}_results.npz",
                percentages=result["percentages"],
                accuracies=result["accuracies"],
                correctness=result["correctness"],
                auc=np.asarray(result["auc"]),
            )
            summary["metrics"][name] = {
                "direction": direction,
                "auc": float(result["auc"]),
                "accuracies": np.asarray(result["accuracies"]).tolist(),
                "percentages": np.asarray(result["percentages"]).tolist(),
            }

        elif name == "saco":
            assert adapter is not None and target_classes is not None
            values = saco(
                adapter=adapter,
                images=dataset.images,
                heatmaps=artifact.heatmaps,
                target_classes=target_classes,
                groups=int(settings.get("groups", 10)),
            )
            summary["metrics"][name] = _save_scores(output_path, name, values)

        elif name == "infidelity":
            assert adapter is not None and target_classes is not None
            values = infidelity(
                adapter=adapter,
                images=dataset.images,
                heatmaps=artifact.heatmaps,
                target_classes=target_classes,
                patch_sizes=[int(v) for v in settings.get("patch_sizes", [8])],
                num_samples=int(settings.get("num_samples", 1000)),
                all_patches=bool(settings.get("all_patches", False)),
                apply_beta=bool(settings.get("apply_beta", True)),
                seed=int(settings.get("seed", config.get("seed", 42))),
                perturbation_batch_size=int(
                    settings.get("batch_size", adapter.batch_size)
                ),
            )
            summary["metrics"][name] = _save_scores(output_path, name, values)

        elif name == "sensitivity":
            assert adapter is not None and target_classes is not None
            explainer = build_explainer(method, adapter, config)
            clip = settings.get("clip_values", [0.0, 1.0])
            if clip == "dataset_range":
                clip_values = (
                    float(np.min(dataset.images)),
                    float(np.max(dataset.images)),
                )
            else:
                clip_values = (
                    None
                    if clip is None
                    else (float(clip[0]), float(clip[1]))
                )
            values = sensitivity(
                images=dataset.images,
                heatmaps=artifact.heatmaps,
                target_classes=target_classes,
                explain_fn=explainer.explain,
                epsilon=float(settings.get("epsilon", 0.1)),
                n_perturbations=int(settings.get("n_perturbations", 50)),
                clip_values=clip_values,
                seed=int(settings.get("seed", config.get("seed", 42))),
            )
            summary["metrics"][name] = _save_scores(output_path, name, values)

    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved explanations without regenerating the original heatmaps"
    )
    parser.add_argument("--config", required=True, help="Dataset JSON configuration")
    parser.add_argument(
        "--solution",
        choices=["banzhaf", "shapley"],
        help="Override QUOVADIS solution concept for Sensitivity",
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "q25", "contamination_aware"],
        help="Override QUOVADIS aggregation for Sensitivity",
    )
    parser.add_argument(
        "--template-configuration",
        choices=["A", "B", "C"],
        help="Override QUOVADIS template set for Sensitivity",
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
    parser.add_argument("--method", required=True, help="Method that generated the saved maps")
    parser.add_argument(
        "--explanations",
        required=True,
        help="Directory containing explanations.npy, e.g. .../5_row",
    )
    parser.add_argument("--output", required=True, help="Directory for raw metric outputs")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(SUPPORTED_METRICS),
        help="Space- or comma-separated metric names",
    )
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
    metrics = _parse_metrics(args.metrics)
    path = evaluate_explanations(
        config=config,
        method_name=args.method,
        explanations_path=args.explanations,
        output=args.output,
        metric_names=metrics,
    )
    print(f"Saved raw metric results to {path}")


if __name__ == "__main__":
    main()

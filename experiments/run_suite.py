#!/usr/bin/env python3
"""Run the reproducible experiment suites used in the thesis chapter.

The script deliberately writes raw explanation artifacts and metric outputs.
It does not generate thesis tables, statistical tests, timing benchmarks, or
figures.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable

from quovadis_repro.config import load_config
from quovadis_repro.evaluate import SUPPORTED_METRICS, evaluate_explanations
from quovadis_repro.generate import generate_explanations
from quovadis_repro.templates import CONFIGURATIONS


DEFAULT_METRICS = ["road_morf", "road_lerf", "saco", "infidelity", "sensitivity", "complexity"]


def _metrics(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_METRICS)
    out: list[str] = []
    for value in values:
        out.extend(part.strip().lower() for part in value.split(",") if part.strip())
    bad = sorted(set(out) - set(SUPPORTED_METRICS))
    if bad:
        raise ValueError(f"Unsupported metrics: {bad}")
    return list(dict.fromkeys(out))


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _run_one(
    base_config: dict[str, Any],
    output_root: Path,
    suite: str,
    variant: str,
    method: str,
    metric_names: list[str],
    limit: int | None,
    chunk_size: int,
    qg_overrides: dict[str, Any] | None = None,
    skip_existing: bool = False,
) -> dict[str, str]:
    config = copy.deepcopy(base_config)
    if qg_overrides:
        config.setdefault("explainers", {}).setdefault("quovadis", {}).update(qg_overrides)
    run_dir = output_root / _safe_name(suite) / _safe_name(variant)
    explanations_dir = run_dir / "explanations"
    metrics_dir = run_dir / "metrics"

    explanation_file = explanations_dir / "explanations.npy"
    if not (skip_existing and explanation_file.exists()):
        generate_explanations(
            config=config,
            method_name=method,
            output=explanations_dir,
            limit=limit,
            chunk_size=chunk_size,
        )
    if metric_names:
        summary_file = metrics_dir / "summary.json"
        if not (skip_existing and summary_file.exists()):
            evaluate_explanations(
                config=config,
                method_name=method,
                explanations_path=explanations_dir,
                output=metrics_dir,
                metric_names=metric_names,
            )

    run_meta = {
        "suite": suite,
        "variant": variant,
        "method": method,
        "qg_overrides": qg_overrides or {},
        "metrics": metric_names,
        "limit": limit,
        "explanations": str(explanations_dir),
        "metric_outputs": str(metrics_dir),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    return {"variant": variant, "run_dir": str(run_dir)}


def _write_manifest(output_root: Path, suite: str, rows: list[dict[str, str]]) -> None:
    path = output_root / _safe_name(suite) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def run_baselines(args, config, metrics):
    dataset = str(config["data"]["name"]).lower()
    if dataset == "imagenet":
        default = ["quovadis", "partition_shap", "shap_bpt", "lime", "occlusion", "integrated_gradients", "gradcam_vit"]
    else:
        default = ["quovadis", "partition_shap", "shap_bpt", "lime", "occlusion", "integrated_gradients", "gradcam", "gradcampp"]
    methods = args.methods or default
    rows = []
    for method in methods:
        rows.append(_run_one(config, args.output, "baselines", method, method, metrics, args.limit, args.chunk_size, skip_existing=args.skip_existing))
    _write_manifest(args.output, "baselines", rows)


def run_feature_removal(args, config, metrics):
    rows = []
    for strategy in ("average_blur", "gaussian_blur", "black"):
        rows.append(_run_one(config, args.output, "feature_removal", strategy, "quovadis", metrics, args.limit, args.chunk_size, {"removal_strategy": strategy}, args.skip_existing))
    _write_manifest(args.output, "feature_removal", rows)


def run_solution(args, config, metrics):
    rows = []
    for solution in ("banzhaf", "shapley"):
        rows.append(_run_one(config, args.output, "solution", solution, "quovadis", metrics, args.limit, args.chunk_size, {"solution": solution}, args.skip_existing))
    _write_manifest(args.output, "solution", rows)


def run_aggregation(args, config, metrics):
    rows = []
    # Keep the contamination-aware rule exactly as implemented in the original
    # experiments; the thesis text can document that implementation directly.
    for aggregation in ("median", "q25", "contamination_aware"):
        rows.append(_run_one(config, args.output, "aggregation", aggregation, "quovadis", metrics, args.limit, args.chunk_size, {"aggregation": aggregation}, args.skip_existing))
    _write_manifest(args.output, "aggregation", rows)


def run_configurations(args, config, metrics):
    rows = []
    for name in ("A", "B", "C"):
        rows.append(_run_one(config, args.output, "template_configurations", name, "quovadis", metrics, args.limit, args.chunk_size, {"configuration": name, "template_names": None}, args.skip_existing))
    _write_manifest(args.output, "template_configurations", rows)


def run_single_templates(args, config, metrics):
    rows = []
    for spec in CONFIGURATIONS["A"]:
        rows.append(_run_one(config, args.output, "single_template", spec.name, "quovadis", metrics, args.limit, args.chunk_size, {"configuration": "A", "template_names": [spec.name]}, args.skip_existing))
    rows.append(_run_one(config, args.output, "single_template", "ensemble_A", "quovadis", metrics, args.limit, args.chunk_size, {"configuration": "A", "template_names": None}, args.skip_existing))
    _write_manifest(args.output, "single_template", rows)


def run_template_count(args, config, metrics):
    names = [spec.name for spec in CONFIGURATIONS["A"]]
    sizes = args.template_counts or [1, 2, 4, 8, 12, 17]
    rng = random.Random(args.subset_seed)
    rows = []
    for size in sizes:
        if size < 1 or size > len(names):
            raise ValueError(f"template count must be in [1,{len(names)}], got {size}")
        reps = 1 if size == len(names) else args.repetitions
        for rep in range(reps):
            selected = names if size == len(names) else sorted(rng.sample(names, size))
            variant = f"R{size}_rep{rep:02d}"
            rows.append(_run_one(config, args.output, "template_count", variant, "quovadis", metrics, args.limit, args.chunk_size, {"configuration": "A", "template_names": selected}, args.skip_existing))
    _write_manifest(args.output, "template_count", rows)


def run_granularity(args, config, metrics):
    by_players: dict[int, list[str]] = {}
    for spec in CONFIGURATIONS["A"]:
        by_players.setdefault(spec.players, []).append(spec.name)
    rows = []
    for players, names in sorted(by_players.items()):
        variant = f"players_{players}"
        rows.append(_run_one(config, args.output, "granularity", variant, "quovadis", metrics, args.limit, args.chunk_size, {"configuration": "A", "template_names": names}, args.skip_existing))
    _write_manifest(args.output, "granularity", rows)


def run_template_robustness(args, config, metrics):
    names = [spec.name for spec in CONFIGURATIONS["A"]]
    if args.subset_size < 1 or args.subset_size > len(names):
        raise ValueError(f"subset-size must be in [1,{len(names)}]")
    rng = random.Random(args.subset_seed)
    rows = []
    for rep in range(args.repetitions):
        selected = sorted(rng.sample(names, args.subset_size))
        rows.append(_run_one(config, args.output, "template_robustness", f"R{args.subset_size}_rep{rep:02d}", "quovadis", metrics, args.limit, args.chunk_size, {"configuration": "A", "template_names": selected}, args.skip_existing))
    _write_manifest(args.output, "template_robustness", rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=[
        "baselines", "feature-removal", "solution", "aggregation", "configurations",
        "single-template", "template-count", "granularity", "template-robustness",
    ])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", help="Default: road_morf road_lerf saco infidelity sensitivity complexity")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--methods", nargs="+", help="Only for the baselines suite")
    parser.add_argument("--template-counts", nargs="+", type=int)
    parser.add_argument("--subset-size", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--subset-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    metrics = _metrics(args.metrics)
    dispatch = {
        "baselines": run_baselines,
        "feature-removal": run_feature_removal,
        "solution": run_solution,
        "aggregation": run_aggregation,
        "configurations": run_configurations,
        "single-template": run_single_templates,
        "template-count": run_template_count,
        "granularity": run_granularity,
        "template-robustness": run_template_robustness,
    }
    dispatch[args.suite](args, config, metrics)


if __name__ == "__main__":
    main()

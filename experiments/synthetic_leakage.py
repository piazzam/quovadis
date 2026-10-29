#!/usr/bin/env python3
"""Synthetic verification of region-level attribution leakage.

Creates additive pixel games with known ground-truth contributions, applies the
same template partitions used by QUO VADIS, and stores raw per-pixel errors for
single templates and ensemble aggregations. No model is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quovadis_repro.templates import CONFIGURATIONS, generate_partition


def uniform_group_map(weights: np.ndarray, partition: np.ndarray) -> np.ndarray:
    result = np.zeros_like(weights, dtype=np.float64)
    for player in np.unique(partition):
        total = float(weights[partition == player].sum())
        result[partition == player] = total
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--distribution",
        choices=["normal", "sparse_positive", "sparse_signed"],
        default="sparse_signed",
    )
    parser.add_argument("--sparsity", type=float, default=0.1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    specs = CONFIGURATIONS["A"]
    partitions = [
        generate_partition(args.height, args.width, spec.players, spec.mode)
        for spec in specs
    ]

    single_mae = np.zeros((args.trials, len(specs)), dtype=np.float64)
    median_mae = np.zeros(args.trials, dtype=np.float64)
    q25_mae = np.zeros(args.trials, dtype=np.float64)
    truth_l1 = np.zeros(args.trials, dtype=np.float64)

    for trial in range(args.trials):
        if args.distribution == "normal":
            weights = rng.normal(size=(args.height, args.width))
        else:
            active = rng.random((args.height, args.width)) < args.sparsity
            magnitudes = rng.exponential(scale=1.0, size=(args.height, args.width))
            if args.distribution == "sparse_positive":
                weights = active * magnitudes
            else:
                signs = rng.choice([-1.0, 1.0], size=(args.height, args.width))
                weights = active * magnitudes * signs

        maps = np.stack([uniform_group_map(weights, p) for p in partitions], axis=0)
        errors = maps - weights[None, ...]
        single_mae[trial] = np.mean(np.abs(errors), axis=(1, 2))
        median_map = np.median(maps, axis=0)
        q25_map = np.quantile(maps, 0.25, axis=0)
        median_mae[trial] = np.mean(np.abs(median_map - weights))
        q25_mae[trial] = np.mean(np.abs(q25_map - weights))
        truth_l1[trial] = np.mean(np.abs(weights))

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "synthetic_leakage.npz",
        single_template_mae=single_mae,
        median_ensemble_mae=median_mae,
        q25_ensemble_mae=q25_mae,
        truth_mean_abs=truth_l1,
        template_names=np.asarray([s.name for s in specs]),
    )
    summary = {
        "trials": args.trials,
        "shape": [args.height, args.width],
        "distribution": args.distribution,
        "sparsity": args.sparsity,
        "seed": args.seed,
        "template_names": [s.name for s in specs],
        "mean_single_template_mae": {
            spec.name: float(single_mae[:, i].mean()) for i, spec in enumerate(specs)
        },
        "mean_median_ensemble_mae": float(median_mae.mean()),
        "mean_q25_ensemble_mae": float(q25_mae.mean()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

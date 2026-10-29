from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .aggregation import aggregate_template_heatmaps
from .io import load_explanations, save_explanations
from .templates import CONFIGURATIONS, generate_partition


def aggregate_saved_shapes(
    input_root: str | Path,
    output: str | Path,
    configuration: str = "A",
    strategy: str = "median",
    solution: str = "banzhaf",
) -> Path:
    """Aggregate separate ``<shape>/explanations.npy`` files.

    No combined template file or saved partition file is required. Partitions
    are reconstructed deterministically from the template names.
    """
    configuration = configuration.upper()
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown template configuration: {configuration}")

    root = Path(input_root)
    maps: list[np.ndarray] = []
    template_names: list[str] = []
    reference_shape: tuple[int, ...] | None = None

    for spec in CONFIGURATIONS[configuration]:
        artifact = load_explanations(root / spec.name)
        current = np.asarray(artifact.heatmaps, dtype=np.float32)
        if reference_shape is None:
            reference_shape = current.shape
        elif current.shape != reference_shape:
            raise ValueError(
                f"All shape files must have the same dimensions. "
                f"Expected {reference_shape}, got {current.shape} for {spec.name}."
            )
        maps.append(current)
        template_names.append(spec.name)

    stacked = np.stack(maps, axis=0)
    partitions = None
    if strategy.lower() in {"contamination", "contamination_aware"}:
        height, width = stacked.shape[-2:]
        partitions = np.stack(
            [
                generate_partition(height, width, spec.players, spec.mode)
                for spec in CONFIGURATIONS[configuration]
            ],
            axis=0,
        )

    aggregate = aggregate_template_heatmaps(
        stacked,
        strategy=strategy,
        partitions=partitions,
    )
    output_path = Path(output)
    metadata: dict[str, Any] = {
        "method": "quovadis",
        "solution": solution,
        "aggregation": strategy,
        "template_configuration": configuration,
        "template_names": template_names,
        "sample_order": "dataset_order",
        "n_samples": int(len(aggregate)),
        "source_layout": "separate_shape_directories",
    }
    save_explanations(output_path, aggregate, metadata)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate QUOVADIS shape folders into one explanation file"
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory containing folders such as 5_row/explanations.npy",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--configuration", choices=["A", "B", "C"], default="A"
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "q25", "contamination_aware"],
        default="median",
    )
    parser.add_argument(
        "--solution", choices=["banzhaf", "shapley"], default="banzhaf"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = aggregate_saved_shapes(
        input_root=args.input_root,
        output=args.output,
        configuration=args.configuration,
        strategy=args.aggregation,
        solution=args.solution,
    )
    print(f"Saved aggregated explanations to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reconstruct the 10,000-image ImageNet evaluation subset used by QUOVADIS.

The supplementary package contains the exact validation-row indices in
``data/imagenet_sample_indices.csv`` but does not include the generated Parquet
file because of its size.

This script downloads only the validation Parquet shards of the gated
``ILSVRC/imagenet-1k`` dataset, selects the rows listed in the CSV file, checks
the labels and class balance, and writes the local Parquet file expected by
``configs/imagenet.json``.

Authentication options, in order of precedence:

1. Replace the placeholder in ``HF_TOKEN`` temporarily.
2. Export the token as the ``HF_TOKEN`` environment variable.
3. Authenticate once with ``hf auth login`` and leave the placeholder intact.

Never commit or submit a real access token.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DATASET_ID = "ILSVRC/imagenet-1k"
HF_TOKEN = "PASTE_YOUR_HUGGING_FACE_TOKEN_HERE"

EXPECTED_SOURCE_ROWS = 50_000
EXPECTED_SAMPLE_ROWS = 10_000
EXPECTED_CLASSES = 1_000
EXPECTED_SAMPLES_PER_CLASS = 10
EXPECTED_INDICES_SHA256 = (
    "0341faf4468bb9a591fa379bfc2f24df1991537992c5eee444561f81cd557d34"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDICES_CSV = REPOSITORY_ROOT / "data" / "imagenet_sample_indices.csv"
DEFAULT_OUTPUT_PARQUET = (
    REPOSITORY_ROOT / "data" / "imagenet1k_val_sampled10K.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the ImageNet-1K validation shards and reconstruct the "
            "exact 10,000-image QUOVADIS evaluation subset."
        )
    )
    parser.add_argument(
        "--indices-csv",
        type=Path,
        default=DEFAULT_INDICES_CSV,
        help="CSV containing sample_position, source_index, and label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help="Destination Parquet file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output Parquet file.",
    )
    return parser.parse_args()


def resolve_token() -> str | bool:
    """Return an explicit token or request the locally cached token."""
    inline = HF_TOKEN.strip()
    if inline and inline != "PASTE_YOUR_HUGGING_FACE_TOKEN_HERE":
        return inline

    environment_token = os.environ.get("HF_TOKEN", "").strip()
    if environment_token:
        return environment_token

    # ``True`` tells huggingface_hub to use the token stored by `hf auth login`.
    return True


def indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes(order="C")
    ).hexdigest()


def load_and_validate_index_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"ImageNet index CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected_columns = ["sample_position", "source_index", "label"]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"Expected CSV columns {expected_columns}, got {reader.fieldnames}."
            )
        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_ROWS} CSV rows, found {len(rows)}."
        )

    positions = np.asarray(
        [int(row["sample_position"]) for row in rows], dtype=np.int64
    )
    indices = np.asarray(
        [int(row["source_index"]) for row in rows], dtype=np.int64
    )
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)

    expected_positions = np.arange(EXPECTED_SAMPLE_ROWS, dtype=np.int64)
    if not np.array_equal(positions, expected_positions):
        raise ValueError("sample_position must contain the ordered range 0..9999.")
    if len(np.unique(indices)) != EXPECTED_SAMPLE_ROWS:
        raise ValueError("source_index contains duplicates.")
    if np.any(indices < 0) or np.any(indices >= EXPECTED_SOURCE_ROWS):
        raise ValueError("source_index contains values outside 0..49999.")
    if not np.all(indices[:-1] < indices[1:]):
        raise ValueError("source_index must be strictly increasing.")
    if np.any(labels < 0) or np.any(labels >= EXPECTED_CLASSES):
        raise ValueError("label contains values outside 0..999.")

    label_counts = Counter(labels.tolist())
    if len(label_counts) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, found {len(label_counts)}."
        )
    invalid_counts = {
        label: count
        for label, count in label_counts.items()
        if count != EXPECTED_SAMPLES_PER_CLASS
    }
    if invalid_counts:
        raise ValueError(
            "The CSV is not class balanced. Invalid class counts: "
            f"{dict(list(invalid_counts.items())[:10])}"
        )

    digest = indices_sha256(indices)
    if digest != EXPECTED_INDICES_SHA256:
        raise ValueError(
            "The source-index sequence does not match the version supplied "
            f"with this package. Expected {EXPECTED_INDICES_SHA256}, got {digest}."
        )

    return indices, labels


def download_validation_shards(token: str | bool, cache_dir: Path | None) -> list[Path]:
    """Download only validation Parquet shards, never the training shards."""
    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            allow_patterns=[
                "data/validation-*.parquet",
                "validation-*.parquet",
            ],
        )
    )

    candidates = sorted(
        path
        for path in snapshot_path.rglob("*.parquet")
        if path.name.startswith("validation-")
    )
    if not candidates:
        raise FileNotFoundError(
            "No validation Parquet shards were found in the downloaded snapshot."
        )

    return candidates


def select_rows_from_shards(
    shard_paths: list[Path],
    source_indices: np.ndarray,
) -> tuple[Any, int]:
    """Select global validation rows while preserving CSV/sample order."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    selected_tables: list[Any] = []
    global_offset = 0
    selected_count = 0

    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)
        shard_rows = int(parquet_file.metadata.num_rows)
        shard_end = global_offset + shard_rows

        in_shard = (source_indices >= global_offset) & (source_indices < shard_end)
        local_indices = source_indices[in_shard] - global_offset

        if local_indices.size:
            table = pq.read_table(shard_path)
            selected_tables.append(
                table.take(pa.array(local_indices, type=pa.int64()))
            )
            selected_count += int(local_indices.size)

        global_offset = shard_end

    if global_offset != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_ROWS} validation rows, found {global_offset}."
        )
    if selected_count != EXPECTED_SAMPLE_ROWS:
        raise ValueError(
            f"Expected to select {EXPECTED_SAMPLE_ROWS} rows, selected {selected_count}."
        )
    if not selected_tables:
        raise RuntimeError("No ImageNet rows were selected.")

    return pa.concat_tables(selected_tables), global_offset


def validate_reconstructed_table(table: Any, expected_labels: np.ndarray) -> None:
    if table.num_rows != EXPECTED_SAMPLE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_ROWS} reconstructed rows, got {table.num_rows}."
        )
    if "image" not in table.column_names or "label" not in table.column_names:
        raise ValueError(
            f"Expected image and label columns, got {table.column_names}."
        )

    observed_labels = np.asarray(table["label"].to_pylist(), dtype=np.int64)
    if not np.array_equal(observed_labels, expected_labels):
        mismatch = int(np.flatnonzero(observed_labels != expected_labels)[0])
        raise ValueError(
            "The downloaded validation ordering does not match the supplied CSV. "
            f"First label mismatch at sample_position={mismatch}: "
            f"expected {expected_labels[mismatch]}, observed {observed_labels[mismatch]}."
        )

    counts = np.bincount(observed_labels, minlength=EXPECTED_CLASSES)
    if not np.array_equal(
        counts,
        np.full(EXPECTED_CLASSES, EXPECTED_SAMPLES_PER_CLASS, dtype=np.int64),
    ):
        raise ValueError("The reconstructed ImageNet subset is not class balanced.")


def main() -> int:
    args = parse_args()
    args.indices_csv = args.indices_csv.resolve()
    args.output = args.output.resolve()
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.resolve()

    if args.output.exists() and not args.force:
        print(f"Output already exists: {args.output}")
        print("Use --force to reconstruct and overwrite it.")
        return 0

    try:
        source_indices, expected_labels = load_and_validate_index_csv(
            args.indices_csv
        )

        print(f"Dataset:        {DATASET_ID}")
        print("Source split:   validation")
        print(f"Index CSV:      {args.indices_csv}")
        print(f"Output Parquet: {args.output}")
        print("Downloading only the validation Parquet shards ...")

        shard_paths = download_validation_shards(
            token=resolve_token(),
            cache_dir=args.cache_dir,
        )
        print(f"Validation shards: {len(shard_paths)}")

        reconstructed, source_rows = select_rows_from_shards(
            shard_paths,
            source_indices,
        )
        validate_reconstructed_table(reconstructed, expected_labels)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq

        pq.write_table(reconstructed, args.output, compression="snappy")

        print("\nImageNet subset reconstructed successfully.")
        print(f"Source validation rows: {source_rows:,}")
        print(f"Selected rows:          {reconstructed.num_rows:,}")
        print(f"Classes:                {EXPECTED_CLASSES:,}")
        print(f"Samples per class:      {EXPECTED_SAMPLES_PER_CLASS}")
        print(f"Indices SHA-256:        {EXPECTED_INDICES_SHA256}")
        print(f"Saved to:               {args.output}")
        return 0

    except Exception as exc:
        print("\nImageNet subset reconstruction failed.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nBefore retrying, accept the access conditions for "
            f"{DATASET_ID} with your Hugging Face account and authenticate "
            "with `hf auth login`. Fine-grained tokens must allow access to "
            "public gated repositories. Never place a real token in the "
            "supplementary archive.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

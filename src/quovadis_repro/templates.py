from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemplateSpec:
    players: int
    mode: str

    @property
    def name(self) -> str:
        return f"{self.players}_{self.mode}"


CONFIGURATIONS: dict[str, tuple[TemplateSpec, ...]] = {
    "A": (
        TemplateSpec(2, "ellipse"),
        TemplateSpec(3, "ellipse"),
        TemplateSpec(5, "ellipse"),
        TemplateSpec(5, "diagonal"),
        TemplateSpec(5, "diagonal_transposed"),
        TemplateSpec(5, "triangle"),
        TemplateSpec(7, "triangle"),
        TemplateSpec(5, "row"),
        TemplateSpec(5, "column"),
        TemplateSpec(5, "centered_row"),
        TemplateSpec(5, "centered_column"),
        TemplateSpec(6, "block"),
        TemplateSpec(6, "block_transposed"),
        TemplateSpec(7, "row"),
        TemplateSpec(7, "column"),
        TemplateSpec(8, "block"),
        TemplateSpec(8, "block_transposed"),
    ),
    "B": (
        TemplateSpec(5, "ellipse"),
        TemplateSpec(5, "diagonal"),
        TemplateSpec(5, "diagonal_transposed"),
        TemplateSpec(5, "triangle"),
        TemplateSpec(5, "row"),
        TemplateSpec(5, "column"),
        TemplateSpec(5, "centered_row"),
        TemplateSpec(5, "centered_column"),
        TemplateSpec(6, "block"),
        TemplateSpec(6, "block_transposed"),
    ),
    "C": (
        TemplateSpec(5, "ellipse"),
        TemplateSpec(5, "diagonal"),
        TemplateSpec(5, "diagonal_transposed"),
        TemplateSpec(7, "triangle"),
        TemplateSpec(7, "row"),
        TemplateSpec(7, "column"),
        TemplateSpec(8, "block"),
        TemplateSpec(8, "block_transposed"),
    ),
}


def generate_partition(height: int, width: int, players: int, mode: str, alpha: float = 2.05) -> np.ndarray:
    partition = np.zeros((height, width), dtype=np.int64)

    if mode in {"row", "column"}:
        axis_length = height if mode == "row" else width
        base, extra = divmod(axis_length, players)
        cursor = 0
        for player in range(players):
            end = cursor + base + (1 if player < extra else 0)
            if mode == "row":
                partition[cursor:end, :] = player
            else:
                partition[:, cursor:end] = player
            cursor = end
        return partition

    if mode in {"block", "block_transposed"}:
        block_rows = int(math.sqrt(players))
        while players % block_rows != 0:
            block_rows -= 1
        block_cols = players // block_rows
        if mode == "block_transposed":
            block_rows, block_cols = block_cols, block_rows
        row_edges = np.linspace(0, height, block_rows + 1, dtype=int)
        col_edges = np.linspace(0, width, block_cols + 1, dtype=int)
        player = 0
        for row in range(block_rows):
            for col in range(block_cols):
                partition[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]] = player
                player += 1
        return partition

    rows, cols = np.indices((height, width))
    center_row, center_col = (height - 1) / 2.0, (width - 1) / 2.0

    if mode == "centered_row":
        y = (rows - center_row) / (center_row if center_row > 0 else 1.0)
        values = (np.sign(y) * np.abs(y) ** alpha + 1.0) / 2.0
    elif mode == "centered_column":
        y = (cols - center_col) / (center_col if center_col > 0 else 1.0)
        values = (np.sign(y) * np.abs(y) ** alpha + 1.0) / 2.0
    elif mode == "diagonal":
        values = (rows + cols) / max(height + width - 2, 1)
    elif mode == "diagonal_transposed":
        values = (rows - cols + width - 1) / max(height + width - 2, 1)
    elif mode == "triangle":
        angles = np.mod(np.arctan2(rows - center_row, cols - center_col) + 2 * np.pi, 2 * np.pi)
        values = angles / (2 * np.pi)
    elif mode == "ellipse":
        row_scale = center_row if center_row > 0 else 1.0
        col_scale = center_col if center_col > 0 else 1.0
        distances = np.sqrt(((rows - center_row) / row_scale) ** 2 + ((cols - center_col) / col_scale) ** 2)
        values = distances / (distances.max() + 1e-12)
    else:
        raise ValueError(f"Unknown template mode: {mode}")

    return np.clip(np.floor(values * players).astype(np.int64), 0, players - 1)

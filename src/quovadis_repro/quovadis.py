from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .adapters import ModelAdapter
from .aggregation import aggregate_template_heatmaps
from .masking import apply_coalition_masks, canonical_removal_strategy, to_channel_last
from .templates import CONFIGURATIONS, TemplateSpec, generate_partition


def all_coalitions(players: int) -> np.ndarray:
    return np.asarray(list(itertools.product([0, 1], repeat=players)), dtype=np.float32)


def exact_banzhaf(values: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
    players = coalitions.shape[1]
    scores = np.zeros(players, dtype=np.float64)
    for player in range(players):
        absent = coalitions[:, player] == 0
        base_masks = coalitions[absent]
        with_player = base_masks.copy()
        with_player[:, player] = 1
        mask_to_index = {tuple(mask.astype(int)): idx for idx, mask in enumerate(coalitions)}
        base_values = values[absent]
        added_values = np.asarray([values[mask_to_index[tuple(mask.astype(int))]] for mask in with_player])
        scores[player] = np.mean(added_values - base_values)
    return scores


def exact_shapley(values: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
    players = coalitions.shape[1]
    mask_to_index = {tuple(mask.astype(int)): idx for idx, mask in enumerate(coalitions)}
    scores = np.zeros(players, dtype=np.float64)
    denominator = math.factorial(players)
    for player in range(players):
        for mask in coalitions:
            if mask[player] == 1:
                continue
            size = int(mask.sum())
            with_player = mask.copy()
            with_player[player] = 1
            weight = math.factorial(size) * math.factorial(players - size - 1) / denominator
            scores[player] += weight * (
                values[mask_to_index[tuple(with_player.astype(int))]]
                - values[mask_to_index[tuple(mask.astype(int))]]
            )
    return scores


@dataclass
class QuovadisExplainer:
    adapter: ModelAdapter
    solution: str = "banzhaf"
    aggregation: str = "median"
    configuration: str = "A"
    blur_kernel: tuple[int, int] = (11, 11)
    removal_strategy: str = "average_blur"
    black_value: float = 0.0
    template_names: tuple[str, ...] | None = None

    def _specs(self) -> tuple[TemplateSpec, ...]:
        base = CONFIGURATIONS[self.configuration.upper()]
        if self.template_names is None:
            return base
        wanted = tuple(self.template_names)
        by_name = {spec.name: spec for spec in CONFIGURATIONS["A"]}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise ValueError(f"Unknown QUO VADIS templates: {missing}")
        return tuple(by_name[name] for name in wanted)

    def _single_template(
        self,
        image: np.ndarray,
        target_class: int,
        spec: TemplateSpec,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_cl = to_channel_last(image, self.adapter.channel_first)
        height, width = image_cl.shape[:2]
        partition = generate_partition(height, width, spec.players, spec.mode)
        coalitions = all_coalitions(spec.players)
        masked_inputs = apply_coalition_masks(
            image=image,
            coalition_masks=coalitions,
            partition=partition,
            kernel_size=self.blur_kernel,
            channel_first=self.adapter.channel_first,
            strategy=canonical_removal_strategy(self.removal_strategy),
            black_value=self.black_value,
        )
        probabilities = self.adapter.predict_proba(masked_inputs)[:, int(target_class)]
        baseline = probabilities[0]
        values = probabilities - baseline

        if self.solution.lower() == "shapley":
            patch_scores = exact_shapley(values, coalitions)
        elif self.solution.lower() == "banzhaf":
            patch_scores = exact_banzhaf(values, coalitions)
            efficiency_gap = values[-1] - patch_scores.sum()
            patch_scores = patch_scores + efficiency_gap / spec.players
        else:
            raise ValueError("solution must be 'banzhaf' or 'shapley'")

        heatmap = patch_scores[partition]
        return heatmap.astype(np.float32), partition

    def explain(
        self,
        images: np.ndarray,
        target_classes: np.ndarray,
    ) -> np.ndarray:
        images = np.asarray(images)
        target_classes = np.asarray(target_classes).reshape(-1)
        if len(images) != len(target_classes):
            raise ValueError("images and target_classes must have the same length")
        specs = self._specs()

        per_template: list[np.ndarray] = []
        partitions: list[np.ndarray] = []
        for spec in specs:
            maps = []
            template_partition = None
            for image, target in zip(images, target_classes):
                heatmap, partition = self._single_template(image, int(target), spec)
                maps.append(heatmap)
                template_partition = partition
            per_template.append(np.stack(maps, axis=0))
            assert template_partition is not None
            partitions.append(template_partition)

        template_heatmaps = np.stack(per_template, axis=0)
        partition_array = np.stack(partitions, axis=0)
        aggregate = aggregate_template_heatmaps(
            template_heatmaps,
            strategy=self.aggregation,
            partitions=partition_array,
        )
        return aggregate

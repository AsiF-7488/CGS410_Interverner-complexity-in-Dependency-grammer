"""Reusable metrics for complexity, entropy, and distribution comparisons."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .settings import ComplexityConfig


def morphological_richness(feats: str) -> int:
    if not feats or feats == "_":
        return 0
    return len([part for part in feats.split("|") if part])


def shannon_entropy(values: list[str]) -> float:
    if not values:
        return float("nan")
    counts = Counter(values)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def complexity_score(
    *,
    arity: int,
    subtree_size: int,
    depth: int,
    upos: str,
    complexity: ComplexityConfig,
) -> float:
    pos_weight = complexity.pos_weights.get(upos, 1.0)
    return (
        complexity.arity_weight * arity
        + complexity.subtree_weight * subtree_size
        + complexity.depth_weight * depth
        + complexity.pos_weight * pos_weight
    )


def efficiency_ratio(dependency_distance: int, complexity_value: float) -> float:
    if complexity_value <= 0:
        return float("nan")
    return dependency_distance / complexity_value


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    mean_diff = np.mean(group_a) - np.mean(group_b)
    n_a = len(group_a)
    n_b = len(group_b)
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    if pooled <= 0:
        return float("nan")
    return mean_diff / math.sqrt(pooled)


def kl_divergence_from_counts(real_counts: dict, comparison_counts: dict) -> float:
    keys = sorted(set(real_counts) | set(comparison_counts))
    if not keys:
        return float("nan")
    real = np.array([real_counts.get(key, 0.0) for key in keys], dtype=float)
    comp = np.array([comparison_counts.get(key, 0.0) for key in keys], dtype=float)
    real = (real + 1e-9) / (real.sum() + 1e-9 * len(real))
    comp = (comp + 1e-9) / (comp.sum() + 1e-9 * len(comp))
    return float(np.sum(real * np.log(real / comp)))

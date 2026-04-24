"""Baseline generators for pre-ML comparison."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .research_features import SentenceStructure


@dataclass(frozen=True)
class BaselineRealization:
    name: str
    replicate: int
    position_map: dict[int, int]


def simplified_position_map(structure: SentenceStructure) -> dict[int, int]:
    original = structure.position_map
    subtree_size = structure.subtree_sizes

    def child_key(parent_id: int, child_id: int) -> tuple[int, int, int]:
        return (
            subtree_size[child_id],
            abs(original[parent_id] - original[child_id]),
            original[child_id],
        )

    def linearize(node_id: int) -> list[int]:
        children = structure.children[node_id]
        left_children = [child for child in children if original[child] < original[node_id]]
        right_children = [child for child in children if original[child] > original[node_id]]
        output: list[int] = []
        for child in sorted(left_children, key=lambda child: child_key(node_id, child), reverse=True):
            output.extend(linearize(child))
        output.append(node_id)
        for child in sorted(right_children, key=lambda child: child_key(node_id, child)):
            output.extend(linearize(child))
        return output

    order: list[int] = []
    for root in sorted(structure.roots, key=lambda token_id: original[token_id]):
        order.extend(linearize(root))
    return {token_id: index for index, token_id in enumerate(order, start=1)}


def fully_random_position_map(structure: SentenceStructure, rng: random.Random) -> dict[int, int]:
    token_ids = [token.token_id for token in structure.tokens]
    rng.shuffle(token_ids)
    return {token_id: index for index, token_id in enumerate(token_ids, start=1)}


def projective_random_position_map(structure: SentenceStructure, rng: random.Random) -> dict[int, int]:
    def linearize(node_id: int) -> list[int]:
        children = structure.children[node_id][:]
        rng.shuffle(children)
        left_children: list[int] = []
        right_children: list[int] = []
        for child in children:
            if rng.random() < 0.5:
                left_children.append(child)
            else:
                right_children.append(child)
        output: list[int] = []
        for child in left_children:
            output.extend(linearize(child))
        output.append(node_id)
        for child in right_children:
            output.extend(linearize(child))
        return output

    roots = structure.roots[:]
    rng.shuffle(roots)
    order: list[int] = []
    for root in roots:
        order.extend(linearize(root))
    return {token_id: index for index, token_id in enumerate(order, start=1)}


def grammar_constrained_position_map(structure: SentenceStructure, rng: random.Random) -> dict[int, int]:
    original = structure.position_map

    def shuffle_bucket(children: list[int]) -> list[int]:
        keyed = {}
        for child in children:
            token = structure.token_by_id[child]
            keyed.setdefault((token.deprel, token.upos), []).append(child)
        order: list[int] = []
        bucket_keys = list(keyed.keys())
        rng.shuffle(bucket_keys)
        for key in bucket_keys:
            bucket = keyed[key]
            rng.shuffle(bucket)
            order.extend(bucket)
        return order

    def linearize(node_id: int) -> list[int]:
        children = structure.children[node_id]
        left_children = [child for child in children if original[child] < original[node_id]]
        right_children = [child for child in children if original[child] > original[node_id]]
        output: list[int] = []
        for child in shuffle_bucket(left_children):
            output.extend(linearize(child))
        output.append(node_id)
        for child in shuffle_bucket(right_children):
            output.extend(linearize(child))
        return output

    order: list[int] = []
    for root in sorted(structure.roots, key=lambda token_id: original[token_id]):
        order.extend(linearize(root))
    return {token_id: index for index, token_id in enumerate(order, start=1)}


def generate_baseline_realizations(
    structure: SentenceStructure,
    *,
    baseline_names: list[str],
    random_runs: int,
    rng: random.Random,
) -> list[BaselineRealization]:
    realizations: list[BaselineRealization] = []
    for baseline_name in baseline_names:
        if baseline_name == "simplified":
            realizations.append(
                BaselineRealization(
                    name="simplified",
                    replicate=0,
                    position_map=simplified_position_map(structure),
                )
            )
            continue

        for replicate in range(random_runs):
            if baseline_name == "fully_random":
                position_map = fully_random_position_map(structure, rng)
            elif baseline_name == "projective_random":
                position_map = projective_random_position_map(structure, rng)
            elif baseline_name == "grammar_constrained":
                position_map = grammar_constrained_position_map(structure, rng)
            else:
                raise ValueError(f"Unsupported baseline: {baseline_name}")
            realizations.append(
                BaselineRealization(
                    name=baseline_name,
                    replicate=replicate,
                    position_map=position_map,
                )
            )
    return realizations

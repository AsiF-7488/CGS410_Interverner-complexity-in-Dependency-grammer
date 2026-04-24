"""Random and simplified baselines for intervener analysis."""

from __future__ import annotations

import random

from .features import SentenceStructure


def random_position_maps(
    structure: SentenceStructure,
    n_runs: int,
    rng: random.Random,
) -> list[dict[int, int]]:
    token_ids = [token.token_id for token in structure.tokens]
    position_maps: list[dict[int, int]] = []
    for _ in range(n_runs):
        shuffled = token_ids[:]
        rng.shuffle(shuffled)
        position_maps.append({token_id: index for index, token_id in enumerate(shuffled, start=1)})
    return position_maps


def simplified_position_map(structure: SentenceStructure) -> dict[int, int]:
    original_position = structure.position_map
    subtree_size = structure.subtree_sizes

    def child_key(parent_id: int, child_id: int) -> tuple[int, int, int]:
        return (
            subtree_size[child_id],
            abs(original_position[parent_id] - original_position[child_id]),
            original_position[child_id],
        )

    def linearize(node_id: int) -> list[int]:
        children = structure.children[node_id]
        left_children = [child_id for child_id in children if original_position[child_id] < original_position[node_id]]
        right_children = [child_id for child_id in children if original_position[child_id] > original_position[node_id]]

        output: list[int] = []
        for child_id in sorted(left_children, key=lambda child_id: child_key(node_id, child_id), reverse=True):
            output.extend(linearize(child_id))
        output.append(node_id)
        for child_id in sorted(right_children, key=lambda child_id: child_key(node_id, child_id)):
            output.extend(linearize(child_id))
        return output

    linearized: list[int] = []
    for root_id in sorted(structure.roots, key=lambda token_id: original_position[token_id]):
        linearized.extend(linearize(root_id))

    return {token_id: index for index, token_id in enumerate(linearized, start=1)}

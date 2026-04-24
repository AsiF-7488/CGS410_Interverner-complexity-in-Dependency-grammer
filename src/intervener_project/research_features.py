"""Advanced feature extraction for the team-ready pre-ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .conllu_io import Sentence, Token
from .metrics import complexity_score, efficiency_ratio, morphological_richness
from .settings import ComplexityConfig, LanguageSpec


@dataclass(slots=True)
class SentenceStructure:
    language: LanguageSpec
    sentence: Sentence
    tokens: list[Token]
    token_by_id: dict[int, Token]
    position_map: dict[int, int]
    children: dict[int, list[int]]
    roots: list[int]
    subtree_sizes: dict[int, int]
    depths: dict[int, int]


def build_sentence_structure(language: LanguageSpec, sentence: Sentence) -> SentenceStructure | None:
    tokens = [token for token in sentence.tokens if token.upos != "PUNCT"]
    if len(tokens) < 2:
        return None

    tokens.sort(key=lambda token: token.token_id)
    token_by_id = {token.token_id: token for token in tokens}
    position_map = {token.token_id: index for index, token in enumerate(tokens, start=1)}
    children = {token.token_id: [] for token in tokens}
    roots: list[int] = []

    for token in tokens:
        if token.head in token_by_id:
            children[token.head].append(token.token_id)
        else:
            roots.append(token.token_id)

    @lru_cache(maxsize=None)
    def subtree_size(node_id: int) -> int:
        return 1 + sum(subtree_size(child_id) for child_id in children[node_id])

    @lru_cache(maxsize=None)
    def depth(node_id: int) -> int:
        token = token_by_id[node_id]
        if token.head not in token_by_id:
            return 0
        return 1 + depth(token.head)

    return SentenceStructure(
        language=language,
        sentence=sentence,
        tokens=tokens,
        token_by_id=token_by_id,
        position_map=position_map,
        children=children,
        roots=sorted(roots, key=lambda token_id: position_map[token_id]),
        subtree_sizes={token_id: subtree_size(token_id) for token_id in token_by_id},
        depths={token_id: depth(token_id) for token_id in token_by_id},
    )


def dependency_direction(position_map: dict[int, int], head_id: int, dependent_id: int) -> str:
    return "right" if position_map[dependent_id] > position_map[head_id] else "left"


def interveners_between(position_map: dict[int, int], head_id: int, dependent_id: int) -> list[int]:
    left, right = sorted((position_map[head_id], position_map[dependent_id]))
    return [
        token_id
        for token_id, position in position_map.items()
        if left < position < right
    ]


def structural_role(intervener: Token, head: Token, dependent: Token) -> str:
    if intervener.head == head.token_id:
        return "modifies_head"
    if intervener.head == dependent.token_id:
        return "modifies_dependent"
    return "neither"


def extract_condition_rows(
    structure: SentenceStructure,
    *,
    position_map: dict[int, int],
    condition: str,
    replicate: int,
    complexity: ComplexityConfig,
    include_zero_distance: bool = False,
) -> tuple[list[dict], list[dict]]:
    intervener_rows: list[dict] = []
    dependency_rows: list[dict] = []

    for dependent in structure.tokens:
        if dependent.head == 0 or dependent.head not in position_map:
            continue

        head = structure.token_by_id[dependent.head]
        between_ids = sorted(
            interveners_between(position_map, head.token_id, dependent.token_id),
            key=lambda token_id: position_map[token_id],
        )
        dependency_distance = len(between_ids)
        if dependency_distance == 0 and not include_zero_distance:
            continue

        direction = dependency_direction(position_map, head.token_id, dependent.token_id)
        edge_complexities: list[float] = []

        for intervener_id in between_ids:
            intervener = structure.token_by_id[intervener_id]
            arity = len(structure.children[intervener_id])
            subtree_size = structure.subtree_sizes[intervener_id]
            depth = structure.depths[intervener_id]
            score = complexity_score(
                arity=arity,
                subtree_size=subtree_size,
                depth=depth,
                upos=intervener.upos,
                complexity=complexity,
            )
            edge_complexities.append(score)

            intervener_rows.append(
                {
                    "language": structure.language.key,
                    "language_label": structure.language.label,
                    "treebank": structure.language.corpus_id,
                    "typology": structure.language.typology,
                    "family": structure.language.family,
                    "region": structure.language.region,
                    "condition": condition,
                    "replicate": replicate,
                    "sentence_id": structure.sentence.sent_id,
                    "source_file": structure.sentence.source_file,
                    "token_id": intervener.token_id,
                    "head_id": head.token_id,
                    "dependent_id": dependent.token_id,
                    "intervener_id": intervener.token_id,
                    "dependency_relation": dependent.deprel,
                    "dependency_distance": dependency_distance,
                    "direction": direction,
                    "intervener_upos": intervener.upos,
                    "head_upos": head.upos,
                    "dependent_upos": dependent.upos,
                    "arity": arity,
                    "subtree_size": subtree_size,
                    "depth": depth,
                    "modifies": structural_role(intervener, head, dependent),
                    "morphological_richness": morphological_richness(intervener.feats),
                    "complexity_score": score,
                    "efficiency_ratio": efficiency_ratio(dependency_distance, score),
                    "head_form": head.form,
                    "dependent_form": dependent.form,
                    "intervener_form": intervener.form,
                }
            )

        dependency_rows.append(
            {
                "language": structure.language.key,
                "language_label": structure.language.label,
                "treebank": structure.language.corpus_id,
                "typology": structure.language.typology,
                "family": structure.language.family,
                "region": structure.language.region,
                "condition": condition,
                "replicate": replicate,
                "sentence_id": structure.sentence.sent_id,
                "head_id": head.token_id,
                "dependent_id": dependent.token_id,
                "dependency_relation": dependent.deprel,
                "dependency_distance": dependency_distance,
                "direction": direction,
                "head_upos": head.upos,
                "dependent_upos": dependent.upos,
                "num_interveners": dependency_distance,
                "mean_complexity_score": (
                    sum(edge_complexities) / len(edge_complexities) if edge_complexities else 0.0
                ),
            }
        )

    return intervener_rows, dependency_rows

"""Feature extraction for dependency edges and intervening tokens."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .conllu_io import Sentence, Token
from .config import LanguageConfig


@dataclass(slots=True)
class SentenceStructure:
    language: LanguageConfig
    sentence: Sentence
    tokens: list[Token]
    token_by_id: dict[int, Token]
    position_map: dict[int, int]
    children: dict[int, list[int]]
    subtree_sizes: dict[int, int]
    roots: list[int]


def build_sentence_structure(language: LanguageConfig, sentence: Sentence) -> SentenceStructure | None:
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

    subtree_sizes = {token_id: subtree_size(token_id) for token_id in token_by_id}

    return SentenceStructure(
        language=language,
        sentence=sentence,
        tokens=tokens,
        token_by_id=token_by_id,
        position_map=position_map,
        children=children,
        subtree_sizes=subtree_sizes,
        roots=sorted(roots, key=lambda token_id: position_map[token_id]),
    )


def _direction(head_position: int, dep_position: int) -> str:
    return "right" if dep_position > head_position else "left"


def _between_token_ids(position_map: dict[int, int], left: int, right: int) -> list[int]:
    return [
        token_id
        for token_id, position in position_map.items()
        if left < position < right
    ]


def extract_dependency_and_intervener_rows(
    structure: SentenceStructure,
    condition: str,
    replicate: int = 0,
    position_map: dict[int, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    if position_map is None:
        position_map = structure.position_map

    dependency_rows: list[dict] = []
    intervener_rows: list[dict] = []

    for dependent in structure.tokens:
        if dependent.head == 0 or dependent.head not in position_map:
            continue

        head = structure.token_by_id[dependent.head]
        head_position = position_map[head.token_id]
        dep_position = position_map[dependent.token_id]
        left, right = sorted((head_position, dep_position))
        between_ids = sorted(
            _between_token_ids(position_map, left, right),
            key=lambda token_id: position_map[token_id],
        )

        intervener_arities: list[int] = []
        intervener_subtree_sizes: list[int] = []

        for intervener_id in between_ids:
            intervener = structure.token_by_id[intervener_id]
            arity = len(structure.children[intervener_id])
            subtree = structure.subtree_sizes[intervener_id]
            intervener_arities.append(arity)
            intervener_subtree_sizes.append(subtree)

            intervener_rows.append(
                {
                    "language": structure.language.key,
                    "language_label": structure.language.label,
                    "treebank": structure.language.corpus_id,
                    "typology": structure.language.typology,
                    "family": structure.language.family,
                    "condition": condition,
                    "replicate": replicate,
                    "sentence_id": structure.sentence.sent_id,
                    "sentence_text": structure.sentence.text,
                    "source_file": structure.sentence.source_file,
                    "head_id": head.token_id,
                    "head_form": head.form,
                    "head_upos": head.upos,
                    "dependent_id": dependent.token_id,
                    "dependent_form": dependent.form,
                    "dependent_upos": dependent.upos,
                    "deprel": dependent.deprel,
                    "dependency_distance": len(between_ids),
                    "direction": _direction(head_position, dep_position),
                    "intervener_id": intervener.token_id,
                    "intervener_form": intervener.form,
                    "intervener_upos": intervener.upos,
                    "intervener_arity": arity,
                    "intervener_subtree_size": subtree,
                    "head_position": head_position,
                    "dependent_position": dep_position,
                    "intervener_position": position_map[intervener_id],
                }
            )

        dependency_rows.append(
            {
                "language": structure.language.key,
                "language_label": structure.language.label,
                "treebank": structure.language.corpus_id,
                "typology": structure.language.typology,
                "family": structure.language.family,
                "condition": condition,
                "replicate": replicate,
                "sentence_id": structure.sentence.sent_id,
                "sentence_text": structure.sentence.text,
                "source_file": structure.sentence.source_file,
                "head_id": head.token_id,
                "head_form": head.form,
                "head_upos": head.upos,
                "dependent_id": dependent.token_id,
                "dependent_form": dependent.form,
                "dependent_upos": dependent.upos,
                "deprel": dependent.deprel,
                "dependency_distance": len(between_ids),
                "direction": _direction(head_position, dep_position),
                "num_interveners": len(between_ids),
                "mean_intervener_arity": (
                    sum(intervener_arities) / len(intervener_arities) if intervener_arities else 0.0
                ),
                "mean_intervener_subtree_size": (
                    sum(intervener_subtree_sizes) / len(intervener_subtree_sizes)
                    if intervener_subtree_sizes
                    else 0.0
                ),
            }
        )

    return dependency_rows, intervener_rows

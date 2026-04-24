"""Minimal CoNLL-U reading utilities without external parser dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(slots=True)
class Token:
    token_id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str


@dataclass(slots=True)
class Sentence:
    sent_id: str
    text: str
    tokens: list[Token]
    source_file: str
    source_index: int


def _parse_token_line(line: str) -> Token | None:
    parts = line.split("\t")
    if len(parts) != 10:
        return None

    raw_id = parts[0]
    if "-" in raw_id or "." in raw_id:
        return None

    token_id = int(raw_id)
    head = 0 if parts[6] == "_" else int(parts[6])
    return Token(
        token_id=token_id,
        form=parts[1],
        lemma=parts[2],
        upos=parts[3],
        xpos=parts[4],
        feats=parts[5],
        head=head,
        deprel=parts[7],
        deps=parts[8],
        misc=parts[9],
    )


def parse_conllu(path: str | Path) -> Iterator[Sentence]:
    path = Path(path)
    comments: dict[str, str] = {}
    tokens: list[Token] = []
    sentence_index = 0

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                if tokens:
                    sentence_index += 1
                    sent_id = comments.get("sent_id", f"{path.stem}-{sentence_index}")
                    text = comments.get("text", "")
                    yield Sentence(
                        sent_id=sent_id,
                        text=text,
                        tokens=tokens,
                        source_file=path.name,
                        source_index=sentence_index,
                    )
                comments = {}
                tokens = []
                continue

            if line.startswith("#"):
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    comments[key.strip()] = value.strip()
                continue

            token = _parse_token_line(line)
            if token is not None:
                tokens.append(token)

    if tokens:
        sentence_index += 1
        sent_id = comments.get("sent_id", f"{path.stem}-{sentence_index}")
        text = comments.get("text", "")
        yield Sentence(
            sent_id=sent_id,
            text=text,
            tokens=tokens,
            source_file=path.name,
            source_index=sentence_index,
        )


def iter_treebank_sentences(treebank_dir: str | Path) -> Iterator[Sentence]:
    treebank_dir = Path(treebank_dir)
    for path in sorted(treebank_dir.rglob("*.conllu")):
        yield from parse_conllu(path)

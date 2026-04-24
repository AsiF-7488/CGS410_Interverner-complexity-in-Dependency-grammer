"""End-to-end pipeline for points 1-3 of the project."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from .analysis import build_analysis_tables
from .baselines import random_position_maps, simplified_position_map
from .config import (
    DEFAULT_RANDOM_RUNS,
    DEFAULT_SEED,
    FIGURES_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    TABLES_DIR,
)
from .conllu_io import iter_treebank_sentences
from .features import build_sentence_structure, extract_dependency_and_intervener_rows
from .treebanks import ensure_selected_treebanks
from .visualize import save_all_figures


def run_full_pipeline(
    language_keys: list[str] | None = None,
    download: bool = True,
    force_download: bool = False,
    max_sentences: int | None = None,
    random_runs: int = DEFAULT_RANDOM_RUNS,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    rng = random.Random(seed)

    available_treebanks, unavailable_languages = ensure_selected_treebanks(
        language_keys=language_keys,
        force_download=force_download if download else False,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dependencies_path = PROCESSED_DIR / "dependencies.csv"
    interveners_path = PROCESSED_DIR / "interveners.csv"
    sentence_metadata_path = PROCESSED_DIR / "sentence_metadata.csv"
    for path in [dependencies_path, interveners_path, sentence_metadata_path]:
        if path.exists():
            path.unlink()

    for language, treebank_dir in available_treebanks:
        dependency_rows: list[dict] = []
        intervener_rows: list[dict] = []
        metadata_rows: list[dict] = []
        sentence_counter = 0
        for sentence in iter_treebank_sentences(treebank_dir):
            structure = build_sentence_structure(language, sentence)
            if structure is None:
                continue

            real_deps, real_ints = extract_dependency_and_intervener_rows(
                structure,
                condition="real",
            )
            dependency_rows.extend(real_deps)
            intervener_rows.extend(real_ints)

            simplified_map = simplified_position_map(structure)
            simple_deps, simple_ints = extract_dependency_and_intervener_rows(
                structure,
                condition="simplified",
                position_map=simplified_map,
            )
            dependency_rows.extend(simple_deps)
            intervener_rows.extend(simple_ints)

            for replicate, position_map in enumerate(random_position_maps(structure, random_runs, rng)):
                random_deps, random_ints = extract_dependency_and_intervener_rows(
                    structure,
                    condition="random",
                    replicate=replicate,
                    position_map=position_map,
                )
                dependency_rows.extend(random_deps)
                intervener_rows.extend(random_ints)

            metadata_rows.append(
                {
                    "language": language.key,
                    "language_label": language.label,
                    "treebank": language.corpus_id,
                    "typology": language.typology,
                    "family": language.family,
                    "sentence_id": sentence.sent_id,
                    "source_file": sentence.source_file,
                    "token_count_no_punct": len(structure.tokens),
                }
            )

            sentence_counter += 1
            if max_sentences is not None and sentence_counter >= max_sentences:
                break

        _append_rows(dependencies_path, dependency_rows)
        _append_rows(interveners_path, intervener_rows)
        _append_rows(sentence_metadata_path, metadata_rows)

    dependencies = pd.read_csv(dependencies_path)
    interveners = pd.read_csv(interveners_path)
    sentence_metadata = pd.read_csv(sentence_metadata_path)

    tables = build_analysis_tables(dependencies, interveners, TABLES_DIR)
    save_all_figures(dependencies, interveners, tables, FIGURES_DIR)
    report_path = _write_report(
        dependencies=dependencies,
        interveners=interveners,
        tables=tables,
        unavailable_languages=unavailable_languages,
        max_sentences=max_sentences,
        random_runs=random_runs,
        seed=seed,
    )

    metadata_path = _write_run_metadata(
        unavailable_languages=unavailable_languages,
        random_runs=random_runs,
        seed=seed,
        max_sentences=max_sentences,
    )

    return {
        "dependencies_path": dependencies_path,
        "interveners_path": interveners_path,
        "sentence_metadata_path": sentence_metadata_path,
        "report_path": report_path,
        "run_metadata_path": metadata_path,
        "figure_dir": FIGURES_DIR,
        "table_dir": TABLES_DIR,
    }


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", header=write_header, index=False)


def _write_run_metadata(
    unavailable_languages: list,
    random_runs: int,
    seed: int,
    max_sentences: int | None,
) -> Path:
    payload = {
        "random_runs": random_runs,
        "seed": seed,
        "max_sentences": max_sentences,
        "unavailable_languages": [
            {
                "key": language.key,
                "label": language.label,
                "notes": language.notes,
            }
            for language in unavailable_languages
        ],
    }
    path = TABLES_DIR / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_report(
    dependencies: pd.DataFrame,
    interveners: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    unavailable_languages: list,
    max_sentences: int | None,
    random_runs: int,
    seed: int,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    real_interveners = interveners[interveners["condition"] == "real"]
    real_dependencies = dependencies[dependencies["condition"] == "real"]
    top_pos = (
        real_interveners["intervener_upos"].value_counts().head(5).to_dict()
        if not real_interveners.empty
        else {}
    )
    mean_arity = float(real_interveners["intervener_arity"].mean()) if not real_interveners.empty else float("nan")
    mean_subtree = (
        float(real_interveners["intervener_subtree_size"].mean()) if not real_interveners.empty else float("nan")
    )
    mean_distance = (
        float(real_dependencies["dependency_distance"].mean()) if not real_dependencies.empty else float("nan")
    )

    report = f"""# Intervener Complexity Analysis Report

## Scope

This report covers points 1-3 of the project:

- multilingual SUD data loading
- intervener feature extraction
- descriptive and inferential analysis with real, random, and simplified baselines

## Hypothesis

Intervening words in dependency relations tend to be structurally simple: low-arity tokens, relatively small subtrees, and a skew toward a small set of UPOS categories.

## Data

- Languages analyzed: `Marathi, Catalan, Galician, Persian, Tamil, Telugu, Indonesian`
- Random baseline runs per sentence: `{random_runs}`
- Random seed: `{seed}`
- Sentence cap per language: `{max_sentences if max_sentences is not None else "None"}`
- Languages requested but unavailable in current public SUD release: `{", ".join(language.label for language in unavailable_languages) if unavailable_languages else "None"}`

## Methodology

For each dependency arc with a non-root head, the pipeline collects all non-punctuation tokens strictly between the head and dependent. For each intervener, it computes:

- UPOS
- arity
- subtree size
- dependency distance of the arc
- left/right direction of the dependency

The analysis compares:

- real sentence order
- random token-order baselines
- a simplified head-proximal baseline

## High-Level Results

- Mean intervener arity in real data: `{mean_arity:.3f}`
- Mean intervener subtree size in real data: `{mean_subtree:.3f}`
- Mean dependency distance in real data: `{mean_distance:.3f}`
- Top real-data intervener POS categories: `{top_pos}`

## Generated Outputs

- Tables: `data/outputs/tables/`
- Figures: `data/outputs/figures/`
- Processed data: `data/processed/`

## Notes

- The typology analysis is implemented generically, but with the requested language set the public SUD data primarily covers `SOV` and `SVO`; a `VSO` comparison will require adding at least one VSO language.
- The Dravidian-focused comparison in this 7-language version contrasts Tamil and Telugu with the non-Dravidian languages in the selected set.
"""
    path = REPORTS_DIR / "analysis_report_points_1_3.md"
    path.write_text(report, encoding="utf-8")
    return path

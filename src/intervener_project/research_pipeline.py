"""Production pre-ML pipeline for distributed, config-driven execution."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .merge import merge_all_results
from .pre_ml_analysis import (
    build_condition_summary,
    build_distribution_data,
    build_intervener_features,
    build_kl_divergence,
    build_language_summary,
    build_statistical_tests,
    build_zscore_results,
)
from .research_baselines import generate_baseline_realizations
from .research_features import build_sentence_structure, extract_condition_rows
from .schemas import KL_COLUMNS, ML_RESULTS_COLUMNS, STATISTICAL_TEST_COLUMNS
from .settings import ProjectSettings
from .treebanks import ensure_configured_treebanks
from .validators import write_validated_csv
from .visualize_pre_ml import save_language_figures
from .conllu_io import iter_treebank_sentences


def run_pre_ml_pipeline(
    settings: ProjectSettings,
    *,
    download: bool = True,
    force_download: bool = False,
    merge_results: bool = False,
) -> dict[str, object]:
    logger = _configure_logging(settings.log_root)
    logger.info("Starting pre-ML pipeline with languages=%s", [language.key for language in settings.languages])
    rng = random.Random(settings.random_seed)

    output_root = settings.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    configured_treebanks = ensure_configured_treebanks(settings.languages, force_download=force_download if download else False)

    language_outputs: dict[str, Path] = {}
    for language, treebank_dir in configured_treebanks:
        logger.info("Processing language=%s treebank=%s", language.key, treebank_dir)
        language_output_dir = output_root / language.key
        language_output_dir.mkdir(parents=True, exist_ok=True)
        real_feature_rows: list[dict] = []
        all_feature_rows: list[dict] = []
        real_dependency_rows: list[dict] = []
        all_dependency_rows: list[dict] = []
        sentence_count = 0

        for sentence in iter_treebank_sentences(treebank_dir):
            structure = build_sentence_structure(language, sentence)
            if structure is None:
                continue

            real_features, real_dependencies = extract_condition_rows(
                structure,
                position_map=structure.position_map,
                condition="real",
                replicate=0,
                complexity=settings.complexity,
                include_zero_distance=settings.include_zero_distance_dependencies,
            )
            real_feature_rows.extend(real_features)
            real_dependency_rows.extend(real_dependencies)
            all_feature_rows.extend(real_features)
            all_dependency_rows.extend(real_dependencies)

            for realization in generate_baseline_realizations(
                structure,
                baseline_names=settings.baselines,
                random_runs=settings.random_runs,
                rng=rng,
            ):
                baseline_features, baseline_dependencies = extract_condition_rows(
                    structure,
                    position_map=realization.position_map,
                    condition=realization.name,
                    replicate=realization.replicate,
                    complexity=settings.complexity,
                    include_zero_distance=settings.include_zero_distance_dependencies,
                )
                all_feature_rows.extend(baseline_features)
                all_dependency_rows.extend(baseline_dependencies)

            sentence_count += 1
            if settings.max_sentences is not None and sentence_count >= settings.max_sentences:
                break

        real_features_df = pd.DataFrame(real_feature_rows)
        all_features_df = pd.DataFrame(all_feature_rows)
        real_dependencies_df = pd.DataFrame(real_dependency_rows)
        all_dependencies_df = pd.DataFrame(all_dependency_rows)

        intervener_features_path = write_validated_csv(
            build_intervener_features(real_features_df),
            language_output_dir / "intervener_features.csv",
            "intervener_features",
        )
        language_summary_path = write_validated_csv(
            build_language_summary(real_features_df, real_dependencies_df),
            language_output_dir / "language_summary.csv",
            "language_summary",
        )
        distribution_path = write_validated_csv(
            build_distribution_data(all_features_df),
            language_output_dir / "distribution_data.csv",
            "distribution_data",
        )
        zscore_path = write_validated_csv(
            build_zscore_results(all_features_df, all_dependencies_df),
            language_output_dir / "zscore_results.csv",
            "zscore_results",
        )

        empty_ml = pd.DataFrame(columns=ML_RESULTS_COLUMNS)
        write_validated_csv(empty_ml, language_output_dir / "ml_results.csv", "ml_results")

        condition_summary = build_condition_summary(all_features_df, all_dependencies_df)
        condition_summary.to_csv(language_output_dir / "condition_summary.csv", index=False)

        statistical_tests = build_statistical_tests(all_features_df)
        if statistical_tests.empty:
            statistical_tests = pd.DataFrame(columns=STATISTICAL_TEST_COLUMNS)
        statistical_tests.to_csv(language_output_dir / "statistical_tests.csv", index=False)

        kl_divergence = build_kl_divergence(all_features_df)
        if kl_divergence.empty:
            kl_divergence = pd.DataFrame(columns=KL_COLUMNS)
        kl_divergence.to_csv(language_output_dir / "kl_divergence.csv", index=False)

        save_language_figures(language_output_dir / "figures", all_features_df, pd.read_csv(zscore_path))
        _write_run_log(
            language_output_dir / "run_log.json",
            language=language.key,
            sentence_count=sentence_count,
            dependency_count=len(real_dependencies_df),
            intervener_count=len(real_features_df),
        )

        language_outputs[language.key] = language_output_dir
        logger.info(
            "Completed language=%s sentences=%s dependencies=%s interveners=%s",
            language.key,
            sentence_count,
            len(real_dependencies_df),
            len(real_features_df),
        )

    merged_outputs = None
    if merge_results:
        merged_outputs = merge_all_results(output_root, settings.final_output_root, settings)
        logger.info("Merged outputs written to %s", settings.final_output_root)

    return {
        "language_outputs": language_outputs,
        "merged_outputs": merged_outputs,
        "output_root": output_root,
    }


def _configure_logging(log_root: Path) -> logging.Logger:
    log_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("intervener_project_pre_ml")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_root / "pre_ml_pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _write_run_log(path: Path, *, language: str, sentence_count: int, dependency_count: int, intervener_count: int) -> None:
    payload = {
        "language": language,
        "sentence_count": sentence_count,
        "dependency_count": dependency_count,
        "intervener_count": intervener_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

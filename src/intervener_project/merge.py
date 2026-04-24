"""Merging and global pre-ML analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from .reporting import write_pre_ml_report
from .schemas import KL_COLUMNS, ML_RESULTS_COLUMNS, STATISTICAL_TEST_COLUMNS
from .validators import read_validated_csv, write_validated_csv
from .visualize_pre_ml import save_global_figures


def merge_all_results(output_root: Path, final_output_root: Path, settings) -> dict[str, Path]:
    final_output_root.mkdir(parents=True, exist_ok=True)
    language_dirs = [path for path in sorted(output_root.iterdir()) if path.is_dir()]

    features = _concat(language_dirs, "intervener_features.csv", "intervener_features")
    summaries = _concat(language_dirs, "language_summary.csv", "language_summary")
    distributions = _concat(language_dirs, "distribution_data.csv", "distribution_data")
    ml_results = _concat(language_dirs, "ml_results.csv", "ml_results")
    zscores = _concat(language_dirs, "zscore_results.csv", "zscore_results")
    statistical_tests = _concat_extra(language_dirs, "statistical_tests.csv", STATISTICAL_TEST_COLUMNS)
    kl_results = _concat_extra(language_dirs, "kl_divergence.csv", KL_COLUMNS)

    all_features_path = write_validated_csv(features, final_output_root / "all_features.csv", "intervener_features")
    all_summary_path = write_validated_csv(summaries, final_output_root / "all_language_summary.csv", "language_summary")
    all_distributions_path = write_validated_csv(distributions, final_output_root / "all_distributions.csv", "distribution_data")
    all_ml_path = write_validated_csv(ml_results, final_output_root / "all_ml_results.csv", "ml_results")
    all_zscores_path = write_validated_csv(zscores, final_output_root / "all_zscores.csv", "zscore_results")

    statistical_tests.to_csv(final_output_root / "all_statistical_tests.csv", index=False)
    kl_results.to_csv(final_output_root / "all_kl_divergence.csv", index=False)

    global_typology = (
        features.groupby(["typology"])
        .agg(
            avg_complexity=("complexity_score", "mean"),
            avg_dependency_length=("dependency_distance", "mean"),
            avg_arity=("arity", "mean"),
            avg_subtree_size=("subtree_size", "mean"),
            avg_depth=("depth", "mean"),
            avg_efficiency_ratio=("efficiency_ratio", "mean"),
        )
        .reset_index()
    )
    global_typology.to_csv(final_output_root / "global_typology_summary.csv", index=False)

    global_left_right = (
        features.groupby(["language", "direction"])
        .agg(
            avg_complexity=("complexity_score", "mean"),
            avg_dependency_length=("dependency_distance", "mean"),
        )
        .reset_index()
    )
    global_left_right.to_csv(final_output_root / "global_left_right_summary.csv", index=False)

    correlations = summaries.select_dtypes(include=["number"]).corr().reset_index()
    correlations.to_csv(final_output_root / "global_correlations.csv", index=False)

    mixed_effects = _fit_mixed_effects(features)
    mixed_effects.to_csv(final_output_root / "mixed_effects_results.csv", index=False)

    save_global_figures(final_output_root, features, summaries, zscores, distributions)
    report_path = write_pre_ml_report(settings, final_output_root, summaries, zscores, correlations)

    return {
        "all_features": all_features_path,
        "all_language_summary": all_summary_path,
        "all_distributions": all_distributions_path,
        "all_ml_results": all_ml_path,
        "all_zscores": all_zscores_path,
        "report": report_path,
    }


def _concat(language_dirs: list[Path], filename: str, schema_name: str) -> pd.DataFrame:
    frames = [read_validated_csv(language_dir / filename, schema_name) for language_dir in language_dirs if (language_dir / filename).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _concat_extra(language_dirs: list[Path], filename: str, columns: list[str]) -> pd.DataFrame:
    frames = []
    for language_dir in language_dirs:
        path = language_dir / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(frames, ignore_index=True)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns + [column for column in frame.columns if column not in columns]]


def _fit_mixed_effects(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or features["language"].nunique() < 2:
        return pd.DataFrame(columns=["term", "coefficient", "std_error", "p_value"])
    try:
        model = smf.mixedlm(
            "complexity_score ~ dependency_distance + C(direction) + C(modifies) + C(typology)",
            data=features,
            groups=features["language"],
        )
        result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        return pd.DataFrame(
            {
                "term": result.params.index,
                "coefficient": result.params.values,
                "std_error": result.bse.values,
                "p_value": result.pvalues.values,
            }
        )
    except Exception as exc:
        return pd.DataFrame(
            [{"term": "model_error", "coefficient": float("nan"), "std_error": float("nan"), "p_value": str(exc)}]
        )

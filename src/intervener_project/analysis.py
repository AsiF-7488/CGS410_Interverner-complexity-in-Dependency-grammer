"""Statistical summaries for intervener-complexity analyses."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_analysis_tables(
    dependencies: pd.DataFrame,
    interveners: pd.DataFrame,
    output_tables_dir: Path,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    tables["arity_histogram"] = (
        interveners.groupby(["condition", "language", "language_label", "intervener_arity"])
        .size()
        .reset_index(name="count")
        .sort_values(["condition", "language_label", "intervener_arity"])
    )

    tables["pos_distribution"] = (
        interveners.groupby(["condition", "language", "language_label", "intervener_upos"])
        .size()
        .reset_index(name="count")
        .sort_values(["condition", "language_label", "count"], ascending=[True, True, False])
    )

    tables["dependency_length_complexity"] = (
        interveners.groupby(["condition", "language", "language_label", "dependency_distance"])
        .agg(
            mean_intervener_arity=("intervener_arity", "mean"),
            mean_intervener_subtree_size=("intervener_subtree_size", "mean"),
            count=("intervener_arity", "size"),
        )
        .reset_index()
        .sort_values(["condition", "language_label", "dependency_distance"])
    )

    tables["language_condition_summary"] = _language_condition_summary(dependencies, interveners)
    tables["z_scores"] = _z_scores(tables["language_condition_summary"])
    tables["statistical_tests"] = _real_vs_random_tests(dependencies, interveners)
    tables["typology_summary"] = _typology_summary(dependencies, interveners)
    tables["directionality_summary"] = _directionality_summary(dependencies, interveners)
    tables["dravidian_summary"] = _dravidian_summary(dependencies, interveners)
    tables["condition_comparison"] = _condition_comparison(dependencies, interveners)

    for name, table in tables.items():
        _write_table(table, output_tables_dir / f"{name}.csv")

    return tables


def _language_condition_summary(dependencies: pd.DataFrame, interveners: pd.DataFrame) -> pd.DataFrame:
    intervener_summary = (
        interveners.groupby(
            ["condition", "replicate", "language", "language_label", "typology", "family"]
        )
        .agg(
            mean_arity=("intervener_arity", "mean"),
            median_arity=("intervener_arity", "median"),
            mean_subtree_size=("intervener_subtree_size", "mean"),
            median_subtree_size=("intervener_subtree_size", "median"),
            mean_dependency_distance=("dependency_distance", "mean"),
            intervener_count=("intervener_id", "size"),
        )
        .reset_index()
    )

    dependency_summary = (
        dependencies.groupby(
            ["condition", "replicate", "language", "language_label", "typology", "family"]
        )
        .agg(
            mean_interveners_per_dependency=("num_interveners", "mean"),
            share_with_interveners=("num_interveners", lambda values: float((values > 0).mean())),
            dependency_count=("head_id", "size"),
        )
        .reset_index()
    )

    return dependency_summary.merge(
        intervener_summary,
        on=["condition", "replicate", "language", "language_label", "typology", "family"],
        how="left",
    )


def _z_scores(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_arity",
        "mean_subtree_size",
        "mean_dependency_distance",
        "mean_interveners_per_dependency",
        "share_with_interveners",
    ]
    rows: list[dict] = []

    for language, subset in summary.groupby("language"):
        real_rows = subset[subset["condition"] == "real"]
        random_rows = subset[subset["condition"] == "random"]
        if real_rows.empty or random_rows.empty:
            continue

        real_row = real_rows.iloc[0]
        for metric in metrics:
            real_value = float(real_row[metric])
            random_values = random_rows[metric].dropna().astype(float)
            random_mean = float(random_values.mean())
            random_std = float(random_values.std(ddof=0))
            if math.isclose(random_std, 0.0):
                z_score = np.nan
            else:
                z_score = (real_value - random_mean) / random_std
            rows.append(
                {
                    "language": language,
                    "language_label": real_row["language_label"],
                    "metric": metric,
                    "real_value": real_value,
                    "random_mean": random_mean,
                    "random_std": random_std,
                    "z_score": z_score,
                }
            )

    return pd.DataFrame(rows)


def _real_vs_random_tests(dependencies: pd.DataFrame, interveners: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = [
        ("intervener_arity", interveners),
        ("intervener_subtree_size", interveners),
        ("dependency_distance", dependencies[dependencies["num_interveners"] > 0]),
        ("num_interveners", dependencies),
    ]

    for language in sorted(set(dependencies["language"])):
        for metric, frame in metrics:
            real_values = frame[(frame["language"] == language) & (frame["condition"] == "real")][metric]
            random_values = frame[(frame["language"] == language) & (frame["condition"] == "random")][metric]
            simplified_values = frame[(frame["language"] == language) & (frame["condition"] == "simplified")][metric]

            for comparison_name, comparison_values in [
                ("real_vs_random", random_values),
                ("real_vs_simplified", simplified_values),
            ]:
                if real_values.empty or comparison_values.empty:
                    continue
                mann = stats.mannwhitneyu(real_values, comparison_values, alternative="two-sided")
                ks = stats.ks_2samp(real_values, comparison_values)
                rows.append(
                    {
                        "language": language,
                        "metric": metric,
                        "comparison": comparison_name,
                        "real_mean": float(real_values.mean()),
                        "comparison_mean": float(comparison_values.mean()),
                        "mannwhitney_u": float(mann.statistic),
                        "mannwhitney_p": float(mann.pvalue),
                        "ks_statistic": float(ks.statistic),
                        "ks_p": float(ks.pvalue),
                    }
                )

    return pd.DataFrame(rows)


def _typology_summary(dependencies: pd.DataFrame, interveners: pd.DataFrame) -> pd.DataFrame:
    dependency_summary = (
        dependencies.groupby(["condition", "typology"])
        .agg(
            mean_distance=("dependency_distance", "mean"),
            mean_interveners=("num_interveners", "mean"),
            dependency_count=("head_id", "size"),
        )
        .reset_index()
    )

    intervener_summary = (
        interveners.groupby(["condition", "typology"])
        .agg(
            mean_arity=("intervener_arity", "mean"),
            mean_subtree_size=("intervener_subtree_size", "mean"),
            intervener_count=("intervener_id", "size"),
        )
        .reset_index()
    )

    return dependency_summary.merge(intervener_summary, on=["condition", "typology"], how="left")


def _directionality_summary(dependencies: pd.DataFrame, interveners: pd.DataFrame) -> pd.DataFrame:
    dependency_summary = (
        dependencies.groupby(["condition", "language", "language_label", "direction"])
        .agg(
            mean_distance=("dependency_distance", "mean"),
            mean_interveners=("num_interveners", "mean"),
            dependency_count=("head_id", "size"),
        )
        .reset_index()
    )

    intervener_summary = (
        interveners.groupby(["condition", "language", "language_label", "direction"])
        .agg(
            mean_arity=("intervener_arity", "mean"),
            mean_subtree_size=("intervener_subtree_size", "mean"),
            intervener_count=("intervener_id", "size"),
        )
        .reset_index()
    )

    return dependency_summary.merge(
        intervener_summary,
        on=["condition", "language", "language_label", "direction"],
        how="left",
    )


def _dravidian_summary(dependencies: pd.DataFrame, interveners: pd.DataFrame) -> pd.DataFrame:
    def group_label(row: pd.Series) -> str:
        if row["language"] == "tamil":
            return "Tamil"
        if row["family"] == "Dravidian":
            return "Other Dravidian"
        return "Non-Dravidian"

    dep = dependencies.copy()
    dep["comparison_group"] = dep.apply(group_label, axis=1)
    ints = interveners.copy()
    ints["comparison_group"] = ints.apply(group_label, axis=1)

    dependency_summary = (
        dep.groupby(["condition", "comparison_group"])
        .agg(
            mean_distance=("dependency_distance", "mean"),
            mean_interveners=("num_interveners", "mean"),
            dependency_count=("head_id", "size"),
        )
        .reset_index()
    )
    intervener_summary = (
        ints.groupby(["condition", "comparison_group"])
        .agg(
            mean_arity=("intervener_arity", "mean"),
            mean_subtree_size=("intervener_subtree_size", "mean"),
            intervener_count=("intervener_id", "size"),
        )
        .reset_index()
    )
    return dependency_summary.merge(
        intervener_summary,
        on=["condition", "comparison_group"],
        how="left",
    )


def _condition_comparison(dependencies: pd.DataFrame, interveners: pd.DataFrame) -> pd.DataFrame:
    dependency_summary = (
        dependencies.groupby(["condition", "language", "language_label"])
        .agg(
            mean_distance=("dependency_distance", "mean"),
            mean_interveners=("num_interveners", "mean"),
            dependency_count=("head_id", "size"),
        )
        .reset_index()
    )
    intervener_summary = (
        interveners.groupby(["condition", "language", "language_label"])
        .agg(
            mean_arity=("intervener_arity", "mean"),
            mean_subtree_size=("intervener_subtree_size", "mean"),
            intervener_count=("intervener_id", "size"),
        )
        .reset_index()
    )
    return dependency_summary.merge(
        intervener_summary,
        on=["condition", "language", "language_label"],
        how="left",
    )

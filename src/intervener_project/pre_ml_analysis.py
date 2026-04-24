"""Per-language analysis helpers for the production pre-ML workflow."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import cohen_d, kl_divergence_from_counts, shannon_entropy


NUMERIC_METRICS = [
    "dependency_distance",
    "arity",
    "subtree_size",
    "depth",
    "complexity_score",
    "efficiency_ratio",
]


def build_intervener_features(real_features: pd.DataFrame) -> pd.DataFrame:
    return real_features.copy()


def build_language_summary(real_features: pd.DataFrame, real_dependencies: pd.DataFrame) -> pd.DataFrame:
    most_common_pos = (
        real_features["intervener_upos"].mode().iloc[0] if not real_features.empty else None
    )
    summary = {
        "language": real_features["language"].iloc[0] if not real_features.empty else real_dependencies["language"].iloc[0],
        "avg_dependency_length": float(real_dependencies["dependency_distance"].mean()) if not real_dependencies.empty else float("nan"),
        "avg_complexity": float(real_features["complexity_score"].mean()) if not real_features.empty else float("nan"),
        "avg_arity": float(real_features["arity"].mean()) if not real_features.empty else float("nan"),
        "avg_subtree_size": float(real_features["subtree_size"].mean()) if not real_features.empty else float("nan"),
        "avg_depth": float(real_features["depth"].mean()) if not real_features.empty else float("nan"),
        "percent_left_dependencies": float((real_dependencies["direction"] == "left").mean()) if not real_dependencies.empty else float("nan"),
        "percent_right_dependencies": float((real_dependencies["direction"] == "right").mean()) if not real_dependencies.empty else float("nan"),
        "most_common_pos": most_common_pos,
        "entropy_pos_distribution": shannon_entropy(real_features["intervener_upos"].tolist()),
        "avg_efficiency_ratio": float(real_features["efficiency_ratio"].dropna().mean()) if not real_features.empty else float("nan"),
    }
    return pd.DataFrame([summary])


def build_distribution_data(all_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for condition, frame in all_features.groupby("condition"):
        for metric in NUMERIC_METRICS:
            for value in frame[metric].dropna().tolist():
                rows.append(
                    {
                        "language": frame["language"].iloc[0],
                        "metric_type": f"{condition}:{metric}",
                        "value": value,
                    }
                )
        for value in frame["intervener_upos"].tolist():
            rows.append(
                {
                    "language": frame["language"].iloc[0],
                    "metric_type": f"{condition}:intervener_upos",
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def aggregate_condition_metrics(features: pd.DataFrame, dependencies: pd.DataFrame) -> dict[str, float]:
    return {
        "avg_dependency_length": float(dependencies["dependency_distance"].mean()) if not dependencies.empty else float("nan"),
        "avg_complexity": float(features["complexity_score"].mean()) if not features.empty else float("nan"),
        "avg_arity": float(features["arity"].mean()) if not features.empty else float("nan"),
        "avg_subtree_size": float(features["subtree_size"].mean()) if not features.empty else float("nan"),
        "avg_depth": float(features["depth"].mean()) if not features.empty else float("nan"),
        "avg_efficiency_ratio": float(features["efficiency_ratio"].dropna().mean()) if not features.empty else float("nan"),
    }


def build_condition_summary(all_features: pd.DataFrame, all_dependencies: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for condition in sorted(all_features["condition"].unique()):
        feature_frame = all_features[all_features["condition"] == condition]
        dep_frame = all_dependencies[all_dependencies["condition"] == condition]
        row = {"language": feature_frame["language"].iloc[0], "condition": condition}
        row.update(aggregate_condition_metrics(feature_frame, dep_frame))
        rows.append(row)
    return pd.DataFrame(rows)


def build_zscore_results(all_features: pd.DataFrame, all_dependencies: pd.DataFrame) -> pd.DataFrame:
    language = all_features["language"].iloc[0]
    real_features = all_features[all_features["condition"] == "real"]
    real_dependencies = all_dependencies[all_dependencies["condition"] == "real"]
    real_metrics = aggregate_condition_metrics(real_features, real_dependencies)

    rows: list[dict] = []
    baseline_conditions = [condition for condition in sorted(all_features["condition"].unique()) if condition != "real"]
    for condition in baseline_conditions:
        condition_features = all_features[all_features["condition"] == condition]
        condition_dependencies = all_dependencies[all_dependencies["condition"] == condition]
        replicate_ids = sorted(condition_features["replicate"].unique())
        if not replicate_ids:
            continue
        replicate_metrics: list[dict[str, float]] = []
        for replicate in replicate_ids:
            feature_frame = condition_features[condition_features["replicate"] == replicate]
            dependency_frame = condition_dependencies[condition_dependencies["replicate"] == replicate]
            replicate_metrics.append(aggregate_condition_metrics(feature_frame, dependency_frame))

        for metric in real_metrics:
            replicate_values = np.array([replicate_metric[metric] for replicate_metric in replicate_metrics], dtype=float)
            random_mean = float(np.nanmean(replicate_values))
            random_std = float(np.nanstd(replicate_values))
            real_value = float(real_metrics[metric])
            if math.isclose(random_std, 0.0):
                z_score = float("nan")
            else:
                z_score = (real_value - random_mean) / random_std
            rows.append(
                {
                    "language": language,
                    "metric": f"{condition}:{metric}",
                    "real_value": real_value,
                    "random_mean": random_mean,
                    "random_std": random_std,
                    "z_score": z_score,
                }
            )
    return pd.DataFrame(rows)


def build_statistical_tests(all_features: pd.DataFrame) -> pd.DataFrame:
    language = all_features["language"].iloc[0]
    rows: list[dict] = []
    conditions = sorted(all_features["condition"].unique())

    for metric in NUMERIC_METRICS:
        metric_groups = [
            all_features[(all_features["condition"] == condition)][metric].dropna().astype(float).to_numpy()
            for condition in conditions
        ]
        metric_groups = [group for group in metric_groups if len(group) > 0]
        if len(metric_groups) >= 2:
            anova = stats.f_oneway(*metric_groups)
            rows.append(
                {
                    "language": language,
                    "metric": metric,
                    "comparison": "all_conditions",
                    "test_name": "anova",
                    "statistic": float(anova.statistic),
                    "p_value": float(anova.pvalue),
                    "effect_size": float("nan"),
                }
            )

        real_values = all_features[all_features["condition"] == "real"][metric].dropna().astype(float).to_numpy()
        for condition in conditions:
            if condition == "real":
                continue
            comparison_values = all_features[all_features["condition"] == condition][metric].dropna().astype(float).to_numpy()
            if len(real_values) == 0 or len(comparison_values) == 0:
                continue
            mann = stats.mannwhitneyu(real_values, comparison_values, alternative="two-sided")
            rows.append(
                {
                    "language": language,
                    "metric": metric,
                    "comparison": f"real_vs_{condition}",
                    "test_name": "mannwhitney_u",
                    "statistic": float(mann.statistic),
                    "p_value": float(mann.pvalue),
                    "effect_size": float(cohen_d(real_values, comparison_values)),
                }
            )

        left_values = all_features[(all_features["condition"] == "real") & (all_features["direction"] == "left")][metric].dropna().astype(float).to_numpy()
        right_values = all_features[(all_features["condition"] == "real") & (all_features["direction"] == "right")][metric].dropna().astype(float).to_numpy()
        if len(left_values) > 0 and len(right_values) > 0:
            mann = stats.mannwhitneyu(left_values, right_values, alternative="two-sided")
            rows.append(
                {
                    "language": language,
                    "metric": metric,
                    "comparison": "real_left_vs_right",
                    "test_name": "mannwhitney_u",
                    "statistic": float(mann.statistic),
                    "p_value": float(mann.pvalue),
                    "effect_size": float(cohen_d(left_values, right_values)),
                }
            )

    return pd.DataFrame(rows)


def build_kl_divergence(all_features: pd.DataFrame) -> pd.DataFrame:
    language = all_features["language"].iloc[0]
    real = all_features[all_features["condition"] == "real"]
    rows: list[dict] = []
    baseline_conditions = [condition for condition in sorted(all_features["condition"].unique()) if condition != "real"]

    real_arity_counts = real["arity"].value_counts().to_dict()
    real_pos_counts = real["intervener_upos"].value_counts().to_dict()

    for condition in baseline_conditions:
        comparison = all_features[all_features["condition"] == condition]
        rows.append(
            {
                "language": language,
                "metric": "arity",
                "comparison": f"real_vs_{condition}",
                "kl_divergence": kl_divergence_from_counts(real_arity_counts, comparison["arity"].value_counts().to_dict()),
            }
        )
        rows.append(
            {
                "language": language,
                "metric": "intervener_upos",
                "comparison": f"real_vs_{condition}",
                "kl_divergence": kl_divergence_from_counts(real_pos_counts, comparison["intervener_upos"].value_counts().to_dict()),
            }
        )
    return pd.DataFrame(rows)


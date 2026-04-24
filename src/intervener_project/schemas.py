"""Schema definitions for standardized outputs."""

from __future__ import annotations


INTERVENER_FEATURE_COLUMNS = [
    "language",
    "sentence_id",
    "token_id",
    "head_id",
    "dependent_id",
    "intervener_id",
    "dependency_relation",
    "dependency_distance",
    "direction",
    "intervener_upos",
    "head_upos",
    "dependent_upos",
    "arity",
    "subtree_size",
    "depth",
    "modifies",
    "complexity_score",
    "efficiency_ratio",
]

LANGUAGE_SUMMARY_COLUMNS = [
    "language",
    "avg_dependency_length",
    "avg_complexity",
    "avg_arity",
    "avg_subtree_size",
    "avg_depth",
    "percent_left_dependencies",
    "percent_right_dependencies",
    "most_common_pos",
    "entropy_pos_distribution",
    "avg_efficiency_ratio",
]

DISTRIBUTION_COLUMNS = [
    "language",
    "metric_type",
    "value",
]

ML_RESULTS_COLUMNS = [
    "language",
    "model_name",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]

ZSCORE_COLUMNS = [
    "language",
    "metric",
    "real_value",
    "random_mean",
    "random_std",
    "z_score",
]

STATISTICAL_TEST_COLUMNS = [
    "language",
    "metric",
    "comparison",
    "test_name",
    "statistic",
    "p_value",
    "effect_size",
]

KL_COLUMNS = [
    "language",
    "metric",
    "comparison",
    "kl_divergence",
]

RUN_LOG_COLUMNS = [
    "language",
    "sentence_count",
    "dependency_count",
    "intervener_count",
    "timestamp",
]

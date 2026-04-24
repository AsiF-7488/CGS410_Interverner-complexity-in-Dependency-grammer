"""Schema validation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import (
    DISTRIBUTION_COLUMNS,
    INTERVENER_FEATURE_COLUMNS,
    KL_COLUMNS,
    LANGUAGE_SUMMARY_COLUMNS,
    ML_RESULTS_COLUMNS,
    RUN_LOG_COLUMNS,
    STATISTICAL_TEST_COLUMNS,
    ZSCORE_COLUMNS,
)


SCHEMAS = {
    "intervener_features": INTERVENER_FEATURE_COLUMNS,
    "language_summary": LANGUAGE_SUMMARY_COLUMNS,
    "distribution_data": DISTRIBUTION_COLUMNS,
    "ml_results": ML_RESULTS_COLUMNS,
    "zscore_results": ZSCORE_COLUMNS,
    "statistical_tests": STATISTICAL_TEST_COLUMNS,
    "kl_divergence": KL_COLUMNS,
    "run_log": RUN_LOG_COLUMNS,
}


def validate_dataframe(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    expected = SCHEMAS[schema_name]
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {schema_name}: {missing}")
    if "language" in df.columns and df["language"].isna().any():
        raise ValueError(f"Language column contains missing values for {schema_name}")
    return df[expected + [column for column in df.columns if column not in expected]]


def write_validated_csv(df: pd.DataFrame, path: str | Path, schema_name: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_dataframe(df.copy(), schema_name)
    validated.to_csv(path, index=False)
    return path


def read_validated_csv(path: str | Path, schema_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_dataframe(frame, schema_name)

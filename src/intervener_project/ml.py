"""Machine-learning analysis for intervener complexity prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR


sns.set_theme(style="whitegrid", context="talk")

NUMERIC_FEATURES = ["dependency_distance"]
CATEGORICAL_FEATURES = ["head_upos", "dependent_upos", "language", "typology", "direction"]


@dataclass(frozen=True)
class MLRunArtifacts:
    ml_results_path: Path
    feature_importance_path: Path
    dataset_with_predictions_path: Path
    report_path: Path


def run_machine_learning_analysis(seed: int = 13, test_size: float = 0.2) -> MLRunArtifacts:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    interveners = pd.read_csv(PROCESSED_DIR / "interveners.csv")
    real = interveners[interveners["condition"] == "real"].copy()
    dataset = build_ml_dataset(real)

    X = dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = dataset["complexity_label"]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        dataset[["language", "complexity_score", "complexity_label"]],
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    results_rows: list[dict] = []
    feature_importance_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results_rows.append(
            build_metric_row(
                language="all",
                model_name=model_name,
                y_true=y_test.to_numpy(),
                y_pred=y_pred,
            )
        )

        per_language = meta_test.copy()
        per_language["prediction"] = y_pred
        for language, frame in per_language.groupby("language"):
            results_rows.append(
                build_metric_row(
                    language=language,
                    model_name=model_name,
                    y_true=frame["complexity_label"].to_numpy(),
                    y_pred=frame["prediction"].to_numpy(),
                )
            )

        prediction_frame = meta_test.copy()
        prediction_frame["model_name"] = model_name
        prediction_frame["prediction"] = y_pred
        prediction_frame["prediction_label"] = np.where(y_pred == 1, "high", "low")
        prediction_frames.append(prediction_frame)

        save_confusion_matrix(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            output_path=FIGURES_DIR / f"{model_name}_confusion_matrix.png",
            title=f"{model_name.replace('_', ' ').title()} Confusion Matrix",
        )

        importance_frame = build_feature_importance_frame(
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            seed=seed,
        )
        feature_importance_frames.append(importance_frame)
        save_feature_importance_plot(
            importance_frame,
            FIGURES_DIR / f"{model_name}_feature_importance.png",
            title=f"{model_name.replace('_', ' ').title()} Feature Importance",
        )

        joblib.dump(pipeline, TABLES_DIR / f"{model_name}_model.joblib")

    ml_results = pd.DataFrame(results_rows)
    ml_results_path = TABLES_DIR / "ml_results.csv"
    ml_results.to_csv(ml_results_path, index=False)

    feature_importance = pd.concat(feature_importance_frames, ignore_index=True)
    feature_importance_path = TABLES_DIR / "ml_feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    dataset_with_predictions_path = TABLES_DIR / "ml_predictions.csv"
    predictions.to_csv(dataset_with_predictions_path, index=False)

    save_model_comparison_plot(ml_results, FIGURES_DIR / "ml_model_comparison.png")
    report_path = write_ml_report(
        dataset=dataset,
        ml_results=ml_results,
        feature_importance=feature_importance,
    )

    return MLRunArtifacts(
        ml_results_path=ml_results_path,
        feature_importance_path=feature_importance_path,
        dataset_with_predictions_path=dataset_with_predictions_path,
        report_path=report_path,
    )


def build_ml_dataset(real_interveners: pd.DataFrame) -> pd.DataFrame:
    dataset = real_interveners.copy()
    dataset["complexity_score"] = dataset["intervener_arity"] + dataset["intervener_subtree_size"]
    threshold = dataset["complexity_score"].median()
    dataset["complexity_label"] = (dataset["complexity_score"] > threshold).astype(int)
    dataset["complexity_label_name"] = np.where(dataset["complexity_label"] == 1, "high", "low")
    dataset["typology"] = dataset["typology"].fillna("unknown")
    return dataset.rename(
        columns={
            "intervener_arity": "arity",
            "intervener_subtree_size": "subtree_size",
        }
    )


def build_metric_row(language: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "language": language,
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def build_feature_importance_frame(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    feature_names = list(X_test.columns)
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=seed,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return frame


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_feature_importance_plot(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    top = frame.head(15).copy()
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top, x="importance_mean", y="feature", color="#2563eb")
    plt.title(title)
    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_model_comparison_plot(ml_results: pd.DataFrame, output_path: Path) -> None:
    overall = ml_results[ml_results["language"] == "all"].copy()
    melted = overall.melt(
        id_vars=["language", "model_name"],
        value_vars=["accuracy", "precision", "recall", "f1_score"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(12, 7))
    sns.barplot(data=melted, x="metric", y="score", hue="model_name")
    plt.ylim(0, 1)
    plt.title("Model Comparison on Global Test Set")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_ml_report(dataset: pd.DataFrame, ml_results: pd.DataFrame, feature_importance: pd.DataFrame) -> Path:
    report_path = TABLES_DIR / "ml_report.md"
    threshold = dataset["complexity_score"].median()
    class_balance = dataset["complexity_label_name"].value_counts(normalize=True).round(4).to_dict()
    overall = ml_results[ml_results["language"] == "all"].copy()
    top_features = (
        feature_importance.groupby("feature", as_index=False)["importance_mean"]
        .mean()
        .sort_values("importance_mean", ascending=False)
        .head(10)
        .to_dict("records")
    )

    text = f"""# Machine Learning Report

## Task

Predict whether an intervener has **low** or **high** complexity.

## Label Definition

- `complexity_score = arity + subtree_size`
- `high` if `complexity_score` is greater than the global median
- Median threshold used: `{threshold:.3f}`

## Features

- dependency distance
- head POS
- dependent POS
- language
- typology
- direction

## Dataset

- Number of real interveners: `{len(dataset)}`
- Class balance: `{class_balance}`

## Model Performance

{overall.to_string(index=False)}

## Most Important Features

{top_features}
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path

"""Plotting utilities for the implemented analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def save_all_figures(
    dependencies: pd.DataFrame,
    interveners: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _arity_histogram(interveners, output_dir / "arity_histogram_real.png")
    _pos_distribution(tables["pos_distribution"], output_dir / "pos_distribution_real.png")
    _dependency_length_complexity(
        tables["dependency_length_complexity"],
        output_dir / "dependency_length_vs_complexity.png",
    )
    _z_score_plot(tables["z_scores"], output_dir / "z_scores_by_language.png")
    _condition_boxplots(dependencies, interveners, output_dir / "real_random_simplified_boxplots.png")
    _typology_plot(tables["typology_summary"], output_dir / "typology_comparison.png")
    _directionality_plot(tables["directionality_summary"], output_dir / "left_vs_right_dependencies.png")
    _dravidian_plot(tables["dravidian_summary"], output_dir / "tamil_dravidian_comparison.png")
    _language_condition_plot(tables["condition_comparison"], output_dir / "language_condition_comparison.png")


def _arity_histogram(interveners: pd.DataFrame, path: Path) -> None:
    frame = interveners[interveners["condition"] == "real"].copy()
    plt.figure(figsize=(14, 7))
    sns.histplot(data=frame, x="intervener_arity", hue="language_label", multiple="stack", discrete=True)
    plt.title("Arity Distribution of Interveners in Real Sentences")
    plt.xlabel("Intervener arity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _pos_distribution(table: pd.DataFrame, path: Path) -> None:
    frame = table[table["condition"] == "real"].copy()
    overall = (
        frame.groupby("intervener_upos")["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
        .head(12)
    )
    plt.figure(figsize=(14, 7))
    sns.barplot(data=overall, x="intervener_upos", y="count", color="#3b82f6")
    plt.title("Most Common Intervener POS Categories")
    plt.xlabel("UPOS")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _dependency_length_complexity(table: pd.DataFrame, path: Path) -> None:
    frame = table[table["dependency_distance"] > 0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    sns.lineplot(
        data=frame,
        x="dependency_distance",
        y="mean_intervener_arity",
        hue="condition",
        ax=axes[0],
    )
    axes[0].set_title("Dependency Distance vs Mean Intervener Arity")
    axes[0].set_xlabel("Dependency distance")
    axes[0].set_ylabel("Mean intervener arity")

    sns.lineplot(
        data=frame,
        x="dependency_distance",
        y="mean_intervener_subtree_size",
        hue="condition",
        ax=axes[1],
    )
    axes[1].set_title("Dependency Distance vs Mean Intervener Subtree Size")
    axes[1].set_xlabel("Dependency distance")
    axes[1].set_ylabel("Mean subtree size")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _z_score_plot(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    pivot = table.pivot(index="language_label", columns="metric", values="z_score")
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0)
    plt.title("Z-scores: Real vs Random Baseline")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _condition_boxplots(dependencies: pd.DataFrame, interveners: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.boxplot(data=interveners, x="condition", y="intervener_arity", ax=axes[0])
    axes[0].set_title("Intervener arity")
    axes[0].set_xlabel("")

    sns.boxplot(data=interveners, x="condition", y="intervener_subtree_size", ax=axes[1])
    axes[1].set_title("Intervener subtree size")
    axes[1].set_xlabel("")

    dep_frame = dependencies[dependencies["num_interveners"] > 0].copy()
    sns.boxplot(data=dep_frame, x="condition", y="dependency_distance", ax=axes[2])
    axes[2].set_title("Dependency distance")
    axes[2].set_xlabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _typology_plot(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(data=table, x="typology", y="mean_arity", hue="condition")
    plt.title("Mean Intervener Arity by Word Order Typology")
    plt.xlabel("Typology")
    plt.ylabel("Mean intervener arity")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _directionality_plot(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    plt.figure(figsize=(14, 7))
    sns.barplot(data=table, x="language_label", y="mean_arity", hue="direction")
    plt.title("Left vs Right Dependency Interveners")
    plt.xlabel("Language")
    plt.ylabel("Mean intervener arity")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _dravidian_plot(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(data=table, x="comparison_group", y="mean_arity", hue="condition")
    plt.title("Tamil-Focused Dravidian Comparison")
    plt.xlabel("")
    plt.ylabel("Mean intervener arity")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _language_condition_plot(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    plt.figure(figsize=(16, 7))
    sns.barplot(data=table, x="language_label", y="mean_subtree_size", hue="condition")
    plt.title("Real vs Simplified vs Random Across Languages")
    plt.xlabel("Language")
    plt.ylabel("Mean intervener subtree size")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

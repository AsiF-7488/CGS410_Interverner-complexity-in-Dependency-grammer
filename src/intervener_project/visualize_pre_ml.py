"""Visualization helpers for per-language and global pre-ML outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


sns.set_theme(style="whitegrid", context="talk")


def save_language_figures(output_dir: Path, all_features: pd.DataFrame, zscores: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    language = all_features["language"].iloc[0]

    plt.figure(figsize=(10, 6))
    sns.histplot(data=all_features[all_features["condition"] == "real"], x="arity", discrete=True, color="#2563eb")
    plt.title(f"{language.title()}: Arity Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "arity_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    pos_counts = (
        all_features[all_features["condition"] == "real"]["intervener_upos"]
        .value_counts()
        .head(12)
        .reset_index()
    )
    pos_counts.columns = ["intervener_upos", "count"]
    sns.barplot(data=pos_counts, x="intervener_upos", y="count", color="#0f766e")
    plt.title(f"{language.title()}: POS Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "pos_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=all_features[all_features["condition"] == "real"],
        x="dependency_distance",
        y="complexity_score",
        scatter_kws={"alpha": 0.25, "s": 18},
        line_kws={"color": "#dc2626"},
    )
    plt.title(f"{language.title()}: Dependency Length vs Complexity")
    plt.tight_layout()
    plt.savefig(output_dir / "dependency_length_vs_complexity.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=all_features, x="condition", y="complexity_score")
    plt.title(f"{language.title()}: Real vs Baselines")
    plt.tight_layout()
    plt.savefig(output_dir / "condition_boxplot.png", dpi=200)
    plt.close()

    if not zscores.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=zscores, x="metric", y="z_score", color="#7c3aed")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{language.title()}: Z-scores")
        plt.tight_layout()
        plt.savefig(output_dir / "zscore_barplot.png", dpi=200)
        plt.close()


def save_global_figures(
    final_output_dir: Path,
    all_features: pd.DataFrame,
    all_language_summary: pd.DataFrame,
    all_zscores: pd.DataFrame,
    all_distributions: pd.DataFrame,
) -> None:
    figures_dir = final_output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    real_features = all_features.copy()

    plt.figure(figsize=(12, 7))
    sns.boxplot(data=real_features, x="typology", y="complexity_score")
    plt.title("Complexity by Typology")
    plt.tight_layout()
    plt.savefig(figures_dir / "typology_complexity_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.boxplot(data=real_features, x="direction", y="complexity_score")
    plt.title("Left vs Right Dependency Complexity")
    plt.tight_layout()
    plt.savefig(figures_dir / "left_right_complexity_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.regplot(
        data=real_features,
        x="dependency_distance",
        y="complexity_score",
        scatter_kws={"alpha": 0.15, "s": 12},
        line_kws={"color": "#dc2626"},
    )
    plt.title("Global Dependency Length vs Complexity")
    plt.tight_layout()
    plt.savefig(figures_dir / "global_dependency_length_vs_complexity.png", dpi=200)
    plt.close()

    if not all_zscores.empty:
        pivot = all_zscores.pivot(index="language", columns="metric", values="z_score")
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, cmap="coolwarm", center=0)
        plt.title("Global Z-score Heatmap")
        plt.tight_layout()
        plt.savefig(figures_dir / "global_zscore_heatmap.png", dpi=200)
        plt.close()

    numeric_summary = all_language_summary.select_dtypes(include=["number"]).copy()
    if len(numeric_summary) >= 2 and numeric_summary.shape[1] >= 2:
        scaled = StandardScaler().fit_transform(numeric_summary)
        pca = PCA(n_components=2).fit_transform(scaled)
        pca_frame = pd.DataFrame(
            {
                "pc1": pca[:, 0],
                "pc2": pca[:, 1],
                "language": all_language_summary["language"],
            }
        )
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=pca_frame, x="pc1", y="pc2", hue="language", s=120)
        plt.title("PCA of Language-Level Summary Features")
        plt.tight_layout()
        plt.savefig(figures_dir / "language_summary_pca.png", dpi=200)
        plt.close()

    dravidian = real_features.copy()
    dravidian["group"] = dravidian["language"].map(
        lambda value: "Tamil" if value == "tamil" else ("Other Dravidian" if value == "telugu" else "Non-Dravidian")
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=dravidian, x="group", y="complexity_score", estimator="mean")
    plt.title("Tamil vs Other Languages")
    plt.tight_layout()
    plt.savefig(figures_dir / "dravidian_focus.png", dpi=200)
    plt.close()

    numeric_distributions = all_distributions[~all_distributions["metric_type"].str.contains("intervener_upos")]
    if not numeric_distributions.empty:
        numeric_distributions = numeric_distributions.copy()
        numeric_distributions["value"] = pd.to_numeric(numeric_distributions["value"], errors="coerce")
        numeric_distributions = numeric_distributions[numeric_distributions["metric_type"].str.contains("complexity_score")]
        if not numeric_distributions.empty:
            numeric_distributions["condition"] = numeric_distributions["metric_type"].str.split(":").str[0]
            plt.figure(figsize=(12, 7))
            sns.violinplot(data=numeric_distributions, x="condition", y="value", inner="quartile")
            plt.title("Complexity Distributions by Condition")
            plt.tight_layout()
            plt.savefig(figures_dir / "condition_complexity_violin.png", dpi=200)
            plt.close()

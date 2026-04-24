#!/usr/bin/env python3
"""Copy the main report figures into plots/report_figures for submission use."""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "plots" / "report_figures"

SOURCES = {
    PROJECT_ROOT / "plots" / "report_figures" / "fig01_arity_histogram_real.png": "fig01_arity_histogram_real.png",
    PROJECT_ROOT / "plots" / "report_figures" / "fig02_pos_distribution_real.png": "fig02_pos_distribution_real.png",
    PROJECT_ROOT / "plots" / "report_figures" / "fig03_dependency_length_vs_complexity.png": "fig03_dependency_length_vs_complexity.png",
    PROJECT_ROOT / "plots" / "report_figures" / "fig04_real_random_simplified_boxplots.png": "fig04_real_random_simplified_boxplots.png",
    PROJECT_ROOT / "plots" / "report_figures" / "fig05_z_scores_by_language.png": "fig05_z_scores_by_language.png",
    PROJECT_ROOT / "plots" / "report_figures" / "fig06_typology_comparison.png": "fig06_typology_comparison.png",
    PROJECT_ROOT / "plots" / "report_figures" / "fig07_left_right_dependencies.png": "fig07_left_right_dependencies.png",
    PROJECT_ROOT / "plots" / "report_figures" / "fig08_tamil_dravidian_comparison.png": "fig08_tamil_dravidian_comparison.png",
    PROJECT_ROOT / "plots" / "report_figures" / "fig09_ml_model_comparison.png": "fig09_ml_model_comparison.png",
}


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source, name in SOURCES.items():
        if source.exists():
            shutil.copy2(source, TARGET / name)
            copied += 1
    print(f"Report figure folder refreshed. Files copied: {copied}")


if __name__ == "__main__":
    main()

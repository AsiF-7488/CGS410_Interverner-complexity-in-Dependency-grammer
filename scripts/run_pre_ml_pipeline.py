#!/usr/bin/env python3
"""Run the production pre-ML research pipeline for selected languages or contributors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intervener_project.research_pipeline import run_pre_ml_pipeline
from intervener_project.settings import load_project_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--contributor", default=None, help="Contributor key from the YAML config.")
    parser.add_argument("--languages", nargs="*", default=None, help="Override language selection.")
    parser.add_argument("--download", action="store_true", help="Download or verify configured treebanks.")
    parser.add_argument("--force-download", action="store_true", help="Redownload configured treebanks.")
    parser.add_argument("--max-sentences", type=int, default=None, help="Optional debug cap; default is full dataset.")
    parser.add_argument("--merge", action="store_true", help="Merge all per-language outputs after the run.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_project_settings(
        args.config,
        contributor=args.contributor,
        languages_override=args.languages,
        max_sentences_override=args.max_sentences,
    )
    outputs = run_pre_ml_pipeline(
        settings,
        download=args.download,
        force_download=args.force_download,
        merge_results=args.merge,
    )
    print("Pre-ML pipeline completed.")
    print(outputs)


if __name__ == "__main__":
    main()

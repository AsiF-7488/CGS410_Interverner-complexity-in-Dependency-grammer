#!/usr/bin/env python3
"""Run the 7-language pre-ML pipeline for one selected language."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from intervener_project.research_pipeline import run_pre_ml_pipeline
from intervener_project.settings import load_project_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, help="Language key, e.g. tamil or persian")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    settings = load_project_settings(
        args.config,
        languages_override=[args.language],
        max_sentences_override=args.max_sentences,
    )
    outputs = run_pre_ml_pipeline(
        settings,
        download=args.download,
        force_download=args.force_download,
        merge_results=args.merge,
    )
    print(outputs)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI entry point for the multilingual intervener-complexity pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intervener_project import run_full_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download configured treebanks before running.")
    parser.add_argument("--force-download", action="store_true", help="Redownload archives and re-extract treebanks.")
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Maximum number of sentences per language to process.",
    )
    parser.add_argument(
        "--random-runs",
        type=int,
        default=5,
        help="Number of random linearization baselines per sentence.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility.")
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional subset of language keys, e.g. tamil telugu persian.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    outputs = run_full_pipeline(
        language_keys=args.languages,
        download=args.download,
        force_download=args.force_download,
        max_sentences=args.max_sentences,
        random_runs=args.random_runs,
        seed=args.seed,
    )

    print("Pipeline completed.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

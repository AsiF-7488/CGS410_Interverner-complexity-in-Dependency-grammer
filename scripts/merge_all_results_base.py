#!/usr/bin/env python3
"""Merge standardized per-language outputs into final cross-language outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intervener_project.merge import merge_all_results
from intervener_project.settings import load_project_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project_config.yaml")
    args = parser.parse_args()
    settings = load_project_settings(args.config)
    outputs = merge_all_results(settings.output_root, settings.final_output_root, settings)
    print("Merged outputs created.")
    print(outputs)


if __name__ == "__main__":
    main()

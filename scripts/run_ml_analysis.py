#!/usr/bin/env python3
"""Run point-4 machine-learning analysis on the point-1-to-3 dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intervener_project.ml import run_machine_learning_analysis


def main() -> None:
    artifacts = run_machine_learning_analysis()
    print("Machine-learning analysis completed.")
    print(artifacts)


if __name__ == "__main__":
    main()

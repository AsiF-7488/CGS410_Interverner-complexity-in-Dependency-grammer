#!/usr/bin/env python3
"""Download or verify the configured 7-language treebanks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intervener_project.settings import load_project_settings
from intervener_project.treebanks import ensure_configured_treebanks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    settings = load_project_settings(args.config, languages_override=args.languages)
    results = ensure_configured_treebanks(settings.languages, force_download=args.force_download)
    print("Treebanks ready:")
    for language, path in results:
        print(f"  {language.key}: {path}")


if __name__ == "__main__":
    main()

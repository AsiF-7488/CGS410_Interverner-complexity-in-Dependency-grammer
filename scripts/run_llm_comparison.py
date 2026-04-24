#!/usr/bin/env python3
"""Run point-5 LLM comparison workflow or generate prompt templates."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intervener_project.llm_compare import generate_llm_prompt_templates, run_llm_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="llm_model")
    parser.add_argument("--language", default=None, help="Language key, e.g. tamil")
    parser.add_argument("--conllu-path", default=None, help="Parsed CoNLL-U file for generated sentences.")
    parser.add_argument("--generate-prompts-only", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.generate_prompts_only:
        path = generate_llm_prompt_templates()
        print(f"Prompt templates written to {path}")
        return

    if not args.language:
        parser.error("--language is required unless --generate-prompts-only is used.")

    artifacts = run_llm_comparison(
        model_name=args.model_name,
        language_key=args.language,
        conllu_path=args.conllu_path,
    )
    print("LLM comparison workflow completed.")
    print(artifacts)


if __name__ == "__main__":
    main()

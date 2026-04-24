"""LLM comparison utilities for parsed generated sentences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, LANGUAGE_BY_KEY, PROJECT_ROOT, TABLES_DIR
from .conllu_io import parse_conllu
from .features import build_sentence_structure, extract_dependency_and_intervener_rows


sns.set_theme(style="whitegrid", context="talk")

LLM_ROOT = PROJECT_ROOT / "data" / "llm"
LLM_PROMPTS_DIR = LLM_ROOT / "prompts"
LLM_PARSED_DIR = LLM_ROOT / "parsed"
LLM_OUTPUT_DIR = LLM_ROOT / "outputs"


@dataclass(frozen=True)
class LLMComparisonArtifacts:
    prompt_file: Path | None
    comparison_summary_path: Path | None
    llm_interveners_path: Path | None
    report_path: Path | None


def generate_llm_prompt_templates(
    languages: list[str] | None = None,
    sentences_per_language: int = 100,
) -> Path:
    if languages is None:
        languages = list(LANGUAGE_BY_KEY.keys())
    LLM_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for language_key in languages:
        language = LANGUAGE_BY_KEY[language_key]
        prompt = (
            f"Generate {sentences_per_language} grammatical, natural sentences in {language.label}. "
            "Keep the sentences diverse in syntactic structure, between 8 and 20 tokens long, "
            "and return one sentence per line without numbering."
        )
        rows.append(
            {
                "language": language.key,
                "language_label": language.label,
                "prompt": prompt,
            }
        )

    path = LLM_PROMPTS_DIR / "llm_generation_prompts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_llm_comparison(
    *,
    model_name: str,
    language_key: str,
    conllu_path: str | Path | None = None,
) -> LLMComparisonArtifacts:
    prompt_file = generate_llm_prompt_templates([language_key])
    if conllu_path is None:
        report_path = write_llm_status_report(
            model_name=model_name,
            language_key=language_key,
            status="waiting_for_parsed_conllu",
        )
        return LLMComparisonArtifacts(
            prompt_file=prompt_file,
            comparison_summary_path=None,
            llm_interveners_path=None,
            report_path=report_path,
        )

    language = LANGUAGE_BY_KEY[language_key]
    llm_interveners = extract_interveners_from_conllu(language_key=language_key, conllu_path=conllu_path)
    output_dir = LLM_OUTPUT_DIR / model_name / language_key
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_interveners_path = output_dir / "llm_interveners.csv"
    llm_interveners.to_csv(llm_interveners_path, index=False)

    real = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "interveners.csv")
    real = real[(real["condition"] == "real") & (real["language"] == language_key)].copy()

    summary = build_llm_comparison_summary(real, llm_interveners, model_name=model_name, language_key=language_key)
    summary_path = output_dir / "llm_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    save_llm_comparison_figures(real, llm_interveners, output_dir, language.label, model_name)
    report_path = write_llm_report(summary, model_name=model_name, language_key=language_key, output_dir=output_dir)

    return LLMComparisonArtifacts(
        prompt_file=prompt_file,
        comparison_summary_path=summary_path,
        llm_interveners_path=llm_interveners_path,
        report_path=report_path,
    )


def extract_interveners_from_conllu(language_key: str, conllu_path: str | Path) -> pd.DataFrame:
    language = LANGUAGE_BY_KEY[language_key]
    rows: list[dict] = []
    conllu_path = Path(conllu_path)

    for sentence in parse_conllu(conllu_path):
        structure = build_sentence_structure(language, sentence)
        if structure is None:
            continue
        _, intervener_rows = extract_dependency_and_intervener_rows(
            structure,
            condition="llm_generated",
        )
        rows.extend(intervener_rows)

    return pd.DataFrame(rows)


def build_llm_comparison_summary(
    real: pd.DataFrame,
    llm_interveners: pd.DataFrame,
    *,
    model_name: str,
    language_key: str,
) -> pd.DataFrame:
    def safe_kl(real_counts: pd.Series, llm_counts: pd.Series) -> float:
        keys = sorted(set(real_counts.index) | set(llm_counts.index))
        if not keys:
            return float("nan")
        real_probs = (real_counts.reindex(keys, fill_value=0) + 1e-9).to_numpy()
        llm_probs = (llm_counts.reindex(keys, fill_value=0) + 1e-9).to_numpy()
        real_probs = real_probs / real_probs.sum()
        llm_probs = llm_probs / llm_probs.sum()
        return float(np.sum(real_probs * np.log(real_probs / llm_probs)))

    rows = [
        {
            "language": language_key,
            "model_name": model_name,
            "metric": "mean_arity",
            "real_value": float(real["intervener_arity"].mean()) if not real.empty else float("nan"),
            "llm_value": float(llm_interveners["intervener_arity"].mean()) if not llm_interveners.empty else float("nan"),
            "kl_divergence": safe_kl(
                real["intervener_arity"].value_counts(),
                llm_interveners["intervener_arity"].value_counts(),
            ),
        },
        {
            "language": language_key,
            "model_name": model_name,
            "metric": "mean_dependency_distance",
            "real_value": float(real["dependency_distance"].mean()) if not real.empty else float("nan"),
            "llm_value": float(llm_interveners["dependency_distance"].mean()) if not llm_interveners.empty else float("nan"),
            "kl_divergence": safe_kl(
                real["dependency_distance"].value_counts(),
                llm_interveners["dependency_distance"].value_counts(),
            ),
        },
        {
            "language": language_key,
            "model_name": model_name,
            "metric": "intervener_upos",
            "real_value": float("nan"),
            "llm_value": float("nan"),
            "kl_divergence": safe_kl(
                real["intervener_upos"].value_counts(),
                llm_interveners["intervener_upos"].value_counts(),
            ),
        },
    ]
    return pd.DataFrame(rows)


def save_llm_comparison_figures(
    real: pd.DataFrame,
    llm_interveners: pd.DataFrame,
    output_dir: Path,
    language_label: str,
    model_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    arity = pd.concat(
        [
            real.assign(source="real")[["intervener_arity", "source"]],
            llm_interveners.assign(source=model_name)[["intervener_arity", "source"]],
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(10, 6))
    sns.histplot(data=arity, x="intervener_arity", hue="source", multiple="dodge", discrete=True)
    plt.title(f"{language_label}: Real vs {model_name} Arity")
    plt.tight_layout()
    plt.savefig(output_dir / "llm_arity_comparison.png", dpi=200)
    plt.close()

    pos = pd.concat(
        [
            real.assign(source="real")[["intervener_upos", "source"]],
            llm_interveners.assign(source=model_name)[["intervener_upos", "source"]],
        ],
        ignore_index=True,
    )
    pos_counts = (
        pos.groupby(["source", "intervener_upos"])
        .size()
        .reset_index(name="count")
    )
    plt.figure(figsize=(12, 7))
    sns.barplot(data=pos_counts, x="intervener_upos", y="count", hue="source")
    plt.title(f"{language_label}: Real vs {model_name} POS Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "llm_pos_comparison.png", dpi=200)
    plt.close()

    dep = pd.concat(
        [
            real.assign(source="real")[["dependency_distance", "source"]],
            llm_interveners.assign(source=model_name)[["dependency_distance", "source"]],
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dep, x="dependency_distance", hue="source", multiple="step", element="step")
    plt.title(f"{language_label}: Real vs {model_name} Dependency Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "llm_dependency_distance_comparison.png", dpi=200)
    plt.close()


def write_llm_status_report(model_name: str, language_key: str, status: str) -> Path:
    output_dir = LLM_OUTPUT_DIR / model_name / language_key
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "llm_status.md"
    text = f"""# LLM Comparison Status

- model_name: `{model_name}`
- language: `{language_key}`
- status: `{status}`

No parsed CoNLL-U input was provided, so this stage generated only the prompt template.

Next step:

1. Generate sentences with an LLM using `data/llm/prompts/llm_generation_prompts.csv`
2. Parse the generated text into CoNLL-U
3. Re-run `run_llm_comparison.py` with `--conllu-path`
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_llm_report(summary: pd.DataFrame, model_name: str, language_key: str, output_dir: Path) -> Path:
    path = output_dir / "llm_report.md"
    text = f"""# LLM Comparison Report

- model_name: `{model_name}`
- language: `{language_key}`

{summary.to_string(index=False)}
"""
    path.write_text(text, encoding="utf-8")
    return path

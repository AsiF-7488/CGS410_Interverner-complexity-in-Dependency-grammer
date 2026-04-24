"""Report generation for the pre-ML pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .settings import ProjectSettings


def write_pre_ml_report(
    settings: ProjectSettings,
    final_output_dir: Path,
    all_language_summary: pd.DataFrame,
    all_zscores: pd.DataFrame,
    global_correlations: pd.DataFrame,
) -> Path:
    report_dir = final_output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    avg_complexity = float(all_language_summary["avg_complexity"].mean()) if not all_language_summary.empty else float("nan")
    avg_dep_length = float(all_language_summary["avg_dependency_length"].mean()) if not all_language_summary.empty else float("nan")
    strongest_z = all_zscores.sort_values("z_score").head(5).to_dict("records") if not all_zscores.empty else []
    correlation_text = global_correlations.to_string(index=False) if not global_correlations.empty else "No correlation table available."

    text = f"""# Pre-ML Research Report

## Abstract

This project studies intervener complexity in dependency grammar using a configurable cross-linguistic pipeline over full SUD treebanks. The current pre-ML system extracts structural properties of intervening tokens, compares real data against multiple baselines, standardizes outputs for distributed team execution, and produces merged global analyses.

## Introduction

The central question is whether languages optimize not only dependency length but also the structural complexity of the material that intervenes between a head and its dependent. This report operationalizes intervener complexity using arity, subtree size, depth, and POS-sensitive weighting.

## Related Work

The framework draws on dependency length minimization, locality constraints, cognitive memory limitations, and information-theoretic approaches to structural optimization.

## Methodology

- Treebank source: `{settings.treebank_source} {settings.release}`
- Languages in this run: `{", ".join(language.label for language in settings.languages)}`
- Random seed: `{settings.random_seed}`
- Random runs per baseline: `{settings.random_runs}`
- Baselines: `{", ".join(settings.baselines)}`
- Complexity score combines arity, subtree size, depth, and configurable POS weight.
- Efficiency is measured as dependency length divided by complexity score.

## Experiments

The pre-ML workflow computes:

- standardized per-language feature tables
- language summaries
- distribution tables
- baseline Z-scores
- ANOVA and Mann-Whitney U tests
- Cohen's d effect sizes
- KL divergence
- global typological comparisons
- mixed visual diagnostics across languages

## Results

- Mean language-level complexity across this run: `{avg_complexity:.4f}`
- Mean language-level dependency length across this run: `{avg_dep_length:.4f}`
- Strongest negative Z-scores (real lower than baseline): `{strongest_z}`

## Discussion

The pipeline is designed to test whether simpler interveners are preferred in real language relative to random and constrained baselines. Because every contributor writes the same schema, the global merge stage makes typological and cross-language analysis straightforward.

## Error Analysis

Potential sources of error include parser annotation differences across treebanks, differences in corpus genre, and the simplifying assumptions used in structural-role assignment and grammar-constrained baseline generation.

## Limitations

- The current report stops before the machine-learning and LLM-comparison stages.
- The baseline family focuses on linearization-based comparisons over observed dependency trees.
- Typology coverage depends on the languages included in the configuration.

## Future Work

Next steps include supervised modeling, cross-lingual transfer, SHAP-based interpretability, representation learning on dependency graphs, and LLM-generated sentence comparisons.

## Conclusion

The system is now ready for distributed team use: contributors can run assigned languages independently, merge results automatically, and obtain comparable pre-ML analyses under a shared research design.

## Correlation Snapshot

```
{correlation_text}
```
"""
    path = report_dir / "pre_ml_research_report.md"
    path.write_text(text, encoding="utf-8")
    return path

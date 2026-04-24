# Intervener Complexity in Dependency Grammar
## A Cross-Linguistic Computational Study — 7-Language Final Submission

This repository is the final deliverable package for the CGS410 project on intervener complexity in dependency grammar.

It keeps the same overall repo style as the larger team repository, but this version is specifically scoped to these 7 languages:

- Marathi
- Catalan
- Galician
- Persian
- Tamil
- Telugu
- Indonesian

## What Is Included

- final 5-page report plus appendix
- modular Python code
- 7-language config
- per-language outputs
- merged global outputs
- report figures
- notebooks
- LLM comparison prompt templates and status files

## Quick Start

```bash
pip install -r requirements.txt
python scripts/download_treebanks.py
python scripts/run_all.py --max-sentences 250
python scripts/merge_all_results.py
```

For the exact reported run used in the paper:

```bash
python scripts/run_analysis.py --download --max-sentences 250 --random-runs 3 --seed 13
python scripts/run_ml_analysis.py
python scripts/generate_course_submission.py
```

## Repository Structure

```text
CGS410--intervener-complexity-asif/
├── config/
│   ├── config.yaml
│   └── languages.yaml
├── src/intervener_project/
├── scripts/
│   ├── download_treebanks.py
│   ├── run_language.py
│   ├── run_all.py
│   ├── merge_all_results.py
│   ├── global_analysis.py
│   ├── generate_report_figures.py
│   ├── run_analysis.py
│   ├── run_ml_analysis.py
│   └── run_llm_comparison.py
├── notebooks/
├── report/
├── plots/report_figures/
├── outputs/<language>/
└── final_outputs/
```

## Final Report Files

- `report/Final_Report.pdf`
- `report/Appendix.pdf`
- `report/Final_Results_Explained.md`

## Important Notes

- This upload-friendly repository excludes the very large `distribution_data.csv` files and `all_distributions.csv`, because those files are too large for a practical GitHub upload.
- The code is still included, so those files can be regenerated locally.
- The per-language `intervener_features.csv` files are included.

## Main Findings

- Real interveners are overwhelmingly low-arity and short-subtree.
- Nouns are the most common intervener type.
- Adverbs are not dominant, so the original hypothesis is only partially supported.
- Real corpora are much simpler than random baselines.
- Dependency distance is the strongest predictor in the ML stage.

# Appendix: Technical Implementation and Code Map

## A1. Reproducibility

- Repository package prepared for upload: `CGS410--intervener-complexity-asif`
- Random seed: `13`
- Random baseline replicates: `3`
- Sentence cap used in the current reported run: `250`

## A2. Commands Used

```bash
python3 scripts/run_analysis.py --download --max-sentences 250 --random-runs 3 --seed 13
python3 scripts/run_ml_analysis.py
python3 scripts/run_llm_comparison.py --generate-prompts-only
python3 scripts/generate_course_submission.py
python3 scripts/create_github_repo_export.py
```

## A3. Method-to-Code Map

| Report method | Main implementation file | What it does |
| --- | --- | --- |
| Treebank loading | `src/intervener_project/treebanks.py` | Resolves configured SUD treebanks and download locations |
| CoNLL-U parsing | `src/intervener_project/conllu_io.py` | Reads SUD `.conllu` sentences without external parser dependencies |
| Sentence structure construction | `src/intervener_project/features.py` | Builds token index, child map, and subtree sizes |
| Intervener extraction | `src/intervener_project/features.py` | Extracts head, dependent, and all tokens between them |
| Random and simplified baselines | `src/intervener_project/baselines.py` | Creates randomized and compressed linear orders |
| Statistical analysis | `src/intervener_project/analysis.py` | Produces histograms, POS summaries, Z-scores, and significance tests |
| Visualization | `src/intervener_project/visualize.py` | Generates the graphs used in the report |
| Machine learning | `src/intervener_project/ml.py` | Builds the dataset, trains the models, and reports metrics |
| Main experiment runner | `scripts/run_analysis.py` | Runs the multilingual corpus analysis |
| ML runner | `scripts/run_ml_analysis.py` | Runs logistic regression and random forest |
| LLM comparison scaffold | `scripts/run_llm_comparison.py` | Prepares prompts and comparison code for future execution |

## A4. Core Equations Implemented

```text
I(h,d) = { i | min(pos(h), pos(d)) < pos(i) < max(pos(h), pos(d)) }
gap(h,d) = |I(h,d)|
arity(i) = number of dependents of i
subtree_size(i) = number of nodes dominated by i
complexity(i) = arity(i) + subtree_size(i)
Z = (X_real - mean(X_random)) / sd(X_random)
label(i) = 1 if complexity(i) > median(complexity), else 0
```

## A5. Repository Tree for GitHub Upload

```text
CGS410--intervener-complexity-asif/
  README.md
  requirements.txt
  .gitignore
  configs/
  scripts/
  src/intervener_project/
  notebooks/
  reports/
  data/outputs/figures/
  data/outputs/tables/
  data/llm/
```

## A6. Files Included for the Uploadable Repository

- Full Python source code
- Notebooks
- Config file for the 7-language setup
- Final report PDF and markdown source
- Appendix markdown and PDF
- Result figures used in the report
- Result tables used in the report
- LLM prompt templates and status notes

Large raw treebanks and the 436 MB `interveners.csv` file are not copied into the GitHub export folder because they are too large for a clean course repository. The code can regenerate them.

## A7. GitHub Upload Note

After uploading `CGS410--intervener-complexity-asif/` to GitHub, replace the placeholder below in the report if needed:

`Public code repository: <add-your-github-link-here>`

## A8. Implementation Check

The appendix and repository are consistent with the report methods:

- the extraction code implements the intervener definition used in the methods section,
- the statistics code implements the Z-score and non-parametric tests described in the methods section,
- the ML code implements the predictive experiment described in the implementation section.

Current reported dataset size:

- real dependencies: `31,007`
- real interveners: `44,881`

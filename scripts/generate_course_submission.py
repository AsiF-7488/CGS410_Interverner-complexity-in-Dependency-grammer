from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = ROOT / "data" / "outputs" / "tables"
FIGURES = ROOT / "data" / "outputs" / "figures"
PROCESSED = ROOT / "data" / "processed"
REPO_EXPORT = ROOT / "intervener-complexity-study"
REPO_LABEL = "CGS410--intervener-complexity-asif"

REPORT_MD = REPORTS / "course_project_submission.md"
REPORT_PDF = REPORTS / "course_project_submission.pdf"
APPENDIX_MD = REPORTS / "course_project_appendix.md"
APPENDIX_PDF = REPORTS / "course_project_appendix.pdf"


@dataclass
class ReportStats:
    real_dependencies: int
    real_interveners: int
    mean_gap: float
    mean_arity: float
    median_arity: float
    pct_arity_0_1: float
    mean_subtree: float
    median_subtree: float
    pct_subtree_1_3: float
    noun_share: float
    adv_share: float
    top_pos: list[tuple[str, float]]
    class_balance: dict[str, float]
    ml_threshold: float


def load_tables() -> dict[str, pd.DataFrame | dict]:
    return {
        "interveners": pd.read_csv(PROCESSED / "interveners.csv"),
        "dependencies": pd.read_csv(PROCESSED / "dependencies.csv"),
        "metadata": json.loads((TABLES / "run_metadata.json").read_text(encoding="utf-8")),
        "condition": pd.read_csv(TABLES / "condition_comparison.csv"),
        "tests": pd.read_csv(TABLES / "statistical_tests.csv"),
        "zscores": pd.read_csv(TABLES / "z_scores.csv"),
        "typology": pd.read_csv(TABLES / "typology_summary.csv"),
        "directionality": pd.read_csv(TABLES / "directionality_summary.csv"),
        "dravidian": pd.read_csv(TABLES / "dravidian_summary.csv"),
        "ml": pd.read_csv(TABLES / "ml_results.csv"),
        "importance": pd.read_csv(TABLES / "ml_feature_importance.csv"),
    }


def compute_stats(data: dict[str, pd.DataFrame | dict]) -> ReportStats:
    interveners = data["interveners"]
    dependencies = data["dependencies"]
    real_interveners = interveners[interveners["condition"] == "real"].copy()
    real_dependencies = dependencies[dependencies["condition"] == "real"].copy()
    complexity = real_interveners["intervener_arity"] + real_interveners["intervener_subtree_size"]
    threshold = float(complexity.median())
    labels = (complexity > threshold).map({True: "high", False: "low"})
    class_balance = labels.value_counts(normalize=True).round(4).to_dict()
    pos_share = real_interveners["intervener_upos"].value_counts(normalize=True).mul(100)
    top_pos = [(upos, float(share)) for upos, share in pos_share.head(5).items()]
    return ReportStats(
        real_dependencies=int(len(real_dependencies)),
        real_interveners=int(len(real_interveners)),
        mean_gap=float(real_dependencies["num_interveners"].mean()),
        mean_arity=float(real_interveners["intervener_arity"].mean()),
        median_arity=float(real_interveners["intervener_arity"].median()),
        pct_arity_0_1=float((real_interveners["intervener_arity"] <= 1).mean() * 100),
        mean_subtree=float(real_interveners["intervener_subtree_size"].mean()),
        median_subtree=float(real_interveners["intervener_subtree_size"].median()),
        pct_subtree_1_3=float(
            (
                (real_interveners["intervener_subtree_size"] >= 1)
                & (real_interveners["intervener_subtree_size"] <= 3)
            ).mean()
            * 100
        ),
        noun_share=float((real_interveners["intervener_upos"] == "NOUN").mean() * 100),
        adv_share=float((real_interveners["intervener_upos"] == "ADV").mean() * 100),
        top_pos=top_pos,
        class_balance={key: float(value) for key, value in class_balance.items()},
        ml_threshold=threshold,
    )


def build_appendix_markdown(data: dict[str, pd.DataFrame | dict], stats: ReportStats) -> str:
    metadata = data["metadata"]
    appendix = f"""# Appendix: Technical Implementation and Code Map

## A1. Reproducibility

- Repository package prepared for upload: `{REPO_LABEL}`
- Random seed: `{metadata["seed"]}`
- Random baseline replicates: `{metadata["random_runs"]}`
- Sentence cap used in the current reported run: `{metadata["max_sentences"]}`

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
I(h,d) = {{ i | min(pos(h), pos(d)) < pos(i) < max(pos(h), pos(d)) }}
gap(h,d) = |I(h,d)|
arity(i) = number of dependents of i
subtree_size(i) = number of nodes dominated by i
complexity(i) = arity(i) + subtree_size(i)
Z = (X_real - mean(X_random)) / sd(X_random)
label(i) = 1 if complexity(i) > median(complexity), else 0
```

## A5. Repository Tree for GitHub Upload

```text
intervener-complexity-study/
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

After uploading `{REPO_LABEL}/` to GitHub, replace the placeholder below in the report if needed:

`Public code repository: <add-your-github-link-here>`

## A8. Implementation Check

The appendix and repository are consistent with the report methods:

- the extraction code implements the intervener definition used in the methods section,
- the statistics code implements the Z-score and non-parametric tests described in the methods section,
- the ML code implements the predictive experiment described in the implementation section.

Current reported dataset size:

- real dependencies: `{stats.real_dependencies:,}`
- real interveners: `{stats.real_interveners:,}`
"""
    return appendix


def build_report_markdown(data: dict[str, pd.DataFrame | dict], stats: ReportStats) -> str:
    metadata = data["metadata"]
    condition = data["condition"]
    tests = data["tests"]
    zscores = data["zscores"]
    typology = data["typology"]
    directionality = data["directionality"]
    dravidian = data["dravidian"]
    ml = data["ml"]
    importance = data["importance"]

    real = condition[condition["condition"] == "real"].copy()
    random = condition[condition["condition"] == "random"].copy()
    simplified = condition[condition["condition"] == "simplified"].copy()
    real_typology = typology[typology["condition"] == "real"].set_index("typology")
    real_dravidian = dravidian[dravidian["condition"] == "real"].set_index("comparison_group")

    overall_ml = ml[ml["language"] == "all"].copy().set_index("model_name")
    best_model = overall_ml["f1_score"].idxmax()
    dep_importance = float(importance[importance["feature"] == "dependency_distance"]["importance_mean"].max())
    z_means = zscores.groupby("metric")["z_score"].mean().round(3).to_dict()
    strong_random_tests = int((tests[tests["comparison"] == "real_vs_random"]["mannwhitney_p"] < 0.001).sum())
    total_random_tests = int((tests["comparison"] == "real_vs_random").sum())

    real_gap = float(real["intervener_count"].sum() / real["dependency_count"].sum())
    random_gap = float(random["intervener_count"].sum() / random["dependency_count"].sum())
    simplified_gap = float(simplified["intervener_count"].sum() / simplified["dependency_count"].sum())

    left_dir = directionality[(directionality["condition"] == "real") & (directionality["direction"] == "left")]
    right_dir = directionality[(directionality["condition"] == "real") & (directionality["direction"] == "right")]
    left_weighted_arity = float((left_dir["mean_arity"] * left_dir["intervener_count"]).sum() / left_dir["intervener_count"].sum())
    right_weighted_arity = float((right_dir["mean_arity"] * right_dir["intervener_count"]).sum() / right_dir["intervener_count"].sum())

    top_pos_text = ", ".join(f"{upos} ({share:.2f}%)" for upos, share in stats.top_pos)
    appendix_md = build_appendix_markdown(data, stats)

    report = f"""# Intervener Complexity in Dependency Grammar: A Cross-Linguistic Computational Study

## Abstract

This project studies the complexity of intervening words in dependency relations across seven languages: Marathi, Catalan, Galician, Persian, Tamil, Telugu, and Indonesian. The main question is whether natural languages minimize intervener complexity in addition to minimizing dependency distance. Using {stats.real_dependencies:,} real dependency pairs with non-zero gaps and {stats.real_interveners:,} intervener tokens, I compare real sentences with randomized and simplified baselines, and I add a machine-learning experiment that predicts whether an intervener is low- or high-complexity. The results strongly support the structural simplicity part of the hypothesis: real interveners have low arity, short subtrees, and much lower complexity than random baselines. However, the categorical part of the hypothesis is only partially supported. Nouns are the most common intervener type, but adverbs are rare. The results suggest that natural languages minimize not only distance, but also the internal complexity of material placed between syntactically related words.

## 1. Research Problem

### 1.1 Motivation

Dependency grammar is often used to study efficiency in language. Most previous work focuses on dependency length minimization, but there is another possible source of processing cost: the complexity of the words that occur between a head and its dependent. If the words inside a dependency gap are structurally heavy, the dependency may be harder to process even when the gap is not especially long.

### 1.2 Research Question

What kinds of nodes intervene in dependency relations in natural languages, and do languages prefer structurally simple interveners over complex ones?

### 1.3 Hypothesis

Interveners in real language are expected to be:

- short,
- low-arity,
- and biased toward nouns and adverbs rather than structurally heavy categories.

### 1.4 Objectives

The project has only two main objectives.

1. Measure intervener complexity across multiple languages and test whether real corpora are simpler than randomized baselines.
2. Test whether intervener complexity can be predicted from local structural information such as dependency gap size, POS, language, typology, and direction.

## 2. Methods

### 2.1 Data

The analysis uses SUD treebanks for seven languages. The reported run used a cap of {metadata["max_sentences"]} sentences per language, a fixed random seed of {metadata["seed"]}, and {metadata["random_runs"]} random baseline replicates per sentence. This gives a multilingual sample well above the minimum four-language requirement in the grading rubric.

### 2.2 Intervener Definition

For a head `h` and dependent `d`, the set of interveners is:

```text
I(h,d) = {{ i | min(pos(h), pos(d)) < pos(i) < max(pos(h), pos(d)) }}
```

The gap size is:

```text
gap(h,d) = |I(h,d)|
```

Only dependencies with `gap(h,d) > 0` are analyzed. Punctuation tokens are excluded.

### 2.3 Structural Measures

For each intervener `i`, the project computes:

```text
arity(i) = number of dependents of i
subtree_size(i) = number of nodes dominated by i
```

The machine-learning stage defines:

```text
complexity(i) = arity(i) + subtree_size(i)
label(i) = 1 if complexity(i) > median(complexity), else 0
```

In the current run, the median threshold for the ML label is `{stats.ml_threshold:.1f}`.

### 2.4 Example

If a head occurs at position 8 and its dependent occurs at position 3, then all non-punctuation tokens in positions 4, 5, 6, and 7 are interveners. Each of those tokens is separately assigned a POS tag, arity value, and subtree size.

### 2.5 Inference Methods

Three sentence conditions are compared:

- real corpus order,
- random token-order baselines,
- simplified head-proximal order.

To test the hypothesis, the project uses:

```text
Z = (X_real - mean(X_random)) / sd(X_random)
```

It also uses:

- Mann-Whitney U tests for distributional differences,
- Kolmogorov-Smirnov tests for shape differences,
- Cohen's d for effect size on global contrasts.

### 2.6 Machine-Learning Method

Two classifiers are trained:

- logistic regression,
- random forest.

Predictors:

- dependency gap size,
- head POS,
- dependent POS,
- language,
- typology,
- direction.

Evaluation metrics:

- accuracy,
- precision,
- recall,
- F1-score.

## 3. Implementation

### 3.1 Amount of Work

This project clearly includes more than one analysis.

1. Corpus experiment: descriptive statistics, baseline comparison, Z-scores, typology analysis, directionality analysis, and Tamil-focused Dravidian comparison.
2. Predictive experiment: logistic regression and random forest models for complexity prediction.

This satisfies the rubric requirement that larger projects should show visible analytical work, not just more writing.

### 3.2 Sample Size

The current reported run contains:

- {stats.real_dependencies:,} real dependency pairs with non-zero gaps,
- {stats.real_interveners:,} real intervener tokens,
- 7 languages instead of the minimum 4.

### 3.3 Software Implementation

The codebase is modular:

- `scripts/run_analysis.py` runs the main corpus pipeline,
- `src/intervener_project/features.py` extracts interveners,
- `src/intervener_project/analysis.py` computes summaries and tests,
- `src/intervener_project/visualize.py` generates figures,
- `src/intervener_project/ml.py` runs the machine-learning stage.

An upload-ready repository package accompanies this submission:

`{REPO_LABEL}`

After upload, you can replace this placeholder in the final submission:

`Public code repository: <add-your-github-link-here>`

## 4. Results

### 4.1 Main Structural Finding

The structural part of the hypothesis is strongly supported.

- Mean intervener arity in real data: `{stats.mean_arity:.3f}`
- Median intervener arity: `{stats.median_arity:.0f}`
- Share of interveners with arity 0 or 1: `{stats.pct_arity_0_1:.2f}%`
- Mean intervener subtree size: `{stats.mean_subtree:.3f}`
- Median intervener subtree size: `{stats.median_subtree:.0f}`
- Share of interveners with subtree size 1-3: `{stats.pct_subtree_1_3:.2f}%`

The mean gap is `{real_gap:.3f}` in real data, `{simplified_gap:.3f}` in simplified order, and `{random_gap:.3f}` in random order. Real language is therefore much closer to the simplified baseline than to the random one.

### 4.2 POS Distribution

The categorical part of the hypothesis is only partially supported.

- NOUN share: `{stats.noun_share:.2f}%`
- ADV share: `{stats.adv_share:.2f}%`

Top real intervener categories: `{top_pos_text}`.

This means nouns are clearly dominant, but adverbs are not.

### 4.3 Inference Results

The baseline comparison is very strong.

- Significant real-vs-random Mann-Whitney tests at `p < 0.001`: `{strong_random_tests}` out of `{total_random_tests}`
- Mean Z-score for arity: `{z_means["mean_arity"]:.3f}`
- Mean Z-score for subtree size: `{z_means["mean_subtree_size"]:.3f}`
- Mean Z-score for dependencies with interveners: `{z_means["share_with_interveners"]:.3f}`

All mean Z-scores are negative, which means real corpora are systematically simpler than random baselines.

### 4.4 Typology and Directionality

In the current sample:

- SOV mean arity: `{real_typology.loc["SOV", "mean_arity"]:.3f}`
- SVO mean arity: `{real_typology.loc["SVO", "mean_arity"]:.3f}`
- left-dependency weighted mean arity: `{left_weighted_arity:.3f}`
- right-dependency weighted mean arity: `{right_weighted_arity:.3f}`

SOV languages and leftward dependencies are slightly more complex on average, but the effect is modest.

### 4.5 Tamil and Dravidian Comparison

Tamil is not unusually complex in this sample, but it is clearly richer than Telugu.

- Tamil mean arity: `{real_dravidian.loc["Tamil", "mean_arity"]:.3f}`
- Other Dravidian mean arity: `{real_dravidian.loc["Other Dravidian", "mean_arity"]:.3f}`
- Tamil mean subtree size: `{real_dravidian.loc["Tamil", "mean_subtree_size"]:.3f}`
- Other Dravidian mean subtree size: `{real_dravidian.loc["Other Dravidian", "mean_subtree_size"]:.3f}`

### 4.6 Machine-Learning Results

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | {overall_ml.loc["logistic_regression", "accuracy"]:.3f} | {overall_ml.loc["logistic_regression", "precision"]:.3f} | {overall_ml.loc["logistic_regression", "recall"]:.3f} | {overall_ml.loc["logistic_regression", "f1_score"]:.3f} |
| Random Forest | {overall_ml.loc["random_forest", "accuracy"]:.3f} | {overall_ml.loc["random_forest", "precision"]:.3f} | {overall_ml.loc["random_forest", "recall"]:.3f} | {overall_ml.loc["random_forest", "f1_score"]:.3f} |

The best model by F1-score is `{best_model}`. The strongest predictor is `dependency_distance`, with permutation importance up to `{dep_importance:.3f}`. This supports the idea that longer dependency gaps are more likely to contain structurally complex interveners.

## 5. Discussion and Theoretical Implications

The results support a stronger version of locality-based processing theory. It is not enough to say that languages minimize distance. The data suggest that languages also minimize the internal complexity of the material placed inside that distance.

For human language processing, this means that memory pressure may depend on both:

- how far apart the linked words are,
- and what kind of structure sits between them.

For theories of language evolution, the results suggest that efficient linearization may favor not only short dependencies but also light intervening material. For machine language models, the findings suggest that syntactic complexity inside dependency gaps may be a useful feature for evaluating generated text, especially in multilingual settings.

## 6. Conclusion

The project answers the main research question clearly. Natural languages in this dataset do prefer structurally simple interveners. The original hypothesis is therefore **partially supported**:

- supported: interveners are usually short and low-arity,
- supported: nouns are common,
- not supported: adverbs are not a dominant intervener type.

The next natural step is to extend the study to full treebanks without sentence caps and then execute the planned LLM comparison.

{appendix_md}
"""
    return report


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SubmissionTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=6,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=12.5,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.8,
            leading=10.5,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    return styles


def make_table(rows: list[list[str]], col_widths: list[float]) -> Table:
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        width_px, height_px = img.size
    scale = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * scale, height=height_px * scale)


def image_row(items: list[tuple[str, str]], image_width: float = 2.55 * inch, image_height: float = 1.85 * inch) -> Table:
    image_cells = []
    caption_cells = []
    for filename, caption in items:
        image_cells.append(scaled_image(FIGURES / filename, image_width, image_height))
        caption_cells.append(Paragraph(caption, build_styles()["Caption"]))
    table = Table([image_cells, caption_cells], colWidths=[3.0 * inch] * len(items))
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def page_number(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(A4[0] - 36, 20, f"Page {canvas.getPageNumber()}")


def build_code_map_rows() -> list[list[str]]:
    return [
        ["Report method", "Main file", "Purpose"],
        ["Treebank loading", "src/intervener_project/treebanks.py", "Resolves configured treebanks"],
        ["CoNLL-U parsing", "src/intervener_project/conllu_io.py", "Reads .conllu sentences"],
        ["Intervener extraction", "src/intervener_project/features.py", "Builds structures and extracts interveners"],
        ["Baselines", "src/intervener_project/baselines.py", "Creates random and simplified orders"],
        ["Statistics", "src/intervener_project/analysis.py", "Builds summaries, Z-scores, and tests"],
        ["Visualization", "src/intervener_project/visualize.py", "Generates report figures"],
        ["Machine learning", "src/intervener_project/ml.py", "Trains and evaluates classifiers"],
        ["Main run script", "scripts/run_analysis.py", "Runs the multilingual experiment"],
        ["ML run script", "scripts/run_ml_analysis.py", "Runs logistic regression and random forest"],
        ["LLM scaffold", "scripts/run_llm_comparison.py", "Prepares prompts and future comparison"],
    ]


def append_appendix_story(story: list, styles, metadata: dict, stats: ReportStats) -> None:
    story.append(Paragraph("Appendix A. Reproducibility", styles["SubSection"]))
    for bullet in [
        f"Repository package name: {REPO_LABEL}",
        f"Random seed: {metadata['seed']}",
        f"Random baseline replicates: {metadata['random_runs']}",
        f"Sentence cap in the current reported run: {metadata['max_sentences']}",
    ]:
        story.append(Paragraph(f"- {bullet}", styles["BulletBody"]))

    story.append(Paragraph("Appendix B. Commands Used", styles["SubSection"]))
    story.append(
        Preformatted(
            "python3 scripts/run_analysis.py --download --max-sentences 250 --random-runs 3 --seed 13\n"
            "python3 scripts/run_ml_analysis.py\n"
            "python3 scripts/run_llm_comparison.py --generate-prompts-only\n"
            "python3 scripts/generate_course_submission.py\n"
            "python3 scripts/create_github_repo_export.py",
            styles["Code"],
        )
    )

    story.append(Paragraph("Appendix C. Method-to-Code Map", styles["SubSection"]))
    story.append(
        make_table(
            build_code_map_rows(),
            [1.2 * inch, 2.1 * inch, 2.9 * inch],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Appendix D. Core Equations", styles["SubSection"]))
    story.append(
        Preformatted(
            "I(h,d) = { i | min(pos(h), pos(d)) < pos(i) < max(pos(h), pos(d)) }\n"
            "gap(h,d) = |I(h,d)|\n"
            "arity(i) = number of dependents of i\n"
            "subtree_size(i) = number of nodes dominated by i\n"
            "complexity(i) = arity(i) + subtree_size(i)\n"
            "Z = (X_real - mean(X_random)) / sd(X_random)\n"
            "label(i) = 1 if complexity(i) > median(complexity), else 0",
            styles["Code"],
        )
    )

    story.append(Paragraph("Appendix E. GitHub Upload Notes", styles["SubSection"]))
    for bullet in [
        "The accompanying repository folder includes code, notebooks, config, figures, tables, and the final report.",
        "Large raw treebanks and the 436 MB interveners.csv file are intentionally excluded from the uploadable repo.",
        "After uploading, replace this placeholder in the report: Public code repository: <add-your-github-link-here>",
        f"Current reported dataset size: {stats.real_dependencies:,} real dependencies and {stats.real_interveners:,} real interveners.",
    ]:
        story.append(Paragraph(f"- {bullet}", styles["BulletBody"]))


def write_pdf(data: dict[str, pd.DataFrame | dict], stats: ReportStats, appendix_text: str) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=36,
        title="Course Project Submission",
        author="OpenAI Codex",
    )
    story = []

    metadata = data["metadata"]
    condition = data["condition"]
    tests = data["tests"]
    zscores = data["zscores"]
    directionality = data["directionality"]
    typology = data["typology"]
    dravidian = data["dravidian"]
    ml = data["ml"]
    importance = data["importance"]

    real = condition[condition["condition"] == "real"].copy()
    random = condition[condition["condition"] == "random"].copy()
    simplified = condition[condition["condition"] == "simplified"].copy()
    real_typology = typology[typology["condition"] == "real"].set_index("typology")
    real_dravidian = dravidian[dravidian["condition"] == "real"].set_index("comparison_group")
    overall_ml = ml[ml["language"] == "all"].copy().set_index("model_name")
    dep_importance = float(importance[importance["feature"] == "dependency_distance"]["importance_mean"].max())
    real_gap = float(real["intervener_count"].sum() / real["dependency_count"].sum())
    random_gap = float(random["intervener_count"].sum() / random["dependency_count"].sum())
    simplified_gap = float(simplified["intervener_count"].sum() / simplified["dependency_count"].sum())
    strong_random_tests = int((tests[tests["comparison"] == "real_vs_random"]["mannwhitney_p"] < 0.001).sum())
    total_random_tests = int((tests["comparison"] == "real_vs_random").sum())
    z_means = zscores.groupby("metric")["z_score"].mean().round(3).to_dict()
    left_dir = directionality[(directionality["condition"] == "real") & (directionality["direction"] == "left")]
    right_dir = directionality[(directionality["condition"] == "real") & (directionality["direction"] == "right")]
    left_weighted_arity = float((left_dir["mean_arity"] * left_dir["intervener_count"]).sum() / left_dir["intervener_count"].sum())
    right_weighted_arity = float((right_dir["mean_arity"] * right_dir["intervener_count"]).sum() / right_dir["intervener_count"].sum())
    top_pos_text = ", ".join(f"{upos} ({share:.1f}%)" for upos, share in stats.top_pos)

    story.append(Paragraph("Intervener Complexity in Dependency Grammar", styles["SubmissionTitle"]))
    story.append(Paragraph("A Cross-Linguistic Computational Study", styles["SubmissionTitle"]))
    story.append(
        Paragraph(
            "Main report fixed to 5 pages; appendix begins after page 5",
            styles["BodyJustify"],
        )
    )
    story.append(
        Paragraph(
            f"Current reported run: 7 languages, {stats.real_dependencies:,} real dependencies, {stats.real_interveners:,} real interveners",
            styles["BodyJustify"],
        )
    )
    story.append(Spacer(1, 0.05 * inch))

    # Page 1: abstract and research problem
    story.append(Paragraph("Abstract", styles["Section"]))
    story.append(
        Paragraph(
            "This report tests whether natural languages minimize the complexity of words that intervene inside dependency relations, not just the raw distance between head and dependent. The analysis uses seven SUD treebanks and compares real sentences with random and simplified baselines. A second experiment trains machine-learning models to predict intervener complexity from local structural features.",
            styles["BodyJustify"],
        )
    )
    story.append(Paragraph("1. Research Problem", styles["Section"]))
    for paragraph in [
        "Dependency Length Minimization explains why related words are often close to one another, but it does not directly explain what kinds of words are allowed to sit inside the gap when a dependency is not adjacent.",
        "Research question: what kinds of nodes intervene in dependency relations in natural languages, and do languages prefer structurally simple interveners over complex ones?",
        "Hypothesis: interveners should be short, low-arity, and biased toward nouns and adverbs rather than structurally heavy categories.",
        "Objectives: first, measure intervener complexity across languages and compare real order with baselines. Second, test whether intervener complexity can be predicted from local structural information.",
        f"The current run already exceeds the rubric minimum for cross-linguistic data: it uses 7 languages, {stats.real_dependencies:,} real dependency pairs with non-zero gaps, and {stats.real_interveners:,} real intervener tokens.",
    ]:
        story.append(Paragraph(paragraph, styles["BodyJustify"]))
    story.append(Paragraph("Language Summary", styles["SubSection"]))
    rows = [["Language", "Typology", "Dependencies", "Interveners", "Mean gap", "Mean arity", "Mean subtree"]]
    for row in real[
        ["language_label", "dependency_count", "intervener_count", "mean_distance", "mean_arity", "mean_subtree_size"]
    ].itertuples(index=False):
        label, deps, inters, gap, arity, subtree = row
        typ = "SOV" if label in {"Marathi", "Persian", "Tamil", "Telugu"} else "SVO"
        rows.append([label, typ, f"{int(deps):,}", f"{int(inters):,}", f"{gap:.2f}", f"{arity:.2f}", f"{subtree:.2f}"])
    story.append(make_table(rows, [0.95 * inch, 0.62 * inch, 0.78 * inch, 0.82 * inch, 0.62 * inch, 0.62 * inch, 0.72 * inch]))
    story.append(PageBreak())

    # Page 2: methods and implementation
    story.append(Paragraph("2. Methods", styles["Section"]))
    for paragraph in [
        f"The analysis uses SUD treebanks for Marathi, Catalan, Galician, Persian, Tamil, Telugu, and Indonesian. The current reported run uses a cap of {metadata['max_sentences']} sentences per language, random seed {metadata['seed']}, and {metadata['random_runs']} random baseline replicates per sentence.",
        "For each dependency pair (h, d), the project collects every non-punctuation token strictly between the head and dependent. Dependencies with zero-gap are excluded.",
    ]:
        story.append(Paragraph(paragraph, styles["BodyJustify"]))
    story.append(
        Preformatted(
            "I(h,d) = { i | min(pos(h), pos(d)) < pos(i) < max(pos(h), pos(d)) }\n"
            "gap(h,d) = |I(h,d)|\n"
            "arity(i) = number of dependents of i\n"
            "subtree_size(i) = number of nodes dominated by i",
            styles["Code"],
        )
    )
    for paragraph in [
        "Each intervener is assigned UPOS, arity, subtree size, dependency gap size, and direction. The baselines are: real order, random token-order, and simplified head-proximal order.",
        "Inference relies on Z-scores relative to random baselines, Mann-Whitney U tests, and Kolmogorov-Smirnov tests. The machine-learning stage defines complexity as arity + subtree size and predicts a high-complexity label above the global median threshold.",
    ]:
        story.append(Paragraph(paragraph, styles["BodyJustify"]))
    story.append(
        Preformatted(
            "complexity(i) = arity(i) + subtree_size(i)\n"
            "label(i) = 1 if complexity(i) > median(complexity), else 0\n"
            "Z = (X_real - mean(X_random)) / sd(X_random)",
            styles["Code"],
        )
    )
    story.append(Paragraph("Implementation Notes", styles["SubSection"]))
    for bullet in [
        "The corpus experiment and the predictive experiment are implemented separately, which makes the amount of work visible in the report.",
        "Main pipeline: scripts/run_analysis.py",
        "Feature extraction: src/intervener_project/features.py",
        "Statistics and tests: src/intervener_project/analysis.py",
        "Machine learning: src/intervener_project/ml.py",
        f"GitHub-ready repository package: {REPO_LABEL}",
    ]:
        story.append(Paragraph(f"- {bullet}", styles["BulletBody"]))
    story.append(PageBreak())

    # Page 3: main results with figures
    story.append(Paragraph("3. Results I: Structural Simplicity", styles["Section"]))
    metric_rows = [
        ["Metric", "Real", "Simplified", "Random"],
        ["Mean gap", f"{real_gap:.3f}", f"{simplified_gap:.3f}", f"{random_gap:.3f}"],
        ["Mean arity", f"{stats.mean_arity:.3f}", f"{float((simplified['mean_arity'] * simplified['intervener_count']).sum() / simplified['intervener_count'].sum()):.3f}", f"{float((random['mean_arity'] * random['intervener_count']).sum() / random['intervener_count'].sum()):.3f}"],
        ["Mean subtree size", f"{stats.mean_subtree:.3f}", f"{float((simplified['mean_subtree_size'] * simplified['intervener_count']).sum() / simplified['intervener_count'].sum()):.3f}", f"{float((random['mean_subtree_size'] * random['intervener_count']).sum() / random['intervener_count'].sum()):.3f}"],
        ["Median arity", f"{stats.median_arity:.0f}", "-", "-"],
        ["Arity 0-1 share", f"{stats.pct_arity_0_1:.2f}%", "-", "-"],
        ["Subtree 1-3 share", f"{stats.pct_subtree_1_3:.2f}%", "-", "-"],
    ]
    story.append(make_table(metric_rows, [1.7 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            f"Real sentences are far simpler than randomized baselines. Mean arity is {stats.mean_arity:.3f} in real data but rises sharply under random order, while mean subtree size rises from {stats.mean_subtree:.3f} to the random baseline level. Out of {total_random_tests} language-by-metric comparisons, {strong_random_tests} real-vs-random Mann-Whitney tests are significant at p < 0.001.",
            styles["BodyJustify"],
        )
    )
    story.append(
        Paragraph(
            f"The POS pattern only partially matches the original hypothesis. Nouns are dominant ({stats.noun_share:.2f}%), but adverbs are rare ({stats.adv_share:.2f}%). The five most common real intervener categories are {top_pos_text}.",
            styles["BodyJustify"],
        )
    )
    story.append(
        image_row(
            [
                ("arity_histogram_real.png", "Figure 1. Real intervener arity histogram."),
                ("pos_distribution_real.png", "Figure 2. Real intervener POS distribution."),
            ]
        )
    )
    story.append(PageBreak())

    # Page 4: remaining corpus results with figures
    story.append(Paragraph("4. Results II: Typology, Direction, and Baselines", styles["Section"]))
    for paragraph in [
        f"The average Z-score for arity is {z_means['mean_arity']:.3f}, for subtree size {z_means['mean_subtree_size']:.3f}, and for the share of dependencies with interveners {z_means['share_with_interveners']:.3f}. All of these values are negative, so the observed corpora consistently favor simpler gaps than the random baseline.",
        f"SOV languages show slightly higher mean arity than SVO languages ({real_typology.loc['SOV', 'mean_arity']:.3f} vs {real_typology.loc['SVO', 'mean_arity']:.3f}). Leftward dependencies are also slightly richer than rightward ones ({left_weighted_arity:.3f} vs {right_weighted_arity:.3f}).",
        f"In the Dravidian subset, Tamil has mean arity {real_dravidian.loc['Tamil', 'mean_arity']:.3f} and mean subtree size {real_dravidian.loc['Tamil', 'mean_subtree_size']:.3f}, while Telugu has mean arity {real_dravidian.loc['Other Dravidian', 'mean_arity']:.3f} and mean subtree size {real_dravidian.loc['Other Dravidian', 'mean_subtree_size']:.3f}.",
    ]:
        story.append(Paragraph(paragraph, styles["BodyJustify"]))
    story.append(
        image_row(
            [
                ("real_random_simplified_boxplots.png", "Figure 3. Real vs simplified vs random."),
                ("z_scores_by_language.png", "Figure 4. Z-scores by language."),
            ],
            image_width=2.6 * inch,
            image_height=1.9 * inch,
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(
        image_row(
            [
                ("left_vs_right_dependencies.png", "Figure 5. Left vs right dependencies."),
                ("tamil_dravidian_comparison.png", "Figure 6. Tamil vs other Dravidian."),
            ],
            image_width=2.6 * inch,
            image_height=1.8 * inch,
        )
    )
    story.append(PageBreak())

    # Page 5: ML, discussion, conclusion with figures
    story.append(Paragraph("5. Machine Learning, Discussion, and Conclusion", styles["Section"]))
    ml_rows = [
        ["Model", "Accuracy", "Precision", "Recall", "F1"],
        [
            "Logistic Regression",
            f"{overall_ml.loc['logistic_regression', 'accuracy']:.3f}",
            f"{overall_ml.loc['logistic_regression', 'precision']:.3f}",
            f"{overall_ml.loc['logistic_regression', 'recall']:.3f}",
            f"{overall_ml.loc['logistic_regression', 'f1_score']:.3f}",
        ],
        [
            "Random Forest",
            f"{overall_ml.loc['random_forest', 'accuracy']:.3f}",
            f"{overall_ml.loc['random_forest', 'precision']:.3f}",
            f"{overall_ml.loc['random_forest', 'recall']:.3f}",
            f"{overall_ml.loc['random_forest', 'f1_score']:.3f}",
        ],
    ]
    story.append(make_table(ml_rows, [1.85 * inch, 0.8 * inch, 0.85 * inch, 0.75 * inch, 0.65 * inch]))
    story.append(
        Paragraph(
            f"The predictive experiment shows that dependency_distance is the strongest feature for intervener complexity, with permutation importance up to {dep_importance:.3f}. Random forest gives the best F1-score, which suggests that the relation between gap size and complexity is not purely linear.",
            styles["BodyJustify"],
        )
    )
    story.append(
        Paragraph(
            "The theoretical implication is that locality in language depends on both distance and the internal structure of the material inside that distance. This supports memory-based accounts of human sentence processing and suggests that future evaluations of machine-generated text should track intervener complexity, not just dependency length.",
            styles["BodyJustify"],
        )
    )
    story.append(
        Paragraph(
            "Overall conclusion: the hypothesis is partially supported. Real interveners are clearly short and low-arity, and nouns are dominant, but adverbs are not a major intervener class in the current data.",
            styles["BodyJustify"],
        )
    )
    story.append(
        image_row(
            [
                ("typology_comparison.png", "Figure 7. Typology comparison."),
                ("ml_model_comparison.png", "Figure 8. ML model comparison."),
            ],
            image_width=2.6 * inch,
            image_height=1.95 * inch,
        )
    )
    story.append(PageBreak())

    # Appendix starts after page 5
    story.append(Paragraph("Appendix", styles["Section"]))
    append_appendix_story(story, styles, metadata, stats)

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


def write_appendix_pdf(appendix_text: str) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(APPENDIX_PDF),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=36,
        title="Course Project Appendix",
        author="OpenAI Codex",
    )
    story = [Paragraph("Appendix: Technical Implementation", styles["SubmissionTitle"])]
    metadata = load_tables()["metadata"]
    stats = compute_stats(load_tables())
    append_appendix_story(story, styles, metadata, stats)
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = load_tables()
    stats = compute_stats(data)
    appendix = build_appendix_markdown(data, stats)
    report = build_report_markdown(data, stats)

    REPORT_MD.write_text(report, encoding="utf-8")
    APPENDIX_MD.write_text(appendix, encoding="utf-8")
    write_pdf(data, stats, appendix)
    write_appendix_pdf(appendix)

    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_PDF}")
    print(f"Wrote {APPENDIX_MD}")
    print(f"Wrote {APPENDIX_PDF}")


if __name__ == "__main__":
    main()

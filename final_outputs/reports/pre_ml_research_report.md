# Pre-ML Research Report

## Abstract

This project studies intervener complexity in dependency grammar using a configurable cross-linguistic pipeline over full SUD treebanks. The current pre-ML system extracts structural properties of intervening tokens, compares real data against multiple baselines, standardizes outputs for distributed team execution, and produces merged global analyses.

## Introduction

The central question is whether languages optimize not only dependency length but also the structural complexity of the material that intervenes between a head and its dependent. This report operationalizes intervener complexity using arity, subtree size, depth, and POS-sensitive weighting.

## Related Work

The framework draws on dependency length minimization, locality constraints, cognitive memory limitations, and information-theoretic approaches to structural optimization.

## Methodology

- Treebank source: `SUD 2.17`
- Languages in this run: `Marathi, Catalan, Galician, Persian, Tamil, Telugu, Indonesian`
- Random seed: `13`
- Random runs per baseline: `3`
- Baselines: `fully_random, projective_random, grammar_constrained, simplified`
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

- Mean language-level complexity across this run: `4.1754`
- Mean language-level dependency length across this run: `3.2516`
- Strongest negative Z-scores (real lower than baseline): `[{'language': 'galician', 'metric': 'fully_random:avg_dependency_length', 'real_value': 3.2028941814892975, 'random_mean': 11.076597593364433, 'random_std': 0.0253130334725976, 'z_score': -311.0533322842841}, {'language': 'catalan', 'metric': 'fully_random:avg_dependency_length', 'real_value': 3.0540447504302928, 'random_mean': 12.840448064913794, 'random_std': 0.0496887323676094, 'z_score': -196.95417548753883}, {'language': 'catalan', 'metric': 'fully_random:avg_subtree_size', 'real_value': 2.687669071235347, 'random_mean': 6.004336541097641, 'random_std': 0.0249471741766881, 'z_score': -132.94762149700907}, {'language': 'persian', 'metric': 'fully_random:avg_arity', 'real_value': 0.6915955446260669, 'random_mean': 0.9381687225607904, 'random_std': 0.0019729476183432, 'z_score': -124.97705242766484}, {'language': 'catalan', 'metric': 'fully_random:avg_arity', 'real_value': 0.5836339044183949, 'random_mean': 0.9362659052448596, 'random_std': 0.0030009791983932, 'z_score': -117.50564649540556}]`

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
                     index  avg_dependency_length  avg_complexity  avg_arity  avg_subtree_size  avg_depth  percent_left_dependencies  percent_right_dependencies  entropy_pos_distribution  avg_efficiency_ratio
     avg_dependency_length               1.000000        0.787539   0.857591          0.798492   0.735278                  -0.308588                    0.308588                 -0.255871              0.764079
            avg_complexity               0.787539        1.000000   0.972213          0.993508   0.994357                  -0.691950                    0.691950                  0.200953              0.792864
                 avg_arity               0.857591        0.972213   1.000000          0.958177   0.953007                  -0.564462                    0.564462                  0.179420              0.830535
          avg_subtree_size               0.798492        0.993508   0.958177          1.000000   0.980407                  -0.680211                    0.680211                  0.142742              0.767587
                 avg_depth               0.735278        0.994357   0.953007          0.980407   1.000000                  -0.738940                    0.738940                  0.256885              0.778614
 percent_left_dependencies              -0.308588       -0.691950  -0.564462         -0.680211  -0.738940                   1.000000                   -1.000000                 -0.537538             -0.570594
percent_right_dependencies               0.308588        0.691950   0.564462          0.680211   0.738940                  -1.000000                    1.000000                  0.537538              0.570594
  entropy_pos_distribution              -0.255871        0.200953   0.179420          0.142742   0.256885                  -0.537538                    0.537538                  1.000000              0.290159
      avg_efficiency_ratio               0.764079        0.792864   0.830535          0.767587   0.778614                  -0.570594                    0.570594                  0.290159              1.000000
```

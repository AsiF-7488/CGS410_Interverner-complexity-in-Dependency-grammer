# Final Results Explained

This file gives a plain-language summary of the final findings from the 7-language project.

## Languages

- Marathi
- Catalan
- Galician
- Persian
- Tamil
- Telugu
- Indonesian

## Main Conclusion

The hypothesis is **partially supported**.

- Supported: interveners are usually short and low-arity.
- Supported: nouns are the single most common intervener type.
- Not supported: adverbs are not dominant.

## Key Quantitative Results

- Real dependency pairs with non-zero gaps: `31,007`
- Real intervener tokens: `44,881`
- Mean real intervener arity: `0.612`
- Median real intervener arity: `0`
- Share of real interveners with arity `0` or `1`: `87.77%`
- Mean real intervener subtree size: `2.881`
- Median real intervener subtree size: `1`
- Share of real interveners with subtree size `1-3`: `78.81%`
- NOUN share among real interveners: `30.43%`
- ADV share among real interveners: `3.04%`

## Interpretation

The strongest result is that real language is much simpler than randomized baselines. This means languages seem to minimize not only dependency length, but also the complexity of what appears inside the dependency gap.

The POS finding is more nuanced. Nouns dominate, but determiners, adpositions, adjectives, and proper nouns are also common. Adverbs are relatively rare, so the original claim "mostly nouns and adverbs" needs to be revised.

## Machine Learning

Two models were tested:

- Logistic Regression
- Random Forest

Global results:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.610 | 0.580 | 0.499 | 0.536 |
| Random Forest | 0.594 | 0.544 | 0.629 | 0.584 |

The strongest predictive feature is `dependency_distance`.

## Typology and Directionality

- SOV languages have slightly higher intervener arity than SVO languages.
- Leftward dependencies are slightly richer than rightward dependencies.
- Tamil is structurally richer than Telugu in this dataset, but not an outlier relative to all other languages.

## What To Say In The Discussion

You can safely say:

1. Natural languages prefer simple interveners.
2. This supports memory-based and locality-based theories of processing.
3. Dependency distance alone is not enough; the internal structure of the gap also matters.
4. The hypothesis should be revised from "mostly nouns and adverbs" to "mostly nouns and other structurally light categories."

# V7-PRE — Decision Layer Check

We compared three ways of turning the existing forecast outputs into cyclone categories.

The thresholds were fixed from the training data:

- 27.5 kt
- 32.5 kt
- 47.5 kt
- 62.5 kt
- 117.5 kt

The 758 validation sequences belong to three storms that were kept completely separate from training. The test set was not used.

| Method | +6h Accuracy | +6h Present F1 | +12h Accuracy | +12h Present F1 | Mean Present F1 |
|---|---:|---:|---:|---:|---:|
| Direct classifier | 0.6623 | 0.4640 | 0.5462 | 0.3860 | **0.4250** |
| Wind buckets | 0.6253 | 0.4485 | 0.5594 | 0.3953 | 0.4219 |
| Hybrid | 0.6623 | 0.4640 | 0.5462 | 0.3860 | **0.4250** |

The direct classifier remains the best overall decision method. The wind-based rule did not improve the combined result, so the decision layer is now locked to the direct V4 classifier.

This experiment used no test data.

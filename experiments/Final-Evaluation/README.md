# Final evaluation

The forecasting model was evaluated on seven held-out storms:

ASHOBAA, Fani, HELEN, KYAAR, LEHAR, NILOFAR and PAWAN.

| Metric | +6h | +12h |
|---|---:|---:|
| Exact six-class accuracy | 54.84% | 42.25% |
| Macro F1 | 0.3988 | 0.3324 |
| Within one intensity class | 97.09% | 92.09% |
| 3-level operational accuracy | 75.34% | 69.61% |

The mean exact six-class accuracy across the two horizons is 48.54%. Mean within-one-class agreement is 94.59%.

The decision parameters were selected before the final test evaluation. No test-set tuning was used.

Within-one-class agreement is reported separately from exact six-class accuracy and should not be interpreted as the same metric.

Wind MAE on the frozen test set was 8.45 kt at +6h and 10.80 kt at +12h.
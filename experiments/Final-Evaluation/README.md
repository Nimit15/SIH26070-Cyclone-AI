# Final test check

The temporal model was run once on the seven held-out test storms after the V4 checkpoint and decision rule had already been fixed.

The test set contains 961 sequences.

| Metric | +6h | +12h |
|---|---:|---:|
| Exact six-class accuracy | 52.97% | 42.25% |
| Macro F1 | 0.3827 | 0.3324 |
| Top-2 accuracy | 82.00% | 69.72% |
| Within one intensity class | 95.73% | 92.09% |
| Wind MAE | 8.45 kt | 10.80 kt |
| Wind RMSE | 10.57 kt | 13.97 kt |

Using the frozen structured decision rule gave 54.84% exact accuracy at +6h and 42.25% at +12h. The mean exact accuracy was 48.54%.

For the project's tolerance-aware view, the mean agreement within one intensity class was 94.59%. This is kept separate from exact six-class accuracy.

The test set was not used to choose the model or decision parameters.

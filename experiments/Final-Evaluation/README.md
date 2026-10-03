# Final evaluation

The forecasting model was evaluated once on the seven held-out test storms:

ASHOBAA, Fani, HELEN, KYAAR, LEHAR, NILOFAR and PAWAN.

| Metric | +6h | +12h |
|---|---:|---:|
| Exact six-class accuracy | 52.97% | 42.25% |
| Macro F1 | 0.3827 | 0.3324 |
| Top-2 accuracy | 82.00% | 69.72% |
| Within one intensity class | 95.73% | 92.09% |
| Wind MAE | 8.45 kt | 10.80 kt |
| Wind RMSE | 10.57 kt | 13.97 kt |

The frozen structured decision rule raised +6h exact accuracy to 54.84%, while +12h remained at 42.25%. The combined exact six-class average was 47.61% for the direct model.

Across both forecast horizons, 93.91% of predictions were within one intensity class of the target.

The wind estimates were more stable than the class predictions. For the >=80 kt subset, the +6h MAE was 9.84 kt and the +12h MAE was 15.91 kt.

The direct classifier did not assign the SuCS label to any of the 67 true SuCS cases at either horizon.

The test set was not used to choose the model, thresholds or decision parameters.

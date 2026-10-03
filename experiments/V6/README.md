# V6 — V4 Stacking / Correction

## What we tested

V6 added a small Logistic Regression correction layer on top of the verified V4 forecast outputs. The idea was to see whether the model could learn useful corrections from the existing predictions, recent wind history, and physical context.

We used 6,802 training sequences from 33 storms and kept the 758 validation sequences from 3 completely unseen storms separate. The test set was not used.

Two versions were compared with 5-fold storm-grouped cross-validation:

- standard Logistic Regression
- class-balanced Logistic Regression

The standard version had the better internal OOF result, with a mean present-class F1 of 0.6902.

## What happened on the unseen validation storms

The internal OOF result did not carry over to the unseen validation storms.

| Model | +6h Accuracy | +6h Present F1 | +12h Accuracy | +12h Present F1 | Mean Present F1 |
|---|---:|---:|---:|---:|---:|
| V4 | 0.6623 | 0.4640 | 0.5462 | 0.3860 | 0.4250 |
| V6 | 0.6530 | 0.4074 | 0.5475 | 0.3844 | 0.3959 |

So V6 was not kept as the final forecasting model.

## What we learned

The result suggests that another generic stacking layer is not solving the main generalization problem. Our next experiments will focus on information that is more directly connected to cyclone structure and evolution, especially spatial-temporal features and additional physical signals.

V4 remains the validated fallback for the forecasting branch.

## Reproducibility

The V6 run used the frozen V4 prediction cache. Neither the validation results nor the model choice used the test set.

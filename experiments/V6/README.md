# V6 — V4 Stacking / Correction Model

**Status: Rejected for final model**

V6 tested a lightweight meta-learning correction layer on top of the verified V4 temporal forecast outputs.

## Setup

- Train sequences: 6,802
- Validation sequences: 758
- Unique training storms: 33
- Meta-features: 53
- Internal selection: 5-fold GroupKFold by storm
- Candidates:
  - standard Logistic Regression
  - class-balanced Logistic Regression
- Test set: **not used**

## Internal OOF selection

The standard Logistic Regression model was selected:

- +6h accuracy: 0.8583
- +6h present-class Macro F1: 0.7104
- +12h accuracy: 0.8118
- +12h present-class Macro F1: 0.6700
- mean present-class Macro F1: **0.6902**

The balanced model scored 0.6845 mean present-class Macro F1 and was not selected.

## Completely unseen validation storms

Verified V4:

- +6h accuracy: 0.6623
- +6h present-class Macro F1: 0.4640
- +12h accuracy: 0.5462
- +12h present-class Macro F1: 0.3860
- mean present-class Macro F1: **0.4250**

V6 meta-model:

- +6h accuracy: 0.6530
- +6h present-class Macro F1: 0.4074
- +12h accuracy: 0.5475
- +12h present-class Macro F1: 0.3844
- mean present-class Macro F1: **0.3959**

Mean validation Macro F1 delta versus V4: **-0.0291**

## Decision

V6 is **not** the final forecasting model.

The experiment demonstrates that strong grouped OOF performance on the 33 training storms did not transfer to the 3 completely unseen validation storms. This supports prioritizing physically grounded features and spatial-temporal modeling over another generic stacking layer.

The V4 checkpoint remains the stronger validated fallback.

## Reproducibility

The experiment used the frozen V4 prediction cache and did not evaluate or tune against the test set.

# V6 — V4 Stacking / Correction Model

Status: **Rejected for final model**

V6 used 53 meta-features built from verified V4 predictions,
current-state logits/winds, wind history and physical context.

Internal grouped 5-fold OOF:
- Mean present-class F1: 0.6902

Completely unseen validation storms:
- V4 mean present-class F1: 0.4250
- V6 mean present-class F1: 0.3959

Therefore V6 was not selected.

The test set was not used for model selection or evaluation.

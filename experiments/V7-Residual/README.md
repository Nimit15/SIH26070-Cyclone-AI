# V7 residual check

The spatial ConvLSTM was tried as a small correction on top of the existing V4 forecast instead of replacing it.

On the three held-out validation storms, the first epoch gave a mean present-class F1 of 0.4292:

- +6h Present F1: 0.4661
- +12h Present F1: 0.3923
- +6h wind MAE: 8.40 kt
- +12h wind MAE: 11.30 kt

The V4 reference was 0.4250 mean present-class F1. Later epochs did not keep the small gain.

The residual model is therefore kept as an experimental checkpoint, but V4 stays the main forecasting reference because the improvement was small and not consistent across epochs.

The test set was not used.

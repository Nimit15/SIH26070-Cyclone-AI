# Current project state

This is the working state at the end of the 4 October session.

## Current baseline

The spatial EfficientNet-B0 baseline is still the main image model.

Validation:
- Accuracy: 66.15%
- Macro F1: 62.37%
- Wind MAE: 7.02 kt
- Wind RMSE: 10.09 kt

Locked storm split:
- Train: 69,048 images / 33 storms
- Validation: 1,362 images / 3 storms
- Test: 2,532 images / 7 storms

## Temporal branch

Strict sequence table:
- 8,521 sequences
- Train: 6,802
- Validation: 758
- Test: 961
- 4 inputs at t-9h, t-6h, t-3h and t
- Targets at +6h and +12h
- Storm split kept separate

Current V4 temporal reference on the 758 validation sequences:
- +6h exact accuracy: 66.23%
- +6h strict six-class Macro F1: 0.3867
- +6h wind MAE: about 8.44 kt
- +12h exact accuracy: 54.62%
- +12h strict six-class Macro F1: 0.3216
- +12h wind MAE: about 11.32 kt
- +6h within one severity class: 95.38%
- +12h within one severity class: 90.37%
- +6h top-2 accuracy: 87.86%

A structured decision check was fitted from training-only grouped data:
- +6h gamma 0.10, delta 0.00
- +12h gamma 0.00, delta 1.40

On validation this gave:
- +6h exact accuracy: 66.89%
- +6h 3-level operational accuracy: 88.52%
- +12h 3-level operational accuracy: 85.62%
- overall 3-level operational accuracy: 87.07%

The 3-level grouping used for that figure is:
- D
- DD
- CS / SCS / VSCS / SuCS

The 100% two-way SuCS split seen during exploration is not used because the validation storms do not contain a meaningful SuCS case.

## Experiments not kept

- V5 full temporal fine-tuning: degraded the V4 result.
- V6 generic stacking: rejected after unseen-storm validation.
- V7 spatial ConvLSTM replacement: rejected.
- V7 spatial residual: epoch 1 reached 0.4292 mean present-class F1, only slightly above V4 0.4250; later epochs fell back, so V4 remains the reference.
- DeepTCEye / EPI: weak transfer on the NIO HURSAT subset.
- DeepTCNet intensity transfer: weak overall and especially poor on strong storms.
- TCIR NIO pretraining: corrected alignment run reached 31.15% validation accuracy and 18.01 kt wind MAE; not kept.

## TCIR data

A compact TCIR NIO subset was successfully created:
- 1,600 frames
- 75 storms
- 4 channels
- Vmax 15–145 kt
- file: `TCIR_NIO_subset.h5`
- metadata: `TCIR_NIO_subset_metadata.csv`

The original 13.92 GB HDF5 extracted under /tmp was deleted after the subset was corrected.

The original compressed archive had been kept under Kaggle working and was used to recreate the corrected subset.

## Feature and context caches

Verified files:
- spatial feature cache: `spatial_features_fp16.dat`
- shape: (12,033, 1,280, 7, 7)
- dtype: fp16
- size: about 1.406 GiB
- baseline logits cache: (12,033, 6)
- baseline wind cache: (12,033,)
- temporal context: (8,521, 20)
- physical context: (8,521, 6)

## Last unfinished task

The final V4 test benchmark has NOT been completed.

The previous attempt tried to reconstruct the V4 model from the checkpoint but did not reproduce the known validation fingerprint. The actual V4 checkpoint has a wrapper structure with:
- `baseline.*`
- `physical_core.*`

The attempted reconstruction produced:
- +6h accuracy 63.32%, Macro F1 0.3136, MAE 9.58 kt
- +12h accuracy 56.99%, Macro F1 0.2874, MAE 12.08 kt

So those outputs must not be used.

The immediate next step is to locate the original V4 implementation or an existing exact V4 test-output cache in the Kaggle environment. Only after reproducing the known validation fingerprint should the 961 test predictions be generated.

Do not tune the test set.

## GitHub

The repository is kept intentionally plain and readable. Experiment notes are written as normal project notes rather than automated-looking change logs.

Latest relevant documentation includes:
- `experiments/V6/README.md`
- `experiments/V7-PRE/README.md`
- `experiments/V7-Residual/README.md`
- `experiments/External-Models/README.md`

The dashboard code in `main.py` no longer contains the old hard-coded 117.5 kt SuCS override.

## Tomorrow

1. Recover the exact V4 inference implementation from Kaggle files/notebooks or locate an existing exact V4 test cache.
2. Reproduce the known V4 validation numbers before evaluating the test set.
3. Generate the 961 frozen V4 test outputs.
4. Apply the already-frozen decision parameters without tuning.
5. Save the final metrics and per-class results.
6. Then finish the dashboard/demo and presentation around the verified numbers.

No test-based tuning has been done so far.

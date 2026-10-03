# External model checks

A few published or external models were tested against our North Indian Ocean data before deciding whether they belonged in the final system.

## DeepTCEye / DeepTCNet

We ran the released DeepTCEye model on 742 original HURSAT-B1 infrared frames from five North Indian Ocean storms already selected for the project.

The eye detector loaded in its original TensorFlow 2.8 / Keras 2.8 environment. We kept that environment separate from the main project setup.

We also tested the released DeepTCNet intensity model on the same frames.

DeepTCEye:

- 742 frames processed
- 70 eye-positive frames
- eye-positive rate: 9.43%

The eye-persistence counts had only weak relationships with the HURSAT wind labels. The strongest correlation was 0.178 for EPI12, and the relationships were weaker on the stronger-storm subset.

DeepTCNet transfer:

- overall MAE: 15.83 kt
- overall RMSE: 22.35 kt
- bias: -7.17 kt
- correlation: 0.655
- MAE for winds at least 80 kt: 32.65 kt
- bias for winds at least 80 kt: -31.6 kt

We left both out of the final forecasting branch. The published models did not transfer cleanly to this North Indian Ocean set, especially at high intensity.

## TCIR North Indian Ocean check

We also pulled the Indian Ocean part of TCIR and kept a compact 1,600-frame subset covering 75 storms and winds from 15 to 145 kt. The stronger cases were deliberately kept because the main model has its biggest weakness in the upper intensity tail.

For a quick transfer check, we trained an EfficientNet-B0 on the TCIR IR channel and kept storm years overlapping the HURSAT training set out of the run.

Validation:

- 321 frames
- 11 storms
- best accuracy: 31.15%
- best wind MAE: 18.01 kt

The transfer was not strong enough to justify replacing the project encoder with a TCIR-pretrained one, so this branch is not being carried forward.

The useful part of the TCIR work is the additional North Indian Ocean satellite data itself. We keep it as a possible representation-learning source, but not as a source of ready-made intensity predictions.

No project test data was used for these checks.

# External eye-structure experiment

## What we tested

We used the released DeepTCEye model on 742 original HURSAT-B1 infrared frames from five North Indian Ocean storms already selected for the project.

The eye detector loaded successfully in its original TensorFlow 2.8 / Keras 2.8 environment. We did not change the main project environment.

We also tested the released DeepTCNet intensity model on the same 742 frames.

## Results

DeepTCEye:

- 742 frames processed
- 70 eye-positive frames
- eye-positive rate: 9.43%

Eye-persistence counts had only weak relationships with the HURSAT wind labels:

- EPI6 count correlation: 0.176
- EPI12 count correlation: 0.178
- EPI18 count correlation: 0.141
- EPI24 count correlation: 0.097

For the 136 frames with wind at least 80 kt, the correlations dropped further:

- EPI6: 0.153
- EPI12: 0.120
- EPI18: 0.031
- EPI24: 0.031

DeepTCNet transfer:

- overall MAE: 15.83 kt
- overall RMSE: 22.35 kt
- bias: -7.17 kt
- correlation: 0.655
- MAE for winds at least 80 kt: 32.65 kt
- bias for winds at least 80 kt: -31.6 kt

## Decision

We are not adding DeepTCEye EPI or the released DeepTCNet wind prediction to the final forecasting model.

The external eye signal was too weak on this North Indian Ocean subset, and the external intensity model substantially underestimated strong storms. The experiment was still useful because it gave us a direct transfer test rather than assuming that a published result would transfer to our task.

The next modeling work will focus on preserving spatial evolution and using a geographically relevant external satellite dataset rather than importing a pretrained intensity output directly.

No project test data was used.

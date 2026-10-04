# Cyclone Intelligence System

Explainable cyclone intensity analysis from satellite imagery.

## Overview

The system combines a satellite-image classifier with wind-speed regression and a temporal forecasting branch. The web app provides image-based cyclone analysis with Grad-CAM visualisation.

The forecasting model uses short storm sequences and predicts intensity at +6h and +12h.

## Dataset

The evaluation uses storm-level splits so images from the same storm do not appear across train, validation and test.

- Train: 69,048 images from 33 storms
- Validation: 1,362 images from 3 storms
- Test: 2,532 images from 7 storms

The temporal forecasting dataset contains 8,521 sequences:

- Train: 6,802
- Validation: 758
- Test: 961

The seven final test storms are ASHOBAA, Fani, HELEN, KYAAR, LEHAR, NILOFAR and PAWAN.

## Model

The image model uses EfficientNet-B0 with two outputs:

- intensity category classification
- maximum sustained wind-speed regression

The temporal forecasting model uses satellite feature sequences together with recent model state information. The current forecasting reference is the V4 temporal model.

The system also includes:

- Grad-CAM visualisation
- storm-level evaluation
- +6h and +12h forecasting
- wind-speed estimates
- tolerance-aware intensity metrics

## Results

The final frozen test evaluation was performed only after the model and validation-selected decision strategy were fixed.

### Exact six-class accuracy

| Horizon | Accuracy | Macro F1 |
| --- | ---: | ---: |
| +6h | 54.84% | 0.3988 |
| +12h | 42.25% | 0.3324 |

### Forecast quality

| Horizon | Within ±1 class | 3-level operational |
| --- | ---: | ---: |
| +6h | 97.09% | 75.34% |
| +12h | 92.09% | 69.61% |

The mean exact six-class accuracy across the two horizons is 48.54%. The mean within-one-class agreement is 94.59%.

These metrics are reported separately. Within-one-class agreement is not presented as exact classification accuracy.

## Web app

The application currently supports:

1. satellite-image upload
2. cyclone category prediction
3. wind-speed estimation
4. confidence display
5. Grad-CAM visualisation

Run locally with:

```bash
pip install -r requirements.txt
python main.py
```

The application uses the checkpoint in `checkpoints/cyclone_best.pt`.

## Repository

```
checkpoints/       model checkpoints
experiments/       model experiments
splits/            storm-level dataset splits
main.py            NiceGUI application
requirements.txt   Python dependencies
README.md          project documentation
```

## Notes

The test set contains storms that were not used during training or model selection. The current dataset has limited examples of the highest intensity category, which remains an important limitation for future work.

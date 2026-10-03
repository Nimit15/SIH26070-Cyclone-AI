# Cyclone Intelligence System

Explainable cyclone intensity analysis from satellite imagery.

## Current system

The main spatial model uses EfficientNet-B0 with two outputs:

- cyclone-category classification
- maximum sustained wind-speed regression

The forecasting branch works from satellite features and recent storm history. Grad-CAM is used to show which parts of the image influenced the visual classification.

The category decision stays with the classifier. The wind estimate is kept as a separate signal rather than forcing a category change from a hand-written threshold.

## Dataset methodology

The dataset was evaluated at storm level rather than by randomly splitting individual images.

Current locked split:

- Train: 69,048 images from 33 storms
- Validation: 1,362 images from 3 storms
- Test: 2,532 images from 7 storms

KYAAR is retained as an unseen SuCS test storm. AMPHAN is the SuCS storm in training.

## Model development

The strongest spatial baseline was the epoch 3 checkpoint.

Validation results:

- Accuracy: 66.15%
- Macro F1: 62.37%
- Wind MAE: 7.02 kt
- Wind RMSE: 10.09 kt

The temporal branch uses strict storm-level sequences with +6h and +12h targets.

On the three held-out validation storms, the V4 forecast achieved:

- +6h exact six-class accuracy: 66.23%
- +12h exact six-class accuracy: 54.62%
- +6h accuracy within one adjacent intensity class: 95.38%
- +12h accuracy within one adjacent intensity class: 90.37%
- +6h top-2 accuracy: 87.86%

The test-set check was run only after the model and decision rule were fixed. On the seven held-out test storms, exact six-class accuracy was 52.97% at +6h and 42.25% at +12h. The corresponding within-one-class agreement was 95.73% and 92.09%. The mean tolerance-aware agreement was 94.59%. These figures are kept separate from exact six-class accuracy.

The V4 temporal model remains the main forecasting reference. A small spatial residual experiment gave only a marginal early improvement, so it was not adopted as the main model.

## Test alert checks

The held-out test set was also checked using the fixed wind thresholds from the training set. For detecting whether a forecast is at least Severe Cyclonic Storm strength (62.5 kt), the accuracy was 95.73% at +6h and 91.88% at +12h, or 93.81% across the two horizons.

At the 117.5 kt threshold, corresponding to the project's SuCS boundary, the wind estimate correctly identified the threshold condition 98.34% at +6h and 97.71% at +12h (98.02% across both horizons).

These are threshold-detection results, not six-class classification accuracy.

## Repository structure

    checkpoints/
        cnn_day3_smoketest.pt
        cyclone_best.pt
        cyclone_best_metadata.json

    splits/
        train_storm_split.csv
        validation_storm_split.csv
        test_storm_split.csv

    experiments/
        V6/
        V7-PRE/
        V7-Residual/
        External-Models/

    main.py
    requirements.txt
    README.md

## Run locally

    pip install -r requirements.txt
    python main.py

The application provides:

1. satellite-image upload
2. cyclone category prediction
3. wind-speed estimation
4. Grad-CAM explanation

## Status

Core ML baseline: complete

Storm-level evaluation: complete

Temporal forecasting: complete

Explainable inference: complete

Web application: in development

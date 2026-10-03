# Cyclone Intelligence System

Explainable cyclone intensity analysis from satellite imagery.

## Current system

The main spatial model uses EfficientNet-B0 with two outputs:

- cyclone-category classification
- maximum sustained wind-speed regression

The forecasting branch is built on top of the satellite features and recent storm history. Grad-CAM is used to show which parts of the image influenced the visual classification.

The category decision currently stays with the classifier. The wind estimate is kept as a separate signal rather than forcing a category change from a hand-written threshold.

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

The temporal branch has been tested separately using strict storm-level sequences with +6h and +12h targets. The current fallback for that branch is the V4 residual model; newer stacking and external-model experiments have not replaced it.

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

Temporal forecasting experiments: in progress

Explainable inference: complete

Web application: in development

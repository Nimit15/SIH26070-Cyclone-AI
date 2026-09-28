# Cyclone Intelligence System

Explainable cyclone intensity analysis from satellite imagery.

## Current system

The prototype uses an EfficientNet-B0 backbone with two prediction heads:

- cyclone-category classification
- maximum sustained wind-speed regression

The inference layer combines the visual classifier with the continuous wind estimate. A wind estimate of 117.5 kt or above triggers the SuCS decision because the training distribution separates VSCS (maximum 115 kt) from SuCS (minimum 120 kt).

Grad-CAM is used to visualize the image regions contributing to the visual classification.

## Dataset methodology

The dataset was evaluated at storm level rather than by randomly splitting individual images.

Current locked split:

- Train: 69,048 images from 33 storms
- Validation: 1,362 images from 3 storms
- Test: 2,532 images from 7 storms

KYAAR is retained as an unseen SuCS test storm. AMPHAN is the SuCS storm in training.

## Model development

The best validation checkpoint was obtained at epoch 3.

Validation results:

- Accuracy: 66.15%
- Macro F1: 62.37%
- Wind MAE: 7.02 kt
- Wind RMSE: 10.09 kt

The model's wind-speed head also showed useful performance on the held-out KYAAR storm.

## Repository structure

    checkpoints/
        cnn_day3_smoketest.pt
        cyclone_best.pt
        cyclone_best_metadata.json

    splits/
        train_storm_split.csv
        validation_storm_split.csv
        test_storm_split.csv

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

Core ML model: complete

Storm-level evaluation: complete

Explainable inference: complete

Web application: in development

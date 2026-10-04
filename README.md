# CYCLO-INTEL

AI-powered cyclone intelligence for satellite-image analysis, short-term intensity forecasting, explainability, official alerts, and regional-language guidance.

## What it does

- Analyses a cyclone satellite image with EfficientNet-B0
- Predicts one of six intensity classes: D, DD, CS, SCS, VSCS, SuCS
- Estimates wind speed separately
- Shows a Grad-CAM visual explanation
- Provides historical +6h / +12h intensity replay using the verified V4 results
- Checks official IMD alerts for the selected State / UT
- Provides regional-language safety guidance
- Links to official IMD, NDMA SACHET, MOSDAC and NASA satellite sources

The current-image decision is classifier-only. The wind estimate is shown separately and does not override the category.

Forecast Replay is a historical validation demonstration, not a live forecast service. It includes both successful and failed cases.

## Model and data

The image model is EfficientNet-B0 with classification and wind-speed regression.

The temporal model uses four ordered satellite frames with temporal attention and physical/context features for +6h and +12h intensity forecasting.

The main image dataset is INCYDE (INSAT-3D, North Indian Ocean). The locked temporal dataset contains 8,521 storm-level sequences:

- Train: 6,802
- Validation: 758
- Test: 961

The final test storms are held out by storm rather than by random image.

IBTrACS was used for storm and intensity metadata during dataset preparation. NOAA HURSAT-B1 was investigated as an additional historical source, but it is not a final V4 predictor input.

## Final frozen V4 test reference

| Horizon | Exact accuracy | Macro F1 | Within ±1 severity |
|---|---:|---:|---:|
| +6h | 52.97% | 0.3827 | 95.73% |
| +12h | 42.25% | 0.3324 | 92.09% |

Wind MAE:

- +6h: 8.45 kt
- +12h: 10.80 kt

Within ±1 severity is a tolerance metric, not exact classification accuracy.

## Run locally

The repository already contains the image-model checkpoint, so no separate model download is needed for the main application.

Tested setup:

- Windows
- Python 3.14
- PyTorch with CUDA when a compatible NVIDIA setup is available

### 1. Clone the repository

```powershell
git clone https://github.com/Nimit15/SIH26070-Cyclone-AI.git
cd SIH26070-Cyclone-AI
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the dashboard

```powershell
python main.py
```

Open:

```text
http://127.0.0.1:8000
```

The model checkpoint is already included here:

```text
checkpoints/cyclone_best.pt
```

CPU inference is supported, but an NVIDIA GPU will be faster.

### Optional: Forecast Replay

The GitHub repository does not include the local `v4_demo_bundle` replay images.

The main dashboard and current-image analysis still run without them.

For the complete historical Forecast Replay, place the supplied `v4_demo_bundle` folder next to `main.py`:

```text
SIH26070-Cyclone-AI/
├── main.py
├── checkpoints/
│   └── cyclone_best.pt
└── v4_demo_bundle/
```

The final demo package contains the replay assets separately.

No training dataset is required to run the application.

## Project structure

```text
main.py
checkpoints/
splits/
experiments/Final-Evaluation/
requirements.txt
README.md
```

## Online deployment

The submission version is a local Python application so it can be demonstrated reliably. It can be deployed later on a GPU-capable host without changing the model itself.

## Safety

CYCLO-INTEL is decision support. It does not replace official IMD, RSMC, NDMA or state-authority warnings.

# CYCLO-INTEL

AI-powered cyclone intelligence for satellite-image analysis, short-term intensity forecasting, explainability, official alerts, and regional-language guidance.

## What it does

- Analyses a cyclone satellite image with EfficientNet-B0
- Predicts one of six intensity classes: D, DD, CS, SCS, VSCS, SuCS
- Estimates wind speed separately
- Shows a Grad-CAM visual explanation
- Uses the verified V4 temporal model for historical +6h / +12h intensity replay
- Checks official IMD alerts for the selected State / UT
- Provides regional-language guidance
- Links to official IMD, NDMA SACHET, MOSDAC and NASA satellite sources

The final decision for a current image is classifier-only. The wind head is shown as a separate estimate and does not override the category.

The Forecast Replay page is a historical validation demonstration, not a live forecast service. It shows both successful and failed cases.

## Model and data

The image model is EfficientNet-B0 with classification and wind-speed regression.

The temporal model uses four ordered satellite frames with temporal attention and physical/context features to forecast intensity at +6h and +12h.

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

Tested setup:

- Windows
- Python 3.14
- PyTorch with CUDA when an NVIDIA GPU is available

Clone the repository and enter it:

```bash
git clone https://github.com/Nimit15/SIH26070-Cyclone-AI.git
cd SIH26070-Cyclone-AI
```

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the application:

```powershell
python main.py
```

Open:

```text
http://127.0.0.1:8000
```

The checkpoint is already included at:

```text
checkpoints/cyclone_best.pt
```

For an NVIDIA GPU, use a PyTorch build compatible with your driver. CPU inference also works, but will be slower.

## Forecast replay assets

The full four-case replay uses the supplied `v4_demo_bundle` folder. Keep it next to `main.py`.

The main image-analysis application does not depend on the replay bundle. If the bundle is missing, the replay page reports that the historical assets are unavailable instead of blocking the rest of the dashboard.

## Project structure

```text
main.py                     NiceGUI application
checkpoints/                final image-model checkpoint
splits/                     storm-level train/validation/test splits
experiments/Final-Evaluation final evaluation notes
requirements.txt            Python dependencies
README.md                   project and run instructions
```

## Online deployment

The submission version is a local Python application so it can be demonstrated reliably. An online deployment can be added after submission on a GPU-capable host or container platform without changing the model itself.

## Safety

CYCLO-INTEL is decision support. It does not replace official IMD, RSMC, NDMA or state-authority warnings.
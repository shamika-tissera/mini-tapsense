# mini-tapsense

Acoustic tap-gesture recognition on a personal device microphone.

**Repo:** `https://github.com/shamika-tissera/mini-tapsense`

## What’s in here
- `main.ipynb`: end-to-end pipeline (load audio/labels → segment taps → FFT features → rule-based + k-NN → dataset-size sweep + confusion matrices).
- `new_data/`: recorded 4-class dataset audio (`Knuckle.m4a`, `Pad.m4a`, `Nail tip tap.m4a`, `Nail tap.m4a`).
- `labels/`: Audacity-style label tracks (`start\tend` per tap) for each class.
- `scripts/auto_annotate_taps.py`: onset-based auto-label generator (used when label files are missing).
- `acm_report.tex`: 2-page ACM-style report draft (includes figures from `report_figs/`).
- `report_figs/`: figures used by the report (exported from notebook outputs).

## Run Notebook
1. Open and run `main.ipynb`.
2. The notebook supports **4 classes**: `knuckle`, `pad`, `nail_tip_tap`, `nail_tap`.
3. The dataset-size sweep will evaluate on:
   - an external test set if you provide separate test recordings, or
   - a stratified holdout split from the extracted features (fallback).
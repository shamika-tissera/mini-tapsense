# mini-tapsense

Acoustic tap-gesture recognition on a personal device microphone.

**Repo:** `https://github.com/shamika-tissera/mini-tapsense`

## Repository layout
- `main.ipynb`: end-to-end pipeline (load audio/labels → segment taps → FFT features → rule-based + k-NN → dataset-size sweep + confusion matrices).
- `auto_annotate_and_visualize.ipynb`: helper notebook for label generation/inspection and visualization.
- `data/train/`: training recordings (`*.m4a`) per gesture class.
- `data/test/`: test recordings (`*-test.m4a`) per gesture class.
- `labels/`: Audacity label tracks (`*.txt`) matching audio basenames (e.g. `data/train/Knuckle.m4a` → `labels/Knuckle.txt`).
- `scripts/auto_annotate_taps.py`: onset-based auto-label generator (writes `.txt` label files).
- `scripts/auto_annotate_taps.md`: usage notes + recommended labeling workflow.

## Run notebook
1. Open and run `main.ipynb`.
2. The current dataset is split into:
   - `data/train/` with labels in `labels/*.txt`
   - `data/test/` with labels in `labels/*-test.txt`
3. The notebook’s 4 gesture classes are derived from the filenames:
   - `Knuckle`, `Pad`, `Nail tip tap`, `Nail tap`

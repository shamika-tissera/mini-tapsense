# Auto-annotation (Audacity label files)

This repo’s tap labels are in Audacity “label track” text format:

`start_time_sec<TAB>end_time_sec<TAB>index`

By default, `scripts/auto_annotate_taps.py` writes a `.txt` file alongside each input audio file. To match this repo layout, pass `--out-dir labels`.

```bash
python3 scripts/auto_annotate_taps.py data/train/Knuckle.m4a --out-dir labels --print-stats
```

To (re)generate labels for the full dataset:

```bash
python3 scripts/auto_annotate_taps.py data/train/*.m4a data/test/*-test.m4a --out-dir labels --print-stats
```

## Best accuracy workflow

1) Record one long file per gesture class (e.g., only knuckle taps).
2) While recording, tap a known number of times (e.g., exactly 30).
3) Run auto-annotation with the expected count:

```bash
python3 scripts/auto_annotate_taps.py my_knuckle.m4a \
  --out-dir labels \
  --expected-count 30 \
  --fixed-dur-ms 110 \
  --pre-ms 10 \
  --min-gap-ms 120 \
  --print-stats
```

4) Import the generated `.txt` into Audacity to visually spot-check and correct rare misses/merges.

## Notes

- Stereo input is handled automatically (the script converts to mono).
- For non-`.wav` inputs (e.g. `.m4a`, `.mp3`), `ffmpeg` is required (default behavior is to use it when available).

# Auto-annotation (Audacity label files)

This repo’s training labels are in Audacity “label track” text format:

`start_time_sec<TAB>end_time_sec<TAB>index`

To generate these automatically from `.wav` recordings, use:

```bash
python3 scripts/auto_annotate_taps.py audio_file/knuckle.wav --print-stats
```

## Best accuracy workflow (recommended)

1) Record one long file per gesture class (e.g., only knuckle taps).
2) While recording, tap a known number of times (e.g., exactly 30).
3) Run auto-annotation with the expected count:

```bash
python3 scripts/auto_annotate_taps.py my_knuckle.wav \
  --expected-count 30 \
  --fixed-dur-ms 110 \
  --pre-ms 10 \
  --min-gap-ms 120 \
  --print-stats
```

4) Import the generated `.txt` into Audacity to visually spot-check and correct rare misses/merges.

## Notes

- Stereo input is handled automatically (the script converts to mono).
- If `ffmpeg` is installed, it’s used for robust decoding; otherwise the script falls back to Python’s built-in `.wav` reader.

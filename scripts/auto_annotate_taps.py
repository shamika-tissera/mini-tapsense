#!/usr/bin/env python3
"""
Generate Audacity-style label tracks for tap-like acoustic events.

Outputs a tab-delimited file with: start_time_sec <TAB> end_time_sec <TAB> index

Optionally uses ffmpeg
for decoding and mono conversion.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    peak: int
    score: float


def _read_wav_via_wave(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        frames = wf.readframes(frame_count)

    if sample_width == 1:
        data_u8 = np.frombuffer(frames, dtype=np.uint8)
        data = (data_u8.astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data_i16 = np.frombuffer(frames, dtype=np.int16)
        data = data_i16.astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8)
        triplets = raw.reshape(-1, 3)
        ints = (
            triplets[:, 0].astype(np.int32)
            | (triplets[:, 1].astype(np.int32) << 8)
            | (triplets[:, 2].astype(np.int32) << 16)
        )
        ints = (ints ^ 0x800000) - 0x800000
        data = ints.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data_i32 = np.frombuffer(frames, dtype=np.int32)
        data = data_i32.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels <= 0:
        raise ValueError(f"Invalid WAV channel count: {channels}")
    if channels == 1:
        mono = data
    else:
        interleaved = data.reshape(-1, channels)
        mono = interleaved.mean(axis=1)
    return mono, sample_rate


def decode_audio_mono(
    path: str,
    *,
    target_sr: int | None,
    use_ffmpeg: bool,
) -> tuple[np.ndarray, int]:
    """
    Returns (mono_signal_float32, sample_rate).
    If ffmpeg is available/selected, we decode via ffmpeg -> PCM16 WAV in memory.
    """
    if use_ffmpeg and shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-vn",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
        ]
        if target_sr:
            cmd += ["-ar", str(target_sr)]
        cmd += ["-f", "wav", "pipe:1"]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg decode failed:\n"
                + proc.stderr.decode("utf-8", errors="replace").strip()
            )
        return _read_wav_via_wave(proc.stdout)

    with open(path, "rb") as f:
        wav_bytes = f.read()
    return _read_wav_via_wave(wav_bytes)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same")


def robust_threshold(x: np.ndarray, z: float) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    sigma = 1.4826 * mad
    return med + z * sigma


def local_maxima(x: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return np.array([], dtype=np.int64)
    left = x[1:-1] > x[:-2]
    right = x[1:-1] >= x[2:]
    return np.nonzero(left & right)[0] + 1


def pick_peaks(
    score: np.ndarray,
    *,
    min_gap: int,
    expected_count: int | None,
    threshold: float,
) -> np.ndarray:
    """
    Returns peak indices (in score domain), sorted ascending.
    """
    if expected_count is not None:
        candidates = local_maxima(score)
        if candidates.size == 0:
            return np.array([], dtype=np.int64)
        cand_scores = score[candidates]
        order = np.argsort(-cand_scores)
        selected: list[int] = []
        for idx in order:
            p = int(candidates[idx])
            if all(abs(p - s) >= min_gap for s in selected):
                selected.append(p)
                if len(selected) >= expected_count:
                    break
        return np.array(sorted(selected), dtype=np.int64)

    above = np.nonzero(score >= threshold)[0]
    if above.size == 0:
        return np.array([], dtype=np.int64)

    # Group contiguous above-threshold runs and pick the max in each run.
    splits = np.nonzero(np.diff(above) > 1)[0] + 1
    groups = np.split(above, splits)
    peaks: list[int] = []
    for g in groups:
        if g.size == 0:
            continue
        p = int(g[np.argmax(score[g])])
        peaks.append(p)

    # Enforce min_gap by merging too-close peaks, keeping the strongest.
    peaks.sort()
    filtered: list[int] = []
    for p in peaks:
        if not filtered:
            filtered.append(p)
            continue
        if p - filtered[-1] >= min_gap:
            filtered.append(p)
            continue
        if score[p] > score[filtered[-1]]:
            filtered[-1] = p
    return np.array(filtered, dtype=np.int64)


def segments_from_peaks(
    x: np.ndarray,
    sr: int,
    peaks: np.ndarray,
    *,
    lag: int,
    onset_search: int,
    onset_frac: float,
    pre: int,
    post: int,
    env_win: int,
    boundary_frac: float,
    min_dur: int,
    max_dur: int,
    fixed_dur: int | None,
    scores: np.ndarray,
    threshold: float,
) -> list[Segment]:
    env = moving_average(np.abs(x), env_win)
    baseline = float(np.percentile(env, 10))

    segments: list[Segment] = []
    n = x.size
    for p_score in peaks:
        peak = int(p_score + lag)
        if peak <= 0 or peak >= n or p_score <= 0 or p_score >= scores.size:
            continue

        # Refine to an onset (earliest threshold/frac crossing before the peak).
        level = max(float(threshold), float(scores[p_score]) * onset_frac)
        search_start = max(1, int(p_score) - onset_search)
        onset_score = int(p_score)
        for i in range(search_start, int(p_score) + 1):
            if scores[i] >= level and scores[i - 1] < level:
                onset_score = i
                break
        onset = int(onset_score + lag)
        onset = min(max(onset, 0), n - 1)
        start_anchor = max(0, onset - pre)

        if fixed_dur is not None:
            start = start_anchor
            end = min(n, start + fixed_dur + post)
            segments.append(
                Segment(start=start, end=end, peak=peak, score=float(scores[p_score]))
            )
            continue

        peak_env = float(env[peak])
        level = baseline + boundary_frac * max(0.0, (peak_env - baseline))

        start = start_anchor
        while start > 0 and env[start] > level:
            start -= 1

        end = peak
        while end < (n - 1) and env[end] > level:
            end += 1
        end = min(n, end + post)

        if end - start < min_dur:
            half = min_dur // 2
            start = max(0, peak - half)
            end = min(n, start + min_dur)

        if end - start > max_dur:
            end = start + max_dur

        segments.append(
            Segment(start=start, end=end, peak=peak, score=float(scores[p_score]))
        )

    segments.sort(key=lambda s: s.start)
    return segments


def annotate_taps(
    x: np.ndarray,
    sr: int,
    *,
    lag: int,
    smooth_win: int,
    env_win: int,
    onset_search: int,
    onset_frac: float,
    pre: int,
    post: int,
    threshold_z: float,
    min_gap: int,
    boundary_frac: float,
    min_dur: int,
    max_dur: int,
    fixed_dur: int | None,
    expected_count: int | None,
) -> tuple[list[Segment], dict]:
    if x.size < max(4, lag + 2):
        return [], {"reason": "audio too short"}

    # High-pass-ish transient score: lagged absolute difference, then smooth.
    d = x[lag:] - x[:-lag]
    score = moving_average(np.abs(d), smooth_win)

    thr = robust_threshold(score, threshold_z)
    peaks = pick_peaks(score, min_gap=min_gap, expected_count=expected_count, threshold=thr)

    segments = segments_from_peaks(
        x,
        sr,
        peaks,
        lag=lag,
        onset_search=onset_search,
        onset_frac=onset_frac,
        pre=pre,
        post=post,
        env_win=env_win,
        boundary_frac=boundary_frac,
        min_dur=min_dur,
        max_dur=max_dur,
        fixed_dur=fixed_dur,
        scores=score,
        threshold=thr,
    )

    stats = {
        "sr": sr,
        "samples": int(x.size),
        "seconds": float(x.size) / float(sr),
        "lag": lag,
        "smooth_win": smooth_win,
        "env_win": env_win,
        "threshold_z": threshold_z,
        "threshold": float(thr),
        "min_gap": min_gap,
        "expected_count": expected_count,
        "segments": len(segments),
    }
    return segments, stats


def write_labels(path: str, segments: list[Segment], sr: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start_s = seg.start / sr
            end_s = seg.end / sr
            f.write(f"{start_s:.6f}\t{end_s:.6f}\t{idx}\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-annotate tap events and emit Audacity label files.",
    )
    p.add_argument("inputs", nargs="+", help="Input .wav (or any audio if ffmpeg is enabled).")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for .txt labels (default: alongside inputs).",
    )
    p.add_argument(
        "--suffix",
        default=".txt",
        help="Output label suffix (default: .txt).",
    )
    p.add_argument(
        "--target-sr",
        type=int,
        default=None,
        help="If set, resample to this rate during decode (ffmpeg mode).",
    )
    p.add_argument(
        "--ffmpeg",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use ffmpeg for decoding/mono conversion (default: auto).",
    )
    p.add_argument(
        "--lag",
        type=int,
        default=10,
        help="Lag in samples for the transient score (default: 10).",
    )
    p.add_argument(
        "--smooth-ms",
        type=float,
        default=2.0,
        help="Smoothing window (ms) for transient score (default: 2.0).",
    )
    p.add_argument(
        "--env-ms",
        type=float,
        default=5.0,
        help="Envelope smoothing window (ms) for boundaries (default: 5.0).",
    )
    p.add_argument(
        "--onset-search-ms",
        type=float,
        default=30.0,
        help="Search window (ms) before each peak to estimate onset (default: 30).",
    )
    p.add_argument(
        "--onset-frac",
        type=float,
        default=0.25,
        help="Onset threshold as a fraction of peak score (default: 0.25).",
    )
    p.add_argument(
        "--pre-ms",
        type=float,
        default=10.0,
        help="Include this many ms before the detected onset (default: 10).",
    )
    p.add_argument(
        "--post-ms",
        type=float,
        default=0.0,
        help="Extend segment end by this many ms (default: 0).",
    )
    p.add_argument(
        "--threshold-z",
        type=float,
        default=6.0,
        help="Median+z*MAD threshold when --expected-count not set (default: 6.0).",
    )
    p.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="If set, pick exactly N strongest peaks separated by --min-gap-ms.",
    )
    p.add_argument(
        "--min-gap-ms",
        type=float,
        default=120.0,
        help="Minimum time between taps (ms) (default: 120).",
    )
    p.add_argument(
        "--boundary-frac",
        type=float,
        default=0.10,
        help="Boundary level as fraction of (peak_env-baseline) (default: 0.10).",
    )
    p.add_argument(
        "--min-dur-ms",
        type=float,
        default=20.0,
        help="Minimum labeled segment duration (ms) (default: 20).",
    )
    p.add_argument(
        "--max-dur-ms",
        type=float,
        default=150.0,
        help="Maximum labeled segment duration (ms) (default: 150).",
    )
    p.add_argument(
        "--fixed-dur-ms",
        type=float,
        default=None,
        help="If set, label each tap with a fixed duration (ms) starting at the detected peak.",
    )
    p.add_argument(
        "--print-stats",
        action="store_true",
        help="Print detection stats to stdout.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    use_ffmpeg = args.ffmpeg == "on" or (args.ffmpeg == "auto" and shutil.which("ffmpeg"))

    exit_code = 0
    for inp in args.inputs:
        x, sr = decode_audio_mono(inp, target_sr=args.target_sr, use_ffmpeg=use_ffmpeg)

        smooth_win = max(1, int(round((args.smooth_ms / 1000.0) * sr)))
        env_win = max(1, int(round((args.env_ms / 1000.0) * sr)))
        onset_search = max(1, int(round((args.onset_search_ms / 1000.0) * sr)))
        pre = max(0, int(round((args.pre_ms / 1000.0) * sr)))
        post = max(0, int(round((args.post_ms / 1000.0) * sr)))
        min_gap = max(1, int(round((args.min_gap_ms / 1000.0) * sr)))
        min_dur = max(1, int(round((args.min_dur_ms / 1000.0) * sr)))
        max_dur = max(min_dur, int(round((args.max_dur_ms / 1000.0) * sr)))
        fixed_dur = (
            None
            if args.fixed_dur_ms is None
            else max(1, int(round((args.fixed_dur_ms / 1000.0) * sr)))
        )

        segments, stats = annotate_taps(
            x,
            sr,
            lag=args.lag,
            smooth_win=smooth_win,
            env_win=env_win,
            onset_search=onset_search,
            onset_frac=args.onset_frac,
            pre=pre,
            post=post,
            threshold_z=args.threshold_z,
            min_gap=min_gap,
            boundary_frac=args.boundary_frac,
            min_dur=min_dur,
            max_dur=max_dur,
            fixed_dur=fixed_dur,
            expected_count=args.expected_count,
        )

        base = os.path.splitext(os.path.basename(inp))[0]
        out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(inp))
        out_path = os.path.join(out_dir, base + args.suffix)
        write_labels(out_path, segments, sr)

        if args.print_stats:
            print(f"[{os.path.basename(inp)}] wrote {len(segments)} segments -> {out_path}")
            print(f"  sr={stats.get('sr')} sec={stats.get('seconds'):.3f} thr={stats.get('threshold'):.6g}")
            if args.expected_count is not None and len(segments) != args.expected_count:
                print(
                    f"  WARNING: expected {args.expected_count} but found {len(segments)}",
                    file=sys.stderr,
                )
                exit_code = 2

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

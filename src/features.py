"""Per-clip feature extraction for the CCL vs healthy classifier.

Reads the lateral-clip list from dataset.py, runs the SuperAnimal-Quadruped
pose pipeline frame-by-frame over each clip's gait window, and reduces the
resulting time series of joint angles + paw positions into a fixed-length
feature vector. The result is written to results/features.csv and is the
input to train.py / evaluate.py.

    * For CCL detection we need (a) angle-magnitude signals around the rear
      stifle and hip (the affected joints) and (b) symmetry signals between
      left and right sides (the primary lameness signature).
    * Per-angle descriptive stats (mean / std / min / max / range) let us capture range of motion
    * |L-R| asymmetry stats encode lameness independant of which leg is closer to camera
    * Stride frequency from the back-paw y-trajectory captures cadence
      irregularity (lame dogs often have shorter stance on bad leg).
    * Stance variance ratio L/R is a proxy for limping — lame dogs put less
      weight on the bad leg

Run:
    python src/features.py                  # full extraction (mega slow on CPU)
    python src/features.py --sample 1       # 1 clip per dog, for testing
    python src/features.py --resume         # skip clips already in features.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

# Local modules
from dataset import Clip, build_clip_index, RESULTS_DIR
from animal_openpose import (
    extract_pose,
    KEYPOINT_NAMES,
    NAME_TO_INDEX,
    ANGLE_TRIPLETS,
    DEFAULT_SCORE_THRESHOLD,
)
FEATURES_CSV = RESULTS_DIR / "features.csv"

ANGLE_NAMES = list(ANGLE_TRIPLETS.keys())  # 10 entries
ANGLE_STAT_NAMES = ["mean", "std", "min", "max", "range"]

ASYM_PAIRS = [
    ("stifle",   "L_stifle",   "R_stifle"),    # rear knees = primary CCL signal?
    ("hip",      "L_hip",      "R_hip"),
    ("elbow",    "L_elbow",    "R_elbow"),
    ("shoulder", "L_shoulder", "R_shoulder"),
]


def feature_columns() -> list[str]:
    """Stable list of feature column names for the output CSV."""
    cols: list[str] = []
    for ang in ANGLE_NAMES:
        for stat in ANGLE_STAT_NAMES:
            cols.append(f"{ang}_{stat}")
    for pair_name, _, _ in ASYM_PAIRS:
        cols.append(f"asym_{pair_name}_mean")
        cols.append(f"asym_{pair_name}_std")
    cols.append("stride_freq_L_back_hz")
    cols.append("stride_freq_R_back_hz")
    cols.append("stance_var_ratio_back")
    cols.append("stance_var_ratio_front")
    cols.append("duration_s")
    cols.append("pose_coverage")        # fraction of frames with >=1 angle defined
    cols.append("direction_LR")         # 1 if LR (right side visible), else 0. maybe does not need to be included here.
    return cols


META_COLUMNS = ["video_path", "dog_id", "dog_name", "label"]
ALL_COLUMNS = META_COLUMNS + feature_columns()

def _iter_clip_frames(clip: Clip):
    """Yield (frame_index, BGR frame) for every frame in [start, end] inclusive.

    Uses sequential reads (not seeking) within the window — much faster than
    seeking to each frame individually.
    """
    cap = cv2.VideoCapture(clip.video_path)
    if not cap.isOpened():
        return
    # Seek to the window start once, then read sequentially.
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)
    for f_idx in range(clip.start_frame, clip.end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        yield f_idx, frame
    cap.release()


def _extract_clip_timeseries(clip: Clip, verbose: bool = True) -> dict[str, np.ndarray]:
    """Run pose on every frame in the gait window. Returns time-series arrays.

    Returned arrays are all length = window_length. Missing values (low-conf
    keypoints, undetected dog) are NaN, which downstream stats handle with
    np.nan* aggregations.

    `verbose=True` prints a progress line every PROGRESS_EVERY frames so a
    long pose run isn't silent.
    """
    win = clip.end_frame - clip.start_frame + 1

    # Time series per angle.
    angles_ts = {a: np.full(win, np.nan, dtype=np.float64) for a in ANGLE_NAMES}
    paw_keys = ["front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw"]
    paws_x = {p: np.full(win, np.nan, dtype=np.float64) for p in paw_keys}
    paws_y = {p: np.full(win, np.nan, dtype=np.float64) for p in paw_keys}
    # Dog-detected per frame (any instances). useful for quantifing completeness
    has_dog = np.zeros(win, dtype=bool)

    PROGRESS_EVERY = 20
    t_start = time.time()
    n_done = 0
    n_with_dog = 0

    for f_idx, frame in _iter_clip_frames(clip):
        t = f_idx - clip.start_frame
        if t < 0 or t >= win:
            continue
        info = extract_pose(frame)
        n_done += 1
        if verbose and (n_done % PROGRESS_EVERY == 0 or n_done == win):
            elapsed = time.time() - t_start
            avg = elapsed / max(n_done, 1)
            remaining = avg * (win - n_done)
            print(f"      pose {n_done:3d}/{win} frames "
                  f"({avg*1000:.0f} ms/frame, "
                  f"~{remaining:.0f}s remaining, "
                  f"{n_with_dog} dogs detected so far)",
                  flush=True)
        if not info["instances"]:
            continue
        n_with_dog += 1
        # Use the first (and hopefully only) detected animal. If there are several,
        # pick the largest bbox which is what we also did in detector.py
        instances = info["instances"]
        if len(instances) > 1:
            areas = [inst["bbox_xywh"][2] * inst["bbox_xywh"][3] for inst in instances]
            inst = instances[int(np.argmax(areas))]
        else:
            inst = instances[0]
        has_dog[t] = True

        # Angle values (None when contributing keypoints fell below threshold).
        for ang_name, val in inst["angles_deg"].items():
            if ang_name in angles_ts and val is not None:
                angles_ts[ang_name][t] = float(val)

        for kp in inst["keypoints"]:
            name = kp["name"]
            if name in paws_x and kp["score"] >= DEFAULT_SCORE_THRESHOLD:
                paws_x[name][t] = float(kp["x"])
                paws_y[name][t] = float(kp["y"])

    return {
        "angles": angles_ts,
        "paws_x": paws_x,
        "paws_y": paws_y,
        "has_dog": has_dog,
    }


def _angle_stats(values: np.ndarray) -> dict[str, float]:
    """Mean/std/min/max/range, ignoring NaNs. NaN if no data."""
    if np.all(np.isnan(values)):
        return {s: np.nan for s in ANGLE_STAT_NAMES}
    mn = float(np.nanmin(values))
    mx = float(np.nanmax(values))
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": mn,
        "max": mx,
        "range": mx - mn,
    }


def _asymmetry_stats(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Mean and std of |left - right| over frames where both are defined."""
    diff = np.abs(left - right)
    if np.all(np.isnan(diff)):
        return (np.nan, np.nan)
    return (float(np.nanmean(diff)), float(np.nanstd(diff)))


def _dominant_frequency(signal: np.ndarray, fps: float) -> float:
    """Return the dominant non-DC frequency in Hz of a 1D signal.

    Drops NaNs by linear-interpolating across them so the FFT sees a clean,
    evenly-sampled series. Returns NaN if the signal is too short or too sparse
    to have a meaningful spectrum.
    """
    if signal.size < 8 or fps <= 0:
        return np.nan
    s = signal.astype(np.float64).copy()
    valid = ~np.isnan(s)
    if valid.sum() < max(8, signal.size * 0.5):
        return np.nan
    idx = np.arange(s.size)
    s[~valid] = np.interp(idx[~valid], idx[valid], s[valid])
    s = s - np.mean(s)
    if s.size >= 4:
        slope = np.polyfit(idx, s, 1)
        s = s - (slope[0] * idx + slope[1])
    spec = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(s.size, d=1.0 / fps)
    # nothing meaningful above ~6 Hz.
    if spec.size <= 1:
        return np.nan
    band = (freqs > 0.3) & (freqs <= 6.0)
    if not band.any():
        return np.nan
    band_spec = spec.copy()
    band_spec[~band] = 0.0
    peak = int(np.argmax(band_spec))
    return float(freqs[peak])


def _variance_ratio(left_signal: np.ndarray, right_signal: np.ndarray) -> float:
    """var(L) / var(R), NaN-safe. Returns 1.0 if both sides are equally noisy
    (no asymmetry); >1 if L is more variable, <1 if R is more variable.
    """
    if np.all(np.isnan(left_signal)) or np.all(np.isnan(right_signal)):
        return np.nan
    vl = float(np.nanvar(left_signal))
    vr = float(np.nanvar(right_signal))
    if vr <= 1e-9 and vl <= 1e-9:
        return 1.0
    if vr <= 1e-9:
        return float("inf")  # one side static, other moving
    return vl / vr


def _aggregate_features(clip: Clip, ts: dict[str, Any]) -> dict[str, float]:
    """Reduce a clip's time series into a 65-element feature dict."""
    feats: dict[str, float] = {}

    for ang in ANGLE_NAMES:
        stats = _angle_stats(ts["angles"][ang])
        for stat_name, val in stats.items():
            feats[f"{ang}_{stat_name}"] = val

    for pair_name, l_ang, r_ang in ASYM_PAIRS:
        m, s = _asymmetry_stats(ts["angles"][l_ang], ts["angles"][r_ang])
        feats[f"asym_{pair_name}_mean"] = m
        feats[f"asym_{pair_name}_std"] = s

    feats["stride_freq_L_back_hz"] = _dominant_frequency(
        ts["paws_y"]["back_left_paw"], clip.fps
    )
    feats["stride_freq_R_back_hz"] = _dominant_frequency(
        ts["paws_y"]["back_right_paw"], clip.fps
    )

    feats["stance_var_ratio_back"] = _variance_ratio(
        ts["paws_y"]["back_left_paw"], ts["paws_y"]["back_right_paw"]
    )
    feats["stance_var_ratio_front"] = _variance_ratio(
        ts["paws_y"]["front_left_paw"], ts["paws_y"]["front_right_paw"]
    )

    win = clip.gait_length
    feats["duration_s"] = win / max(clip.fps, 1.0)
    feats["pose_coverage"] = float(np.mean(ts["has_dog"])) if win > 0 else 0.0
    feats["direction_LR"] = 1.0 if clip.direction == "LR" else 0.0

    return feats

def extract_clip_row(clip: Clip) -> dict[str, Any]:
    """Run pose on a single clip and return one CSV row's worth of data."""
    label_str = "CCL" if clip.label == 1 else "Norm"
    print(f"    > {label_str:4s}  {clip.dog_id}  ({clip.direction})  "
          f"{Path(clip.video_path).name}  win={clip.gait_length} frames "
          f"@ {clip.fps:.1f}fps  ({clip.gait_length / max(clip.fps, 1):.1f}s)",
          flush=True)
    t0 = time.time()
    ts = _extract_clip_timeseries(clip)
    elapsed = time.time() - t0
    coverage = float(np.mean(ts["has_dog"]))

    feats = _aggregate_features(clip, ts)
    row: dict[str, Any] = {
        "video_path": clip.video_path,
        "dog_id": clip.dog_id,
        "dog_name": clip.dog_name,
        "label": clip.label,
        **feats,
    }
    # Show two of the diagnostic features so the user can sanity-check the run
    # without opening csv
    stif_l = feats.get("L_stifle_mean", float("nan"))
    stif_r = feats.get("R_stifle_mean", float("nan"))
    asym_stifle = feats.get("asym_stifle_mean", float("nan"))
    print(f"    < done in {elapsed:.1f}s  pose_coverage={coverage:.0%}  "
          f"L_stifle_mean={stif_l:.1f}°  R_stifle_mean={stif_r:.1f}°  "
          f"|L-R|_stifle={asym_stifle:.1f}°",
          flush=True)
    return row


def build_feature_matrix(
    clips: list[Clip],
    resume: bool = False,
) -> pd.DataFrame:
    """Run feature extraction across all clips. Returns a DataFrame and writes
    `results/features.csv` (overwriting unless --resume).

    `resume=True` skips clips whose video_path is already in the existing CSV.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing: pd.DataFrame | None = None
    if resume and FEATURES_CSV.exists():
        existing = pd.read_csv(FEATURES_CSV)
        already = set(existing["video_path"].tolist())
        clips_to_run = [c for c in clips if c.video_path not in already]
        print(f"[resume] {len(already)} clips already extracted; "
              f"{len(clips_to_run)}/{len(clips)} new clips remaining.")
    else:
        clips_to_run = clips

    rows: list[dict[str, Any]] = []
    overall_t0 = time.time()
    for i, clip in enumerate(clips_to_run, start=1):
        # Estimated time remaining based on elapsed average.
        elapsed_total = time.time() - overall_t0
        if i > 1:
            avg_per_clip = elapsed_total / (i - 1)
            eta_sec = avg_per_clip * (len(clips_to_run) - i + 1)
            eta_str = f"  (avg {avg_per_clip:.0f}s/clip, ETA {eta_sec/60:.1f} min)"
        else:
            eta_str = ""
        print(f"\n[clip {i}/{len(clips_to_run)}]{eta_str}", flush=True)
        try:
            rows.append(extract_clip_row(clip))
        except Exception as e:
            print(f"  ! FAILED on {clip.video_path}: {e}", flush=True)
        # Persist progressively so a kill mid run doesnt lose hours of work. Learned the hard way.
        if rows and (i % 5 == 0 or i == len(clips_to_run)):
            partial = pd.DataFrame(rows, columns=ALL_COLUMNS)
            if existing is not None:
                partial = pd.concat([existing, partial], ignore_index=True)
            partial.to_csv(FEATURES_CSV, index=False)
            print(f"  [checkpoint] features.csv now has {len(partial)} rows.",
                  flush=True)

    new_df = pd.DataFrame(rows, columns=ALL_COLUMNS) if rows else pd.DataFrame(columns=ALL_COLUMNS)
    if existing is not None and not new_df.empty:
        df = pd.concat([existing, new_df], ignore_index=True)
    elif existing is not None:
        df = existing
    else:
        df = new_df

    # Persist incrementally too: write at end. (The view-index cache in
    # dataset.py already protects pose-detection work; if extraction is
    # interrupted, --resume lets us pick up where we left off.)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"\nwrote {FEATURES_CSV}  ({len(df)} rows total)")
    return df


def main():
    p = argparse.ArgumentParser(description="Extract per-clip features for the CCL classifier.")
    p.add_argument("--sample", type=int, default=None,
                   help="Cap to N clips per dog in the index (smoke test).")
    p.add_argument("--limit-per-dog", type=int, default=None,
                   help="Only motion-analyze at most N videos per dog "
                        "(smoke test — keeps the dataset scan fast).")
    p.add_argument("--refresh-views", action="store_true",
                   help="Re-run motion analysis (ignore view_index.json cache).")
    p.add_argument("--resume", action="store_true",
                   help="Skip clips already present in features.csv.")
    args = p.parse_args()

    print("=== Building lateral-clip index ===")
    clips = build_clip_index(
        sample_per_dog=args.sample,
        limit_per_dog=args.limit_per_dog,
        refresh=args.refresh_views,
    )
    print(f"{len(clips)} usable clips")
    print()
    print("=== Extracting features ===")
    df = build_feature_matrix(clips, resume=args.resume)
    print()
    print("Label distribution:")
    print(df.groupby("label")["dog_id"].nunique().rename("dogs").to_string())
    print()
    print("Per-dog clip counts:")
    print(df.groupby(["dog_id", "label"]).size().rename("clips").to_string())


if __name__ == "__main__":
    main()

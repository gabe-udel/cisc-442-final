# features.py - runs pose on every clip and reduces the results to a flat feature table

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dataset import Clip, build_clip_index, RESULTS_DIR
from animal_openpose import (
    extract_pose,
    KEYPOINT_NAMES,
    NAME_TO_INDEX,
    ANGLE_TRIPLETS,
    DEFAULT_SCORE_THRESHOLD,
)


# where the output csv goes
FEATURES_CSV = RESULTS_DIR / "features.csv"

# all 10 angle names from the pose model
ANGLE_NAMES = list(ANGLE_TRIPLETS.keys())

# stats computed per angle over the clip
ANGLE_STAT_NAMES = ["mean", "std", "min", "max", "range"]

# left/right pairs used to measure asymmetry
ASYM_PAIRS = [
    ("stifle",   "L_stifle",   "R_stifle"),
    ("hip",      "L_hip",      "R_hip"),
    ("elbow",    "L_elbow",    "R_elbow"),
    ("shoulder", "L_shoulder", "R_shoulder"),
]


# return the stable list of 65 feature column names
def feature_columns():
    cols = []
    # 50 angle stats (10 angles x 5 stats each)
    for ang in ANGLE_NAMES:
        for stat in ANGLE_STAT_NAMES:
            cols.append(f"{ang}_{stat}")
    # 8 left/right asymmetry stats (4 pairs x 2 stats each)
    for pair_name, _, _ in ASYM_PAIRS:
        cols.append(f"asym_{pair_name}_mean")
        cols.append(f"asym_{pair_name}_std")
    # 2 stride frequency features
    cols.append("stride_freq_L_back_hz")
    cols.append("stride_freq_R_back_hz")
    # 2 stance asymmetry features
    cols.append("stance_var_ratio_back")
    cols.append("stance_var_ratio_front")
    # 3 metadata features
    cols.append("duration_s")
    cols.append("pose_coverage")
    cols.append("direction_LR")
    return cols


META_COLUMNS = ["video_path", "dog_id", "dog_name", "label"]
ALL_COLUMNS = META_COLUMNS + feature_columns()


# yield each frame in the clip's gait window
def _iter_clip_frames(clip):
    cap = cv2.VideoCapture(clip.video_path)
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)
    for f_idx in range(clip.start_frame, clip.end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        yield f_idx, frame
    cap.release()


# run pose on every frame and collect time-series arrays
def _extract_clip_timeseries(clip, verbose=True):
    win = clip.end_frame - clip.start_frame + 1

    # one array per angle, filled with nan where pose was missing
    angles_ts = {}
    for a in ANGLE_NAMES:
        angles_ts[a] = np.full(win, np.nan, dtype=np.float64)

    # paw position arrays for stride and stance analysis
    paw_keys = ["front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw"]
    paws_x = {}
    paws_y = {}
    for p in paw_keys:
        paws_x[p] = np.full(win, np.nan, dtype=np.float64)
        paws_y[p] = np.full(win, np.nan, dtype=np.float64)

    # track which frames had a detected dog
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

        # if multiple animals detected, use the largest bounding box
        instances = info["instances"]
        if len(instances) > 1:
            areas = []
            for inst in instances:
                area = inst["bbox_xywh"][2] * inst["bbox_xywh"][3]
                areas.append(area)
            inst = instances[int(np.argmax(areas))]
        else:
            inst = instances[0]
        has_dog[t] = True

        # record angle values (None means the keypoints were low confidence)
        for ang_name, val in inst["angles_deg"].items():
            if ang_name in angles_ts and val is not None:
                angles_ts[ang_name][t] = float(val)

        # record paw positions for high-confidence keypoints only
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


# compute mean/std/min/max/range for one angle's time series
def _angle_stats(values):
    if np.all(np.isnan(values)):
        result = {}
        for s in ANGLE_STAT_NAMES:
            result[s] = np.nan
        return result
    mn = float(np.nanmin(values))
    mx = float(np.nanmax(values))
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": mn,
        "max": mx,
        "range": mx - mn,
    }


# compute mean and std of |left - right| over frames where both are valid
def _asymmetry_stats(left, right):
    diff = np.abs(left - right)
    if np.all(np.isnan(diff)):
        return (np.nan, np.nan)
    return (float(np.nanmean(diff)), float(np.nanstd(diff)))


# return the dominant non-dc frequency in hz from a 1d signal
def _dominant_frequency(signal, fps):
    if signal.size < 8 or fps <= 0:
        return np.nan
    s = signal.astype(np.float64).copy()
    valid = ~np.isnan(s)
    if valid.sum() < max(8, signal.size * 0.5):
        return np.nan

    # fill nan gaps with linear interpolation so fft gets a clean signal
    idx = np.arange(s.size)
    s[~valid] = np.interp(idx[~valid], idx[valid], s[valid])

    # remove mean and linear trend so dc and slow drift don't dominate
    s = s - np.mean(s)
    if s.size >= 4:
        slope = np.polyfit(idx, s, 1)
        s = s - (slope[0] * idx + slope[1])

    spec = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(s.size, d=1.0 / fps)

    if spec.size <= 1:
        return np.nan

    # look only in the biological range (dogs walk/trot at 1-5 hz)
    band = (freqs > 0.3) & (freqs <= 6.0)
    if not band.any():
        return np.nan

    band_spec = spec.copy()
    band_spec[~band] = 0.0
    peak = int(np.argmax(band_spec))
    return float(freqs[peak])


# compute var(left) / var(right) as a proxy for stance asymmetry
def _variance_ratio(left_signal, right_signal):
    if np.all(np.isnan(left_signal)) or np.all(np.isnan(right_signal)):
        return np.nan
    vl = float(np.nanvar(left_signal))
    vr = float(np.nanvar(right_signal))
    if vr <= 1e-9 and vl <= 1e-9:
        return 1.0
    if vr <= 1e-9:
        return float("inf")
    return vl / vr


# reduce a clip's time series into a flat dict of 65 features
def _aggregate_features(clip, ts):
    feats = {}

    # 50 per-angle stats
    for ang in ANGLE_NAMES:
        stats = _angle_stats(ts["angles"][ang])
        for stat_name, val in stats.items():
            feats[f"{ang}_{stat_name}"] = val

    # 8 left/right asymmetry stats
    for pair_name, l_ang, r_ang in ASYM_PAIRS:
        m, s = _asymmetry_stats(ts["angles"][l_ang], ts["angles"][r_ang])
        feats[f"asym_{pair_name}_mean"] = m
        feats[f"asym_{pair_name}_std"] = s

    # 2 stride frequency features
    feats["stride_freq_L_back_hz"] = _dominant_frequency(ts["paws_y"]["back_left_paw"], clip.fps)
    feats["stride_freq_R_back_hz"] = _dominant_frequency(ts["paws_y"]["back_right_paw"], clip.fps)

    # 2 stance asymmetry features
    feats["stance_var_ratio_back"] = _variance_ratio(ts["paws_y"]["back_left_paw"], ts["paws_y"]["back_right_paw"])
    feats["stance_var_ratio_front"] = _variance_ratio(ts["paws_y"]["front_left_paw"], ts["paws_y"]["front_right_paw"])

    # 3 metadata features
    win = clip.gait_length
    feats["duration_s"] = win / max(clip.fps, 1.0)
    feats["pose_coverage"] = float(np.mean(ts["has_dog"])) if win > 0 else 0.0
    feats["direction_LR"] = 1.0 if clip.direction == "LR" else 0.0

    return feats


# run pose on one clip and return a dict ready to write as a csv row
def extract_clip_row(clip):
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
    row = {
        "video_path": clip.video_path,
        "dog_id": clip.dog_id,
        "dog_name": clip.dog_name,
        "label": clip.label,
    }
    row.update(feats)

    # show a quick sanity check of the key diagnostic features
    stif_l = feats.get("L_stifle_mean", float("nan"))
    stif_r = feats.get("R_stifle_mean", float("nan"))
    asym_stifle = feats.get("asym_stifle_mean", float("nan"))
    print(f"    < done in {elapsed:.1f}s  pose_coverage={coverage:.0%}  "
          f"L_stifle_mean={stif_l:.1f}°  R_stifle_mean={stif_r:.1f}°  "
          f"|L-R|_stifle={asym_stifle:.1f}°",
          flush=True)
    return row


# run feature extraction for all clips and write results/features.csv
def build_feature_matrix(clips, resume=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing = None
    if resume and FEATURES_CSV.exists():
        existing = pd.read_csv(FEATURES_CSV)
        already = set(existing["video_path"].tolist())
        clips_to_run = []
        for c in clips:
            if c.video_path not in already:
                clips_to_run.append(c)
        print(f"[resume] {len(already)} clips already extracted; "
              f"{len(clips_to_run)}/{len(clips)} new clips remaining.")
    else:
        clips_to_run = clips

    rows = []
    overall_t0 = time.time()
    for i, clip in enumerate(clips_to_run, start=1):
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

        # save to csv every 5 clips so we don't lose work if interrupted
        if rows and (i % 5 == 0 or i == len(clips_to_run)):
            partial = pd.DataFrame(rows, columns=ALL_COLUMNS)
            if existing is not None:
                partial = pd.concat([existing, partial], ignore_index=True)
            partial.to_csv(FEATURES_CSV, index=False)
            print(f"  [checkpoint] features.csv now has {len(partial)} rows.", flush=True)

    if rows:
        new_df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    else:
        new_df = pd.DataFrame(columns=ALL_COLUMNS)

    if existing is not None and not new_df.empty:
        df = pd.concat([existing, new_df], ignore_index=True)
    elif existing is not None:
        df = existing
    else:
        df = new_df

    df.to_csv(FEATURES_CSV, index=False)
    print(f"\nwrote {FEATURES_CSV}  ({len(df)} rows total)")
    return df


def main():
    p = argparse.ArgumentParser(description="Extract per-clip features for the CCL classifier.")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--limit-per-dog", type=int, default=None)
    p.add_argument("--refresh-views", action="store_true")
    p.add_argument("--resume", action="store_true")
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
    label_groups = df.groupby("label")["dog_id"]
    dog_counts = label_groups.nunique()
    dog_counts = dog_counts.rename("dogs")
    print(dog_counts.to_string())
    print()
    print("Per-dog clip counts:")
    dog_label_groups = df.groupby(["dog_id", "label"])
    clip_counts = dog_label_groups.size()
    clip_counts = clip_counts.rename("clips")
    print(clip_counts.to_string())


if __name__ == "__main__":
    main()

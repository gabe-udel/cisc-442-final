"""Run the full pose + CCL-classifier pipeline on a single video.

Workflow:
    1. Motion-analyze the video (lateral check + gait window + direction).
    2. Pose every frame in the gait window and aggregate to a feature vector.
       Also writes a pose-overlaid copy of the gait window to results/.
    3. Load results/model.joblib (saved by train.py) and predict CCL vs Normal.
    4. Print the verdict and probability to the console.

Usage:
    python src/predict.py path/to/video.mov
    python src/predict.py "videos/cisc442 dog videos/CCL Cases/.../IMG_1234.MOV"
    python src/predict.py --force path/to/non_lateral.mov
    python src/predict.py --no-overlay path/to/video.mov   # skip overlay write
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from animal_openpose import DEFAULT_SCORE_THRESHOLD, extract_pose, get_pose
from dataset import Clip, RESULTS_DIR, _analyze_motion
from features import (
    ANGLE_NAMES,
    _aggregate_features,
    extract_clip_row,
    feature_columns,
)
from train import MODEL_PATH
from video_io import VideoWriter


def _read_video_metadata(video_path: Path) -> dict:
    """Pull n_frames / fps / size straight from the container — used as a
    fallback when motion analysis bails before populating those fields."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {"n_frames": n, "fps": fps}


def _pose_with_overlay(clip: Clip, overlay_path: Path) -> dict:
    """Run pose on every gait-window frame, write an overlaid mp4, and return
    the time-series dict in the same shape `features._extract_clip_timeseries`
    produces — so we can hand it straight to `_aggregate_features`.

    Single-pass: pose inference happens once per frame and feeds both the
    skeleton drawing and the feature time series (avoids running pose twice).
    """
    win = clip.end_frame - clip.start_frame + 1
    angles_ts = {a: np.full(win, np.nan, dtype=np.float64) for a in ANGLE_NAMES}
    paw_keys = ["front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw"]
    paws_x = {p: np.full(win, np.nan, dtype=np.float64) for p in paw_keys}
    paws_y = {p: np.full(win, np.nan, dtype=np.float64) for p in paw_keys}
    has_dog = np.zeros(win, dtype=bool)

    cap = cv2.VideoCapture(clip.video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {clip.video_path} for overlay write.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)

    PROGRESS_EVERY = 20
    t_start = time.time()
    n_done = 0
    n_with_dog = 0

    with VideoWriter(overlay_path, fps=clip.fps, size=(width, height)) as writer:
        try:
            for f_idx in range(clip.start_frame, clip.end_frame + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                t = f_idx - clip.start_frame
                info = extract_pose(frame)
                writer.write(get_pose(frame, pose_data=info))
                n_done += 1

                if n_done % PROGRESS_EVERY == 0 or n_done == win:
                    elapsed = time.time() - t_start
                    avg = elapsed / max(n_done, 1)
                    remaining = avg * (win - n_done)
                    print(f"      pose {n_done:3d}/{win} frames "
                          f"({avg*1000:.0f} ms/frame, ~{remaining:.0f}s remaining, "
                          f"{n_with_dog} dogs detected so far)", flush=True)

                if not info["instances"]:
                    continue
                n_with_dog += 1
                instances = info["instances"]
                if len(instances) > 1:
                    areas = [inst["bbox_xywh"][2] * inst["bbox_xywh"][3] for inst in instances]
                    inst = instances[int(np.argmax(areas))]
                else:
                    inst = instances[0]
                has_dog[t] = True
                for ang_name, val in inst["angles_deg"].items():
                    if ang_name in angles_ts and val is not None:
                        angles_ts[ang_name][t] = float(val)
                for kp in inst["keypoints"]:
                    name = kp["name"]
                    if name in paws_x and kp["score"] >= DEFAULT_SCORE_THRESHOLD:
                        paws_x[name][t] = float(kp["x"])
                        paws_y[name][t] = float(kp["y"])
        finally:
            cap.release()

    return {
        "angles": angles_ts,
        "paws_x": paws_x,
        "paws_y": paws_y,
        "has_dog": has_dog,
    }


def predict_video(video_path: Path, force: bool = False, save_overlay: bool = True) -> dict:
    """Run pose + classifier on `video_path` and return the prediction dict.

    `force=True` skips the lateral-gait gate. Motion analysis still runs (we
    use its direction + gait window when available), but a non-lateral verdict
    no longer aborts — instead we fall back to the full video as the gait
    window and default to direction='LR' if the analyzer couldn't decide.

    `save_overlay=True` writes results/<stem>_overlaid.mp4 — the gait-window
    frames with the pose skeleton drawn on top. Pose inference still runs only
    once per frame (the same call feeds both feature extraction and the overlay).
    """
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Trained model not found at {MODEL_PATH}. Run train.py first."
        )

    # --- 1. Motion analysis ------------------------------------------------
    # Gives us direction (LR/RL) and the gait-window frames the model expects.
    print(f"=== [1/4] Motion analysis: {video_path.name} ===")
    info = _analyze_motion(video_path)
    if not info.get("is_lateral"):
        if not force:
            raise SystemExit(
                f"Video is not a usable lateral gait clip "
                f"(reason: {info.get('reason', 'unknown')}). The model was only "
                f"trained on lateral views, so a prediction would be unreliable. "
                f"Pass --force to override."
            )
        # Override path: backfill anything _analyze_motion didn't populate.
        # n_frames/fps may be missing if analysis failed before metadata read.
        if "n_frames" not in info or "fps" not in info:
            info.update(_read_video_metadata(video_path))
        info.setdefault("start_frame", 0)
        info.setdefault("end_frame", max(0, info["n_frames"] - 1))
        info.setdefault("direction", "LR")  # arbitrary default; affects 1 feature
        print(f"  WARNING: lateral check failed ({info.get('reason', '?')}) — "
              f"forcing prediction. Treat the result with skepticism.")
    print(f"  direction:     {info['direction']}")
    print(f"  gait window:   frames {info['start_frame']}-{info['end_frame']} "
          f"of {info['n_frames']}")
    print(f"  fps:           {info['fps']:.1f}")
    print(f"  duration:      {(info['end_frame'] - info['start_frame'] + 1) / info['fps']:.1f}s")

    # --- 2. Feature extraction --------------------------------------------
    # Build a Clip record so we can reuse features.extract_clip_row, which is
    # the exact function used during training — guarantees feature parity.
    # `label` is a placeholder; predict_proba doesn't read it.
    clip = Clip(
        video_path=str(video_path),
        dog_id="<single-clip>",
        dog_name=video_path.stem,
        label=0,
        direction=info["direction"],
        start_frame=int(info["start_frame"]),
        end_frame=int(info["end_frame"]),
        n_frames=int(info["n_frames"]),
        fps=float(info["fps"]),
    )

    print(f"\n=== [2/4] Pose extraction + feature aggregation ===")
    overlay_path: Path | None = None
    if save_overlay:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        overlay_path = RESULTS_DIR / f"{video_path.stem}_overlaid.mp4"
        print(f"  overlay output: {overlay_path}")
        ts = _pose_with_overlay(clip, overlay_path)
        feats = _aggregate_features(clip, ts)
        # Mirror extract_clip_row's output shape so the rest of the function
        # treats `row` identically whether overlay was on or off.
        row = {
            "video_path": clip.video_path,
            "dog_id": clip.dog_id,
            "dog_name": clip.dog_name,
            "label": clip.label,
            **feats,
        }
    else:
        row = extract_clip_row(clip)

    # --- 3. Load model ----------------------------------------------------
    print(f"\n=== [3/4] Loading classifier ===")
    artifact = joblib.load(MODEL_PATH)
    pipe = artifact["pipeline"]
    feat_cols = artifact.get("feature_columns", feature_columns())
    model_name = artifact.get("model_name", "?")
    print(f"  model: {model_name}")

    # --- 4. Predict --------------------------------------------------------
    # Wrap the feature row in a 1-row DataFrame with stable column order so
    # the pipeline's imputer/scaler see exactly what they were fit on.
    X = pd.DataFrame([{c: row.get(c, np.nan) for c in feat_cols}])
    proba_ccl = float(pipe.predict_proba(X)[0, 1])
    pred = int(proba_ccl >= 0.5)
    confidence = max(proba_ccl, 1.0 - proba_ccl)

    coverage = float(row.get("pose_coverage", float("nan")))
    stif_l = float(row.get("L_stifle_mean", float("nan")))
    stif_r = float(row.get("R_stifle_mean", float("nan")))
    asym_stifle = float(row.get("asym_stifle_mean", float("nan")))

    print(f"\n=== [4/4] Result ===")
    print(f"  Video:           {video_path.name}")
    print(f"  Model:           {model_name}")
    print(f"  Pose coverage:   {coverage:.0%}")
    print(f"  L stifle (mean): {stif_l:.1f}°")
    print(f"  R stifle (mean): {stif_r:.1f}°")
    print(f"  |L-R| stifle:    {asym_stifle:.1f}°")
    print(f"  P(CCL):          {proba_ccl:.3f}")
    print(f"  Prediction:      {'CCL (injured)' if pred == 1 else 'Normal (healthy)'}")
    print(f"  Confidence:      {confidence:.1%}")
    if overlay_path is not None:
        print(f"  Overlay video:   {overlay_path}")

    return {
        "video": str(video_path),
        "model": model_name,
        "p_ccl": proba_ccl,
        "prediction": "CCL" if pred == 1 else "Normal",
        "confidence": confidence,
        "pose_coverage": coverage,
        "direction": info["direction"],
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
    }


def main():
    p = argparse.ArgumentParser(
        description="Predict CCL vs Normal for a single dog gait video."
    )
    p.add_argument("video", type=Path,
                   help="Path to the video file (.mov / .mp4).")
    p.add_argument("--force", action="store_true",
                   help="Skip the lateral-gait check and run the model anyway. "
                        "Use this for non-lateral or stationary clips — the "
                        "result will be unreliable but won't abort.")
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip writing the pose-overlaid mp4 to results/. "
                        "Slightly faster; otherwise an overlay video named "
                        "results/<stem>_overlaid.mp4 is saved.")
    args = p.parse_args()
    predict_video(args.video, force=args.force, save_overlay=not args.no_overlay)


if __name__ == "__main__":
    main()

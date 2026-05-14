"""End-to-end driver for the CCL-vs-healthy dog gait pipeline.

Stages (toggle each via the RUN_* flags below):
    1. VISUALIZE       Run pose on a single video and dump overlaid/raw videos
                       plus a per-frame keypoint+angle text log to results/.
    2. BUILD_INDEX     Walk the dataset, motion-analyze each video, and cache
                       the list of usable lateral clips to view_index.json.
    3. EXTRACT_FEATURES  Run pose on every indexed clip's gait window and
                       aggregate per-clip features into results/features.csv.
    4. TRAIN           LOO-CV over three classifiers, save the winner to
                       results/model.joblib.
    5. EVALUATE        Score the saved winner on the two held-out dogs.

To configure a run, edit the CONFIGURATION section below — no CLI args needed.

!!!!!!!!!!!!!
NOTE: this file is specifically for breaking down the dataset, generating the pose inforation, and training the model.
To TEST the model on one video, use predict.py: this will generate pose output videos in results and display the model's prediction vis stdout
!!!!!!!!!!!!!!

"""

from pathlib import Path

import numpy as np

from animal_openpose import extract_pose, get_pose
from video_io import read_video, writer_matching


# CONFIGURATION
# Edit these globals to control what main() does. Each RUN_* flag turns a
# pipeline stage on or off; the per-stage settings below tune the stage.


# Project paths (resolved relative to this file so cwd doesn't matter) 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR   = PROJECT_ROOT / "videos"
RESULTS_DIR  = PROJECT_ROOT / "results"

#Stage toggles 
RUN_VISUALIZE        = True   # Stage 1: render pose on a single demo video
RUN_BUILD_INDEX      = False  # Stage 2: scan dataset for usable lateral clips
RUN_EXTRACT_FEATURES = False  # Stage 3: pose -> per-clip feature CSV
RUN_TRAIN            = False  # Stage 4: LOO-CV + persist winning classifier
RUN_EVALUATE         = False  # Stage 5: holdout-set scoring of saved model

#  Stage 1: single-video visualization 
# Path to the clip the visualization stage will pose-annotate. Relative paths
# are resolved against VIDEOS_DIR.
VISUALIZE_VIDEO   = "trial_sample.MOV"
# Side outputs: the overlaid view (skeleton on original frame) and the raw
# view (skeleton on a black canvas) are written when these are True.
SAVE_OVERLAID     = True
SAVE_RAW          = True

#  Stage 2: dataset / clip index -
# Cap motion-analysis to N candidate videos per dog. None = no cap (slow:
# full scan over ~450 videos can take >1 hr on CPU).
INDEX_LIMIT_PER_DOG = None
# Cap the FINAL kept clips per dog after motion analysis. None = keep all.
INDEX_SAMPLE_PER_DOG = None
# True = ignore view_index.json cache and re-run motion analysis on every video.
INDEX_REFRESH_VIEWS = False

#  Stage 3: feature extraction 
# True = skip clips already present in features.csv (resume an interrupted run).
FEATURES_RESUME = True

# --- Stage 4: training ------------------------------------------------------
# Dog ids (folder names like "Reggie Bell 927479") to withhold for the final
# test set. None for either = pick the lexicographically last dog of that class.
HOLDOUT_CCL_DOG    = None
HOLDOUT_NORMAL_DOG = None


# STAGE 1 — single-video pose visualization

def format_pose_info(pose_info: dict, frame_index: int) -> str:
    """Render one frame's pose dict as a human-readable text block."""
    lines = [f"=== Frame {frame_index} ==="]
    instances = pose_info["instances"]
    lines.append(f"Detected {len(instances)} instance(s).")

    for i, inst in enumerate(instances):
        x, y, w, h = inst["bbox_xywh"]
        lines.append("")
        lines.append(f"[Instance {i}]")
        lines.append(f"  bbox (x, y, w, h): ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")

        lines.append("  keypoints:")
        for kp in inst["keypoints"]:
            lines.append(
                f"    {kp['name']:<14} x={kp['x']:7.1f}  y={kp['y']:7.1f}  score={kp['score']:.2f}"
            )

        lines.append("  joint angles (deg):")
        for name, val in inst["angles_deg"].items():
            val_str = f"{val:6.1f}" if val is not None else "  n/a "
            lines.append(f"    {name:<18}: {val_str}")

    lines.append("")
    return "\n".join(lines) + "\n"


def analyze_single_video(path: Path) -> list[dict]:
    """Pose-annotate one video and write outputs alongside results/.

    Outputs (named after the input stem):
        <stem>_overlaid.mp4   skeleton drawn on the original frames
        <stem>_pose_raw.mp4   skeleton on a black canvas (motion-only view)
        <stem>_pose_data.txt  per-frame keypoints + joint angles
    Returns the list of per-frame pose_info dicts so callers can post-process.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base = path.stem
    out_overlaid_path = RESULTS_DIR / f"{base}_overlaid.mp4"
    out_raw_path      = RESULTS_DIR / f"{base}_pose_raw.mp4"
    out_data_path     = RESULTS_DIR / f"{base}_pose_data.txt"

    reader = read_video(path)
    print(
        f"Opened {path.name}: {reader.frame_count} frames @ {reader.fps:.2f} fps, "
        f"{reader.width}x{reader.height}"
    )

    poses_each_frame: list[dict] = []

    # Open every output sink in one `with` so a mid-video crash still flushes
    # and releases all of them cleanly.
    with reader, \
         writer_matching(out_overlaid_path, reader) as overlaid_writer, \
         writer_matching(out_raw_path, reader) as raw_writer, \
         open(out_data_path, "w", encoding="utf-8") as data_file:

        for i, frame in enumerate(reader):
            pose_info = extract_pose(frame)

            # Reuse pose_info in both get_pose() calls so inference runs once.
            if SAVE_OVERLAID:
                overlaid = get_pose(frame, pose_data=pose_info)
                overlaid_writer.write(overlaid)
            if SAVE_RAW:
                blank = np.zeros_like(frame)
                pose_only = get_pose(blank, pose_data=pose_info)
                raw_writer.write(pose_only)

            data_file.write(format_pose_info(pose_info, i))
            poses_each_frame.append(pose_info)

            if (i + 1) % 10 == 0 or (i + 1) == reader.frame_count:
                print(f"  processed {i + 1}/{reader.frame_count} frames")

    if SAVE_OVERLAID:
        print(f"wrote {out_overlaid_path}")
    if SAVE_RAW:
        print(f"wrote {out_raw_path}")
    print(f"wrote {out_data_path}")
    return poses_each_frame


def run_visualize() -> None:
    """Stage 1 dispatcher — resolves the configured demo path and runs pose."""
    video_path = Path(VISUALIZE_VIDEO)
    if not video_path.is_absolute():
        video_path = VIDEOS_DIR / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Visualization video not found: {video_path}")
    analyze_single_video(video_path)


# STAGE 2 — dataset / clip index
def run_build_index():
    """Stage 2 dispatcher — walk the dataset and cache lateral-clip metadata."""
    # Imported lazily so stages we don't run don't pay their import cost
    # (sklearn/xgboost in particular are slow to load).
    from dataset import build_clip_index, _print_summary

    clips = build_clip_index(
        sample_per_dog=INDEX_SAMPLE_PER_DOG,
        limit_per_dog=INDEX_LIMIT_PER_DOG,
        refresh=INDEX_REFRESH_VIEWS,
    )
    _print_summary(clips)
    return clips


# STAGE 3 — feature extraction
def run_extract_features():
    """Stage 3 dispatcher — pose every indexed clip and build features.csv."""
    from dataset import build_clip_index
    from features import build_feature_matrix

    clips = build_clip_index(
        sample_per_dog=INDEX_SAMPLE_PER_DOG,
        limit_per_dog=INDEX_LIMIT_PER_DOG,
        refresh=INDEX_REFRESH_VIEWS,
    )
    print(f"{len(clips)} usable clips")
    return build_feature_matrix(clips, resume=FEATURES_RESUME)


# STAGE 4 — training

def run_train():
    """Stage 4 dispatcher — call train.py's main with configured holdout dogs."""
    # train.py's CLI uses argparse; rather than modify it we
    # patch sys.argv to mimic the command-line invocation. Keeps train.py the
    # single source of truth for training behavior.
    # yes this is kind of hacky ;} but it's nice to be able to run train.py from command line too
    import sys
    import train

    argv_save = sys.argv[:]
    sys.argv = ["train.py"]
    if HOLDOUT_CCL_DOG is not None:
        sys.argv += ["--holdout-ccl", HOLDOUT_CCL_DOG]
    if HOLDOUT_NORMAL_DOG is not None:
        sys.argv += ["--holdout-normal", HOLDOUT_NORMAL_DOG]
    try:
        train.main()
    finally:
        sys.argv = argv_save


# STAGE 5 — evaluation

def run_evaluate():
    """Stage 5 dispatcher — score the saved winning model on the holdout dogs."""
    import evaluate
    evaluate.main()


def main():
    if RUN_VISUALIZE:
        print("\n=== [1/5] Visualize single video ===")
        run_visualize()

    if RUN_BUILD_INDEX:
        print("\n=== [2/5] Build lateral-clip index ===")
        run_build_index()

    if RUN_EXTRACT_FEATURES:
        print("\n=== [3/5] Extract per-clip features ===")
        run_extract_features()

    if RUN_TRAIN:
        print("\n=== [4/5] Train classifier ===")
        run_train()

    if RUN_EVALUATE:
        print("\n=== [5/5] Evaluate on holdout ===")
        run_evaluate()


if __name__ == "__main__":
    main()

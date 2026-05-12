# main pipeline driver - edit the RUN_* flags below to control what runs

from pathlib import Path

import numpy as np

from animal_openpose import extract_pose, get_pose
from video_io import read_video, writer_matching


# project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR   = PROJECT_ROOT / "videos"
RESULTS_DIR  = PROJECT_ROOT / "results"

# toggle each stage on or off
RUN_VISUALIZE        = True
RUN_BUILD_INDEX      = True
RUN_EXTRACT_FEATURES = True
RUN_TRAIN            = True
RUN_EVALUATE         = True

# which video to run pose on for the visualization stage
VISUALIZE_VIDEO = "trial_sample.MOV"
SAVE_OVERLAID   = True
SAVE_RAW        = True

# dataset scan limits (None = no limit)
INDEX_LIMIT_PER_DOG  = None
INDEX_SAMPLE_PER_DOG = None
INDEX_REFRESH_VIEWS  = False

# skip clips already in features.csv when resuming
FEATURES_RESUME = True

# holdout dog overrides - None means pick automatically
HOLDOUT_CCL_DOG    = None
HOLDOUT_NORMAL_DOG = None


# format one frame's pose result as readable text
def format_pose_info(pose_info, frame_index):
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
            if val is not None:
                val_str = f"{val:6.1f}"
            else:
                val_str = "  n/a "
            lines.append(f"    {name:<18}: {val_str}")

    lines.append("")
    return "\n".join(lines) + "\n"


# run pose on a single video and write output files to results/
def analyze_single_video(path):
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

    poses_each_frame = []

    with reader, \
         writer_matching(out_overlaid_path, reader) as overlaid_writer, \
         writer_matching(out_raw_path, reader) as raw_writer, \
         open(out_data_path, "w", encoding="utf-8") as data_file:

        for i, frame in enumerate(reader):
            pose_info = extract_pose(frame)

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


# stage 1: pose-annotate one demo video
def run_visualize():
    video_path = Path(VISUALIZE_VIDEO)
    if not video_path.is_absolute():
        video_path = VIDEOS_DIR / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Visualization video not found: {video_path}")
    analyze_single_video(video_path)


# stage 2: scan the dataset and cache which clips are lateral
def run_build_index():
    from dataset import build_clip_index, _print_summary
    clips = build_clip_index(
        sample_per_dog=INDEX_SAMPLE_PER_DOG,
        limit_per_dog=INDEX_LIMIT_PER_DOG,
        refresh=INDEX_REFRESH_VIEWS,
    )
    _print_summary(clips)
    return clips


# stage 3: run pose on every clip and save features to csv
def run_extract_features():
    from dataset import build_clip_index
    from features import build_feature_matrix
    clips = build_clip_index(
        sample_per_dog=INDEX_SAMPLE_PER_DOG,
        limit_per_dog=INDEX_LIMIT_PER_DOG,
        refresh=INDEX_REFRESH_VIEWS,
    )
    print(f"{len(clips)} usable clips")
    return build_feature_matrix(clips, resume=FEATURES_RESUME)


# stage 4: run cross-validation and save the best model
def run_train():
    import train
    train.main(holdout_ccl=HOLDOUT_CCL_DOG, holdout_normal=HOLDOUT_NORMAL_DOG)


# stage 5: score the saved model on the holdout dogs
def run_evaluate():
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

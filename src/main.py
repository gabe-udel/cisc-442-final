from pathlib import Path

from animal_openpose import extract_pose, get_pose
from video_io import read_video, writer_matching


VIDEOS_DIR = Path(__file__).parent.parent / "videos"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Pick the first .mp4 in videos/ — avoids hardcoding the fullwidth-pipe character
# in the current filename. Swap for an explicit path or argv when ready.
TEST = next(VIDEOS_DIR.glob("*.mp4"))

#    HEY MIKUL!!

# CURRENT STATE:
# THE SHIT WORKS. kind of. it runs on all quadrupeds DECENTLY.
# Now it walks a video frame-by-frame, draws the pose on each frame,
# writes results/out.mp4 and dumps per-frame keypoints+angles to
# results/pose_data.txt.

# TO DO:
# - command line argument for input path
# - tune score threshold per-clip if poses look noisy


def format_pose_info(pose_info: dict, frame_index: int) -> str:
    """Render one frame's pose data as a human-readable text block."""
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_video_path = RESULTS_DIR / "out.mp4"
    out_data_path = RESULTS_DIR / "pose_data.txt"

    reader = read_video(TEST)
    print(
        f"Opened {TEST.name}: {reader.frame_count} frames @ {reader.fps:.2f} fps, "
        f"{reader.width}x{reader.height}"
    )

    with reader, \
         writer_matching(out_video_path, reader) as writer, \
         open(out_data_path, "w", encoding="utf-8") as data_file:

        for i, frame in enumerate(reader):
            pose_info = extract_pose(frame)
            pose = get_pose(frame, pose_data=pose_info)

            data_file.write(format_pose_info(pose_info, i))
            writer.write(pose)

            if (i + 1) % 10 == 0 or (i + 1) == reader.frame_count:
                print(f"  processed {i + 1}/{reader.frame_count} frames")

    print(f"wrote {out_video_path}")
    print(f"wrote {out_data_path}")


if __name__ == "__main__":
    main()

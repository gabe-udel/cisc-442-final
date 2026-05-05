# Entry point for the pose-estimation pipeline.
#
# What this script does, end-to-end:
#   1. Picks an .mp4 from the project's `videos/` folder.
#   2. Opens it for reading frame-by-frame (BGR numpy arrays via OpenCV).
#   3. For each frame, runs the animal-pose pipeline from `animal_openpose.py`
#      to detect quadrupeds and locate 17 AP-10K keypoints on each one.
#   4. Writes an annotated copy of the video (skeleton drawn on top) to
#      `results/out.mp4` and a human-readable text dump of every frame's
#      keypoints + joint angles to `results/pose_data.txt`.

# Path is the standard library's object-oriented filesystem path API.
# We use it instead of raw strings so path math (joining, parents, globbing)
# stays portable across operating systems.
from pathlib import Path

# The two custom modules that do the real work:
#   - extract_pose: runs detection + pose inference, returns structured data.
#   - get_pose:     runs (or reuses) extract_pose and draws the skeleton on the frame.
from animal_openpose import extract_pose, get_pose
#   - read_video:       opens an mp4 and yields BGR frames.
#   - writer_matching:  builds an output writer with the same fps/size as the reader.
from video_io import read_video, writer_matching


# Resolve the project's `videos/` and `results/` directories relative to THIS
# file. `__file__` is the path to main.py; `.parent` is `src/`; `.parent.parent`
# is the repo root. Doing it this way means the script works no matter what
# working directory you launch it from.
VIDEOS_DIR = Path(__file__).parent.parent / "videos"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Pick the first .mp4 in videos/ — avoids hardcoding the fullwidth-pipe character
# in the current filename. Swap for an explicit path or argv when ready.
# `glob("*.mp4")` returns a generator; `next(...)` pulls the first match.
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
    """Render one frame's pose data as a human-readable text block.

    `pose_info` is the dict returned by `extract_pose(...)`. It contains a list
    of detected animal "instances", and for each one: a bounding box, the 17
    keypoints with (x, y, score), and a few computed joint angles in degrees.

    We turn that into a plain-text section that gets appended to pose_data.txt
    so the run leaves behind a readable log alongside the annotated video.
    """
    # Start a list of lines we'll join with newlines at the end. Building a list
    # and joining once is cleaner than repeatedly concatenating strings.
    lines = [f"=== Frame {frame_index} ==="]
    # `instances` is a list — one entry per detected quadruped in this frame.
    instances = pose_info["instances"]
    lines.append(f"Detected {len(instances)} instance(s).")

    # Loop over each detected animal. `enumerate` gives us a stable per-frame
    # index `i` so we can label them as Instance 0, Instance 1, ...
    for i, inst in enumerate(instances):
        # Unpack the bounding box. It's stored as (x, y, width, height) where
        # (x, y) is the top-left corner — same convention OpenCV/COCO use.
        x, y, w, h = inst["bbox_xywh"]
        lines.append("")  # blank line between instances for readability
        lines.append(f"[Instance {i}]")
        # `:.1f` formats the float with one digit after the decimal point.
        lines.append(f"  bbox (x, y, w, h): ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")

        # Each instance has 17 keypoints (eyes, nose, paws, hips, etc.). Dump
        # them one per line with name, position, and detection confidence.
        lines.append("  keypoints:")
        for kp in inst["keypoints"]:
            # Format string breakdown:
            #   {kp['name']:<14}  -> left-justified in a 14-char column
            #   x={kp['x']:7.1f}  -> width 7, one decimal place
            #   score={kp['score']:.2f} -> two decimals (range 0.0–1.0)
            lines.append(
                f"    {kp['name']:<14} x={kp['x']:7.1f}  y={kp['y']:7.1f}  score={kp['score']:.2f}"
            )

        # Joint angles (elbows, knees, hips, shoulders, neck) in degrees.
        # The pose module sets a value to None when at least one of the three
        # contributing keypoints was below the confidence threshold.
        lines.append("  joint angles (deg):")
        for name, val in inst["angles_deg"].items():
            # If the angle is None, print "n/a" with the same column width so
            # everything stays nicely aligned in the output file.
            val_str = f"{val:6.1f}" if val is not None else "  n/a "
            lines.append(f"    {name:<18}: {val_str}")

    # Trailing blank line so consecutive frames don't run together visually.
    lines.append("")
    # Join with newlines and add a final newline so the next frame's "===" header
    # always lands on a fresh line.
    return "\n".join(lines) + "\n"


def main():
    # Make sure results/ exists before we try to write into it.
    # `parents=True` creates intermediate dirs; `exist_ok=True` silences the
    # error if the directory is already there.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Output paths for the annotated video and the text data dump.
    out_video_path = RESULTS_DIR / "out.mp4"
    out_data_path = RESULTS_DIR / "pose_data.txt"

    # Open the input video. `reader` exposes .fps, .width, .height, .frame_count
    # and is iterable: each iteration yields one BGR frame.
    reader = read_video(TEST)
    print(
        f"Opened {TEST.name}: {reader.frame_count} frames @ {reader.fps:.2f} fps, "
        f"{reader.width}x{reader.height}"
    )

    # Open three resources at once with a single `with` statement. The trailing
    # backslashes let us split this across lines for readability. Doing it this
    # way guarantees that — even if pose inference crashes mid-video — every
    # resource (input video, output video, text file) is closed/released cleanly.
    with reader, \
         writer_matching(out_video_path, reader) as writer, \
         open(out_data_path, "w", encoding="utf-8") as data_file:

        # Iterate over BGR frames, indexed so we can label them in the data file
        # and report progress to the terminal.
        for i, frame in enumerate(reader):
            # 1) Run detection + pose inference once. This is the expensive step.
            pose_info = extract_pose(frame)
            # 2) Draw the skeleton on a copy of the frame. We pass `pose_info` in
            #    so `get_pose` reuses the inference result instead of rerunning it.
            pose = get_pose(frame, pose_data=pose_info)

            # 3) Append this frame's pose data to the text file.
            data_file.write(format_pose_info(pose_info, i))
            # 4) Write the annotated frame to the output video.
            writer.write(pose)

            # Progress report every 10 frames, plus one final report on the last
            # frame so the user sees a clean "100%" line at the end.
            if (i + 1) % 10 == 0 or (i + 1) == reader.frame_count:
                print(f"  processed {i + 1}/{reader.frame_count} frames")

    # Once the `with` block exits, both video files and the text file are flushed
    # and closed. Print where everything ended up so the user can find the output.
    print(f"wrote {out_video_path}")
    print(f"wrote {out_data_path}")


# Standard Python idiom: only run main() if this file is executed directly
# (e.g. `python src/main.py`). If something else imports this file as a module,
# main() is NOT auto-run — that's what makes the helpers above safely reusable.
if __name__ == "__main__":
    main()

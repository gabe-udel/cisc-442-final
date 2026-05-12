# dataset.py - finds and filters usable lateral gait clips from the dog video folder

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# where the dog videos live (can be overridden with the DOG_DATA_ROOT env variable)
DEFAULT_DATA_ROOT = Path(os.environ.get(
    "DOG_DATA_ROOT",
    str(PROJECT_ROOT / "videos" / "cisc442 dog videos"),
))

RESULTS_DIR = PROJECT_ROOT / "results"
VIEW_INDEX_PATH = RESULTS_DIR / "view_index.json"


# one usable clip - a single video with metadata
class Clip:
    def __init__(self, video_path, dog_id, dog_name, label, direction, start_frame, end_frame, n_frames, fps):
        self.video_path = video_path
        self.dog_id = dog_id
        self.dog_name = dog_name
        self.label = label           # 1 = ccl injured, 0 = healthy
        self.direction = direction   # 'LR' or 'RL'
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.n_frames = n_frames
        self.fps = fps

    @property
    def gait_length(self):
        # number of frames in the walking window
        return self.end_frame - self.start_frame + 1


# return True if this folder name refers to a baseline visit
def _is_baseline_folder(name):
    return "baseline" in name.lower()


# find the "gait videos" subfolder inside a dog folder
def _gait_videos_dir(dog_dir):
    for sub in dog_dir.iterdir():
        if sub.is_dir() and "gait videos" in sub.name.lower():
            return sub
    return None


# collect all video files under a folder recursively
def _all_movs_under(root):
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".mov", ".mp4"):
            out.append(p)
    return out


# walk the dataset and return a list of (video_path, dog_id, dog_name, label)
def list_candidate_videos(data_root=DEFAULT_DATA_ROOT, limit_per_dog=None):
    out = []

    # take only N evenly spaced items from a list (for smoke testing)
    def take_limited(items):
        if limit_per_dog is None or limit_per_dog >= len(items):
            return items
        result = []
        step = len(items) / limit_per_dog
        for i in range(limit_per_dog):
            result.append(items[int(i * step)])
        return result

    # ccl cases - only use baseline visit folders, skip post-op
    ccl_root = data_root / "CCL Cases"
    if ccl_root.is_dir():
        for dog_dir in sorted(ccl_root.iterdir()):
            if not dog_dir.is_dir():
                continue
            dog_id = dog_dir.name
            dog_name = " ".join(dog_id.split()[:-1])
            if dog_name == "":
                dog_name = dog_id
            gait_dir = _gait_videos_dir(dog_dir)
            if gait_dir is None:
                continue
            dog_movs = []
            for visit in gait_dir.iterdir():
                if visit.is_dir() and _is_baseline_folder(visit.name):
                    dog_movs.extend(_all_movs_under(visit))
            for mov in take_limited(sorted(dog_movs)):
                out.append((mov, dog_id, dog_name, 1))

    # healthy normals - use all videos
    normals_root = data_root / "Normals"
    if normals_root.is_dir():
        for dog_dir in sorted(normals_root.iterdir()):
            if not dog_dir.is_dir():
                continue
            dog_id = dog_dir.name
            dog_name = " ".join(dog_id.split()[:-1])
            if dog_name == "":
                dog_name = dog_id
            gait_dir = _gait_videos_dir(dog_dir)
            if gait_dir is None:
                continue
            dog_movs = sorted(_all_movs_under(gait_dir))
            for mov in take_limited(dog_movs):
                out.append((mov, dog_id, dog_name, 0))

    return out


# settings for motion analysis
N_MOTION_SAMPLES = 12        # frames to sample per video
LATERAL_RATIO_THRESHOLD = 2.5  # horizontal / vertical travel must exceed this
MIN_HORIZONTAL_TRAVEL = 0.20   # dog must cross at least 20% of frame width


# return evenly spaced frame indices across a video
def _sample_frame_indices(n_frames, n_samples):
    if n_frames <= 0:
        return []
    if n_samples >= n_frames:
        return list(range(n_frames))
    result = []
    for i in range(n_samples):
        result.append(int(round(i * (n_frames - 1) / (n_samples - 1))))
    return result


# read specific frames by index from a video file
def _read_frames_at(video_path, indices):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        else:
            frames.append(None)
    cap.release()
    return frames


# run the detector on each sampled frame and get the dog's center point
def _detect_dog_centers(frames):
    from animal_openpose import _detect_quadrupeds
    centers = []
    for f in frames:
        if f is None:
            centers.append(None)
            continue
        pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        boxes = _detect_quadrupeds(pil)
        if boxes.shape[0] == 0:
            centers.append(None)
            continue
        # if multiple animals, pick the largest box
        areas = boxes[:, 2] * boxes[:, 3]
        i = int(np.argmax(areas))
        x, y, w, h = boxes[i]
        centers.append((float(x + w / 2.0), float(y + h / 2.0)))
    return centers


# check if a video is a usable lateral clip and find the walking window
def _analyze_motion(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"is_lateral": False, "reason": "could not open"}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if n < 8:
        return {"is_lateral": False, "n_frames": n, "fps": fps, "reason": "too short"}

    sample_idx = _sample_frame_indices(n, N_MOTION_SAMPLES)
    frames = _read_frames_at(video_path, sample_idx)
    centers = _detect_dog_centers(frames)

    # need at least 4 detections to measure motion
    valid = []
    for idx, c in zip(sample_idx, centers):
        if c is not None:
            valid.append((idx, c))

    if len(valid) < 4:
        return {
            "is_lateral": False, "n_frames": n, "fps": fps,
            "frame_size": [w, h],
            "reason": f"only {len(valid)}/{len(sample_idx)} frames had a dog detection",
        }

    idxs = np.array([v[0] for v in valid], dtype=np.float64)
    cxs = np.array([v[1][0] for v in valid], dtype=np.float64)
    cys = np.array([v[1][1] for v in valid], dtype=np.float64)

    # measure horizontal vs vertical travel to classify view
    dx_tot = float(np.abs(np.diff(cxs)).sum())
    dy_tot = float(np.abs(np.diff(cys)).sum())
    dx_norm = dx_tot / w
    dy_norm = dy_tot / max(h, 1)
    ratio = dx_tot / max(dy_tot, 1.0)

    is_lateral = (ratio >= LATERAL_RATIO_THRESHOLD) and (dx_norm >= MIN_HORIZONTAL_TRAVEL)

    if not is_lateral:
        return {
            "is_lateral": False, "n_frames": n, "fps": fps,
            "frame_size": [w, h],
            "reason": f"ratio={ratio:.2f} (need >={LATERAL_RATIO_THRESHOLD}), "
                      f"dx_norm={dx_norm:.2f} (need >={MIN_HORIZONTAL_TRAVEL})",
            "dx_total_px": dx_tot, "dy_total_px": dy_tot,
        }

    # positive slope means dog moved left to right
    slope = float(np.polyfit(idxs, cxs, 1)[0])
    direction = "LR" if slope > 0 else "RL"

    # trim to the portion where the dog is actively walking
    seg_vel = np.abs(np.diff(cxs)) / np.maximum(np.diff(idxs), 1)
    if seg_vel.size == 0 or seg_vel.max() <= 0:
        start_frame = int(idxs[0])
        end_frame = int(idxs[-1])
    else:
        active_thresh = max(seg_vel.max() * 0.25, 1.0)
        active = seg_vel >= active_thresh
        if active.any():
            first = int(np.argmax(active))
            last = int(len(active) - 1 - np.argmax(active[::-1]))
            start_frame = int(idxs[first])
            end_frame = int(idxs[min(last + 1, len(idxs) - 1)])
        else:
            start_frame = int(idxs[0])
            end_frame = int(idxs[-1])

    start_frame = max(0, start_frame)
    end_frame = min(n - 1, end_frame)
    if end_frame - start_frame < 4:
        return {
            "is_lateral": False, "n_frames": n, "fps": fps,
            "frame_size": [w, h],
            "reason": "gait window too short after trimming",
        }

    return {
        "is_lateral": True,
        "direction": direction,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "n_frames": n,
        "fps": fps,
        "frame_size": [w, h],
        "ratio": ratio,
        "dx_norm": dx_norm,
        "dy_norm": dy_norm,
    }


# load the cached per-video analysis from disk
def _load_view_cache():
    if VIEW_INDEX_PATH.exists():
        with open(VIEW_INDEX_PATH) as f:
            return json.load(f)
    return {}


# save the per-video analysis cache to disk
def _save_view_cache(cache):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(VIEW_INDEX_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


# main entry point - returns a list of usable Clip objects
def build_clip_index(data_root=DEFAULT_DATA_ROOT, sample_per_dog=None, limit_per_dog=None, refresh=False, progress=True):
    candidates = list_candidate_videos(data_root, limit_per_dog=limit_per_dog)
    if not candidates:
        raise RuntimeError(
            f"No videos found under {data_root}. Set DOG_DATA_ROOT or check the path."
        )

    if refresh:
        cache = {}
    else:
        cache = _load_view_cache()

    new_entries = 0
    cached_hits = 0
    scan_t0 = time.time()

    # analyze any video not already in the cache
    for i, (path, dog_id, dog_name, label) in enumerate(candidates):
        key = str(path)
        if key in cache:
            cached_hits += 1
            continue
        if progress:
            if new_entries > 0:
                avg = (time.time() - scan_t0) / new_entries
                remaining_count = 0
                for p, _, _, _ in candidates[i:]:
                    if str(p) not in cache:
                        remaining_count += 1
                eta = avg * remaining_count
                eta_str = f"  (avg {avg:.1f}s/vid, ETA {eta/60:.1f} min for {remaining_count} new)"
            else:
                eta_str = ""
            label_tag = "CCL " if label == 1 else "Norm"
            print(f"  [{i+1}/{len(candidates)}] [{label_tag}] {dog_id}  {path.name}{eta_str}", flush=True)
        t_v = time.time()
        try:
            info = _analyze_motion(path)
        except Exception as e:
            info = {"is_lateral": False, "reason": f"exception: {e}"}
        info["dog_id"] = dog_id
        info["dog_name"] = dog_name
        info["label"] = label
        cache[key] = info
        new_entries += 1
        if progress:
            if info.get("is_lateral"):
                print(f"      -> LATERAL ({info['direction']})  "
                      f"frames {info['start_frame']}-{info['end_frame']} of {info['n_frames']}  "
                      f"ratio={info.get('ratio', 0):.2f}  "
                      f"({time.time() - t_v:.1f}s)", flush=True)
            else:
                print(f"      -> dropped: {info.get('reason', '?')}  "
                      f"({time.time() - t_v:.1f}s)", flush=True)
        # save every 10 videos so we don't lose progress if interrupted
        if new_entries % 10 == 0:
            _save_view_cache(cache)

    if new_entries:
        _save_view_cache(cache)
    if progress:
        scan_elapsed = time.time() - scan_t0
        print(f"  scan finished: {new_entries} newly analyzed, {cached_hits} cache hits, "
              f"{scan_elapsed:.1f}s total ({scan_elapsed/60:.1f} min)", flush=True)

    # turn cache entries into Clip objects, grouped by dog
    by_dog = {}
    for path, dog_id, dog_name, label in candidates:
        info = cache[str(path)]
        if not info.get("is_lateral"):
            continue
        clip = Clip(
            video_path=str(path),
            dog_id=dog_id,
            dog_name=dog_name,
            label=label,
            direction=info["direction"],
            start_frame=int(info["start_frame"]),
            end_frame=int(info["end_frame"]),
            n_frames=int(info["n_frames"]),
            fps=float(info["fps"]),
        )
        if dog_id not in by_dog:
            by_dog[dog_id] = []
        by_dog[dog_id].append(clip)

    clips = []
    for dog_id, dog_clips in by_dog.items():
        if sample_per_dog is not None and sample_per_dog < len(dog_clips):
            # take evenly spaced subset to avoid biasing toward early filenames
            kept = []
            step = len(dog_clips) / sample_per_dog
            for i in range(sample_per_dog):
                kept.append(dog_clips[int(i * step)])
            clips.extend(kept)
        else:
            clips.extend(dog_clips)

    return clips


# print a per-dog summary of how many clips we have
def _print_summary(clips):
    by_dog = {}
    for c in clips:
        if c.dog_id not in by_dog:
            by_dog[c.dog_id] = []
        by_dog[c.dog_id].append(c)
    print()
    print(f"=== {len(clips)} usable lateral clips across {len(by_dog)} dogs ===")
    for dog_id in sorted(by_dog):
        dc = by_dog[dog_id]
        lr = 0
        rl = 0
        for c in dc:
            if c.direction == "LR":
                lr += 1
            else:
                rl += 1
        label = "CCL " if dc[0].label == 1 else "Norm"
        print(f"  [{label}] {dog_id:35s}  total={len(dc):3d}  LR={lr:3d}  RL={rl:3d}")


def main():
    parser = argparse.ArgumentParser(description="Build/inspect the lateral-clip index.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--limit-per-dog", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    clips = build_clip_index(
        data_root=args.data_root,
        sample_per_dog=args.sample,
        limit_per_dog=args.limit_per_dog,
        refresh=args.refresh,
        progress=not args.no_progress,
    )
    _print_summary(clips)


if __name__ == "__main__":
    main()

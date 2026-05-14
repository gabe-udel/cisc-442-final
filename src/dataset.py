"""Discover usable training clips from the CISC-442 dog-video dataset.

Goal of this module
-------------------
Produce a tidy list of Clip records — one per usable video — that downstream
feature extraction can iterate over without worrying about folder layout or
view filtering.

Source layout:
    <DATA_ROOT>/
        CCL Cases/<DogName ID>/<DogName> Gait Videos/
            <Visit-folder>/        # we keep ONLY *Baseline* folders
                possibly nested by view (L Sag/R Sag/etc.) or flat IMG_xxxx.MOV
        Normals/<DogName ID>/<DogName> Gait Videos/
            <flat or per-view folders>

What we keep
------------
* CCL: only files inside *Baseline* visit folders (skip Surgery/Post-op).
* Normal: every gait video.
* Lateral (left/right side) views only — front/back views are filtered out by
  per-video bbox-motion analysis (horizontal motion must dominate vertical).
* We also crop each lateral video to the contiguous frame range where the dog
  is actually moving (drops standing/intro/outro padding).

The motion analysis runs once per video and is cached to
`results/view_index.json` so repeated training runs don't redo it.

Run as a module to populate / inspect the cache:
    python src/dataset.py             # full scan, cached
    python src/dataset.py --sample 1  # 1 video per dog 
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


# Paths
# Project root resolved from this file so cwd doesn't matter.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset lives at <repo>/videos/cisc442 dog videos. DOG_DATA_ROOT env var
# overrides for machines where it lives elsewhere.
DEFAULT_DATA_ROOT = Path(os.environ.get(
    "DOG_DATA_ROOT",
    str(PROJECT_ROOT / "videos" / "cisc442 dog videos"),
))

RESULTS_DIR = PROJECT_ROOT / "results"
VIEW_INDEX_PATH = RESULTS_DIR / "view_index.json"

@dataclass
class Clip:
    """One video clip we want to extract pose features from.

    Attributes:
        video_path: absolute path to the .MOV
        dog_id:     unique per-dog string (folder name including the trailing
                    numeric ID — this is what we group on for LOOCV)
        dog_name:   human-readable dog name (folder name minus the ID)
        label:      1 = CCL injury, 0 = healthy normal
        direction:  'LR' (dog walks left→right, camera sees right side) or
                    'RL' (dog walks right→left, camera sees left side)
        start_frame, end_frame: inclusive gait-window range. Pose extraction
                    should iterate frames in [start, end].
        n_frames:   total frame count in the source video (for sanity).
        fps:        source fps (for stride-frequency conversion).
    """
    video_path: str
    dog_id: str
    dog_name: str
    label: int
    direction: str
    start_frame: int
    end_frame: int
    n_frames: int
    fps: float

    @property
    def gait_length(self) -> int:
        return self.end_frame - self.start_frame + 1


# Filesystem walk: enumerate candidate videos before any motion analysis
def _is_baseline_folder(name: str) -> bool:
    """Folder name says 'baseline' (case-insensitive). Visits like 'Surgery',
    '6 Weeks Post Op', '8 Weeks rads', '6 months pos-op' all fail this check."""
    return "baseline" in name.lower()


def _gait_videos_dir(dog_dir: Path) -> Path | None:
    """Return the dog's '<Name> Gait Videos' subfolder, or None if missing."""
    for sub in dog_dir.iterdir():
        if sub.is_dir() and "gait videos" in sub.name.lower():
            return sub
    return None


def _all_movs_under(root: Path) -> list[Path]:
    """Recursively gather every .MOV / .mp4 under `root` (case-insensitive)."""
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".mov", ".mp4"):
            out.append(p)
    return out


def list_candidate_videos(
    data_root: Path = DEFAULT_DATA_ROOT,
    limit_per_dog: int | None = None,
) -> list[tuple[Path, str, str, int]]:
    """Walk the dataset and return (video_path, dog_id, dog_name, label) tuples.

    A "candidate" hasn't been view-classified yet — we only know it's a gait
    video from a relevant dog/visit. The motion analysis later decides if it's
    actually a usable lateral clip.

    `dog_id` is the full folder name (e.g. "Ash Swider 912908"); using it as
    the LOOCV grouping key avoids mixing two different dogs that happen to
    share a first name.

    `limit_per_dog`: if set, take at most N evenly-spaced candidates per dog
    (BEFORE motion analysis). This makes smoke tests tractable — the full
    motion-analysis scan over ~450 videos can take >1 hour on CPU.
    """
    out: list[tuple[Path, str, str, int]] = []

    def _take(items: list[Path]) -> list[Path]:
        if limit_per_dog is None or limit_per_dog >= len(items):
            return items
        step = len(items) / limit_per_dog
        return [items[int(i * step)] for i in range(limit_per_dog)]

    ccl_root = data_root / "CCL Cases"
    if ccl_root.is_dir():
        for dog_dir in sorted(ccl_root.iterdir()):
            if not dog_dir.is_dir():
                continue
            dog_id = dog_dir.name
            dog_name = " ".join(dog_id.split()[:-1]) or dog_id  # drop trailing numeric ID
            gait_dir = _gait_videos_dir(dog_dir)
            if gait_dir is None:
                continue
            # Only the Baseline visit folder(s) for CCL dogs.
            dog_movs: list[Path] = []
            for visit in gait_dir.iterdir():
                if visit.is_dir() and _is_baseline_folder(visit.name):
                    dog_movs.extend(_all_movs_under(visit))
            for mov in _take(sorted(dog_movs)):
                out.append((mov, dog_id, dog_name, 1))

    normals_root = data_root / "Normals"
    if normals_root.is_dir():
        for dog_dir in sorted(normals_root.iterdir()):
            if not dog_dir.is_dir():
                continue
            dog_id = dog_dir.name
            dog_name = " ".join(dog_id.split()[:-1]) or dog_id
            gait_dir = _gait_videos_dir(dog_dir)
            if gait_dir is None:
                continue
            # Normals don't have visit subfolders separating baseline from
            # post-op (they have no surgery), so take everything under the
            # gait-videos folder.
            dog_movs = sorted(_all_movs_under(gait_dir))
            for mov in _take(dog_movs):
                out.append((mov, dog_id, dog_name, 0))

    return out


# Per-video motion analysis: lateral filter + gait window

# We sample this many evenly-spaced frames per video for the cheap motion
# analysis. More samples = better gait-window estimation but slower. 12 is a
# good tradeoff for 5-15s gait clips, but it really is flexible so long as you stay 0-50 range
N_MOTION_SAMPLES = 12

# Minimum dominance ratio for "lateral" classification. dx_total / dy_total
# must exceed this. Front/back-walking dogs have ratio near 1.
# Note this only works bc the camer is still.
LATERAL_RATIO_THRESHOLD = 2.5

# Minimum bbox-center horizontal travel (as a fraction of frame width) for the
# clip to count as "actually walking". Filters out stationary shots even if
# they're lateral in setup.
MIN_HORIZONTAL_TRAVEL = 0.20


def _sample_frame_indices(n_frames: int, n_samples: int) -> list[int]:
    """Evenly spaced integer frame indices in [0, n_frames-1]."""
    if n_frames <= 0:
        return []
    if n_samples >= n_frames:
        return list(range(n_frames))
    return [int(round(i * (n_frames - 1) / (n_samples - 1))) for i in range(n_samples)]


def _read_frames_at(video_path: Path, indices: list[int]) -> list[np.ndarray]:
    """Read specified frames from a video. Returns BGR numpy arrays."""
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


def _detect_dog_centers(frames: list[np.ndarray]) -> list[tuple[float, float] | None]:
    """Run RT-DETR on each sampled frame; return bbox-center (cx, cy) or None.

    Importing the detector here (lazy) so just listing candidates doesn't pay
    the model-load cost.
    """
    from animal_openpose import _detect_quadrupeds  # reuse the existing detector
    centers: list[tuple[float, float] | None] = []
    for f in frames:
        if f is None:
            centers.append(None)
            continue
        pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        boxes = _detect_quadrupeds(pil)
        if boxes.shape[0] == 0:
            centers.append(None)
            continue
        # If multiple animals are detected, pick the largest box.
        # This works pretty well in our case.
        areas = boxes[:, 2] * boxes[:, 3]
        i = int(np.argmax(areas))
        x, y, w, h = boxes[i]
        centers.append((float(x + w / 2.0), float(y + h / 2.0)))
    return centers


def _analyze_motion(
    video_path: Path,
) -> dict:
    """Decide if a video is a usable lateral-gait clip and find the gait window.

    Returns a dict with:
        is_lateral:    bool
        direction:     'LR' | 'RL' | None
        start_frame:   int  (inclusive, in source-video frame indices)
        end_frame:     int  (inclusive)
        n_frames:      int
        fps:           float
        frame_size:    [width, height]
        reason:        diagnostic note for clips we drop
    """
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

    # Need at least 4 valid detections to estimate motion meaningfully.
    valid = [(idx, c) for idx, c in zip(sample_idx, centers) if c is not None]
    if len(valid) < 4:
        return {
            "is_lateral": False, "n_frames": n, "fps": fps,
            "frame_size": [w, h],
            "reason": f"only {len(valid)}/{len(sample_idx)} frames had a dog detection",
        }

    idxs = np.array([v[0] for v in valid], dtype=np.float64)
    cxs = np.array([v[1][0] for v in valid], dtype=np.float64)
    cys = np.array([v[1][1] for v in valid], dtype=np.float64)

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

    # Direction is determined by the slope of cx over time
    slope = float(np.polyfit(idxs, cxs, 1)[0])
    direction = "LR" if slope > 0 else "RL"

    # Gait window: trim to the longest contiguous span where the per-segment velocity exceeds a low threshold n compute |dx/dframe|
    # between consecutive samples, find the first/last sample exceeding 25%
    # of the median active velocity, and use those as window boundaries.
    # there is also probably a better way to do this, and do we need to to do this either?
    seg_vel = np.abs(np.diff(cxs)) / np.maximum(np.diff(idxs), 1)
    if seg_vel.size == 0 or seg_vel.max() <= 0:
        start_frame = int(idxs[0]); end_frame = int(idxs[-1])
    else:
        active_thresh = max(seg_vel.max() * 0.25, 1.0)
        active = seg_vel >= active_thresh
        if active.any():
            first = int(np.argmax(active))                     # first True
            last = int(len(active) - 1 - np.argmax(active[::-1]))  # last True
            start_frame = int(idxs[first])
            end_frame = int(idxs[min(last + 1, len(idxs) - 1)])
        else:
            start_frame = int(idxs[0]); end_frame = int(idxs[-1])

    # clamp to valid range and ensure long enough window length.
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


def _load_view_cache() -> dict:
    if VIEW_INDEX_PATH.exists():
        with open(VIEW_INDEX_PATH) as f:
            return json.load(f)
    return {}


def _save_view_cache(cache: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(VIEW_INDEX_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def build_clip_index(
    data_root: Path = DEFAULT_DATA_ROOT,
    sample_per_dog: int | None = None,
    limit_per_dog: int | None = None,
    refresh: bool = False,
    progress: bool = True,
) -> list[Clip]:
    """Return a list of usable Clip records.

    Args:
        data_root: dataset root (overridable via DOG_DATA_ROOT env var).
        sample_per_dog: if not None, keep at most this many lateral clips per
            dog AFTER motion analysis — controls the size of the returned list.
        limit_per_dog: if not None, only run motion analysis on this many
            candidates per dog — controls how much work the SCAN does. Use
            this for fast smoke tests.
        refresh: if True, ignore the existing view-index cache and re-analyze.
        progress: print one line per video while analyzing.
    """
    candidates = list_candidate_videos(data_root, limit_per_dog=limit_per_dog)
    if not candidates:
        raise RuntimeError(
            f"No videos found under {data_root}. Set DOG_DATA_ROOT or check the path."
        )

    cache = {} if refresh else _load_view_cache()
    new_entries = 0
    cached_hits = 0
    scan_t0 = time.time()

    # ensure every candidate has motion-analysis info.
    for i, (path, dog_id, dog_name, label) in enumerate(candidates):
        key = str(path)
        if key in cache:
            cached_hits += 1
            continue
        if progress:
            #  eta based on average analysis time so far
            if new_entries > 0:
                avg = (time.time() - scan_t0) / new_entries
                remaining_count = sum(1 for p, _, _, _ in candidates[i:]
                                      if str(p) not in cache)
                eta = avg * remaining_count
                eta_str = f"  (avg {avg:.1f}s/vid, ETA {eta/60:.1f} min for {remaining_count} new)"
            else:
                eta_str = ""
            label_tag = "CCL " if label == 1 else "Norm"
            print(f"  [{i+1}/{len(candidates)}] [{label_tag}] {dog_id}  "
                  f"{path.name}{eta_str}", flush=True)
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
        # show classification result inline so the log shows what's keeping vs dropping.
        if progress:
            if info.get("is_lateral"):
                print(f"      -> LATERAL ({info['direction']})  "
                      f"frames {info['start_frame']}-{info['end_frame']} of {info['n_frames']}  "
                      f"ratio={info.get('ratio', 0):.2f}  "
                      f"({time.time() - t_v:.1f}s)",
                      flush=True)
            else:
                print(f"      -> dropped: {info.get('reason', '?')}  "
                      f"({time.time() - t_v:.1f}s)",
                      flush=True)
        if new_entries % 10 == 0:
            _save_view_cache(cache)

    if new_entries:
        _save_view_cache(cache)
    if progress:
        scan_elapsed = time.time() - scan_t0
        print(f"  scan finished: {new_entries} newly analyzed, {cached_hits} cache hits, "
              f"{scan_elapsed:.1f}s total ({scan_elapsed/60:.1f} min)", flush=True)

    # Second pass: turn keep-able cache entries into Clip records, optionally
    # capped per dog.
    by_dog: dict[str, list[Clip]] = {}
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
        by_dog.setdefault(dog_id, []).append(clip)

    clips: list[Clip] = []
    for dog_id, dog_clips in by_dog.items():
        if sample_per_dog is not None:
            # Spread the cap across the per-dog list so we don't bias to early
            # filenames. Take an evenly-spaced subsample.
            if sample_per_dog >= len(dog_clips):
                kept = dog_clips
            else:
                step = len(dog_clips) / sample_per_dog
                kept = [dog_clips[int(i * step)] for i in range(sample_per_dog)]
            clips.extend(kept)
        else:
            clips.extend(dog_clips)
    return clips


# CLI for inspecting / building the cache
def _print_summary(clips: list[Clip]) -> None:
    by_dog: dict[str, list[Clip]] = {}
    for c in clips:
        by_dog.setdefault(c.dog_id, []).append(c)
    print()
    print(f"=== {len(clips)} usable lateral clips across {len(by_dog)} dogs ===")
    for dog_id in sorted(by_dog):
        dc = by_dog[dog_id]
        lr = sum(1 for c in dc if c.direction == "LR")
        rl = sum(1 for c in dc if c.direction == "RL")
        label = "CCL " if dc[0].label == 1 else "Norm"
        print(f"  [{label}] {dog_id:35s}  total={len(dc):3d}  LR={lr:3d}  RL={rl:3d}")


def main():
    parser = argparse.ArgumentParser(description="Build/inspect the lateral-clip index.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sample", type=int, default=None,
                        help="Keep at most N clips per dog in the final index.")
    parser.add_argument("--limit-per-dog", type=int, default=None,
                        help="Only motion-analyze at most N videos per dog "
                             "(smoke-test mode; full scan is slow).")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-analyze all videos even if cached.")
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

"""Animal OpenPose - dog (and quadruped) pose estimation.

Uses RT-DETR for animal detection and ViTPose+ on the AP-10K head for
17-keypoint quadruped pose estimation, then draws the skeleton over the
input image.

High-level pipeline:
    BGR image (numpy)
        -> detect quadruped bounding boxes  (RT-DETR)
        -> for each box, locate 17 keypoints (ViTPose+, AP-10K head)
        -> compute named joint angles in degrees from triplets of keypoints
        -> (optionally) draw the skeleton onto the image
"""

# Defer evaluation of type annotations so we can write `dict | None` etc. on
# Python versions that don't natively support PEP 604 in runtime annotations.
from __future__ import annotations

# Standard library:
import os
# functools.lru_cache is used here as a "compute once and reuse" decorator —
# we cache the device choice and the (heavy) loaded models so we never load
# them twice in the same process.
from functools import lru_cache

# Quiet down HuggingFace before its modules are imported — these env vars are
# read at import time, so they have to be set BEFORE the `transformers` import below.
#   TRANSFORMERS_VERBOSITY=error -> only error logs, no info/warning spam
#   HF_HUB_DISABLE_SYMLINKS_WARNING -> silence the Windows symlink notice
# `setdefault` means we only set them if the user hasn't already overridden them.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Third-party deps:
import numpy as np            # array math (keypoints, vectors, angles)
import torch                  # the deep-learning backend that runs the models
import cv2                    # image color conversion and drawing primitives
from PIL import Image         # transformers' processors expect PIL images
from transformers import (
    AutoProcessor,                 # auto-detects the right preprocessor for a model
    RTDetrForObjectDetection,      # the detector model class
    VitPoseForPoseEstimation,      # the pose-estimation model class
)


# HuggingFace model IDs. Both will be downloaded and cached on first run.
DETECTOR_ID = "PekingU/rtdetr_r50vd_coco_o365"   # COCO-trained RT-DETR detector
POSE_ID = "usyd-community/vitpose-plus-base"     # ViTPose+ with multi-dataset heads

# AP-10K is dataset_index 3 in ViTPose+'s mixture-of-experts head.
# ViTPose+ is trained jointly on several pose datasets and exposes them via a
# `dataset_index` argument; index 3 selects the AP-10K (animals) head we want.
AP10K_DATASET_INDEX = 3

# COCO class names that we accept as "quadrupeds we'd like to pose-estimate".
# RT-DETR is trained on COCO, so its outputs use these names.
QUADRUPED_LABELS = {
    "dog", "cat", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

# AP-10K 17-keypoint names, by index.
# The order MUST match what the model outputs — index 0 is L_eye, 1 is R_eye, etc.
KEYPOINT_NAMES = [
    "L_eye", "R_eye", "nose", "neck", "tail_root",
    "L_shoulder", "L_elbow", "L_front_paw",
    "R_shoulder", "R_elbow", "R_front_paw",
    "L_hip", "L_knee", "L_back_paw",
    "R_hip", "R_knee", "R_back_paw",
]

# Fallback skeleton if model.config.edges is missing.
# Each tuple (a, b) means "draw a line between keypoint a and keypoint b".
# These define the visual skeleton: head triangle, neck-to-tail spine, then
# each of the four legs.
AP10K_EDGES = [
    (0, 1), (0, 2), (1, 2), (2, 3), (3, 4),    # head + spine: eyes, eyes-to-nose, nose-to-neck, neck-to-tail
    (3, 5), (5, 6), (6, 7),                    # left front leg:  neck->shoulder->elbow->paw
    (3, 8), (8, 9), (9, 10),                   # right front leg: neck->shoulder->elbow->paw
    (4, 11), (11, 12), (12, 13),               # left back leg:   tail->hip->knee->paw
    (4, 14), (14, 15), (15, 16),               # right back leg:  tail->hip->knee->paw
]

# Named joint angles as (point_a, vertex, point_c) keypoint indices.
# The angle is measured at the vertex.
# Example: "L_front_elbow" = angle at keypoint 6 (L_elbow), formed by the rays
# going to keypoint 5 (shoulder) and keypoint 7 (paw). Always in [0, 180] degrees.
ANGLE_TRIPLETS = {
    "L_front_elbow":    (5, 6, 7),
    "R_front_elbow":    (8, 9, 10),
    "L_back_knee":      (11, 12, 13),
    "R_back_knee":      (14, 15, 16),
    "L_front_shoulder": (3, 5, 6),
    "R_front_shoulder": (3, 8, 9),
    "L_back_hip":       (4, 11, 12),
    "R_back_hip":       (4, 14, 15),
    "neck":             (2, 3, 4),
}

# BGR (not RGB!) colors used to differentiate multiple animals in the same frame.
# Cycled through with modulo: instance i uses PER_INSTANCE_COLORS[i % len(...)].
PER_INSTANCE_COLORS = [
    (0, 165, 255),    # orange
    (50, 200, 50),    # green
    (200, 0, 200),    # magenta
    (0, 255, 255),    # yellow
    (255, 100, 100),  # light blue
    (50, 50, 255),    # red
]

# Below this confidence we treat a keypoint as unreliable: don't draw it,
# don't include it in angle calculations.
DEFAULT_SCORE_THRESHOLD = 0.3


@lru_cache(maxsize=1)
def _device() -> str:
    """Prefer CUDA, but verify it actually works before committing.

    `torch.cuda.is_available()` returns True if CUDA *appears* available, but on
    misconfigured systems the first real CUDA op can still raise. We do a tiny
    allocation to confirm, and fall back to CPU on failure. lru_cache makes this
    a one-time check per process.
    """
    if torch.cuda.is_available():
        try:
            # Tiny allocation — cheap, but real enough to surface driver issues.
            torch.zeros(1, device="cuda")
            print("[animal_openpose] using device: cuda")
            return "cuda"
        except RuntimeError as e:
            print(f"[animal_openpose] CUDA detected but unusable ({e}); falling back to CPU.")
    print("[animal_openpose] using device: cpu")
    return "cpu"


def _move_to_device(model):
    """Move model to the preferred device; fall back to CPU on CUDA failure (e.g. OOM).

    Even if `_device()` chose CUDA, .to('cuda') can still fail at model-load time
    when the GPU is out of memory. Catch that case and degrade gracefully.
    """
    device = _device()
    if device == "cuda":
        try:
            return model.to("cuda"), "cuda"
        except RuntimeError as e:
            print(f"[animal_openpose] failed to move model to CUDA ({e}); using CPU.")
            # Free up whatever was partially allocated before retrying on CPU.
            torch.cuda.empty_cache()
    return model.to("cpu"), "cpu"


@lru_cache(maxsize=1)
def _load_detector():
    # Loads the detector once per process (cached by lru_cache).
    # Returns (preprocessor, model, device-string).
    proc = AutoProcessor.from_pretrained(DETECTOR_ID)         # image preprocessor
    model = RTDetrForObjectDetection.from_pretrained(DETECTOR_ID)
    model, device = _move_to_device(model)
    # eval() turns off dropout/batchnorm-train-mode; we're only doing inference.
    model.eval()
    return proc, model, device


@lru_cache(maxsize=1)
def _load_pose_model():
    # Same pattern as the detector loader, for the pose model. Cached separately.
    proc = AutoProcessor.from_pretrained(POSE_ID)
    model = VitPoseForPoseEstimation.from_pretrained(POSE_ID)
    model, device = _move_to_device(model)
    model.eval()
    return proc, model, device


def _detect_quadrupeds(image: Image.Image, threshold: float = 0.3) -> np.ndarray:
    """Return Nx4 boxes in (x, y, w, h) format for quadrupeds in the image.

    Steps:
        1. Preprocess the PIL image (resize, normalize, to tensor).
        2. Forward pass through RT-DETR.
        3. Post-process to get (label, score, box-in-pixels) at threshold.
        4. Keep only boxes whose label is in QUADRUPED_LABELS.
        5. Convert from (x1, y1, x2, y2) to (x, y, w, h) — the format the
           pose model wants.
    """
    proc, model, device = _load_detector()

    # `proc(...)` returns a dict of tensors (pixel_values, etc.). `.to(device)`
    # moves them to the same device as the model — required for the forward pass.
    inputs = proc(images=image, return_tensors="pt").to(device)
    # `torch.no_grad()` disables autograd bookkeeping. Saves memory + a little
    # time during inference; we don't need gradients here.
    with torch.no_grad():
        outputs = model(**inputs)

    # The detector output is in normalized space; we need pixel coordinates.
    # `target_sizes` tells the post-processor the original image size so it can
    # rescale boxes back to pixels. Note: PIL gives (height, width).
    target_sizes = torch.tensor([(image.height, image.width)])
    # `[0]` because there's only one image in the batch — results is a list of
    # length 1 with one dict per image.
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    # id2label maps integer class ids (0, 1, 2, ...) to string names ("person", "dog", ...).
    id2label = model.config.id2label
    boxes_xywh: list[list[float]] = []
    # Walk through each detection. .tolist() pulls Python ints/floats out of
    # the torch tensors so we can iterate normally.
    for label_id, box in zip(results["labels"].tolist(), results["boxes"].tolist()):
        # Lowercase to be safe — COCO labels are already lowercase, but defensive.
        name = id2label.get(label_id, "").lower()
        if name in QUADRUPED_LABELS:
            # RT-DETR outputs (x1, y1, x2, y2) — opposite-corner coordinates.
            # ViTPose+ wants (x, y, w, h), so convert.
            x1, y1, x2, y2 = box
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    # Return an empty (0,4) array (rather than None or []) so the caller can
    # check `.shape[0] == 0` uniformly without special-casing the empty case.
    if not boxes_xywh:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(boxes_xywh, dtype=np.float32)


def _estimate_pose(image: Image.Image, boxes_xywh: np.ndarray):
    """Run the pose model on each detection box and return its results + skeleton edges."""
    proc, model, device = _load_pose_model()

    # The processor accepts a list-of-lists of boxes (one inner list per image).
    # We have one image so it's `[boxes_xywh]`.
    inputs = proc(image, boxes=[boxes_xywh], return_tensors="pt").to(device)
    # ViTPose+ uses a mixture-of-experts: tell it which dataset's head to use
    # for each box. We want AP-10K (animals) for every box, so repeat that
    # index once per detection.
    dataset_index = torch.tensor([AP10K_DATASET_INDEX] * len(boxes_xywh), device=device)

    with torch.no_grad():
        outputs = model(**inputs, dataset_index=dataset_index)

    # Post-process turns raw heatmaps into (keypoints, scores) per box.
    # Again, [0] because of the batch dimension.
    pose_results = proc.post_process_pose_estimation(outputs, boxes=[boxes_xywh])[0]
    # Some checkpoints expose their canonical skeleton as `model.config.edges`;
    # use it when available, otherwise fall back to our hardcoded AP-10K edges.
    edges = getattr(model.config, "edges", None) or AP10K_EDGES
    return pose_results, edges


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b in degrees, given points a, b, c as (x, y).

    Math: the angle between vectors (a-b) and (c-b) is arccos of their normalized
    dot product. We clamp the cosine into [-1, 1] before arccos because tiny
    floating-point drift can push it just outside that domain and produce NaN.
    """
    # Convert to float64 to dodge precision issues with float32 keypoint coords.
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    # If either vector has zero length (two points coincide), the angle is undefined.
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    # Clamp to [-1, 1] to keep arccos happy regardless of float drift.
    cosine = max(-1.0, min(1.0, cosine))
    # arccos returns radians; convert to degrees for human readability.
    return float(np.degrees(np.arccos(cosine)))


def _build_instances(
    pose_results,
    boxes_xywh: np.ndarray,
    score_threshold: float,
) -> list[dict]:
    """Turn raw model output into a clean list of per-animal dicts.

    Each dict has:
        bbox_xywh:  the original detection box
        keypoints:  named, with x/y/score
        angles_deg: named joint angles (or None if any contributing keypoint
                    fell below score_threshold)
    """
    instances: list[dict] = []
    # Iterate over (one pose result, one bbox) pairs — one per detected animal.
    for pose, box in zip(pose_results, boxes_xywh):
        # Cast to numpy with a known dtype so downstream math is predictable.
        kpts = np.asarray(pose["keypoints"], dtype=np.float32)   # shape (17, 2)
        scores = np.asarray(pose["scores"], dtype=np.float32)    # shape (17,)

        # Build a list of {name, x, y, score} dicts — easier to read than parallel arrays.
        keypoints_named = []
        for idx, (kpt, sc) in enumerate(zip(kpts, scores)):
            # Defensive fallback: if the model ever returns more than 17 points,
            # name the extras kpt_17, kpt_18, etc. instead of crashing.
            name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"kpt_{idx}"
            keypoints_named.append({
                "name": name,
                "x": float(kpt[0]),
                "y": float(kpt[1]),
                "score": float(sc),
            })

        # Compute every named joint angle. If any of the three contributing
        # keypoints is below the threshold, set the angle to None to flag it
        # as unreliable rather than reporting a misleading number.
        angles: dict[str, float | None] = {}
        for name, (ai, bi, ci) in ANGLE_TRIPLETS.items():
            if min(scores[ai], scores[bi], scores[ci]) < score_threshold:
                angles[name] = None
            else:
                angles[name] = _angle_deg(kpts[ai], kpts[bi], kpts[ci])

        # Convert the box (numpy array) to a plain list of Python floats so the
        # whole instance dict is JSON-serializable / easy to print.
        instances.append({
            "bbox_xywh": [float(v) for v in box.tolist()],
            "keypoints": keypoints_named,
            "angles_deg": angles,
        })
    return instances


def _draw_pose(
    bgr_image: np.ndarray,
    instances: list[dict],
    edges,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> np.ndarray:
    """Mutate `bgr_image` in place: draw skeleton lines + keypoint dots.

    Returns the same image for convenience (so callers can chain). The caller
    is responsible for passing a copy if they want to preserve the original.
    """
    # Image dimensions, used to skip points that landed outside the visible area.
    h, w = bgr_image.shape[:2]
    for i, inst in enumerate(instances):
        # Re-stack the named keypoints back into plain numpy arrays for fast indexing.
        kpts = np.array([[k["x"], k["y"]] for k in inst["keypoints"]], dtype=np.float32)
        scores = np.array([k["score"] for k in inst["keypoints"]], dtype=np.float32)
        # Pick a color for this animal; cycle if there are more animals than colors.
        color = PER_INSTANCE_COLORS[i % len(PER_INSTANCE_COLORS)]

        # Draw skeleton edges first, so keypoint dots end up on top.
        for a, b in edges:
            # Skip edges that reference keypoints we don't have (defensive — the
            # AP-10K skeleton uses indices 0..16, but we don't trust user-provided edge lists).
            if a >= len(scores) or b >= len(scores):
                continue
            # Skip the edge if either endpoint is below the confidence threshold.
            if scores[a] < score_threshold or scores[b] < score_threshold:
                continue
            x1, y1 = int(kpts[a, 0]), int(kpts[a, 1])
            x2, y2 = int(kpts[b, 0]), int(kpts[b, 1])
            # Skip edges with any endpoint off-screen — guards against weird model
            # outputs leaving artifacts at the image edge.
            if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                continue
            # Thickness 3 + LINE_AA for anti-aliased smooth edges.
            cv2.line(bgr_image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

        # Now draw a dot at each high-confidence keypoint.
        for kpt, sc in zip(kpts, scores):
            if sc < score_threshold:
                continue
            x, y = int(kpt[0]), int(kpt[1])
            if not (0 <= x < w and 0 <= y < h):
                continue
            # Two circles to make the dot stand out against any background:
            #   filled red interior  (thickness=-1 means filled)
            #   thin white outline   (thickness=1)
            cv2.circle(bgr_image, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(bgr_image, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

    return bgr_image


def extract_pose(
    image: np.ndarray,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> dict:
    """Run detection + pose estimation and return structured, JSON-serializable data.

    Args:
        image: BGR image array (as produced by cv2.imread).
        score_threshold: per-keypoint confidence below which joint angles are
            reported as None.

    Returns:
        A dict of the form::

            {
                "image_size": {"width": int, "height": int},
                "edges": [(a, b), ...],          # skeleton index pairs
                "keypoint_names": [...],         # names by index (AP-10K order)
                "instances": [
                    {
                        "bbox_xywh": [x, y, w, h],
                        "keypoints": [
                            {"name": "L_eye", "x": float, "y": float, "score": float},
                            ...
                        ],
                        "angles_deg": {
                            "L_front_elbow": float | None,
                            ...
                        },
                    },
                    ...
                ],
            }
    """
    # The transformers processors expect PIL images in RGB. OpenCV gives us
    # numpy arrays in BGR. Convert color order then wrap as PIL.
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Step 1: detect quadrupeds.
    boxes = _detect_quadrupeds(pil_image)
    # If detection turned up nothing, fall back to a single full-frame "box".
    # That way the pose model still gets a chance — useful for tight crops where
    # the detector might be unsure but a clearly-visible animal fills the frame.
    if boxes.shape[0] == 0:
        boxes = np.array(
            [[0.0, 0.0, float(pil_image.width), float(pil_image.height)]],
            dtype=np.float32,
        )

    # Step 2: run pose estimation on each box.
    pose_results, edges = _estimate_pose(pil_image, boxes)
    # Step 3: combine boxes + raw pose output into a clean per-instance structure.
    instances = _build_instances(pose_results, boxes, score_threshold)

    # Final return: everything a caller could want, all JSON-friendly types.
    return {
        "image_size": {"width": pil_image.width, "height": pil_image.height},
        # Convert edge tuples to lists so json.dumps works without a custom encoder.
        "edges": [list(e) for e in edges],
        # Copy of the names list so callers can't accidentally mutate our module-level constant.
        "keypoint_names": list(KEYPOINT_NAMES),
        "instances": instances,
    }


def get_pose(image: np.ndarray, pose_data: dict | None = None) -> np.ndarray:
    """Draw quadruped pose on top of `image` and return the annotated copy.

    Args:
        image: BGR image array (as produced by cv2.imread).
        pose_data: optional output of `extract_pose(image)`. Pass it in to
            avoid running inference twice when you want both the data and
            the annotated image.

    Returns:
        BGR image array with the detected pose skeleton(s) drawn on top.
    """
    # If the caller didn't already run extract_pose, do it now. main.py passes
    # in the cached result so we don't run the heavy models a second time per frame.
    if pose_data is None:
        pose_data = extract_pose(image)
    # `image.copy()` so we never mutate the caller's original frame — _draw_pose
    # draws in place. Cheap (one numpy memcpy) and avoids surprising bugs.
    return _draw_pose(image.copy(), pose_data["instances"], pose_data["edges"])

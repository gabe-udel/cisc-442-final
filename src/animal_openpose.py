"""Animal OpenPose - dog (and quadruped) pose estimation.

Uses RT-DETR for animal detection and ViTPose+ on the AP-10K head for
17-keypoint quadruped pose estimation, then draws the skeleton over the
input image.
"""

from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import cv2
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)


DETECTOR_ID = "PekingU/rtdetr_r50vd_coco_o365"
POSE_ID = "usyd-community/vitpose-plus-base"

# AP-10K is dataset_index 3 in ViTPose+'s mixture-of-experts head.
AP10K_DATASET_INDEX = 3

QUADRUPED_LABELS = {
    "dog", "cat", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

# AP-10K 17-keypoint names, by index.
KEYPOINT_NAMES = [
    "L_eye", "R_eye", "nose", "neck", "tail_root",
    "L_shoulder", "L_elbow", "L_front_paw",
    "R_shoulder", "R_elbow", "R_front_paw",
    "L_hip", "L_knee", "L_back_paw",
    "R_hip", "R_knee", "R_back_paw",
]

# Fallback skeleton if model.config.edges is missing.
AP10K_EDGES = [
    (0, 1), (0, 2), (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7),
    (3, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13),
    (4, 14), (14, 15), (15, 16),
]

# Named joint angles as (point_a, vertex, point_c) keypoint indices.
# The angle is measured at the vertex.
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

PER_INSTANCE_COLORS = [
    (0, 165, 255),
    (50, 200, 50),
    (200, 0, 200),
    (0, 255, 255),
    (255, 100, 100),
    (50, 50, 255),
]

DEFAULT_SCORE_THRESHOLD = 0.3


@lru_cache(maxsize=1)
def _device() -> str:
    """Prefer CUDA, but verify it actually works before committing."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            print("[animal_openpose] using device: cuda")
            return "cuda"
        except RuntimeError as e:
            print(f"[animal_openpose] CUDA detected but unusable ({e}); falling back to CPU.")
    print("[animal_openpose] using device: cpu")
    return "cpu"


def _move_to_device(model):
    """Move model to the preferred device; fall back to CPU on CUDA failure (e.g. OOM)."""
    device = _device()
    if device == "cuda":
        try:
            return model.to("cuda"), "cuda"
        except RuntimeError as e:
            print(f"[animal_openpose] failed to move model to CUDA ({e}); using CPU.")
            torch.cuda.empty_cache()
    return model.to("cpu"), "cpu"


@lru_cache(maxsize=1)
def _load_detector():
    proc = AutoProcessor.from_pretrained(DETECTOR_ID)
    model = RTDetrForObjectDetection.from_pretrained(DETECTOR_ID)
    model, device = _move_to_device(model)
    model.eval()
    return proc, model, device


@lru_cache(maxsize=1)
def _load_pose_model():
    proc = AutoProcessor.from_pretrained(POSE_ID)
    model = VitPoseForPoseEstimation.from_pretrained(POSE_ID)
    model, device = _move_to_device(model)
    model.eval()
    return proc, model, device


def _detect_quadrupeds(image: Image.Image, threshold: float = 0.3) -> np.ndarray:
    """Return Nx4 boxes in (x, y, w, h) format for quadrupeds in the image."""
    proc, model, device = _load_detector()

    inputs = proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    id2label = model.config.id2label
    boxes_xywh: list[list[float]] = []
    for label_id, box in zip(results["labels"].tolist(), results["boxes"].tolist()):
        name = id2label.get(label_id, "").lower()
        if name in QUADRUPED_LABELS:
            x1, y1, x2, y2 = box
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    if not boxes_xywh:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(boxes_xywh, dtype=np.float32)


def _estimate_pose(image: Image.Image, boxes_xywh: np.ndarray):
    proc, model, device = _load_pose_model()

    inputs = proc(image, boxes=[boxes_xywh], return_tensors="pt").to(device)
    dataset_index = torch.tensor([AP10K_DATASET_INDEX] * len(boxes_xywh), device=device)

    with torch.no_grad():
        outputs = model(**inputs, dataset_index=dataset_index)

    pose_results = proc.post_process_pose_estimation(outputs, boxes=[boxes_xywh])[0]
    edges = getattr(model.config, "edges", None) or AP10K_EDGES
    return pose_results, edges


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b in degrees, given points a, b, c as (x, y)."""
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def _build_instances(
    pose_results,
    boxes_xywh: np.ndarray,
    score_threshold: float,
) -> list[dict]:
    instances: list[dict] = []
    for pose, box in zip(pose_results, boxes_xywh):
        kpts = np.asarray(pose["keypoints"], dtype=np.float32)
        scores = np.asarray(pose["scores"], dtype=np.float32)

        keypoints_named = []
        for idx, (kpt, sc) in enumerate(zip(kpts, scores)):
            name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"kpt_{idx}"
            keypoints_named.append({
                "name": name,
                "x": float(kpt[0]),
                "y": float(kpt[1]),
                "score": float(sc),
            })

        angles: dict[str, float | None] = {}
        for name, (ai, bi, ci) in ANGLE_TRIPLETS.items():
            if min(scores[ai], scores[bi], scores[ci]) < score_threshold:
                angles[name] = None
            else:
                angles[name] = _angle_deg(kpts[ai], kpts[bi], kpts[ci])

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
    h, w = bgr_image.shape[:2]
    for i, inst in enumerate(instances):
        kpts = np.array([[k["x"], k["y"]] for k in inst["keypoints"]], dtype=np.float32)
        scores = np.array([k["score"] for k in inst["keypoints"]], dtype=np.float32)
        color = PER_INSTANCE_COLORS[i % len(PER_INSTANCE_COLORS)]

        for a, b in edges:
            if a >= len(scores) or b >= len(scores):
                continue
            if scores[a] < score_threshold or scores[b] < score_threshold:
                continue
            x1, y1 = int(kpts[a, 0]), int(kpts[a, 1])
            x2, y2 = int(kpts[b, 0]), int(kpts[b, 1])
            if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                continue
            cv2.line(bgr_image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

        for kpt, sc in zip(kpts, scores):
            if sc < score_threshold:
                continue
            x, y = int(kpt[0]), int(kpt[1])
            if not (0 <= x < w and 0 <= y < h):
                continue
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
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    boxes = _detect_quadrupeds(pil_image)
    if boxes.shape[0] == 0:
        boxes = np.array(
            [[0.0, 0.0, float(pil_image.width), float(pil_image.height)]],
            dtype=np.float32,
        )

    pose_results, edges = _estimate_pose(pil_image, boxes)
    instances = _build_instances(pose_results, boxes, score_threshold)

    return {
        "image_size": {"width": pil_image.width, "height": pil_image.height},
        "edges": [list(e) for e in edges],
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
    if pose_data is None:
        pose_data = extract_pose(image)
    return _draw_pose(image.copy(), pose_data["instances"], pose_data["edges"])

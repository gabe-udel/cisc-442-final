"""Animal pose estimation — RT-DETR detector + SuperAnimal-Quadruped HRNet-W32.

Why this stack? CCL (cranial cruciate ligament) injury manifests as asymmetric
rear-stifle (knee) mechanics. The SuperAnimal-Quadruped model from the
DeepLabCut model zoo predicts 39 keypoints — multiple per limb segment, with
distinct rear-leg "knee" (true stifle) and "thai/thigh" landmarks — which is
the resolution we need for the downstream healthy-vs-injured classifier.
The earlier ViTPose+/AP-10K head only had 17 points (one rear knee per side),
giving the classifier far less to work with.

Pipeline:
    BGR image (numpy)
        -> RT-DETR detects animal bounding boxes
        -> for each box: top-down crop -> HRNet-W32 -> 39 keypoint heatmaps
        -> compute named joint angles (rear stifle / hip emphasized for CCL)
        -> (optionally) draw skeleton onto image

Why we run HRNet ourselves instead of the `deeplabcut` package:
    `deeplabcut` pins numpy<2 and pulls in `tables`/`blosc2`, both of which
    fail to install cleanly on Python 3.13 / aarch64. We use `dlclibrary`
    (lightweight) only to fetch the official PyTorch checkpoint from
    HuggingFace, then load it into a timm HRNet backbone + a 1x1 deconv
    pose head — which is exactly what DLC's PyTorch implementation runs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Quiet HF/transformers warnings before they import.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image

# Detector (animal bounding boxes) — kept from the previous pipeline because
# RT-DETR already works well on COCO animals and avoids re-introducing yet
# another model dependency.
from transformers import AutoProcessor, RTDetrForObjectDetection


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

# RT-DETR: COCO-trained detector. Same as before.
DETECTOR_ID = "PekingU/rtdetr_r50vd_coco_o365"

# SuperAnimal-Quadruped pose checkpoint name (looked up via dlclibrary).
# The .pt file is fetched lazily from HuggingFace to ~/.cache/superanimal/.
POSE_MODEL_NAME = "superanimal_quadruped_hrnet_w32"
POSE_CACHE_DIR = Path.home() / ".cache" / "superanimal"


# ---------------------------------------------------------------------------
# Keypoint metadata — SuperAnimal-Quadruped's 39-point AP-36K-derived schema.
# Order MUST match the model's output channels exactly.
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
    "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
    "neck_base", "neck_end", "throat_base", "throat_end",
    "back_base", "back_end", "back_middle",
    "tail_base", "tail_end",
    "front_left_thai", "front_left_knee", "front_left_paw",
    "front_right_thai", "front_right_knee", "front_right_paw",
    "back_left_paw", "back_left_thai", "back_right_thai",
    "back_left_knee", "back_right_knee", "back_right_paw",
    "belly_bottom", "body_middle_right", "body_middle_left",
]
NAME_TO_INDEX = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

# Quadruped class labels we accept from the COCO-trained detector. Anything
# else is ignored (people, vehicles, etc.).
QUADRUPED_LABELS = {
    "dog", "cat", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

# Skeleton edges defined by name pairs (easier to read than indices) and then
# resolved to indices once at module load.
EDGES_BY_NAME = [
    # Head/face
    ("right_eye", "nose"), ("left_eye", "nose"),
    ("right_eye", "right_earbase"), ("right_earbase", "right_earend"),
    ("left_eye", "left_earbase"), ("left_earbase", "left_earend"),
    ("nose", "upper_jaw"), ("upper_jaw", "lower_jaw"),
    # Spine
    ("nose", "neck_base"),
    ("neck_base", "neck_end"), ("neck_end", "back_base"),
    ("back_base", "back_middle"), ("back_middle", "back_end"),
    ("back_end", "tail_base"), ("tail_base", "tail_end"),
    # Front legs (note SA calls front mid-joint "knee" but it's biologically the elbow)
    ("neck_base", "front_left_thai"),
    ("front_left_thai", "front_left_knee"), ("front_left_knee", "front_left_paw"),
    ("neck_base", "front_right_thai"),
    ("front_right_thai", "front_right_knee"), ("front_right_knee", "front_right_paw"),
    # Back legs (rear "knee" IS the stifle — the joint CCL affects)
    ("back_end", "back_left_thai"),
    ("back_left_thai", "back_left_knee"), ("back_left_knee", "back_left_paw"),
    ("back_end", "back_right_thai"),
    ("back_right_thai", "back_right_knee"), ("back_right_knee", "back_right_paw"),
]
EDGES = [(NAME_TO_INDEX[a], NAME_TO_INDEX[b]) for a, b in EDGES_BY_NAME]

# Joint angles relevant to gait / lameness analysis. Angle is measured at the
# vertex (middle name) using the rays to the other two points. Always in [0, 180].
# Rear stifle and hip are the CCL-diagnostic angles — those should drive the
# downstream classifier the hardest.
ANGLE_TRIPLETS_BY_NAME = {
    # Rear (most diagnostic for CCL)
    "L_stifle":   ("back_left_thai",  "back_left_knee",  "back_left_paw"),
    "R_stifle":   ("back_right_thai", "back_right_knee", "back_right_paw"),
    "L_hip":      ("back_base",       "back_left_thai",  "back_left_knee"),
    "R_hip":      ("back_base",       "back_right_thai", "back_right_knee"),
    # Front (compensatory loading shows up here)
    "L_elbow":    ("front_left_thai",  "front_left_knee",  "front_left_paw"),
    "R_elbow":    ("front_right_thai", "front_right_knee", "front_right_paw"),
    "L_shoulder": ("neck_base", "front_left_thai",  "front_left_knee"),
    "R_shoulder": ("neck_base", "front_right_thai", "front_right_knee"),
    # Spine/posture
    "neck":       ("nose",       "neck_base",   "back_base"),
    "back":       ("neck_base",  "back_middle", "tail_base"),
}
ANGLE_TRIPLETS = {
    name: (NAME_TO_INDEX[a], NAME_TO_INDEX[b], NAME_TO_INDEX[c])
    for name, (a, b, c) in ANGLE_TRIPLETS_BY_NAME.items()
}

# Per-instance overlay colors, BGR. Cycled if more animals than colors.
PER_INSTANCE_COLORS = [
    (0, 165, 255), (50, 200, 50), (200, 0, 200),
    (0, 255, 255), (255, 100, 100), (50, 50, 255),
]

# Heatmap-peak score below which we treat a keypoint as unreliable: skip
# drawing it, and exclude any angle that depends on it.
DEFAULT_SCORE_THRESHOLD = 0.33

# Pose model input size. HRNet-W32 is fully convolutional so it accepts any
# size, but matching the training configuration (256x256 for SuperAnimal HRNet)
# gives the cleanest accuracy.
POSE_INPUT_SIZE = 256
# Heatmaps come out at 1/4 of the input resolution (HRNet's high-res branch).
POSE_HEATMAP_STRIDE = 4

# ImageNet normalization stats — SuperAnimal training uses standard ImageNet
# mean/std on the cropped, resized RGB tensor.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _device() -> str:
    """Prefer CUDA but verify it actually works before committing."""
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
    """Move model to preferred device; fall back to CPU on CUDA OOM/failure."""
    device = _device()
    if device == "cuda":
        try:
            return model.to("cuda"), "cuda"
        except RuntimeError as e:
            print(f"[animal_openpose] failed to move model to CUDA ({e}); using CPU.")
            torch.cuda.empty_cache()
    return model.to("cpu"), "cpu"


# ---------------------------------------------------------------------------
# Detector loader (RT-DETR — unchanged from prior version)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_detector():
    proc = AutoProcessor.from_pretrained(DETECTOR_ID)
    model = RTDetrForObjectDetection.from_pretrained(DETECTOR_ID)
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


# ---------------------------------------------------------------------------
# SuperAnimal-Quadruped pose model: timm HRNet-W32 + 1x1 deconv head
# ---------------------------------------------------------------------------

class SuperAnimalQuadrupedPose(nn.Module):
    """Top-down pose estimator: HRNet-W32 high-res branch + 39-channel head.

    Mirrors DeepLabCut's PyTorch SuperAnimal-Quadruped construction exactly so
    the published checkpoint loads with strict=True (after we strip the
    `backbone.model.` / `heads.bodypart.heatmap_head.` prefixes used by DLC).
    """

    def __init__(self, num_keypoints: int = 39):
        super().__init__()
        # Lazy timm import — avoids a hard dependency for users who only want
        # to read the constants/utilities at the top of this file.
        import timm
        self.backbone = timm.create_model("hrnet_w32", pretrained=False)
        # Disable timm's built-in classification path: setting these to None
        # makes `forward_features` return the raw 4-branch HRNet output (a list
        # of feature maps at decreasing resolutions). We only need the
        # high-resolution one (32 channels at 1/4 input scale).
        self.backbone.incre_modules = None
        self.backbone.downsamp_modules = None
        # The pose head is a single 1x1 transposed conv mapping the 32-channel
        # high-res branch to `num_keypoints` heatmap channels — that's literally
        # all DLC's bodypart head is for SuperAnimal-Quadruped HRNet-W32.
        self.heatmap_head = nn.ConvTranspose2d(32, num_keypoints, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `forward_features` returns a list of 4 feature maps because we nulled
        # out incre_modules/downsamp_modules above. Index 0 is the highest-res
        # branch (32 channels at 1/4 input resolution) — the canonical input
        # for HRNet pose heads.
        yl = self.backbone.forward_features(x)
        return self.heatmap_head(yl[0])


def _ensure_pose_weights() -> Path:
    """Download the SuperAnimal-Quadruped HRNet-W32 .pt if not already cached."""
    POSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    weight_path = POSE_CACHE_DIR / f"{POSE_MODEL_NAME}.pt"
    if weight_path.exists():
        return weight_path
    # dlclibrary handles the HuggingFace fetch + extraction. It writes the .pt
    # into POSE_CACHE_DIR using its internal naming.
    import dlclibrary
    dlclibrary.download_huggingface_model(POSE_MODEL_NAME, target_dir=str(POSE_CACHE_DIR))
    # dlclibrary names the file consistently with the model name.
    if not weight_path.exists():
        # Fallback: glob in case the name differs by version.
        candidates = list(POSE_CACHE_DIR.glob("*.pt"))
        if not candidates:
            raise RuntimeError(f"Pose weights not found in {POSE_CACHE_DIR} after download.")
        weight_path = candidates[0]
    return weight_path


def _strip_dlc_prefixes(state_dict: dict) -> dict:
    """Remap DLC's flat `backbone.model.` / `heads.bodypart.heatmap_head.`
    prefixes to the names our module expects (`backbone.` / `heatmap_head.`).
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.model."):
            out["backbone." + k[len("backbone.model."):]] = v
        elif k.startswith("heads.bodypart.heatmap_head.deconv_layers.0."):
            # Single-layer deconv head -> our `heatmap_head` Conv parameters.
            suffix = k[len("heads.bodypart.heatmap_head.deconv_layers.0."):]
            out["heatmap_head." + suffix] = v
        # Any other prefixes (shouldn't appear for this checkpoint) are dropped.
    return out


@lru_cache(maxsize=1)
def _load_pose_model():
    """Construct, load weights, and move the pose model to the chosen device."""
    weight_path = _ensure_pose_weights()
    print(f"[animal_openpose] loading pose weights: {weight_path.name}")
    ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = SuperAnimalQuadrupedPose(num_keypoints=len(KEYPOINT_NAMES))
    remapped = _strip_dlc_prefixes(state_dict)
    # strict=False because timm's HRNet has a few classifier-side parameters
    # (final_layer, classifier) we don't load — we only care about the backbone
    # branches and our pose head, both of which DO get loaded.
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # Sanity: anything classifier-related is fine to be missing; anything else
    # signals a real architecture mismatch.
    bad_missing = [k for k in missing
                   if not k.startswith(("backbone.classifier", "backbone.final_layer"))]
    if bad_missing or unexpected:
        print(f"[animal_openpose] WARNING — missing: {bad_missing[:3]}{'...' if len(bad_missing) > 3 else ''}, "
              f"unexpected: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    model, device = _move_to_device(model)
    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# Top-down pose inference: crop around bbox -> resize -> network -> decode
# ---------------------------------------------------------------------------

def _square_padded_crop_box(
    bbox_xywh: np.ndarray,
    img_w: int,
    img_h: int,
    padding: float = 1.25,
) -> tuple[int, int, int, int]:
    """Expand a tight bbox into a square with `padding` margin, clamped to image.

    Returns (x0, y0, side, side). HRNet pose accuracy depends on the animal
    being roughly centered with a bit of context around it, hence the 1.25x
    expansion of the longer side.
    """
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * padding
    half = side / 2.0
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = int(round(cx + half)); y1 = int(round(cy + half))
    # Don't shrink the crop on edges — let the resize step deal with off-image
    # pixels (we'll black-pad below). Keeping the box square preserves the
    # spatial calibration we use to map heatmaps back to image coordinates.
    return x0, y0, x1 - x0, y1 - y0


def _crop_with_padding(image_rgb: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """Crop a `side`x`side` region, zero-padding any out-of-image area."""
    h, w = image_rgb.shape[:2]
    out = np.zeros((side, side, 3), dtype=image_rgb.dtype)
    # Compute valid intersection between the requested crop and the image.
    sx0 = max(0, x0); sy0 = max(0, y0)
    sx1 = min(w, x0 + side); sy1 = min(h, y0 + side)
    if sx1 <= sx0 or sy1 <= sy0:
        return out  # crop is entirely outside the image — return blank
    # Where in the output buffer the valid region lands.
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = image_rgb[sy0:sy1, sx0:sx1]
    return out


def _decode_heatmaps(heatmaps: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Turn KxHxW heatmaps into K (x, y) keypoints and K scalar scores.

    We use plain argmax (no Gaussian sub-pixel refinement) for simplicity —
    accurate enough at HRNet's 1/4 stride, and any extra decoding precision
    will be dwarfed by detection-bbox jitter anyway.
    """
    K, H, W = heatmaps.shape
    flat = heatmaps.reshape(K, -1)
    scores, idx = flat.max(dim=1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    # Coordinates are in heatmap space. Caller multiplies by stride to get
    # input-image-pixel coords.
    coords = torch.stack([xs, ys], dim=1).cpu().numpy().astype(np.float32)
    scores_np = scores.cpu().numpy().astype(np.float32)
    # Clip very-negative logits to 0 so threshold comparisons behave intuitively.
    # SuperAnimal heatmaps are roughly in [0, 1+] for confident peaks; below 0
    # never represents a real detection.
    scores_np = np.clip(scores_np, 0.0, None)
    return coords, scores_np


def _estimate_pose_for_boxes(
    image_bgr: np.ndarray,
    boxes_xywh: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run pose inference for each bbox; returns list of (kpts_xy, scores) per animal.

    Coordinates are in the original image's pixel space.
    """
    model, device = _load_pose_model()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]

    # Build a batch of square crops, all resized to POSE_INPUT_SIZE.
    crops_tensor = []
    crop_meta = []  # (x0, y0, side) per box, for mapping heatmaps back to image space
    for box in boxes_xywh:
        x0, y0, side, _ = _square_padded_crop_box(box, W, H)
        if side <= 0:
            # Degenerate bbox — skip (pose inference would fail or return garbage).
            crop_meta.append(None)
            continue
        crop = _crop_with_padding(image_rgb, x0, y0, side)
        # Resize to model input. INTER_AREA is the right choice when downsizing.
        crop_resized = cv2.resize(crop, (POSE_INPUT_SIZE, POSE_INPUT_SIZE),
                                  interpolation=cv2.INTER_AREA)
        # Normalize to [0,1] then ImageNet-standardize, channels-first.
        t = torch.from_numpy(crop_resized).float().div_(255.0)
        t = (t - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
        t = t.permute(2, 0, 1)  # HWC -> CHW
        crops_tensor.append(t)
        crop_meta.append((x0, y0, side))

    if not crops_tensor:
        return []

    batch = torch.stack(crops_tensor).to(device)
    with torch.no_grad():
        heatmaps = model(batch)  # shape (N, 39, H/4, W/4) where H=W=256 → 64x64

    # Decode each heatmap stack and map back to image coords.
    results = []
    valid_idx = 0
    for meta in crop_meta:
        if meta is None:
            results.append((np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float32),
                            np.zeros(len(KEYPOINT_NAMES), dtype=np.float32)))
            continue
        x0, y0, side = meta
        kpts_hm, scores = _decode_heatmaps(heatmaps[valid_idx])
        valid_idx += 1
        # Heatmap coords -> input-image coords (multiply by 4: HRNet stride).
        # Then -> crop coords (multiply by side/POSE_INPUT_SIZE).
        # Then -> original image coords (add x0/y0).
        scale = side / float(POSE_INPUT_SIZE)
        kpts_img = kpts_hm * POSE_HEATMAP_STRIDE * scale
        kpts_img[:, 0] += x0
        kpts_img[:, 1] += y0
        results.append((kpts_img.astype(np.float32), scores))
    return results


# ---------------------------------------------------------------------------
# Geometry + per-instance assembly
# ---------------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b in degrees, given points a, b, c as (x, y)."""
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cosine = max(-1.0, min(1.0, float(np.dot(v1, v2) / (n1 * n2))))
    return float(np.degrees(np.arccos(cosine)))


def _build_instances(
    pose_results: list[tuple[np.ndarray, np.ndarray]],
    boxes_xywh: np.ndarray,
    score_threshold: float,
) -> list[dict]:
    instances: list[dict] = []
    for (kpts, scores), box in zip(pose_results, boxes_xywh):
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


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

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

        # Skeleton lines first so keypoint dots end up on top.
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

        # Per-keypoint dots: red fill + white outline for visibility on any background.
        for kpt, sc in zip(kpts, scores):
            if sc < score_threshold:
                continue
            x, y = int(kpt[0]), int(kpt[1])
            if not (0 <= x < w and 0 <= y < h):
                continue
            cv2.circle(bgr_image, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(bgr_image, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

    return bgr_image


# ---------------------------------------------------------------------------
# Public API — same shape as the prior ViTPose-based version, so main.py is
# unaffected by the model swap.
# ---------------------------------------------------------------------------

def extract_pose(
    image: np.ndarray,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> dict:
    """Run detection + pose estimation and return structured, JSON-serializable data.

    Args:
        image: BGR image array (as produced by cv2.imread).
        score_threshold: per-keypoint heatmap-peak threshold below which joint
            angles are reported as None and keypoints are skipped during drawing.

    Returns:
        A dict shaped exactly like the previous AP-10K version:
            {
                "image_size": {"width": int, "height": int},
                "edges": [(a, b), ...],
                "keypoint_names": [...39 names...],
                "instances": [
                    {
                        "bbox_xywh": [x, y, w, h],
                        "keypoints": [{"name": str, "x": float, "y": float, "score": float}, ...],
                        "angles_deg": {"L_stifle": float | None, ...},
                    },
                    ...
                ],
            }
        On frames with no detected dog/quadruped, "instances" is empty (no fallback
        full-frame pose), so downstream drawing leaves the frame untouched.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    boxes = _detect_quadrupeds(pil_image)
    if boxes.shape[0] == 0:
        return {
            "image_size": {"width": pil_image.width, "height": pil_image.height},
            "edges": [list(e) for e in EDGES],
            "keypoint_names": list(KEYPOINT_NAMES),
            "instances": [],
        }

    pose_results = _estimate_pose_for_boxes(image, boxes)
    instances = _build_instances(pose_results, boxes, score_threshold)

    return {
        "image_size": {"width": pil_image.width, "height": pil_image.height},
        "edges": [list(e) for e in EDGES],
        "keypoint_names": list(KEYPOINT_NAMES),
        "instances": instances,
    }


def get_pose(image: np.ndarray, pose_data: dict | None = None) -> np.ndarray:
    """Draw the quadruped skeleton onto a copy of `image` and return it.

    Pass in `pose_data` (the result of `extract_pose(image)`) to avoid running
    inference twice when you want both the data and the annotated image.
    """
    if pose_data is None:
        pose_data = extract_pose(image)
    return _draw_pose(image.copy(), pose_data["instances"], pose_data["edges"])

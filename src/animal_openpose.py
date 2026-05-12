# animal_openpose.py - dog detection (rt-detr) + pose estimation (hrnet-w32 / superanimal-quadruped)

import os
from pathlib import Path

# silence noisy huggingface warnings before they load
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image

from transformers import AutoProcessor, RTDetrForObjectDetection


# model identifiers
DETECTOR_ID = "PekingU/rtdetr_r50vd_coco_o365"
POSE_MODEL_NAME = "superanimal_quadruped_hrnet_w32"
POSE_CACHE_DIR = Path.home() / ".cache" / "superanimal"


# all 39 keypoints the pose model outputs, in order
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

# quick lookup from name to index
NAME_TO_INDEX = {}
for i, name in enumerate(KEYPOINT_NAMES):
    NAME_TO_INDEX[name] = i

# animal classes we accept from the detector (ignore people, cars, etc.)
QUADRUPED_LABELS = {
    "dog", "cat", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

# skeleton edges defined by name pairs
EDGES_BY_NAME = [
    ("right_eye", "nose"), ("left_eye", "nose"),
    ("right_eye", "right_earbase"), ("right_earbase", "right_earend"),
    ("left_eye", "left_earbase"), ("left_earbase", "left_earend"),
    ("nose", "upper_jaw"), ("upper_jaw", "lower_jaw"),
    ("nose", "neck_base"),
    ("neck_base", "neck_end"), ("neck_end", "back_base"),
    ("back_base", "back_middle"), ("back_middle", "back_end"),
    ("back_end", "tail_base"), ("tail_base", "tail_end"),
    ("neck_base", "front_left_thai"),
    ("front_left_thai", "front_left_knee"), ("front_left_knee", "front_left_paw"),
    ("neck_base", "front_right_thai"),
    ("front_right_thai", "front_right_knee"), ("front_right_knee", "front_right_paw"),
    ("back_end", "back_left_thai"),
    ("back_left_thai", "back_left_knee"), ("back_left_knee", "back_left_paw"),
    ("back_end", "back_right_thai"),
    ("back_right_thai", "back_right_knee"), ("back_right_knee", "back_right_paw"),
]

# convert edge names to index pairs
EDGES = []
for a, b in EDGES_BY_NAME:
    EDGES.append((NAME_TO_INDEX[a], NAME_TO_INDEX[b]))

# joint angle triplets used for ccl analysis (angle is measured at the middle joint)
ANGLE_TRIPLETS_BY_NAME = {
    "L_stifle":   ("back_left_thai",  "back_left_knee",  "back_left_paw"),
    "R_stifle":   ("back_right_thai", "back_right_knee", "back_right_paw"),
    "L_hip":      ("back_base",       "back_left_thai",  "back_left_knee"),
    "R_hip":      ("back_base",       "back_right_thai", "back_right_knee"),
    "L_elbow":    ("front_left_thai",  "front_left_knee",  "front_left_paw"),
    "R_elbow":    ("front_right_thai", "front_right_knee", "front_right_paw"),
    "L_shoulder": ("neck_base", "front_left_thai",  "front_left_knee"),
    "R_shoulder": ("neck_base", "front_right_thai", "front_right_knee"),
    "neck":       ("nose",       "neck_base",   "back_base"),
    "back":       ("neck_base",  "back_middle", "tail_base"),
}

# convert angle triplet names to index triplets
ANGLE_TRIPLETS = {}
for name, (a, b, c) in ANGLE_TRIPLETS_BY_NAME.items():
    ANGLE_TRIPLETS[name] = (NAME_TO_INDEX[a], NAME_TO_INDEX[b], NAME_TO_INDEX[c])

# colors used when drawing multiple detected animals
PER_INSTANCE_COLORS = [
    (0, 165, 255), (50, 200, 50), (200, 0, 200),
    (0, 255, 255), (255, 100, 100), (50, 50, 255),
]

# keypoints below this confidence score are ignored
DEFAULT_SCORE_THRESHOLD = 0.33

# pose model input size (pixels)
POSE_INPUT_SIZE = 256
# hrnet outputs heatmaps at 1/4 of the input size
POSE_HEATMAP_STRIDE = 4

# imagenet normalization values used during training
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# cached models - loaded once when first needed, then reused
_chosen_device = None
_detector_proc = None
_detector_model = None
_detector_device = None
_pose_model_obj = None
_pose_device = None


# pick cpu or gpu depending on what's available
def _get_device():
    global _chosen_device
    if _chosen_device is not None:
        return _chosen_device
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            print("[animal_openpose] using device: cuda")
            _chosen_device = "cuda"
            return _chosen_device
        except RuntimeError as e:
            print(f"[animal_openpose] CUDA detected but unusable ({e}); falling back to CPU.")
    print("[animal_openpose] using device: cpu")
    _chosen_device = "cpu"
    return _chosen_device


# load the rt-detr detector (only runs once)
def _load_detector():
    global _detector_proc, _detector_model, _detector_device
    if _detector_model is not None:
        return _detector_proc, _detector_model, _detector_device
    _detector_proc = AutoProcessor.from_pretrained(DETECTOR_ID)
    raw_model = RTDetrForObjectDetection.from_pretrained(DETECTOR_ID)
    device = _get_device()
    if device == "cuda":
        try:
            raw_model = raw_model.to("cuda")
            _detector_device = "cuda"
        except RuntimeError as e:
            print(f"[animal_openpose] failed to move detector to CUDA ({e}); using CPU.")
            torch.cuda.empty_cache()
            raw_model = raw_model.to("cpu")
            _detector_device = "cpu"
    else:
        raw_model = raw_model.to("cpu")
        _detector_device = "cpu"
    raw_model.eval()
    _detector_model = raw_model
    return _detector_proc, _detector_model, _detector_device


# run the detector and return bounding boxes for any quadrupeds found
def _detect_quadrupeds(image, threshold=0.3):
    proc, model, device = _load_detector()
    inputs = proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([(image.height, image.width)])
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]
    id2label = model.config.id2label
    boxes_xywh = []
    for label_id, box in zip(results["labels"].tolist(), results["boxes"].tolist()):
        name = id2label.get(label_id, "").lower()
        if name in QUADRUPED_LABELS:
            x1, y1, x2, y2 = box
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
    if not boxes_xywh:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(boxes_xywh, dtype=np.float32)


# the hrnet-w32 pose model with a 39-keypoint output head
class SuperAnimalQuadrupedPose(nn.Module):

    def __init__(self, num_keypoints=39):
        super().__init__()
        import timm
        self.backbone = timm.create_model("hrnet_w32", pretrained=False)
        # disable timm's classification path so we get the raw feature maps
        self.backbone.incre_modules = None
        self.backbone.downsamp_modules = None
        # single 1x1 conv that turns 32 feature channels into keypoint heatmaps
        self.heatmap_head = nn.ConvTranspose2d(32, num_keypoints, kernel_size=1, stride=1)

    def forward(self, x):
        # index 0 is the highest-res branch (32 channels at 1/4 input resolution)
        feature_maps = self.backbone.forward_features(x)
        return self.heatmap_head(feature_maps[0])


# download the pose weights from huggingface if not already cached
def _ensure_pose_weights():
    POSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    weight_path = POSE_CACHE_DIR / f"{POSE_MODEL_NAME}.pt"
    if weight_path.exists():
        return weight_path
    import dlclibrary
    dlclibrary.download_huggingface_model(POSE_MODEL_NAME, target_dir=str(POSE_CACHE_DIR))
    if not weight_path.exists():
        candidates = list(POSE_CACHE_DIR.glob("*.pt"))
        if not candidates:
            raise RuntimeError(f"Pose weights not found in {POSE_CACHE_DIR} after download.")
        weight_path = candidates[0]
    return weight_path


# remap dlc checkpoint key names to match our module layout
def _strip_dlc_prefixes(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.model."):
            new_key = "backbone." + k[len("backbone.model."):]
            out[new_key] = v
        elif k.startswith("heads.bodypart.heatmap_head.deconv_layers.0."):
            suffix = k[len("heads.bodypart.heatmap_head.deconv_layers.0."):]
            out["heatmap_head." + suffix] = v
    return out


# load the pose model (only runs once)
def _load_pose_model():
    global _pose_model_obj, _pose_device
    if _pose_model_obj is not None:
        return _pose_model_obj, _pose_device
    weight_path = _ensure_pose_weights()
    print(f"[animal_openpose] loading pose weights: {weight_path.name}")
    ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    model = SuperAnimalQuadrupedPose(num_keypoints=len(KEYPOINT_NAMES))
    remapped = _strip_dlc_prefixes(state_dict)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    bad_missing = []
    for k in missing:
        if not k.startswith(("backbone.classifier", "backbone.final_layer")):
            bad_missing.append(k)
    if bad_missing or unexpected:
        print(f"[animal_openpose] WARNING — missing: {bad_missing[:3]}, unexpected: {unexpected[:3]}")
    device = _get_device()
    if device == "cuda":
        try:
            model = model.to("cuda")
            _pose_device = "cuda"
        except RuntimeError as e:
            print(f"[animal_openpose] failed to move pose model to CUDA ({e}); using CPU.")
            torch.cuda.empty_cache()
            model = model.to("cpu")
            _pose_device = "cpu"
    else:
        model = model.to("cpu")
        _pose_device = "cpu"
    model.eval()
    _pose_model_obj = model
    return _pose_model_obj, _pose_device


# expand a bounding box into a square with padding
def _square_padded_crop_box(bbox_xywh, padding=1.25):
    x, y, w, h = bbox_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * padding
    half = side / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = int(round(cx + half))
    y1 = int(round(cy + half))
    return x0, y0, x1 - x0, y1 - y0


# crop a square region from an image, filling any out-of-bounds area with black
def _crop_with_padding(image_rgb, x0, y0, side):
    h, w = image_rgb.shape[:2]
    out = np.zeros((side, side, 3), dtype=image_rgb.dtype)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x0 + side)
    sy1 = min(h, y0 + side)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    dx0 = sx0 - x0
    dy0 = sy0 - y0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = image_rgb[sy0:sy1, sx0:sx1]
    return out


# find the peak location in each keypoint heatmap
def _decode_heatmaps(heatmaps):
    K, H, W = heatmaps.shape
    flat = heatmaps.reshape(K, -1)
    scores, idx = flat.max(dim=1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    coords = torch.stack([xs, ys], dim=1).cpu().numpy().astype(np.float32)
    scores_np = scores.cpu().numpy().astype(np.float32)
    # clip negatives to zero - scores below zero never represent a real detection
    scores_np = np.clip(scores_np, 0.0, None)
    return coords, scores_np


# run pose inference for each bounding box and return keypoints in image coordinates
def _estimate_pose_for_boxes(image_bgr, boxes_xywh):
    model, device = _load_pose_model()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    crops_tensor = []
    crop_meta = []

    for box in boxes_xywh:
        x0, y0, side, _ = _square_padded_crop_box(box)
        if side <= 0:
            crop_meta.append(None)
            continue
        crop = _crop_with_padding(image_rgb, x0, y0, side)
        crop_resized = cv2.resize(crop, (POSE_INPUT_SIZE, POSE_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        # normalize to [0,1] then apply imagenet standardization
        t = torch.from_numpy(crop_resized).float().div_(255.0)
        t = (t - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
        t = t.permute(2, 0, 1)  # hwc to chw
        crops_tensor.append(t)
        crop_meta.append((x0, y0, side))

    if not crops_tensor:
        return []

    batch = torch.stack(crops_tensor).to(device)
    with torch.no_grad():
        heatmaps = model(batch)

    results = []
    valid_idx = 0
    for meta in crop_meta:
        if meta is None:
            results.append((
                np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float32),
                np.zeros(len(KEYPOINT_NAMES), dtype=np.float32)
            ))
            continue
        x0, y0, side = meta
        kpts_hm, scores = _decode_heatmaps(heatmaps[valid_idx])
        valid_idx += 1
        # map from heatmap coordinates back to original image coordinates
        scale = side / float(POSE_INPUT_SIZE)
        kpts_img = kpts_hm * POSE_HEATMAP_STRIDE * scale
        kpts_img[:, 0] += x0
        kpts_img[:, 1] += y0
        results.append((kpts_img.astype(np.float32), scores))
    return results


# compute the angle at point b (in degrees) given three points a, b, c
def _angle_deg(a, b, c):
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cosine = max(-1.0, min(1.0, float(np.dot(v1, v2) / (n1 * n2))))
    return float(np.degrees(np.arccos(cosine)))


# build a list of instance dicts with keypoints and angles
def _build_instances(pose_results, boxes_xywh, score_threshold):
    instances = []
    for (kpts, scores), box in zip(pose_results, boxes_xywh):
        keypoints_named = []
        for idx, (kpt, sc) in enumerate(zip(kpts, scores)):
            if idx < len(KEYPOINT_NAMES):
                name = KEYPOINT_NAMES[idx]
            else:
                name = f"kpt_{idx}"
            keypoints_named.append({
                "name": name,
                "x": float(kpt[0]),
                "y": float(kpt[1]),
                "score": float(sc),
            })

        angles = {}
        for name, (ai, bi, ci) in ANGLE_TRIPLETS.items():
            if min(scores[ai], scores[bi], scores[ci]) < score_threshold:
                angles[name] = None
            else:
                angles[name] = _angle_deg(kpts[ai], kpts[bi], kpts[ci])

        bbox_list = []
        for v in box.tolist():
            bbox_list.append(float(v))
        instances.append({
            "bbox_xywh": bbox_list,
            "keypoints": keypoints_named,
            "angles_deg": angles,
        })
    return instances


# draw skeleton and keypoints onto an image
def _draw_pose(bgr_image, instances, edges, score_threshold=DEFAULT_SCORE_THRESHOLD):
    h, w = bgr_image.shape[:2]
    for i, inst in enumerate(instances):
        kpts = np.array([[k["x"], k["y"]] for k in inst["keypoints"]], dtype=np.float32)
        scores = np.array([k["score"] for k in inst["keypoints"]], dtype=np.float32)
        color = PER_INSTANCE_COLORS[i % len(PER_INSTANCE_COLORS)]

        # draw skeleton lines first so dots appear on top
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

        # draw a dot at each confident keypoint
        for kpt, sc in zip(kpts, scores):
            if sc < score_threshold:
                continue
            x, y = int(kpt[0]), int(kpt[1])
            if not (0 <= x < w and 0 <= y < h):
                continue
            cv2.circle(bgr_image, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(bgr_image, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

    return bgr_image


# run detection + pose on one frame and return structured results
def extract_pose(image, score_threshold=DEFAULT_SCORE_THRESHOLD):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    boxes = _detect_quadrupeds(pil_image)

    # no animal detected - return empty instances
    # convert edge tuples to lists for json compatibility
    edges_as_lists = []
    for e in EDGES:
        edges_as_lists.append(list(e))

    if boxes.shape[0] == 0:
        return {
            "image_size": {"width": pil_image.width, "height": pil_image.height},
            "edges": edges_as_lists,
            "keypoint_names": list(KEYPOINT_NAMES),
            "instances": [],
        }

    pose_results = _estimate_pose_for_boxes(image, boxes)
    instances = _build_instances(pose_results, boxes, score_threshold)

    return {
        "image_size": {"width": pil_image.width, "height": pil_image.height},
        "edges": edges_as_lists,
        "keypoint_names": list(KEYPOINT_NAMES),
        "instances": instances,
    }


# draw the skeleton onto a copy of the image and return it
def get_pose(image, pose_data=None):
    if pose_data is None:
        pose_data = extract_pose(image)
    return _draw_pose(image.copy(), pose_data["instances"], pose_data["edges"])

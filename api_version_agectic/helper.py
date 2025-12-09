import os

import matplotlib.pyplot as plt
# ====================================================
# FACE DETECTION
# ====================================================
# helper.py
from PIL import Image
from deepface import DeepFace


def is_human_face(image: Image.Image, min_conf=0.5) -> bool:
    """
    Try multiple detectors with a softer threshold.
    Return True on first reasonable detection.
    """
    arr = np.array(image)

    # Order: retinaface (best), mtcnn (robust), opencv (fast), ssd (backup)
    backends = ["retinaface", "mtcnn", "opencv", "ssd"]

    for be in backends:
        try:
            dets = DeepFace.extract_faces(
                arr,
                enforce_detection=False,      # <- don't throw if none
                detector_backend=be
            )
            # Some backends don’t return a confidence; treat any box as detection
            for d in dets or []:
                score = d.get("confidence") or d.get("probability") or 1.0
                fa = d.get("facial_area") or {}
                w = fa.get("w", 0); h = fa.get("h", 0)
                # require a reasonable face box size
                if (w * h) >= 32 * 32 and float(score) >= min_conf:
                    return True
        except Exception as e:
            print(f"[DeepFace {be} warn] {e}")
            continue

    return False



# ====================================================
# COLOR PALETTE MAPS (STATIC)
# ====================================================
season_tone_palettes = {
    'Autumn_Deep':   ['#7C482B', '#A0522D', '#C68642', '#8B5C42', '#5C3317'],
    'Autumn_Soft':   ['#CFA18D', '#B68C75', '#9C7862', '#7F604D', '#6B4E3D'],
    'Autumn_Warm':   ['#D2691E', '#FF8C00', '#FFD700', '#CD853F', '#B8860B'],
    'Spring_Clear':  ['#FFB6C1', '#FFA07A', '#FF69B4', '#FF6347', '#FF4500'],
    'Spring_Light':  ['#FAD6A5', '#FFDAB9', '#FFFACD', '#E6E6FA', '#F0E68C'],
    'Spring_Warm':   ['#F5DEB3', '#FFD700', '#FFA07A', '#FA8072', '#FFE4B5'],
    'Summer_Cool':   ['#ADD8E6', '#87CEFA', '#B0E0E6', '#E0FFFF', '#AFEEEE'],
    'Summer_Light':  ['#F5F5DC', '#FFFACD', '#E6E6FA', '#E0FFFF', '#FFE4E1'],
    'Summer_Soft':   ['#C0C0C0', '#D8BFD8', '#D3D3D3', '#E0FFFF', '#F5F5F5'],
    'Winter_Clear':  ['#0000FF', '#4169E1', '#8A2BE2', '#9370DB', '#6A5ACD'],
    'Winter_Cool':   ['#4682B4', '#5F9EA0', '#708090', '#778899', '#2F4F4F'],
    'Winter_Deep':   ['#191970', '#00008B', '#8B0000', '#2F4F4F', '#4B0082']
}


# ====================================================
# HEX TO RGB VECTOR
# ====================================================
def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return [int(h[i:i+2], 16)/255. for i in (0, 2, 4)]


def palette_to_vector(palette_hex):
    return np.array([hex_to_rgb(c) for c in palette_hex]).flatten()


# ====================================================
# SAVE COLOR PALETTE AS IMAGE
# ====================================================
def save_palette_image(palette_hex, label, input_filename):
    os.makedirs("palette_outputs", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    save_path = os.path.join("palette_outputs", f"{base_name}_palette.png")

    fig, ax = plt.subplots(figsize=(6, 1))
    for i, color in enumerate(palette_hex):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, len(palette_hex))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title(f"Color Palette for {label}", fontsize=12)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return save_path


# ====================================================
# COLOR MOOD ESTIMATION (OPTIONAL)
# ====================================================
def estimate_emotion_from_palette(hex_palette):
    """Estimate overall mood from color psychology."""
    if not hex_palette:
        return {"color_mood": "neutral", "confidence": 0.5, "features": []}

    rgb_vals = np.array([hex_to_rgb(c) for c in hex_palette])
    brightness = np.mean(rgb_vals)
    warmth = np.mean(rgb_vals[:, 0]) - np.mean(rgb_vals[:, 2])

    if brightness > 0.7 and warmth > 0.1:
        mood = "Happy/Energetic"
    elif brightness < 0.4 and warmth < -0.05:
        mood = "Calm/Sad"
    elif warmth > 0.15:
        mood = "Warm/Comforting"
    elif warmth < -0.15:
        mood = "Cool/Serious"
    else:
        mood = "Neutral/Soft"

    confidence = float(np.clip(abs(warmth) + abs(brightness - 0.5), 0, 1))
    return {"color_mood": mood, "confidence": confidence, "features": {"brightness": float(brightness), "warmth": float(warmth)}}

def crop_face_for_emotion(image: Image.Image) -> Image.Image:
    """
    Returns a tight crop of the first detected face for expression analysis.
    Falls back to the original image if no face box is found.
    """
    try:
        arr = np.array(image)
        # fast + robust; don't crash when no face
        dets = DeepFace.extract_faces(arr, enforce_detection=False, detector_backend="opencv")
        if dets:
            # DeepFace returns items with 'facial_area' dict {x,y,w,h}
            fa = dets[0].get("facial_area") or {}
            x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)
            x2, y2 = x + w, y + h
            h_img, w_img = arr.shape[:2]
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if (x2 > x) and (y2 > y):
                face = arr[y:y2, x:x2]
                if face.size > 0:
                    return Image.fromarray(face)
    except Exception as e:
        print(f"[DeepFace crop warning] {e}")
    # Fallback: use original image
    return image

import colorsys

def is_achromatic(hex_palette, min_gray=3, sat_thresh=0.10):
    """
    Returns True if >= min_gray colors have saturation < sat_thresh (achromatic).
    """
    if not hex_palette:
        return False
    cnt = 0
    for h in hex_palette:
        hh = h.lstrip('#')
        r, g, b = int(hh[0:2], 16)/255.0, int(hh[2:4], 16)/255.0, int(hh[4:6], 16)/255.0
        s = colorsys.rgb_to_hsv(r, g, b)[1]
        if s < sat_thresh:
            cnt += 1
    return cnt >= min_gray

import numpy as np
import cv2
from PIL import Image

def _to_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def _ensure_rgb(arr):
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return arr

def build_skin_midtone_mask(rgb_uint8: np.ndarray, skin_mask_uint8: np.ndarray,
                            v_range=(0.15, 0.85), s_min=0.10) -> np.ndarray:
    """
    Keep only skin pixels that are in midtones and not extremely desaturated.
    Returns uint8 mask {0,255} of same HxW.
    """
    assert rgb_uint8.ndim == 3 and rgb_uint8.shape[2] == 3
    H, W, _ = rgb_uint8.shape

    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    Hc, Sc, Vc = hsv[..., 0] / 179.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0

    skin = (skin_mask_uint8 > 0)
    mid = (Vc >= v_range[0]) & (Vc <= v_range[1])
    sat = (Sc >= s_min)

    keep = (skin & mid & sat).astype(np.uint8) * 255
    return keep

def white_balance_gray_world_on_skin(rgb_uint8: np.ndarray, skin_mid_mask_uint8: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance computed only on skin midtone region.
    Scales R,G,B so their means are equal in that region.
    """
    rgb_uint8 = _ensure_rgb(rgb_uint8)
    mask = (skin_mid_mask_uint8 > 0)

    if mask.sum() < 50:   # too few pixels; skip
        return rgb_uint8

    # compute per-channel means in skin midtones
    means = rgb_uint8[mask].reshape(-1, 3).mean(axis=0)  # [R,G,B] in uint8 space
    ref = means.mean() + 1e-6
    gains = ref / (means + 1e-6)                        # 3 gains

    balanced = rgb_uint8.astype(np.float32)
    balanced[..., 0] *= gains[0]
    balanced[..., 1] *= gains[1]
    balanced[..., 2] *= gains[2]
    balanced = _to_uint8(balanced)
    return balanced

def normalize_exposure_on_skin(rgb_uint8: np.ndarray, skin_mid_mask_uint8: np.ndarray,
                               target_V=0.55) -> np.ndarray:
    """
    Adjust global exposure to bring the skin midtone value near target_V (0..1).
    Works in HSV; applies a smooth gamma to the V channel.
    """
    rgb_uint8 = _ensure_rgb(rgb_uint8)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    V = hsv[..., 2] / 255.0

    mask = (skin_mid_mask_uint8 > 0)
    if mask.sum() < 50:
        return rgb_uint8

    current = float(V[mask].mean())
    current = np.clip(current, 1e-3, 0.999)

    # gamma to map current -> target: V_out = V_in^(gamma)
    # solve gamma = log(target)/log(current)
    gamma = np.log(max(target_V, 1e-3)) / np.log(current)
    gamma = float(np.clip(gamma, 0.5, 2.0))  # avoid extreme shifts

    V_out = np.power(V, gamma)
    hsv[..., 2] = np.clip(V_out * 255.0, 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def stabilize_skin_only_rgb(rgb_uint8: np.ndarray, skin_mask_uint8: np.ndarray,
                            v_range=(0.15, 0.85), s_min=0.10, target_V=0.55,
                            blend_alpha: float = 1.0) -> np.ndarray:  # <-- ADD blend_alpha
    """
    Full pipeline:
      1) build midtone mask on skin
      2) gray-world white balance on skin midtones
      3) exposure normalization on skin midtones
      4) Blend with original image using blend_alpha (1.0 = full effect, 0.0 = no effect)
    Returns stabilized RGB uint8 image.
    """
    rgb_uint8 = _ensure_rgb(rgb_uint8)
    skin_mask_uint8 = (skin_mask_uint8 > 0).astype(np.uint8) * 255
    original_float = rgb_uint8.astype(np.float32)

    mid = build_skin_midtone_mask(rgb_uint8, skin_mask_uint8, v_range=v_range, s_min=s_min)
    step1 = white_balance_gray_world_on_skin(rgb_uint8, mid)
    step2 = normalize_exposure_on_skin(step1, mid, target_V=target_V)

    # ----------------------------------------------------
    # NEW: Blending logic (float math is safer for blending)
    # ----------------------------------------------------
    if 0.0 <= blend_alpha < 1.0:
        stabilized_float = step2.astype(np.float32)
        # Final = (1 - alpha) * Original + alpha * Stabilized
        final_float = (1.0 - blend_alpha) * original_float + blend_alpha * stabilized_float
        return _to_uint8(final_float)

    # If alpha is 1.0 (default), return full stabilization
    return step2

def pil_from_rgb(arr_uint8: np.ndarray) -> Image.Image:
    return Image.fromarray(_ensure_rgb(arr_uint8))

import os, json, uuid, datetime
from typing import List, Tuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def make_run_dir(base_dir: str = "runs", stem: str | None = None) -> str:
    """
    Create a unique folder for this request, e.g. runs/2025-10-18_14-03-07_4f2a/
    Return absolute path.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uid = uuid.uuid4().hex[:4]
    name = stem or "request"
    run_dir = os.path.abspath(os.path.join(base_dir, f"{ts}_{uid}_{name}"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_pil(img: Image.Image, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)
    return path

def save_np_mask(mask_uint8: np.ndarray, path: str) -> str:
    """
    Save a single-channel uint8 mask as PNG.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(mask_uint8).save(path)
    return path

def save_side_by_side(images: List[Image.Image], labels: List[str], path: str, cols: int = 2):
    """
    Save a comparison grid (no need to display). Uses matplotlib.
    """
    assert len(images) == len(labels)
    n = len(images)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 4, rows * 4))
    for i, (im, title) in enumerate(zip(images, labels), 1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def write_json(data: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

from PIL import Image, ImageOps

def fix_orientation_to_portrait(img: Image.Image) -> Image.Image:
    """
    1) Apply EXIF orientation (handles phone-rotated photos).
    2) If the image is still landscape (width > height), rotate 90° CCW
       to make it portrait. (Simple, deterministic rule.)
    """
    img = ImageOps.exif_transpose(img)  # respects EXIF Orientation tag
    if img.width > img.height:
        img = img.rotate(90, expand=True)  # CCW to portrait
    return img

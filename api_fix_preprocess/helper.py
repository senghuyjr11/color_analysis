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
    'Autumn_Deep': ['#7C482B', '#A0522D', '#C68642', '#8B5C42', '#5C3317'],
    'Autumn_Soft': ['#CFA18D', '#B68C75', '#9C7862', '#7F604D', '#6B4E3D'],
    'Autumn_Warm': ['#D2691E', '#FF8C00', '#FFD700', '#CD853F', '#B8860B'],

    'Spring_Bright': ['#FFB6C1', '#FFA07A', '#FF69B4', '#FF6347', '#FF4500'],  # was Spring_Clear
    'Spring_Light': ['#FAD6A5', '#FFDAB9', '#FFFACD', '#E6E6FA', '#F0E68C'],
    'Spring_Warm': ['#F5DEB3', '#FFD700', '#FFA07A', '#FA8072', '#FFE4B5'],

    'Summer_Cool': ['#ADD8E6', '#87CEFA', '#B0E0E6', '#E0FFFF', '#AFEEEE'],
    'Summer_Light': ['#F5F5DC', '#FFFACD', '#E6E6FA', '#E0FFFF', '#FFE4E1'],
    'Summer_Soft': ['#C0C0C0', '#D8BFD8', '#D3D3D3', '#E0FFFF', '#F5F5F5'],

    'Winter_Bright': ['#0000FF', '#4169E1', '#8A2BE2', '#9370DB', '#6A5ACD'],  # was Winter_Clear
    'Winter_Cool': ['#4682B4', '#5F9EA0', '#708090', '#778899', '#2F4F4F'],
    'Winter_Deep': ['#191970', '#00008B', '#8B0000', '#2F4F4F', '#4B0082']
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
def save_palette_image(palette_hex, label, input_filename, run_dir: str):  # <-- ADD run_dir argument
    """
    Saves a visualization of the color palette directly into the run directory.
    """
    # Define the save path directly within the run_dir
    pal_name = f"04_palette_{label}.png"
    save_path = os.path.join(run_dir, pal_name)  # <-- Use run_dir

    fig, ax = plt.subplots(figsize=(6, 1))
    for i, color in enumerate(palette_hex):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, len(palette_hex))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title(f"Color Palette for {label}", fontsize=12)

    # Ensure the run_dir exists (though make_run_dir should handle this)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return save_path


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
                            v_range=(0.20, 0.85), s_min=0.12) -> np.ndarray:
    """
    Keep only skin pixels that are in midtones and not extremely desaturated.
    Returns uint8 mask {0,255} of same HxW.
    """
    assert rgb_uint8.ndim == 3 and rgb_uint8.shape[2] == 3
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    Sc = hsv[..., 1] / 255.0
    Vc = hsv[..., 2] / 255.0
    skin = (skin_mask_uint8 > 0)
    mid = (Vc >= v_range[0]) & (Vc <= v_range[1])
    sat = (Sc >= s_min)
    keep = (skin & mid & sat).astype(np.uint8) * 255
    return keep


def white_balance_gray_world_on_skin(rgb_uint8: np.ndarray,
                                     skin_mid_mask_uint8: np.ndarray,
                                     wb_strength: float = 0.4,
                                     gain_min: float = 0.85,
                                     gain_max: float = 1.15) -> np.ndarray:
    """
    Robust gray-world white balance on skin midtones with gentle blending.
    """
    rgb_uint8 = _ensure_rgb(rgb_uint8)
    mask = (skin_mid_mask_uint8 > 0)

    # Too few pixels? Bail.
    if mask.sum() < 500:   # raise from 50 -> 500 to avoid unstable stats
        return rgb_uint8

    skin = rgb_uint8[mask].reshape(-1, 3).astype(np.float32)

    # Trim extremes per channel (robust to makeup, highlights, shadows)
    lo = np.percentile(skin, 10, axis=0)
    hi = np.percentile(skin, 90, axis=0)
    mid = skin[(skin[:,0] >= lo[0]) & (skin[:,0] <= hi[0]) &
               (skin[:,1] >= lo[1]) & (skin[:,1] <= hi[1]) &
               (skin[:,2] >= lo[2]) & (skin[:,2] <= hi[2])]

    if mid.shape[0] < 500:
        return rgb_uint8

    means = mid.mean(axis=0)  # [R,G,B]
    ref = means.mean() + 1e-6
    gains = np.clip(ref / (means + 1e-6), gain_min, gain_max)

    balanced = rgb_uint8.astype(np.float32)
    balanced[..., 0] *= gains[0]
    balanced[..., 1] *= gains[1]
    balanced[..., 2] *= gains[2]
    balanced = _to_uint8(balanced)

    # Gentle blend toward balanced instead of full replacement
    if wb_strength <= 0.0:
        return rgb_uint8
    if wb_strength >= 1.0:
        return balanced
    out = (1.0 - wb_strength) * rgb_uint8.astype(np.float32) + wb_strength * balanced.astype(np.float32)
    return _to_uint8(out)

def normalize_exposure_on_skin(rgb_uint8: np.ndarray,
                               skin_mid_mask_uint8: np.ndarray,
                               target_V_nominal: float = 0.50,
                               adapt_mix: float = 0.7,
                               exp_strength: float = 0.5,
                               gamma_min: float = 0.85,
                               gamma_max: float = 1.15) -> np.ndarray:
    """
    Exposure tweak in HSV that nudges skin midtones toward an adaptive target.
    Uses blending so the effect is gentle.
    """
    rgb_uint8 = _ensure_rgb(rgb_uint8)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    V = hsv[..., 2] / 255.0

    mask = (skin_mid_mask_uint8 > 0)
    if mask.sum() < 500:
        return rgb_uint8

    current = float(np.clip(V[mask].mean(), 1e-3, 0.999))
    # Adaptive target: mostly current, a bit of nominal
    target = float(np.clip(adapt_mix * current + (1.0 - adapt_mix) * target_V_nominal, 1e-3, 0.999))

    gamma = np.log(target) / np.log(current)
    gamma = float(np.clip(gamma, gamma_min, gamma_max))  # much tighter than 0.5–2.0

    V_out = np.power(np.clip(V, 0, 1), gamma)
    hsv_out = hsv.copy()
    hsv_out[..., 2] = np.clip(V_out * 255.0, 0, 255)

    corrected = cv2.cvtColor(hsv_out.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Gentle blend
    if exp_strength <= 0.0:
        return rgb_uint8
    if exp_strength >= 1.0:
        return corrected
    out = (1.0 - exp_strength) * rgb_uint8.astype(np.float32) + exp_strength * corrected.astype(np.float32)
    return _to_uint8(out)

def stabilize_skin_only_rgb(rgb_uint8: np.ndarray,
                            skin_mask_uint8: np.ndarray,
                            wb_strength: float = 0.4,
                            exp_strength: float = 0.5) -> np.ndarray:
    """
    Gentle pipeline:
      1) build midtone mask on skin (stricter)
      2) robust WB on skin midtones (trimmed, clamped, blended)
      3) adaptive exposure normalization (tight gamma, blended)
    """
    rgb_uint8 = _ensure_rgb(rgb_uint8)
    skin_mask_uint8 = (skin_mask_uint8 > 0).astype(np.uint8) * 255

    mid = build_skin_midtone_mask(rgb_uint8, skin_mask_uint8, v_range=(0.20, 0.85), s_min=0.12)
    if (mid > 0).sum() < 500:
        return rgb_uint8

    step1 = white_balance_gray_world_on_skin(rgb_uint8, mid, wb_strength=wb_strength,
                                             gain_min=0.85, gain_max=1.15)
    step2 = normalize_exposure_on_skin(step1, mid,
                                       target_V_nominal=0.50,
                                       adapt_mix=0.7,
                                       exp_strength=exp_strength,
                                       gamma_min=0.85, gamma_max=1.15)
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
    Smart orientation fix:
    1) Apply EXIF orientation (handles phone-rotated photos)
    2) If landscape, detect face orientation and rotate accordingly
    3) Only rotate if face is sideways (width > height in face box)
    """
    from deepface import DeepFace

    # Step 1: Apply EXIF orientation
    img = ImageOps.exif_transpose(img)

    # Step 2: If already portrait, we're done
    if img.width <= img.height:
        return img

    # Step 3: Image is landscape - check face orientation
    try:
        arr = np.array(img)
        dets = DeepFace.extract_faces(arr, enforce_detection=False, detector_backend="opencv")

        if dets:
            fa = dets[0].get("facial_area") or {}
            face_w = fa.get("w", 0)
            face_h = fa.get("h", 0)

            # If face box is taller than wide, face is upright
            # → image should be portrait, rotate it
            if face_h > face_w:
                print("[INFO] Landscape image with upright face detected - rotating to portrait")
                img = img.rotate(90, expand=True)
            else:
                print("[INFO] Landscape image with horizontal face - keeping landscape orientation")
        else:
            # No face detected, keep original orientation
            print("[INFO] No face detected for orientation check - keeping original")

    except Exception as e:
        print(f"[WARNING] Face detection for orientation failed: {e} - keeping original")

    return img

def gentle_brighten_underexposed(rgb_uint8: np.ndarray,
                                 skin_mask_uint8: np.ndarray,
                                 low_thresh: float = 0.42,     # if skin midtone V < this → brighten
                                 target_V: float = 0.48,       # nudge toward this, not all the way
                                 gamma_floor: float = 0.90,    # never stronger than this (mild)
                                 exp_strength: float = 0.35,   # overall blend amount
                                 shadow_power: float = 1.25,   # >1 → more boost to darker pixels
                                 min_pixels: int = 500) -> np.ndarray:
    """
    Keep the image as-is unless skin midtones are dim.
    If dim, apply a gentle, shadows-weighted exposure lift.
    - No white balance; color stays natural.
    - Protects highlights by weighting boost more in dark regions.
    """
    rgb_uint8 = _ensure_rgb(rgb_uint8)
    skin = (skin_mask_uint8 > 0)

    # Build a midtone-only mask on skin to measure 'true' perceived brightness
    mid = build_skin_midtone_mask(rgb_uint8, skin_mask_uint8, v_range=(0.20, 0.85), s_min=0.12)

    if (mid > 0).sum() < min_pixels:
        return rgb_uint8  # not enough skin midtones to judge → do nothing

    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    V = np.clip(hsv[..., 2] / 255.0, 0.0, 1.0)

    current = float(np.clip(V[mid > 0].mean(), 1e-3, 0.999))

    # If not actually dark, keep the original image
    if current >= low_thresh:
        return rgb_uint8

    # Compute a very mild gamma that moves current slightly toward target_V
    target = float(np.clip(target_V, 1e-3, 0.999))
    gamma = np.log(target) / np.log(current)  # <1 brightens, >1 darkens
    gamma = float(np.clip(gamma, gamma_floor, 1.0))  # never darken, never over-brighten

    # Candidate brightened image (HSV gamma on V)
    V_out = np.power(V, gamma)
    hsv_out = hsv.copy()
    hsv_out[..., 2] = np.clip(V_out * 255.0, 0, 255)
    bright_rgb = cv2.cvtColor(hsv_out.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Shadows-weighted per-pixel blend: darker pixels get more of the brightened result
    # weight = exp_strength * (1 - V)^shadow_power, but only inside skin region (optional).
    weight_map = (1.0 - V) ** shadow_power
    # Optional: prefer to boost inside skin and midtones; elsewhere use a lighter weight
    prefer = (skin_mask_uint8 > 0).astype(np.float32)
    weight_map = (0.75 * prefer + 0.25 * (1.0 - prefer)) * weight_map  # mostly inside-skin, still global a bit
    weight_map = np.clip(exp_strength * weight_map, 0.0, 1.0)[..., None]  # HxWx1

    out = (1.0 - weight_map) * rgb_uint8.astype(np.float32) + weight_map * bright_rgb.astype(np.float32)
    return _to_uint8(out)

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from deepface import DeepFace


# ====================================================
# FACE DETECTION
# ====================================================
# helper.py
from PIL import Image
from deepface import DeepFace
import numpy as np

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

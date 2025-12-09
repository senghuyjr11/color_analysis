import io
import os
import random
import sys
from helper import save_pil, save_np_mask, fix_orientation_to_portrait
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Tuple, Dict, Any, List

from helper import stabilize_skin_only_rgb, pil_from_rgb

# ---------------- Determinism ----------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# Face parsing import & paths
# =========================
# NOTE: This line assumes your face_parsing directory is a sibling of the current directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from face_parsing.model import BiSeNet  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "..", "models")
WEIGHTS = os.path.join(PROJECT_ROOT, "..", "face_parsing", "res", "cp", "79999_iter.pth")

# ---------------- Gemini tone-only API ----------------
from gemini_agent import analyze_tone_skin_only  # assumes gemini_agent.py is importable

# =========================
# CONFIGURATION
# =========================
# Threshold for local tone model confidence before falling back to Gemini
T_TONE_FALLBACK_CONF = 0.55

# segmentation transform
to_tensor = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# --------- load face segmenter once (global) ----------
_FACE_SEGMENTER = None


def _get_face_segmenter():
    global _FACE_SEGMENTER
    if _FACE_SEGMENTER is None:
        if not os.path.exists(WEIGHTS):
            print(f"❌ ERROR: BiSeNet weights not found at: {os.path.abspath(WEIGHTS)}")
            raise FileNotFoundError(f"BiSeNet weights not found: {os.path.abspath(WEIGHTS)}")

        print("💡 Attempting to load BiSeNet (Face Parsing) model...")

        model = BiSeNet(n_classes=19)
        model.to(DEVICE)
        model.load_state_dict(torch.load(os.path.abspath(WEIGHTS), map_location=DEVICE))
        model.eval()
        _FACE_SEGMENTER = model

        print("✅ BiSeNet (Face Parsing) model loaded successfully.")

    return _FACE_SEGMENTER


def _morph_close(mask, k=9, iters=1):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=iters)


def _ellipse_fallback_mask(image_pil: Image.Image) -> np.ndarray:
    """If parsing fails, build a soft elliptical mask from the detected face box."""
    import numpy as np
    import cv2
    from deepface import DeepFace

    rgb = np.array(image_pil)
    H, W = rgb.shape[:2]
    dets = []
    try:
        # Use a reliable detector like retinaface for robust detection
        dets = DeepFace.extract_faces(rgb, enforce_detection=False, detector_backend="retinaface")
    except Exception:
        pass

    mask = np.zeros((H, W), np.uint8)
    if not dets:
        return mask  # no face → return empty (caller will handle)

    fa = dets[0].get("facial_area") or {}
    x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)
    x = max(0, x - int(0.08 * w));
    y = max(0, y - int(0.08 * h))
    w = int(w * 1.16);
    h = int(h * 1.12)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x or y2 <= y:
        return mask

    # draw ellipse inside the face bbox
    cx, cy = (x + x2) // 2, (y + y2) // 2
    ax, ay = max(1, (x2 - x) // 2), max(1, (y2 - y) // 2)
    ellipse = np.zeros_like(mask)
    cv2.ellipse(ellipse, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return ellipse


def segment_face(image_pil: Image.Image):
    """
    Robust skin mask using BiSeNet parsing.
    Returns: filled_rgb_pil, skin_mask_uint8
    """
    image_pil = fix_orientation_to_portrait(image_pil)
    model = _get_face_segmenter()

    # 1) run parsing at 512
    original_size = image_pil.size  # (w, h)
    img_resized = image_pil.resize((512, 512), Image.BILINEAR)
    tensor = to_tensor(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)  # 512x512

    # 2) skin classes: skin(1) + neck(14)
    skin_512 = np.isin(parsing, [1, 14]).astype(np.uint8) * 255

    # 3) upsample to original and clean
    skin_mask = cv2.resize(skin_512, original_size, interpolation=cv2.INTER_NEAREST)
    skin_mask = _morph_close(skin_mask, k=11, iters=1)
    skin_mask_3ch = cv2.merge([skin_mask] * 3)

    # 4) sanity check (too small → fallback ellipse)
    area_ratio = float((skin_mask > 0).sum()) / (skin_mask.size + 1e-6)
    if area_ratio < 0.02:  # <2% of pixels → almost certainly a failure
        fb = _ellipse_fallback_mask(image_pil)
        if fb.sum() > skin_mask.sum():  # only replace if better
            skin_mask = fb
            skin_mask_3ch = cv2.merge([skin_mask] * 3)

    # 5) fill non-skin with median skin color
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    skin_pixels = img_cv[skin_mask > 0]
    if skin_pixels.size > 0:
        median_bgr = np.median(skin_pixels, axis=0).astype(np.uint8)
    else:
        median_bgr = np.array([128, 128, 128], dtype=np.uint8)
    bg = np.full_like(img_cv, median_bgr)
    filled = np.where(skin_mask_3ch > 0, img_cv, bg)

    filled_rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    return Image.fromarray(filled_rgb), skin_mask


# ==================================
# CONCEPTUAL LOCAL TONE PREDICTION
# ==================================
def predict_tone_local(stabilized_pil: Image.Image) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    *** REPLACE WITH YOUR ACTUAL LOCAL MODEL INFERENCE CODE (e.g., ConvNeXt) ***
    Simulated local prediction for demonstration.
    Returns: (predicted_season, confidence, top_2_subtones)
    """
    # For demonstration: high confidence 70% of the time, low otherwise
    if random.random() < 0.7:
        season = random.choice(["autumn", "winter"])
        conf = random.uniform(0.70, 0.90)
        subtype = "deep" if season == "winter" else "warm"
    else:
        season = random.choice(["spring", "summer"])
        conf = random.uniform(0.30, 0.50)
        subtype = "light" if season == "spring" else "cool"

    subtones = [(subtype, conf), ("neutral", 1.0 - conf)]
    return season, conf, subtones


def predict_skin_tone(image_pil: Image.Image, run_dir: str | None = None):
    # 1) segmentation & stabilization
    filled_pil, skin_mask = segment_face(image_pil)
    if run_dir:
        save_pil(filled_pil, os.path.join(run_dir, "02_segmented_filled.jpg"))
        save_np_mask(skin_mask, os.path.join(run_dir, "02a_skin_mask.png"))

    # 2) AWB + exposure normalization (light blend)
    filled_np = np.array(filled_pil)
    SLIGHT_BLEND_ALPHA = 0.2
    stabilized_np = stabilize_skin_only_rgb(
        rgb_uint8=filled_np,
        skin_mask_uint8=skin_mask,
        v_range=(0.15, 0.85),
        s_min=0.10,
        target_V=0.55,
        blend_alpha=SLIGHT_BLEND_ALPHA
    )
    stabilized_pil = pil_from_rgb(stabilized_np)
    if run_dir:
        save_pil(stabilized_pil, os.path.join(run_dir, "03_stabilized_for_gemini.jpg"))

    # 3) === Local Prediction FIRST ===
    local_season, local_confidence, local_subtones = predict_tone_local(stabilized_pil)

    tone_out: Dict[str, Any] = {}

    if local_confidence >= T_TONE_FALLBACK_CONF:
        # Use Local Prediction
        print(f"✅ Tone: Using Local Model ({local_season.capitalize()}, Conf: {local_confidence:.2f})")

        # Format the local result to match Gemini's output structure
        tone_out = {
            "season": local_season.lower(),
            "subtype": local_subtones[0][0],
            "confidence": local_confidence,
            "season_confidences": {local_season.lower(): local_confidence},
            "top_2_seasons": [(local_season.lower(), local_confidence), ("neutral", 0.0)],
            "top_2_subtones": local_subtones,
            "hex_palette": ["#7C482B", "#A0522D", "#C68642", "#8B5C42", "#5C3317"],
            # Fallback or look up a standard palette
            "reasoning": f"Local model predicted {local_season.capitalize()} with confidence {local_confidence:.2f}."
        }
    else:
        # 4) Fallback to Gemini
        print(f"⚠️ Tone: Local confidence is low ({local_confidence:.2f}), falling back to Gemini API...")
        buf = io.BytesIO()
        stabilized_pil.save(buf, format="JPEG", quality=85)
        img_bytes = buf.getvalue()
        tone_out = analyze_tone_skin_only(img_bytes)

    # 5) map outputs (common for both local and gemini results)
    main_season = tone_out.get("season", "spring")
    confidence = float(tone_out.get("confidence", 0.0))
    top_2_subtones = []
    for it in tone_out.get("top_2_subtones", []) or []:
        if isinstance(it, (list, tuple)) and len(it) == 2:
            top_2_subtones.append((str(it[0]), float(it[1])))
    sc = tone_out.get("season_confidences", {}) or {}
    season_confidences = {str(k): float(v) for k, v in sc.items()}
    top_2_seasons = []
    for it in tone_out.get("top_2_seasons", []) or []:
        if isinstance(it, (list, tuple)) and len(it) == 2:
            top_2_seasons.append((str(it[0]), float(it[1])))

    gemini_out = {
        "hex_palette": tone_out.get("hex_palette"),
        "reasoning": tone_out.get("reasoning"),
    }

    return (
        main_season, confidence, top_2_subtones, season_confidences, top_2_seasons, gemini_out,
        filled_pil, stabilized_pil, skin_mask
    )
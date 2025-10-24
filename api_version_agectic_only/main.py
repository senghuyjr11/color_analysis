from helper import make_run_dir, save_pil, save_side_by_side, write_json
from helper import fix_orientation_to_portrait
import io
import os
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from helper import (
    season_tone_palettes,
    palette_to_vector,
    save_palette_image,
    is_achromatic,
)
# Import the new combined function
from predictor import analyze_full

# ==============================
# CONFIG & FASTAPI APP (remain the same)
# ==============================
client_initialized = False

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI(title="Skin Tone Analysis API (Gemini-powered)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# ROOT
# ==============================
@app.get("/")
def root():
    global client_initialized
    client_initialized = True
    return {"message": "Ready. Using Gemini for combined tone and emotion analysis."}


# ==============================
# COMBINED ANALYSIS ENDPOINT
# ==============================
@app.post("/analyze_full")
async def analyze_full_endpoint(file: UploadFile = File(...)):
    if not client_initialized:
        return {"error": "Call GET / first to initialize."}

    try:
        # ---- set up run dir (use original filename stem for readability)
        stem = os.path.splitext(file.filename or "image")[0]
        run_dir = make_run_dir(base_dir="outputs", stem=stem)

        # ---- read & save original
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = fix_orientation_to_portrait(image)
        save_pil(image, os.path.join(run_dir, "01_original.jpg"))

        # ---- Run Combined Analysis
        (
            main_season, confidence, subtones, season_confidences, top_2_seasons, gemini_tone_out,
            seg_filled_pil, stabilized_pil, skin_mask,
            emotion_result
        ) = analyze_full(image, run_dir=run_dir)

        # --------------------
        # TONE-SPECIFIC LOGIC (Copied from old /skin_tone endpoint)
        # --------------------
        warnings = emotion_result["warnings"]  # Start combined warnings with emotion warnings

        # Normalize tone confidences and find predicted label
        if season_confidences:
            total = sum(season_confidences.values()) or 1.0
            season_confidences = {k: float(v) / total for k, v in season_confidences.items()}
        if season_confidences:
            _sorted = sorted(season_confidences.items(), key=lambda kv: kv[1], reverse=True)
            top_2_seasons = [(k, v) for k, v in _sorted[:2]]

        predicted_label = f"{main_season.capitalize()}_{subtones[0][0].capitalize()}"

        # Palette choose + save palette image
        palette_hex = gemini_tone_out.get("hex_palette") if isinstance(gemini_tone_out, dict) else None
        if not palette_hex:
            palette_hex = season_tone_palettes.get(predicted_label, [])
        if is_achromatic(palette_hex):
            fallback_key = f"{main_season.capitalize()}_{subtones[0][0].capitalize()}"
            palette_hex = season_tone_palettes.get(fallback_key, palette_hex)
            warnings.append("Palette was achromatic; used seasonal fallback.")

        palette_vec = palette_to_vector(palette_hex)

        # --- MODIFIED PALETTE SAVING LOGIC ---
        # 1. Pass the run_dir to save the palette image directly inside it.
        palette_path = save_palette_image(palette_hex, predicted_label, file.filename, run_dir=run_dir)


        # Save a side-by-side collage
        compare_path = os.path.join(run_dir, "05_comparison_grid.jpg")
        save_side_by_side(
            images=[image, seg_filled_pil, stabilized_pil],
            labels=["Original", "Segmented (Filled)", "Stabilized (WB+Exposure)"],
            path=compare_path,
            cols=3
        )

        # --------------------
        # COMBINED SUMMARY JSON
        # --------------------
        summary = {
            "tone": {
                "season": main_season,
                "subtype": subtones[0][0],
                "confidence": confidence,
                "season_confidences": season_confidences,
                "top_2_seasons": top_2_seasons,
                "top_2_subtones": subtones,
                "predicted_label": predicted_label,
                "reasoning_tone": (gemini_tone_out.get("reasoning") if isinstance(gemini_tone_out, dict) else None)
            },
            "palette": {
                "hex": palette_hex,
                "rgb_vector_15d": palette_vec.tolist(),
                "image_path": palette_path,
            },
            "emotion": {
                "label": emotion_result["emotion"],
                "confidence": emotion_result["confidence"],
                "reasoning_emotion": emotion_result["reasoning"],
            },
            "warnings": warnings,
            "artifacts": {
                "original": os.path.join(run_dir, "01_original.jpg"),
                "segmented_filled": os.path.join(run_dir, "02_segmented_filled.jpg"),
                "skin_mask": os.path.join(run_dir, "02a_skin_mask.png"),
                "stabilized_for_gemini": os.path.join(run_dir, "03_stabilized_for_gemini.jpg"),
                "cropped_face_for_emotion": os.path.join(run_dir, "04_cropped_face_for_emotion.jpg"),
                "palette_image": palette_path,
                "comparison_grid": compare_path,
                "run_dir": run_dir
            }
        }
        summary_path = os.path.join(run_dir, "result_summary_combined.json")
        write_json(summary, summary_path)

        # ---- return API response
        return summary

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}
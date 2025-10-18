# gemini_agent.py
import os, json, hashlib
from typing import Dict, Any, List
from google import genai
from google.genai import types
from dotenv import load_dotenv

_CLIENT = None

def _client() -> genai.Client:
    global _CLIENT
    if _CLIENT: return _CLIENT
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT

# ----------------------------
# Common settings
# ----------------------------
SEVEN = ["surprise","fear","disgust","happy","sad","angry","neutral"]

# ====== TONE (skin-only) ======
_TONE_PROMPT = """
You analyze SKIN TONE ONLY. Non-skin areas are masked/filled; ignore them completely.
Return tone and a 5-color palette. Do NOT return any emotion fields.

Fields:
- season: one of ["spring","summer","autumn","winter"]
- subtype: one of ["warm","cool","deep","bright","soft","light"]
- confidence: number 0..1 (confidence in season)
- season_confidences: object with keys spring, summer, autumn, winter (0..1; not necessarily normalized)
- top_2_seasons: array of [season, score]
- top_2_subtones: array of [subtype, score]
- hex_palette: exactly 5 HEX strings (e.g. "#AABBCC")
- reasoning: 1–2 sentences (why these tones)
"""

_TONE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "season": types.Schema(type=types.Type.STRING, enum=["spring","summer","autumn","winter"]),
        "subtype": types.Schema(type=types.Type.STRING, enum=["warm","cool","deep","bright","soft","light"]),
        "confidence": types.Schema(type=types.Type.NUMBER),
        "season_confidences": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "spring": types.Schema(type=types.Type.NUMBER),
                "summer": types.Schema(type=types.Type.NUMBER),
                "autumn": types.Schema(type=types.Type.NUMBER),
                "winter": types.Schema(type=types.Type.NUMBER),
            },
            required=["spring","summer","autumn","winter"]
        ),
        "top_2_seasons": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(any_of=[types.Schema(type=types.Type.STRING), types.Schema(type=types.Type.NUMBER)]),
                min_items=2, max_items=2
            )
        ),
        "top_2_subtones": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(any_of=[types.Schema(type=types.Type.STRING), types.Schema(type=types.Type.NUMBER)]),
                min_items=2, max_items=2
            )
        ),
        "hex_palette": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), min_items=5, max_items=5),
        "reasoning": types.Schema(type=types.Type.STRING),
    },
    required=["season","subtype","confidence","season_confidences","top_2_seasons","top_2_subtones","hex_palette","reasoning"]
)

def analyze_tone_skin_only(segmented_jpeg_bytes: bytes, model_name: str = "gemini-2.5-flash") -> dict:
    c = _client()
    img = types.Part.from_bytes(data=segmented_jpeg_bytes, mime_type="image/jpeg")
    resp = c.models.generate_content(
        model=model_name,
        contents=[img, _TONE_PROMPT],
        config={"temperature": 0.0, "response_mime_type": "application/json", "response_schema": _TONE_SCHEMA},
    )

    text = getattr(resp, "text", "") or ""
    if not text:
        raise RuntimeError("Empty tone response from Gemini")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON tone response: {text[:200]}")

# ====== EMOTION (face crop) ======
_EMO_PROMPT = f"""
You classify FACIAL EXPRESSION ONLY from an unmasked face crop.
Use exactly these labels (no others): {SEVEN}.
Return:
- label: one of {SEVEN}
- confidence: 0..1
Keep temperature 0 behavior; do not randomize.
"""

_EMO_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "label": types.Schema(type=types.Type.STRING, enum=SEVEN),
        "confidence": types.Schema(type=types.Type.NUMBER),
    },
    required=["label","confidence"]
)

def infer_expression_facecrop(face_jpeg_bytes: bytes, model_name: str = "gemini-2.5-flash") -> dict:
    c = _client()
    img = types.Part.from_bytes(face_jpeg_bytes, mime_type="image/jpeg")
    resp = c.models.generate_content(
        model=model_name,
        contents=[img, _EMO_PROMPT],
        config={"temperature": 0.0, "response_mime_type": "application/json", "response_schema": _EMO_SCHEMA},
    )
    text = getattr(resp, "text", "") or ""
    if not text:
        raise RuntimeError("Empty emotion response from Gemini")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON emotion response: {text[:200]}")

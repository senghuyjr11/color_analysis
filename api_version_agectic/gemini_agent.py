import os, json, hashlib
from typing import Dict, Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

# -------- Lazy client init (avoids import-time crashes) --------
_client = None
def _get_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client
    load_dotenv()  # load .env at call-time
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Put it in .env or OS env.")
    _client = genai.Client(api_key=api_key)
    return _client

# -------- Deterministic prompt --------
PROMPT = """
You are a deterministic personal color and emotion analyst.
Analyze the given face image and return a JSON strictly following the schema.

Important:
- Non-skin regions are masked or filled; IGNORE any black/gray/filled areas.
- Base your season/subtype and palette ONLY on SKIN undertone cues.
- Be consistent and do not randomize outputs.

Tasks:
1) Identify the most probable SEASON and SUBTYPE of personal color tone.
2) Describe the EMOTION reflected by the overall image (expression + color mood)
   from: [happy, calm, sad, energetic, neutral, serious, surprised].
3) Provide concise REASONING (1–2 sentences) explaining both color and emotion.

Return only JSON that matches the schema exactly.
Use numeric confidences in the 0–1 range.
Set "rgb_vector_15d" to null (server computes it locally).
"""

# -------- Strict schema for JSON output --------
SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "predicted_label": types.Schema(type=types.Type.STRING),
        "season": types.Schema(type=types.Type.STRING, enum=["spring","summer","autumn","winter"]),
        "subtype": types.Schema(type=types.Type.STRING, enum=["warm","cool","deep","bright","soft","light"]),
        "confidence": types.Schema(type=types.Type.NUMBER),
        "top_2_subtones": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(any_of=[types.Schema(type=types.Type.STRING), types.Schema(type=types.Type.NUMBER)]),
                min_items=2, max_items=2
            )
        ),
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
        "hex_palette": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), min_items=5, max_items=5),
        "rgb_vector_15d": types.Schema(any_of=[
            types.Schema(type=types.Type.NULL),
            types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), min_items=15, max_items=15),
        ]),
        "palette_image_path": types.Schema(type=types.Type.STRING),
        "reasoning": types.Schema(type=types.Type.STRING),
        "emotion": types.Schema(type=types.Type.STRING, enum=["happy","calm","sad","energetic","neutral","serious","surprised"]),
        "emotion_confidence": types.Schema(type=types.Type.NUMBER),
    },
    required=[
        "predicted_label","season","subtype","confidence",
        "top_2_subtones","season_confidences","top_2_seasons",
        "hex_palette","rgb_vector_15d","palette_image_path",
        "reasoning","emotion","emotion_confidence"
    ],
)

_CACHE_FILE = "gemini_cache.json"
def _load_cache() -> Dict[str, Any]:
    if not os.path.exists(_CACHE_FILE):
        return {}
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def analyze_with_gemini(segmented_face_bytes: bytes, model_name: str = "gemini-2.5-flash") -> dict:
    """Deterministic Gemini call (temperature=0) with schema + caching."""
    # cache by bytes of the segmented image
    key = hashlib.md5(segmented_face_bytes).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key]

    client = _get_client()
    image_part = types.Part.from_bytes(data=segmented_face_bytes, mime_type="image/jpeg")

    resp = client.models.generate_content(
        model=model_name,
        contents=[image_part, PROMPT],
        config={
            "temperature": 0.0,  # deterministic
            "response_mime_type": "application/json",
            "response_schema": SCHEMA
        }
    )

    raw = getattr(resp, "text", "") or ""
    if not raw:
        # attempt salvage (rare)
        try:
            cand = resp.candidates[0]
            parts = getattr(cand, "content", {}).parts if hasattr(cand, "content") else []
            if parts and hasattr(parts[0], "text"):
                raw = parts[0].text
        except Exception:
            pass

    if not raw:
        raise RuntimeError("Gemini returned empty response or was safety-blocked.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Gemini did not return valid JSON.\nRaw:\n{raw}")

    cache[key] = data
    _save_cache(cache)
    return data

# gemini_agent.py
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

_CLIENT = None


def _client() -> genai.Client:
    """Initialize and return singleton Gemini client."""
    global _CLIENT
    if _CLIENT:
        return _CLIENT
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


# ----------------------------
# Common settings
# ----------------------------
SEVEN = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# ====== TONE (skin-only) ======
_TONE_PROMPT = """
You analyze SKIN TONE ONLY. Non-skin areas are masked/filled; ignore them completely.
Return tone and a 5-color palette.

Fields:
- season: one of ["spring","summer","autumn","winter"]
- subtype: one of ["warm","cool","deep","bright","soft","light"]
- confidence: number 0..1 (confidence in season)
- season_confidences: object with keys spring, summer, autumn, winter (0..1; not necessarily normalized)
- top_2_seasons: array of [season, score]
- top_2_subtones: array of [subtype, score]
- hex_palette: exactly 5 HEX strings (e.g. "#AABBCC")
- reasoning: 1–2 sentences (why these tones
"""

_TONE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "season": types.Schema(type=types.Type.STRING, enum=["spring", "summer", "autumn", "winter"]),
        "subtype": types.Schema(type=types.Type.STRING, enum=["warm", "cool", "deep", "bright", "soft", "light"]),
        "confidence": types.Schema(type=types.Type.NUMBER),
        "season_confidences": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "spring": types.Schema(type=types.Type.NUMBER),
                "summer": types.Schema(type=types.Type.NUMBER),
                "autumn": types.Schema(type=types.Type.NUMBER),
                "winter": types.Schema(type=types.Type.NUMBER),
            },
            required=["spring", "summer", "autumn", "winter"]
        ),
        "top_2_seasons": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(any_of=[types.Schema(type=types.Type.STRING), types.Schema(type=types.Type.NUMBER)]),
                min_items=2,
                max_items=2
            )
        ),
        "top_2_subtones": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(any_of=[types.Schema(type=types.Type.STRING), types.Schema(type=types.Type.NUMBER)]),
                min_items=2,
                max_items=2
            )
        ),
        "hex_palette": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            min_items=5,
            max_items=5
        ),
        "reasoning": types.Schema(type=types.Type.STRING),
    },
    required=["season", "subtype", "confidence", "season_confidences", "top_2_seasons", "top_2_subtones", "hex_palette",
              "reasoning"]
)


def analyze_tone_skin_only(segmented_jpeg_bytes: bytes, model_name: str = "gemini-2.0-flash-exp") -> dict:
    """
    Analyze skin tone from segmented/masked image using Gemini.

    Args:
        segmented_jpeg_bytes: JPEG bytes of skin-segmented image
        model_name: Gemini model to use

    Returns:
        dict with season, subtype, confidence, palette, etc.
    """
    c = _client()

    # Create Part with inline_data
    img = types.Part(
        inline_data=types.Blob(
            mime_type="image/jpeg",
            data=segmented_jpeg_bytes
        )
    )

    resp = c.models.generate_content(
        model=model_name,
        contents=[img, _TONE_PROMPT],
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": _TONE_SCHEMA
        },
    )

    text = getattr(resp, "text", "") or ""
    if not text:
        raise RuntimeError("Empty tone response from Gemini")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON tone response: {text[:200]}")

# ----------------------------
# Common settings
# ----------------------------
SEVEN = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# ====== EMOTION (face-only) ======
_EMOTION_PROMPT = """
Analyze the emotion expressed by the person in the image.
The input image is a tight crop of the face.

**Instructions for Occlusion:** Pay attention to non-occluded features. If the person is wearing **glasses**, focus on the mouth, jawline, and the visible shape of the eyes and eyebrows around the frames. If parts of the face are covered by hair or a scarf, focus on the visible expressions.

Return the primary emotion and its confidence.

Fields:
- emotion: one of ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
- confidence: number 0..1
- reasoning: 1–2 sentences explaining the classification, **mentioning any occlusions (e.g., glasses) that made the analysis challenging or specific features used.**
"""

_EMOTION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "emotion": types.Schema(type=types.Type.STRING, enum=SEVEN),
        "confidence": types.Schema(type=types.Type.NUMBER),
        "reasoning": types.Schema(type=types.Type.STRING),
    },
    required=["emotion", "confidence", "reasoning"]
)

def classify_emotion_from_face(face_jpeg_bytes: bytes, model_name: str = "gemini-2.0-flash-exp") -> dict:
    """
    Classify emotion from a face crop image using Gemini.

    Args:
        face_jpeg_bytes: JPEG bytes of the face-cropped image.
        model_name: Gemini model to use.

    Returns:
        dict with emotion, confidence, and reasoning.
    """
    c = _client()

    # Create Part with inline_data
    img = types.Part(
        inline_data=types.Blob(
            mime_type="image/jpeg",
            data=face_jpeg_bytes
        )
    )

    resp = c.models.generate_content(
        model=model_name,
        contents=[img, _EMOTION_PROMPT],
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": _EMOTION_SCHEMA
        },
    )

    text = getattr(resp, "text", "") or ""
    if not text:
        raise RuntimeError("Empty emotion response from Gemini")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON emotion response: {text[:200]}")
from PIL import Image
from deepface import DeepFace
import os
import numpy as np
import matplotlib.pyplot as plt

def is_human_face(image: Image.Image) -> bool:
    try:
        img_array = np.array(image)
        results = DeepFace.extract_faces(img_array, enforce_detection=True)
        print(f"[DeepFace] Detected {len(results)} face(s).")
        return len(results) > 0
    except Exception as e:
        print(f"[DeepFace ERROR] {e}")
        return False

# === Season tone palettes (12-class) ===
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

# === Convert HEX to RGB vector ===
def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return [int(h[i:i+2], 16)/255. for i in (0, 2, 4)]

def palette_to_vector(palette_hex):
    return np.array([hex_to_rgb(c) for c in palette_hex]).flatten()

# === Save palette to image file ===
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

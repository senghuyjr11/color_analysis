import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import matplotlib.pyplot as plt
import os


# === Config ===
image_path = "clustered_seasons_kmeans/Winter/Deep/14311.png"
output_folder = "palette_outputs"
os.makedirs(output_folder, exist_ok=True)
model_path = "model/color_convnext_fafl.pth"
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Fixed Class Names Based on Your Dataset (12 classes) ===
class_names = [
    'Autumn_Deep', 'Autumn_Soft', 'Autumn_Warm',
    'Spring_Clear', 'Spring_Light', 'Spring_Warm',
    'Summer_Cool', 'Summer_Light', 'Summer_Soft',
    'Winter_Clear', 'Winter_Cool', 'Winter_Deep'
]

label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# === Color Palettes per Label ===
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

# === Utility Functions ===
def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return [int(h[i:i+2], 16)/255. for i in (0, 2, 4)]

def palette_to_vector(palette_hex):
    return np.array([hex_to_rgb(c) for c in palette_hex]).flatten()


def show_and_save_palette(palette_hex, label, input_path, output_folder):
    fig, ax = plt.subplots(figsize=(6, 1))
    for i, color in enumerate(palette_hex):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))

    ax.set_xlim(0, len(palette_hex))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title(f"Color Palette for {label}", fontsize=12)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_folder, f"{base_name}_palette.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Palette image saved to:", out_path)



# === Load Image and Predict ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# === Load Model ===
model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
in_features = model.classifier[2].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Flatten(1),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(in_features, len(class_names))
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === Prediction ===
with torch.no_grad():
    output = model(image_tensor)
    pred_idx = output.argmax(dim=1).item()
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]

# === Get Palette ===
palette_hex = season_tone_palettes[predicted_label]
palette_vec = palette_to_vector(palette_hex)

# === Output ===
print("Predicted Label:", predicted_label)
print("HEX Palette:", palette_hex)
print("RGB Vector (15D):", palette_vec)

# === Visualize Palette ===
show_and_save_palette(palette_hex, predicted_label, image_path, output_folder)



# evaluate_season.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_large
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ======================
# CONFIG
# ======================
DATASET_DIR = "merge_dataset_cropped_segmented"
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")
MODEL_PATH = os.path.join(DATASET_DIR, "convnext_season_large.pth")

# Fix the canonical order to match the paper
SEASON_LABELS = ["autumn", "spring", "summer", "winter"]

BATCH_SIZE = 32
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# DATA
# ======================
df_test = pd.read_csv(TEST_CSV)
df_test["season"] = df_test["label"].apply(lambda x: x.split("_")[0])

class SeasonDataset(Dataset):
    def __init__(self, df, root_dir, transform, label_encoder):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.le = label_encoder
        self.df["label_encoded"] = self.le.transform(self.df["season"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filename"]).replace("\\", "/")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(row["label_encoded"], dtype=torch.long)

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def main():
    print("\n🔍 Evaluating MAIN Season Model (4 classes)\n")

    # Label encoder with fixed order
    le = LabelEncoder()
    le.fit(SEASON_LABELS)

    ds = SeasonDataset(df_test, DATASET_DIR, test_tf, le)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Load model
    model = convnext_large(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing (Season)"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Accuracy
    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100.0
    print(f"✅ Test Accuracy (Season 4-class): {acc:.2f}%")

    # ---- Table 2 CSV ----
    target_names = le.classes_.tolist()
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    rep_df = pd.DataFrame(report).T
    out_csv = os.path.join(DATASET_DIR, "table2_main_season_classification_report.csv")
    rep_df.to_csv(out_csv, float_format="%.4f")
    print(f"💾 Saved Table 2 CSV → {out_csv}")

    # ---- Figure 5 (Confusion Matrix) ----
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(target_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix – Main Season Classification")
    fig_path = os.path.join(DATASET_DIR, "figure5_confusion_matrix.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🖼️ Saved Figure 5 → {fig_path}")

if __name__ == "__main__":
    main()

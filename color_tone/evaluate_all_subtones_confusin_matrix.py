# evaluate_subtone.py
import os
import json
import argparse
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
BATCH_SIZE = 32
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# DATA
# ======================
df_test = pd.read_csv(TEST_CSV)
df_test['season'] = df_test['label'].apply(lambda x: x.split('_')[0])
df_test['subtone'] = df_test['label'].apply(lambda x: x.split('_')[1])

class SubtoneDataset(Dataset):
    def __init__(self, df, root_dir, transform, label_encoder):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.le = label_encoder
        self.df['label_encoded'] = self.le.transform(self.df['subtone'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename']).replace("\\", "/")
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(row['label_encoded'], dtype=torch.long)

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def main(season):
    season = season.lower()
    print(f"\n🔍 Evaluating Subtone Model for {season.upper()}")

    season_df = df_test[df_test['season'] == season].reset_index(drop=True)

    # Label encoder from the test subset of this season
    le = LabelEncoder()
    le.fit(season_df['subtone'])
    class_names = list(le.classes_)
    print(f"Classes for {season.upper()}: {class_names}")

    ds = SubtoneDataset(season_df, DATASET_DIR, test_tf, le)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Load model
    model_path = os.path.join(DATASET_DIR, f"convnext_subtone_{season}.pth")
    model = convnext_large(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Testing ({season} subtone)"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = (np.array(all_preds) == np.array(all_labels)).mean() * 100.0
    print(f"✅ Test Accuracy for {season.upper()} subtone: {acc:.2f}%")

    # ---- Classification report CSV ----
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    rep_df = pd.DataFrame(report).T
    out_csv = os.path.join(DATASET_DIR, f"classification_report_subtone_{season}.csv")
    rep_df.to_csv(out_csv, float_format="%.4f")
    print(f"💾 Saved {season.upper()} subtone report → {out_csv}")

    # ---- Confusion matrix image (per-season) ----
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{season.capitalize()} Subtone – Confusion Matrix")
    fig_path = os.path.join(DATASET_DIR, f"{season}_subtone_confusion_matrix.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🖼️ Saved {season.upper()} subtone confusion matrix → {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True,
                        choices=["spring", "summer", "autumn", "winter"],
                        help="Which season's subtone model to evaluate.")
    args = parser.parse_args()
    main(args.season)

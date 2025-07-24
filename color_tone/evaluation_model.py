import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# === Config ===
DATA_DIR = './ORIGINAL_RGB_NOT_PROCESSED'
TEST_CSV = f'{DATA_DIR}/cleaned_test.csv'
MODEL_PATH = 'ORIGINAL_RGB_NOT_PROCESSED/best_vit_model.pth'
NUM_CLASSES = 12
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Label encoding (must match training) ===
def encode_labels(labels):
    classes = sorted(list(set(labels)))
    label2idx = {cls: idx for idx, cls in enumerate(classes)}
    idx2label = {idx: cls for cls, idx in label2idx.items()}
    encoded = [label2idx[label] for label in labels]
    return encoded, label2idx, idx2label

# === Dataset ===
class TestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.labels, self.label2idx, self.idx2label = encode_labels(self.df['label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = f"{DATA_DIR}/{self.df.iloc[idx]['path']}"
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# === Transforms ===
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# === Load Data ===
test_dataset = TestDataset(TEST_CSV, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
idx2label = test_dataset.idx2label

# === Load Model ===
model = vit_b_16(weights=None)
model.heads.head = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),
    torch.nn.Linear(model.heads.head.in_features, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# === Evaluate ===
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# === Report ===
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(NUM_CLASSES)]))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[idx2label[i] for i in range(NUM_CLASSES)],
            yticklabels=[idx2label[i] for i in range(NUM_CLASSES)])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

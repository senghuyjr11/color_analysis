#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model import BiSeNet

import torch
import pandas as pd
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


def extract_skin_region(image, parsing_anno, save_path='skin_only_hsv.png'):
    SKIN_LABEL = 1

    # Resize parsing mask to match image
    parsing_anno = cv2.resize(parsing_anno.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)

    # Create skin mask
    skin_mask = (parsing_anno == SKIN_LABEL).astype(np.uint8) * 255

    # Convert PIL image to NumPy (RGB)
    image_np = np.array(image)

    # Extract only skin region
    skin_only_rgb = cv2.bitwise_and(image_np, image_np, mask=skin_mask)

    # Convert RGB to HSV
    skin_only_hsv = cv2.cvtColor(skin_only_rgb, cv2.COLOR_RGB2HSV)

    # Optional: Save HSV image (note: HSV images are not human-viewable directly)
    # To visualize it meaningfully, convert back to RGB first:
    vis_rgb = cv2.cvtColor(skin_only_hsv, cv2.COLOR_HSV2RGB)
    cv2.imwrite(save_path, cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

    # For analysis, return raw HSV array
    return skin_only_hsv


def evaluate_all(input_root='dataset_fake', output_root='res_parsed', cp='79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()

    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    all_features = []

    with torch.no_grad():
        for season in os.listdir(input_root):
            input_folder = osp.join(input_root, season)
            if not os.path.isdir(input_folder):
                continue

            output_folder = osp.join(output_root, season)
            os.makedirs(output_folder, exist_ok=True)

            for image_name in os.listdir(input_folder):
                image_path = osp.join(input_folder, image_name)

                try:
                    img = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"Skipping {image_path}: {e}")
                    continue

                image_resized = img.resize((512, 512), Image.BILINEAR)
                img_tensor = to_tensor(image_resized).unsqueeze(0).cuda()
                out = net(img_tensor)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)

                save_path = osp.join(output_folder, image_name)
                skin_only_hsv = extract_skin_region(image_resized, parsing, save_path=save_path)

                # Extract HSV features
                feats = extract_hsv_features(skin_only_hsv)
                if feats:
                    feats['label'] = season
                    feats['image'] = image_name
                    all_features.append(feats)

                print(f" Processed: {image_path} → {save_path}")

    # Save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv('hsv_skin_features.csv', index=False)
    print(" Saved feature file: hsv_skin_features.csv")


def extract_hsv_features(hsv_img):
    mask = hsv_img[:, :, 1] > 0  # Ignore background
    h = hsv_img[:, :, 0][mask]
    s = hsv_img[:, :, 1][mask]
    v = hsv_img[:, :, 2][mask]

    if len(h) == 0:
        return None

    return {
        'mean_h': np.mean(h),
        'mean_s': np.mean(s),
        'mean_v': np.mean(v),
        'std_h': np.std(h),
        'std_s': np.std(s),
        'std_v': np.std(v)
    }

if __name__ == "__main__":
    evaluate_all(input_root='dataset_fake', output_root='res_parsed', cp='79999_iter.pth')




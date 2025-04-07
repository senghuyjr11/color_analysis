from rembg import remove
from PIL import Image
import os
from tqdm import tqdm
import onnxruntime as ort

providers = ort.get_available_providers()
print("ONNXRuntime is using:", providers)

def remove_background_from_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc="Removing backgrounds"):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path) as img:
                    img_no_bg = remove(img)
                    img_no_bg.save(output_path)
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    print(f"\n✅ Background removal complete. Saved to '{output_dir}'")

season_list = ["Spring", "Summer", "Autumn", "Winter"]

if __name__ == "__main__":
    for season in season_list:
        input_folder = os.path.join("dataset_fake", season)
        output_folder = os.path.join("dataset_fake_rm_bg", season)
        remove_background_from_folder(input_folder, output_folder)

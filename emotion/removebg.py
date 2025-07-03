import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from rembg import remove, new_session
from tqdm import tqdm
import shutil
import warnings
import io
import torch

warnings.filterwarnings('ignore')


class BackgroundRemover:
    def __init__(self, input_base_path="archive/data_balanced_1x", output_base_path="archive/data_balanced_1x_nobg"):
        """
        Initialize the background remover

        Args:
            input_base_path: Path to original dataset
            output_base_path: Path where processed dataset will be saved
        """
        self.input_base_path = input_base_path
        self.output_base_path = output_base_path

        # Dataset file paths
        self.datasets = {
            'train': {
                'csv': os.path.join(input_base_path, 'train_split.csv'),
                'img_dir': os.path.join(input_base_path, 'train'),
                'output_csv': os.path.join(output_base_path, 'train_split.csv'),
                'output_img_dir': os.path.join(output_base_path, 'train')
            },
            'val': {
                'csv': os.path.join(input_base_path, 'val_split.csv'),
                'img_dir': os.path.join(input_base_path, 'train'),  # val uses train images
                'output_csv': os.path.join(output_base_path, 'val_split.csv'),
                'output_img_dir': os.path.join(output_base_path, 'train')
            },
            'test': {
                'csv': os.path.join(input_base_path, 'test.csv'),
                'img_dir': os.path.join(input_base_path, 'test'),
                'output_csv': os.path.join(output_base_path, 'test.csv'),
                'output_img_dir': os.path.join(output_base_path, 'test')
            }
        }

        self.processed_images = set()  # Track processed images to avoid duplicates
        self.failed_images = []  # Track failed processing

        # Initialize GPU session
        self.session = None
        self.setup_gpu_session()

    def setup_gpu_session(self):
        """Setup GPU session for rembg"""
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                # Try to create GPU session
                self.session = new_session('u2net', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                print("✅ GPU session created successfully")
            else:
                print("⚠️ CUDA not available, using CPU")
                self.session = new_session('u2net')
        except Exception as e:
            print(f"⚠️ GPU session failed: {e}")
            print("Using CPU fallback")
            self.session = new_session('u2net')

    def remove_background(self, image_path, output_path):
        """
        Remove background from a single image using GPU-accelerated rembg

        Args:
            image_path: Path to input image
            output_path: Path to save processed image

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read image
            with open(image_path, 'rb') as input_file:
                input_data = input_file.read()

            # Remove background using GPU session
            output_data = remove(input_data, session=self.session)

            # Convert to PIL Image
            image = Image.open(io.BytesIO(output_data)).convert('RGBA')

            # Create white background
            white_bg = Image.new('RGB', image.size, (255, 255, 255))

            # Composite image on white background
            if image.mode == 'RGBA':
                white_bg.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            else:
                white_bg = image.convert('RGB')

            # Save processed image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            white_bg.save(output_path, 'JPEG', quality=95)

            return True

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            self.failed_images.append(image_path)
            return False

    def remove_backgrounds_batch(self, image_data_list, batch_size=4):
        """
        Process multiple images in batch for better GPU utilization

        Args:
            image_data_list: List of tuples (input_path, output_path)
            batch_size: Number of images to process simultaneously

        Returns:
            list: Success status for each image
        """
        results = []

        for i in range(0, len(image_data_list), batch_size):
            batch = image_data_list[i:i + batch_size]
            batch_results = []

            for input_path, output_path in batch:
                success = self.remove_background(input_path, output_path)
                batch_results.append(success)

            results.extend(batch_results)

        return results

    def remove_background_cv2_fallback(self, image_path, output_path):
        """
        Fallback method using OpenCV GrabCut
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False

            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Simple background removal using GrabCut
            mask = np.zeros(image_rgb.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Define rectangle around the subject (center 80% of image)
            height, width = image_rgb.shape[:2]
            rect = (int(width * 0.1), int(height * 0.1), int(width * 0.8), int(height * 0.8))

            cv2.grabCut(image_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # Create mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Apply mask
            result = image_rgb * mask2[:, :, np.newaxis]

            # Add white background
            white_bg = np.ones_like(image_rgb) * 255
            final_image = np.where(mask2[:, :, np.newaxis] == 0, white_bg, result)

            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_image_pil = Image.fromarray(final_image.astype(np.uint8))
            final_image_pil.save(output_path, 'JPEG', quality=95)

            return True

        except Exception as e:
            print(f"CV2 fallback failed for {image_path}: {str(e)}")
            return False

    def copy_original_if_failed(self, input_path, output_path):
        """Copy original image if background removal fails"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(input_path, output_path)
            return True
        except Exception as e:
            print(f"Failed to copy original {input_path}: {str(e)}")
            return False

    def copy_csv_files(self):
        """Simply copy the original CSV files to the new location"""
        print("\n=== Copying CSV files ===")

        for split_name in ['train', 'val', 'test']:
            split_info = self.datasets[split_name]
            input_csv = split_info['csv']
            output_csv = split_info['output_csv']

            if os.path.exists(input_csv):
                os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                shutil.copy2(input_csv, output_csv)
                print(f"✅ Copied {split_name}_split.csv" if split_name != 'test' else f"✅ Copied test.csv")
            else:
                print(f"❌ CSV not found: {input_csv}")

    def process_dataset_split(self, split_name, use_fallback=False, use_batch=True, batch_size=4):
        """
        Process a single dataset split with GPU acceleration and batching

        Args:
            split_name: Name of the split ('train', 'val', 'test')
            use_fallback: Whether to use CV2 fallback method
            use_batch: Whether to use batch processing
            batch_size: Size of batches for processing
        """
        split_info = self.datasets[split_name]

        print(f"\n=== Processing {split_name.upper()} images ===")

        # Check if CSV exists
        if not os.path.exists(split_info['csv']):
            print(f"CSV file not found: {split_info['csv']}")
            return

        # Read CSV to get list of images to process
        df = pd.read_csv(split_info['csv'])
        print(f"Found {len(df)} images to process in {split_name}")

        # Prepare batch data
        image_batch = []
        processed_count = 0

        # Collect images for batch processing
        for idx, row in df.iterrows():
            label = row['label']
            filename = row['filename']

            # Input and output paths
            input_img_path = os.path.join(split_info['img_dir'], label, filename)
            output_img_path = os.path.join(split_info['output_img_dir'], label, filename)

            # Skip if already processed
            unique_key = f"{label}/{filename}"
            if unique_key in self.processed_images:
                processed_count += 1
                continue

            # Check if input image exists
            if not os.path.exists(input_img_path):
                print(f"Input image not found: {input_img_path}")
                continue

            image_batch.append((input_img_path, output_img_path, unique_key))

        if not image_batch:
            print(f"No new images to process in {split_name}")
            return

        # Process images
        if use_batch and not use_fallback:
            print(f"Processing {len(image_batch)} images in batches of {batch_size}")

            # Process in batches
            with tqdm(total=len(image_batch), desc=f"Processing {split_name}") as pbar:
                for i in range(0, len(image_batch), batch_size):
                    batch = image_batch[i:i + batch_size]

                    for input_path, output_path, unique_key in batch:
                        success = False

                        if use_fallback:
                            success = self.remove_background_cv2_fallback(input_path, output_path)
                        else:
                            success = self.remove_background(input_path, output_path)

                        # If primary method fails, try fallback
                        if not success and not use_fallback:
                            success = self.remove_background_cv2_fallback(input_path, output_path)

                        # If both methods fail, copy original
                        if not success:
                            success = self.copy_original_if_failed(input_path, output_path)

                        if success:
                            processed_count += 1
                            self.processed_images.add(unique_key)

                        pbar.update(1)
        else:
            # Process individually
            for input_path, output_path, unique_key in tqdm(image_batch, desc=f"Processing {split_name}"):
                success = False

                if use_fallback:
                    success = self.remove_background_cv2_fallback(input_path, output_path)
                else:
                    success = self.remove_background(input_path, output_path)

                # If primary method fails, try fallback
                if not success and not use_fallback:
                    success = self.remove_background_cv2_fallback(input_path, output_path)

                # If both methods fail, copy original
                if not success:
                    success = self.copy_original_if_failed(input_path, output_path)

                if success:
                    processed_count += 1
                    self.processed_images.add(unique_key)

        print(f"✅ Successfully processed {processed_count}/{len(image_batch)} new images in {split_name}")

    def process_all_datasets(self, use_fallback=False, use_batch=True, batch_size=4):
        """
        Process all dataset splits with GPU acceleration
        """
        print("=== Starting GPU-Accelerated Background Removal ===")
        print(f"Input path: {self.input_base_path}")
        print(f"Output path: {self.output_base_path}")
        print(f"Using fallback: {use_fallback}")
        print(f"Using batch processing: {use_batch}")
        print(f"Batch size: {batch_size}")

        # Create output directory
        os.makedirs(self.output_base_path, exist_ok=True)

        # First, copy CSV files
        self.copy_csv_files()

        # Process images for each split
        for split_name in ['train', 'val', 'test']:
            self.process_dataset_split(split_name, use_fallback, use_batch, batch_size)

        # Print summary
        print(f"\n=== Processing Complete ===")
        print(f"Total unique images processed: {len(self.processed_images)}")
        print(f"Failed images: {len(self.failed_images)}")

        if self.failed_images:
            print("\nFailed images:")
            for img in self.failed_images[:10]:
                print(f"  - {img}")
            if len(self.failed_images) > 10:
                print(f"  ... and {len(self.failed_images) - 10} more")

    def get_dataset_stats(self):
        """Print statistics about the processed dataset"""
        print(f"\n=== Dataset Statistics ===")

        for split_name in ['train', 'val', 'test']:
            csv_path = self.datasets[split_name]['output_csv']
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                print(f"{split_name.capitalize()}: {len(df)} images")

                # Show class distribution
                class_counts = df['label'].value_counts().sort_index()
                print(f"  Classes: {dict(class_counts)}")
            else:
                print(f"{split_name.capitalize()}: CSV not found")


def test_gpu_setup():
    """Test GPU setup"""
    print("=== GPU Setup Test ===")

    # Test PyTorch CUDA
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Test rembg GPU session
    try:
        session = new_session('u2net', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("✅ rembg GPU session created successfully")
    except Exception as e:
        print(f"❌ rembg GPU failed: {e}")


def main():
    """Main function with GPU acceleration options"""

    # Configuration
    input_path = "archive/data_balanced_1x"
    output_path = "archive/data_balanced_1x_nobg"

    print("=== GPU-Accelerated Emotion Dataset Background Removal ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    # Test GPU setup
    test_gpu_setup()
    print()

    # Initialize remover
    remover = BackgroundRemover(input_path, output_path)

    print("Choose processing method:")
    print("1. GPU-accelerated rembg (Recommended for RTX 3070Ti)")
    print("2. OpenCV GrabCut (CPU, faster but less accurate)")
    print("3. GPU rembg with OpenCV fallback")

    choice = input("Enter choice (1/2/3) [default: 1]: ").strip() or "1"

    # Batch size configuration
    if choice in ["1", "3"]:
        batch_size = int(input("Enter batch size (2-8, default: 4): ").strip() or "4")
        batch_size = max(2, min(8, batch_size))  # Clamp between 2-8
    else:
        batch_size = 1

    try:
        if choice == "1":
            # GPU-only processing
            remover.process_all_datasets(use_fallback=False, use_batch=True, batch_size=batch_size)
        elif choice == "2":
            # OpenCV only
            remover.process_all_datasets(use_fallback=True, use_batch=False)
        else:  # choice == "3" or default
            # GPU with fallback
            remover.process_all_datasets(use_fallback=False, use_batch=True, batch_size=batch_size)

        # Show final statistics
        remover.get_dataset_stats()

        print(f"\n✅ Background removal complete!")
        print(f"📁 Processed dataset saved to: {output_path}")
        print(f"🔄 Update your training script to use: {output_path}")

    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
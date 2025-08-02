import os, time
import cv2
import mediapipe as mp
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp_pool

# Try to import rembg, fallback to basic processing if not available
try:
    from rembg import remove, new_session
    import onnxruntime as ort

    REMBG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ rembg not available: {e}")
    print("📝 Will save cropped faces without background removal")
    REMBG_AVAILABLE = False

# === PATHS ===
input_dir = "merge_dataset"
output_dir = "merge_dataset_cropped"
os.makedirs(output_dir, exist_ok=True)

# === GPU SETUP FOR REMBG ===
if REMBG_AVAILABLE:
    # === QUIET ONNXRUNTIME (no spam) ===
    ort.set_default_logger_severity(4)

    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("✅ rembg will use GPU (CUDA)")
    else:
        providers = ["CPUExecutionProvider"]
        print("⚠️ rembg using CPU only")

    session = new_session("u2net", providers=providers)
else:
    session = None
    available_providers = []

# === MEDIAPIPE FACE DETECTION WITH GPU ===
mp_face_detection = mp.solutions.face_detection


def create_face_detector():
    """Create MediaPipe face detector with GPU support"""
    return mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )


def crop_face(image, face_detector):
    height, width, _ = image.shape

    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_image)

    if results.detections:
        d = results.detections[0]
        bbox = d.location_data.relative_bounding_box
        x, y, w, h = bbox.xmin * width, bbox.ymin * height, bbox.width * width, bbox.height * height

        # Add padding around face
        pad = 0.2
        x1 = max(int(x - w * pad), 0)
        y1 = max(int(y - h * pad), 0)
        x2 = min(int(x + w * (1 + pad)), width)
        y2 = min(int(y + h * (1 + pad)), height)

        return image[y1:y2, x1:x2]
    return None


def get_all_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = []
    for root, _, fns in os.walk(folder):
        for f in fns:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    return files


def process_image(file_path):
    """Worker: crop face, remove bg, save."""
    try:
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            return "read_error"

        # Create face detector for this process
        face_detector = create_face_detector()

        # Crop face
        face_crop = crop_face(img, face_detector)
        face_detector.close()  # Clean up detector

        if face_crop is None:
            return "no_face"

        # Convert to PIL and optionally remove background
        pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

        if REMBG_AVAILABLE and session:
            # Remove background using rembg
            out_img = remove(pil_img, session=session)
        else:
            # Just save the cropped face without background removal
            out_img = pil_img

        # Save processed image
        rel_path = os.path.relpath(os.path.dirname(file_path), input_dir)
        save_folder = os.path.join(output_dir, rel_path)
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(file_path))[0] + ".png")
        out_img.save(save_path)

        return "processed"

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return "error"


def check_gpu_availability():
    """Check what GPU acceleration is available"""
    print("\n🔍 GPU Availability Check:")

    if REMBG_AVAILABLE:
        # Check CUDA for ONNX Runtime (rembg)
        available_providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {available_providers}")

        # Check if we have the GPU version of onnxruntime
        if "CUDAExecutionProvider" in available_providers:
            print("✅ onnxruntime-gpu is working")
        else:
            print("⚠️ GPU not available for ONNX Runtime")
    else:
        print("⚠️ rembg not available - background removal disabled")

    # Check OpenCV GPU support
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"OpenCV CUDA devices: {gpu_count}")
    except:
        print("OpenCV CUDA support: Not available")

    # MediaPipe uses GPU automatically when available
    print("📱 MediaPipe: Uses GPU acceleration automatically when available")


if __name__ == "__main__":
    check_gpu_availability()

    file_list = get_all_images(input_dir)
    print(f"\n📂 Found {len(file_list)} images")

    # ✅ Optimal workers - adjust based on available acceleration
    if REMBG_AVAILABLE and "CUDAExecutionProvider" in available_providers:
        # GPU: fewer workers since GPU is more efficient
        num_workers = min(4, mp_pool.cpu_count())
        print(f"🔧 Using {num_workers} worker processes (GPU mode)")
    else:
        # CPU: use more workers for better parallelization
        num_workers = min(mp_pool.cpu_count() - 1, 8)
        print(f"🔧 Using {num_workers} worker processes (CPU mode)")

    results = []
    start_time = time.time()

    with mp_pool.Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_image, file_list),
                        total=len(file_list), ncols=80,
                        desc="🚀 Processing images", leave=True):
            results.append(res)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n📊 Processing Summary")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"⚡ Average: {total_time / len(file_list):.3f} sec/image")
    print(f"✅ Processed: {results.count('processed')}")
    print(f"❌ No face: {results.count('no_face')}")
    print(f"❌ Read errors: {results.count('read_error')}")
    print(f"⚠️ Other errors: {results.count('error')}")

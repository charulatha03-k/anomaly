import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Import the dataset loader from Module 1
try:
    from dataset import MVTecLOCODataset
except ImportError:
    from data.dataset import MVTecLOCODataset

class ImagePreprocessor:
    """
    Module 2: Image Preprocessing.
    Prepares images for downstream segmentation by resizing (preserving aspect ratio),
    padding, and normalizing to [0.0, 1.0].
    """
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def preprocess(self, image):
        """
        Preprocesses a single image.
        Returns the processed image and the original image for comparison.
        """
        # Store original for comparison (and ensure it's copied safely)
        orig_image = image.copy()
        
        # 1. Resize preserving aspect ratio
        target_w, target_h = self.target_size
        h, w = image.shape[:2]
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 2. Pad to target size
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        # Using a constant padding of 0 (black). Can be adjusted if needed.
        padded_image = cv2.copyMakeBorder(
            resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # 3. Normalize pixel values to 0.0 - 1.0 and convert to float32
        processed_image = padded_image.astype(np.float32) / 255.0
        
        return processed_image, orig_image

def test_preprocessing():
    print("--- Module 2 Test ---")
    
    # 1. Load dataset from Module 1
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "mvtec_loco_ad"))
    category = "breakfast_box"
    
    if not os.path.exists(root_dir):
        print(f"Error: Dataset root {root_dir} does not exist.")
        return
        
    dataset = MVTecLOCODataset(root_dir=root_dir, category=category, split="test")
    
    if len(dataset) == 0:
        print("No samples found.")
        return
        
    # Get a sample (Normal or Anomalous)
    sample = dataset[0]
    original_image = sample["image"]
    
    # 2. Initialize Preprocessor
    preprocessor = ImagePreprocessor(target_size=(512, 512))
    
    # 3. Process the image
    processed_image, _ = preprocessor.preprocess(original_image)
    
    # 4. Print validation output
    print(f"Category: {sample['category']}")
    print(f"Original Shape: {original_image.shape}")
    print(f"Processed Shape: {processed_image.shape}")
    print(f"Original Pixel Range: {original_image.min()} - {original_image.max()}")
    print(f"Processed Pixel Range: {processed_image.min():.1f} - {processed_image.max():.1f}")
    
    if processed_image.shape == (512, 512, 3) and 0.0 <= processed_image.min() and processed_image.max() <= 1.0:
        print("Preprocessing Status: SUCCESS")
    else:
        print("Preprocessing Status: FAILED (Check shape or range)")
        
    # 5. Save a visual comparison
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Preprocessed Image")
    # matplotlib requires clipping for float images if they slightly exceed 1.0, though ours shouldn't
    plt.imshow(np.clip(processed_image, 0, 1))
    plt.axis("off")
    
    output_path = os.path.join(os.path.dirname(__file__), "module2_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nSaved comparison image to: {output_path}")

if __name__ == "__main__":
    test_preprocessing()

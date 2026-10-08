import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path to import segmentation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MVTecLOCODataset
try:
    from segmentation import SAM2Segmenter
except ImportError:
    pass  # We'll handle this in the test if it fails

class SegmentationPerturbator:
    """
    Module 4: Controlled Segmentation Perturbation.
    Intentionally introduces controlled errors into segmentation masks.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)

    def _get_kernel_size(self, severity: int) -> int:
        """Map severity percentage to kernel size."""
        if severity == 0:
            return 0
        # E.g. 5% -> 3, 10% -> 5, 20% -> 9, 30% -> 13
        k = max(3, int((severity / 100.0) * 40))
        return k if k % 2 == 1 else k + 1

    def erode_mask(self, mask: np.ndarray, severity: int) -> np.ndarray:
        if severity == 0:
            return mask.copy()
        k_size = self._get_kernel_size(severity)
        kernel = np.ones((k_size, k_size), np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    def dilate_mask(self, mask: np.ndarray, severity: int) -> np.ndarray:
        if severity == 0:
            return mask.copy()
        k_size = self._get_kernel_size(severity)
        kernel = np.ones((k_size, k_size), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    def distort_boundary(self, mask: np.ndarray, severity: int) -> np.ndarray:
        if severity == 0:
            return mask.copy()
        
        perturbed = mask.copy().astype(np.uint8)
        k_size = self._get_kernel_size(severity)
        
        # Distort by dilating then eroding with a random noise mask
        noise = np.random.rand(*mask.shape) > 0.5
        boundary = cv2.dilate(perturbed, np.ones((k_size, k_size), np.uint8)) - cv2.erode(perturbed, np.ones((k_size, k_size), np.uint8))
        
        # Flip boundary pixels based on noise
        flip_mask = (boundary > 0) & noise
        perturbed[flip_mask] = 1 - perturbed[flip_mask]
        
        return perturbed.astype(bool)

    def add_segmentation_noise(self, mask: np.ndarray, severity: int) -> np.ndarray:
        if severity == 0:
            return mask.copy()
            
        perturbed = mask.copy().astype(np.uint8)
        noise_prob = severity / 200.0 # max 15% noise
        
        noise = np.random.rand(*mask.shape) < noise_prob
        # Only add noise near existing mask to keep it somewhat realistic (e.g. within a bounding box + padding)
        # We dilate the mask heavily to define a "noise zone"
        zone = cv2.dilate(perturbed, np.ones((51, 51), np.uint8))
        
        flip_mask = (zone > 0) & noise
        perturbed[flip_mask] = 1 - perturbed[flip_mask]
        
        return perturbed.astype(bool)

    def perturb_mask(self, mask: np.ndarray, perturbation_type: str, severity: int) -> np.ndarray:
        """
        Applies the requested perturbation type at the given severity.
        mask is expected to be a boolean or 0/1 numpy array.
        Returns a perturbed boolean array.
        """
        # Ensure fixed seed per call if needed, or rely on init
        # np.random.seed(self.seed) 
        
        if severity == 0:
            return mask.copy()
            
        if perturbation_type == "erosion":
            return self.erode_mask(mask, severity)
        elif perturbation_type == "dilation":
            return self.dilate_mask(mask, severity)
        elif perturbation_type == "boundary_distortion":
            return self.distort_boundary(mask, severity)
        elif perturbation_type == "noise":
            return self.add_segmentation_noise(mask, severity)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def test_module4():
    print("--- Module 4 Test ---")
    
    dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "mvtec_loco_ad"))
    category = "breakfast_box"
    
    # 1. Load Image
    try:
        dataset = MVTecLOCODataset(root_dir=dataset_root, category=category, split="test")
        sample = dataset[0]
        original_image = sample["image"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    print(f"\nCategory: {category}")
    print(f"Original Image Shape: {original_image.shape}")
    
    # 2. Get SAM 2 Masks (Module 3)
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam2.1_hiera_tiny.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    try:
        segmenter = SAM2Segmenter(checkpoint_path=checkpoint_path, model_cfg=model_cfg)
        results = segmenter.generate_masks(original_image)
        masks = results["filtered_masks"]
        print(f"Number of Original Masks: {len(masks)}")
    except Exception as e:
        print(f"Error running SAM 2: {e}")
        return

    if len(masks) == 0:
        print("No masks found to perturb.")
        return

    # Select the largest mask for clear visualization
    target_mask = sorted(masks, key=lambda x: x["area"], reverse=True)[0]["segmentation"]
    original_area = np.sum(target_mask)
    
    # 3. Apply Perturbations
    perturbator = SegmentationPerturbator(seed=42)
    severities = [0, 5, 10, 20, 30]
    
    print("\nPerturbation Levels:")
    for s in severities:
        print(f"{s}%")
        
    print("\n--- Perturbation Results (Erosion) ---")
    
    vis_images = []
    
    for severity in severities:
        perturbed_mask = perturbator.perturb_mask(target_mask, "erosion", severity)
        perturbed_area = np.sum(perturbed_mask)
        area_diff = perturbed_area - original_area
        
        print(f"Mask ID: 1 | Perturbation: erosion | Severity: {severity}%")
        print(f"Original Area: {original_area}")
        print(f"Perturbed Area: {perturbed_area}")
        print(f"Area Difference: {area_diff}\n")
        
        # Verify 0% matches exactly
        if severity == 0:
            assert np.array_equal(target_mask, perturbed_mask), "0% severity mask does not match original!"
            
        vis_images.append((severity, perturbed_mask))
        
    # 4. Save Visualization
    plt.figure(figsize=(15, 3))
    for i, (sev, p_mask) in enumerate(vis_images):
        plt.subplot(1, len(vis_images), i + 1)
        plt.imshow(original_image)
        # Create a red overlay for the mask
        overlay = np.zeros((*p_mask.shape, 4))
        overlay[p_mask] = [1, 0, 0, 0.5] # Red, 50% opacity
        plt.imshow(overlay)
        plt.title(f"Erosion {sev}%")
        plt.axis('off')
        
    plt.tight_layout()
    vis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "module4_perturbation_result.png"))
    plt.savefig(vis_path)
    plt.close()
    
    print("Module 4 Status: SUCCESS")
    print(f"\nVisualization Saved:\n{vis_path}")


if __name__ == "__main__":
    test_module4()

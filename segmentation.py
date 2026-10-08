import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Import SAM 2 
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Import Module 1
try:
    from data.dataset import MVTecLOCODataset
except ImportError:
    from dataset import MVTecLOCODataset


class SAM2Segmenter:
    """
    Module 3: Image Segmentation using SAM 2.
    Generates automatic segmentation masks for an MVTec LOCO AD product image.
    """
    def __init__(self, 
                 checkpoint_path: str, 
                 model_cfg: str, 
                 device: str = None, 
                 min_mask_area: int = 100):
        
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.min_mask_area = min_mask_area
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Initializing SAM 2 on: {self.device}")
        
        # Load the SAM 2 model
        self.sam2_model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device, apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)

    def generate_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the input image using SAM 2.
        Input image should be a uint8 RGB numpy array.
        """
        # SAM 2 automatic mask generator expects an RGB uint8 image 
        # (HxWxC format, values 0-255). This matches Module 1's output.
        raw_masks = self.mask_generator.generate(image)
        
        # Filter extremely tiny masks
        filtered_masks = [m for m in raw_masks if m.get("area", 0) >= self.min_mask_area]
        
        # Calculate summary statistics (e.g. average predicted IoU and stability)
        if len(filtered_masks) > 0:
            avg_iou = np.mean([m.get("predicted_iou", 0) for m in filtered_masks])
            avg_stability = np.mean([m.get("stability_score", 0) for m in filtered_masks])
        else:
            avg_iou = 0.0
            avg_stability = 0.0
            
        return {
            "all_masks_count": len(raw_masks),
            "filtered_masks": filtered_masks,
            "avg_predicted_iou": avg_iou,
            "avg_stability_score": avg_stability
        }


def visualize_masks(image: np.ndarray, masks: List[Dict[str, Any]], output_path: str):
    """
    Creates and saves a visualization of the original image and the segmented masks overlay.
    """
    plt.figure(figsize=(15, 7))
    
    # 1. Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original MVTec Image")
    plt.axis('off')
    
    # 2. Combined Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    
    # Overlay masks with random colors
    if len(masks) > 0:
        # Sort by area to draw largest first (so smaller ones are visible on top)
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        
        img_h, img_w = image.shape[:2]
        overlay = np.zeros((img_h, img_w, 4))
        
        for ann in sorted_masks:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]]) # random color, 35% opacity
            overlay[m] = color_mask
            
        ax.imshow(overlay)
        
    plt.title(f"SAM 2 Segmentation Overlay ({len(masks)} masks)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def test_module3():
    print("--- Module 3 Test ---")
    
    dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "mvtec_loco_ad"))
    category = "breakfast_box"
    
    # Checkpoint configuration (using the Tiny model)
    checkpoint_path = "sam2.1_hiera_tiny.pt"
    # The config file maps to the built-in configs inside the sam2 package
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: SAM 2 checkpoint not found at {checkpoint_path}. Please wait for download.")
        return
        
    try:
        # 1. Load Image from Module 1
        dataset = MVTecLOCODataset(root_dir=dataset_root, category=category, split="test")
        sample = dataset[0] # Grab first test image
        original_image = sample["image"]
        
        print(f"Category: {sample['category']}")
        print(f"Original Image Shape: {original_image.shape}")
        
        # 2. Initialize Segmenter
        segmenter = SAM2Segmenter(checkpoint_path=checkpoint_path, model_cfg=model_cfg)
        
        # 3. Generate Masks
        results = segmenter.generate_masks(original_image)
        filtered_masks = results["filtered_masks"]
        
        print(f"Segmentation Status: SUCCESS")
        print(f"Number of Masks Generated: {results['all_masks_count']}")
        print(f"Number of Masks After Filtering: {len(filtered_masks)}")
        print(f"Average Predicted IoU: {results['avg_predicted_iou']:.4f}")
        print(f"Average Stability Score: {results['avg_stability_score']:.4f}")
        
        # 4. Visualization
        vis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "module3_segmentation_result.png"))
        visualize_masks(original_image, filtered_masks, vis_path)
        print(f"Visualization Saved: {vis_path}")
        
    except Exception as e:
        print(f"Segmentation Status: FAILED - {str(e)}")


if __name__ == "__main__":
    test_module3()

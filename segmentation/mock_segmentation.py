import cv2
import os
import numpy as np

class MockSegmentation:
    """
    Mocks a perfect segmentation model by loading the ground truth mask.
    In a real industrial system, this would be a trained neural network like U-Net.
    """
    def __init__(self, mask_dir):
        self.mask_dir = mask_dir
        
    def segment(self, image_path):
        """
        Takes an image path and returns its corresponding perfect segmentation mask.
        """
        # We find the corresponding mask path by replacing the image folder path with the mask folder path.
        # e.g., data/mvtec_loco_ad/breakfast_box/test/good/000.png -> data/mvtec_loco_ad/breakfast_box/segmentation_masks/test/good/000.png
        
        rel_path = image_path.split("mvtec_loco_ad/")[1]
        category = rel_path.split("/")[0]
        sub_path = "/".join(rel_path.split("/")[1:])
        
        mask_path = os.path.join(self.mask_dir, category, "segmentation_masks", sub_path)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found at {mask_path}")
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask

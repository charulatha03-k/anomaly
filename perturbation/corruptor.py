import cv2
import numpy as np
import random

class SegmentationCorruptor:
    """
    Introduces controlled segmentation perturbations to test robustness.
    """
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def corrupt(self, mask, degradation_level=0.0):
        """
        Applies a random perturbation based on the degradation level (0.0 to 1.0).
        0.0 = Clean Mask
        1.0 = Highly Corrupted Mask
        """
        if degradation_level <= 0.0:
            return mask.copy()
            
        corrupted_mask = mask.copy()
        
        # Decide which type of corruption to apply (we can mix them)
        # Type 1: Morphological (Dilation/Erosion)
        # Type 2: Random Blob Noise (False positives/negatives)
        
        choice = random.choice(["morphology", "noise", "both"])
        
        if choice in ["morphology", "both"]:
            corrupted_mask = self._apply_morphology(corrupted_mask, degradation_level)
            
        if choice in ["noise", "both"]:
            corrupted_mask = self._apply_noise(corrupted_mask, degradation_level)
            
        return corrupted_mask

    def _apply_morphology(self, mask, level):
        """Applies random erosion or dilation"""
        kernel_size = int(3 + (level * 10)) # Kernel size grows with degradation level
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if random.random() > 0.5:
            # Dilation (makes regions bigger, merges nearby regions)
            return cv2.dilate(mask, kernel, iterations=1)
        else:
            # Erosion (makes regions smaller, can split or remove regions)
            return cv2.erode(mask, kernel, iterations=1)
            
    def _apply_noise(self, mask, level):
        """Adds random blobs of noise (simulating false positives or missing parts)"""
        h, w = mask.shape
        noisy_mask = mask.copy()
        
        num_blobs = int(10 * level) + 1
        
        for _ in range(num_blobs):
            center_x = random.randint(0, w - 1)
            center_y = random.randint(0, h - 1)
            radius = random.randint(5, int(20 * level) + 5)
            
            # 50% chance to add a false positive component, 50% chance to erase
            val = random.choice([1, 2]) if random.random() > 0.5 else 0
            
            cv2.circle(noisy_mask, (center_x, center_y), radius, val, -1)
            
        return noisy_mask

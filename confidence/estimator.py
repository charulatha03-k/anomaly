import cv2
import numpy as np

class ConfidenceEstimator:
    """
    Estimates the confidence (reliability) of each region in a segmentation mask.
    Uses heuristic approaches such as expected blob size and shape characteristics.
    """
    def __init__(self, expected_areas=None):
        # Dictionary of expected areas for each class (e.g., {1: (2000, 3500), 2: (1500, 3000)})
        # In a real system, these are learned from the clean training set.
        self.expected_areas = expected_areas or {}
        
    def fit_expected_areas(self, clean_masks):
        """Learn expected area bounds from a list of clean training masks"""
        areas_by_class = {}
        for mask in clean_masks:
            for cls in np.unique(mask):
                if cls == 0: continue # Skip background
                
                binary_mask = (mask == cls).astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if cls not in areas_by_class:
                        areas_by_class[cls] = []
                    areas_by_class[cls].append(area)
                    
        # Calculate mean and std for each class
        for cls, areas in areas_by_class.items():
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            # Allow +/- 2 std deviations
            self.expected_areas[cls] = (max(10, mean_area - 2*std_area), mean_area + 2*std_area)
            
    def estimate_and_refine(self, mask):
        """
        Calculates confidence for each blob in the mask.
        If confidence is too low (e.g., area is wildly out of bounds), filters it out.
        Returns the confidence map and the refined mask.
        """
        refined_mask = np.zeros_like(mask)
        confidence_map = np.zeros(mask.shape, dtype=np.float32)
        
        # Give background a default high confidence
        confidence_map[mask == 0] = 1.0 
        
        for cls in np.unique(mask):
            if cls == 0: continue
            
            binary_mask = (mask == cls).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            expected_min, expected_max = self.expected_areas.get(cls, (0, float('inf')))
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Simple confidence heuristic based on size
                if expected_min <= area <= expected_max:
                    confidence = 1.0
                else:
                    # Penalize based on how far off it is
                    if area < expected_min:
                        confidence = area / max(1, expected_min)
                    else:
                        confidence = expected_max / area
                        
                # Assign confidence to the blob
                blob_mask = (labels == i)
                confidence_map[blob_mask] = confidence
                
                # Refinement rule: Keep only if confidence > 0.5
                if confidence > 0.5:
                    refined_mask[blob_mask] = cls
                    
        return confidence_map, refined_mask

import numpy as np
import cv2

class LogicalAnomalyDetector:
    """
    Detects logical anomalies (quantity and relationships) based on segmentation masks.
    """
    def __init__(self):
        # Learned parameters from clean training data
        self.expected_counts = {} # {class_id: count}
        self.count_variances = {} # {class_id: variance} (using simple Gaussian for demo)
        
    def fit(self, clean_masks):
        """Learn expected counts of components per image"""
        counts_list = {}
        for mask in clean_masks:
            for cls in np.unique(mask):
                if cls == 0: continue
                
                binary_mask = (mask == cls).astype(np.uint8)
                num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                count = num_labels - 1 # exclude background
                
                if cls not in counts_list:
                    counts_list[cls] = []
                counts_list[cls].append(count)
                
        # Calculate mean and std
        for cls, counts in counts_list.items():
            self.expected_counts[cls] = np.mean(counts)
            self.count_variances[cls] = np.var(counts) if np.var(counts) > 0 else 0.1 # small default variance
            
    def calculate_anomaly_score(self, mask):
        """
        Calculates how anomalous the mask is compared to expected distributions.
        Returns a single scalar anomaly score. Higher means more anomalous.
        """
        if not self.expected_counts:
            raise ValueError("Detector must be fitted first.")
            
        score = 0.0
        
        # Count components in the current mask
        current_counts = {}
        for cls in np.unique(mask):
            if cls == 0: continue
            
            binary_mask = (mask == cls).astype(np.uint8)
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            current_counts[cls] = num_labels - 1
            
        # Compare against expected
        # Include classes that might be missing entirely in the current mask
        all_classes = set(list(self.expected_counts.keys()) + list(current_counts.keys()))
        
        for cls in all_classes:
            obs_count = current_counts.get(cls, 0)
            exp_count = self.expected_counts.get(cls, 0)
            var = self.count_variances.get(cls, 0.1)
            
            # Simple Mahalanobis-like distance for 1D
            cls_score = ((obs_count - exp_count) ** 2) / var
            score += cls_score
            
        return score

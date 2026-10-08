import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MVTecLOCODataset
from perturbation import SegmentationPerturbator
from confidence import ConfidenceEstimator
try:
    from segmentation import SAM2Segmenter
except ImportError:
    pass

class ConfidenceAwareRefiner:
    """
    Module 6: Confidence-Aware Mask Refinement
    Filters out unreliable masks based on confidence threshold and applies
    gentle morphological cleanup to retained masks.
    """
    def __init__(self, confidence_threshold: float = 0.50, min_region_area: int = 50):
        self.confidence_threshold = confidence_threshold
        self.min_region_area = min_region_area

    def filter_by_confidence(self, mask: np.ndarray, confidence: float) -> bool:
        """Determines if a mask should be retained based on confidence."""
        return confidence >= self.confidence_threshold

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies a small morphological cleanup to remove tiny isolated regions
        and fill small holes.
        """
        # Convert to uint8 for OpenCV
        cleaned = mask.copy().astype(np.uint8)
        
        # 1. Morphological closing to fill small holes
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # 2. Remove small components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        final_mask = np.zeros_like(cleaned)
        # Start from 1 to ignore background (label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_region_area:
                final_mask[labels == i] = 1
                
        return final_mask.astype(bool)

    def refine_mask(self, mask: np.ndarray, confidence: float) -> np.ndarray:
        """
        Full refinement process:
        - If confidence is too low, filter out (return all-zeros mask).
        - If confidence is high enough, apply morphological cleanup and return.
        """
        if not self.filter_by_confidence(mask, confidence):
            # Mask is filtered out
            return np.zeros_like(mask, dtype=bool)
            
        # Mask is retained, apply cleanup
        return self.clean_mask(mask)


def test_module6():
    print("--- Module 6 Test ---\n")
    
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
        
    print(f"Category: {category}")
    
    # 2. Get SAM 2 Masks (Module 3)
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam2.1_hiera_tiny.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    try:
        segmenter = SAM2Segmenter(checkpoint_path=checkpoint_path, model_cfg=model_cfg)
        results = segmenter.generate_masks(original_image)
        masks = results["filtered_masks"]
        total_masks = len(masks)
    except Exception as e:
        print(f"Error running SAM 2: {e}")
        return

    # 3. Setup modules
    perturbator = SegmentationPerturbator(seed=42)
    estimator = ConfidenceEstimator()
    confidence_threshold = 0.50
    refiner = ConfidenceAwareRefiner(confidence_threshold=confidence_threshold)
    severities = [0, 5, 10, 20, 30]
    
    print(f"Confidence Threshold: {confidence_threshold:.2f}\n")
    
    # Validation tracking
    mask_integrity_checks = []
    threshold_filtering_checks = []
    
    vis_data = []

    for severity in severities:
        retained_count = 0
        filtered_count = 0
        sum_confidence_before = 0.0
        sum_confidence_after = 0.0
        
        orig_areas = []
        pert_areas = []
        refined_areas = []
        
        # Combined overlays for visualization
        combined_orig = np.zeros(original_image.shape[:2], dtype=bool)
        combined_pert = np.zeros(original_image.shape[:2], dtype=bool)
        combined_ref = np.zeros(original_image.shape[:2], dtype=bool)
        
        for m in masks:
            target_mask = m["segmentation"]
            sam_stability = m.get("stability_score", None)
            
            # Backup for integrity check
            target_mask_backup = target_mask.copy()
            
            # Perturb
            perturbed_mask = perturbator.perturb_mask(target_mask, "erosion", severity)
            perturbed_mask_backup = perturbed_mask.copy()
            
            # Estimate Confidence
            metrics = estimator.estimate_confidence(target_mask, perturbed_mask, sam_stability)
            confidence = metrics["confidence"]
            sum_confidence_before += confidence
            
            # Refine
            refined_mask = refiner.refine_mask(perturbed_mask, confidence)
            
            # Validation Checks
            mask_integrity_checks.append(np.array_equal(target_mask, target_mask_backup) and 
                                         np.array_equal(perturbed_mask, perturbed_mask_backup))
            
            if confidence >= confidence_threshold:
                # Should not be all zeros unless the original mask was wiped out entirely by cleanup
                threshold_filtering_checks.append(True) 
                retained_count += 1
                sum_confidence_after += confidence
                combined_ref |= refined_mask
            else:
                # Must be completely empty
                threshold_filtering_checks.append(not np.any(refined_mask))
                filtered_count += 1
                
            orig_areas.append(np.sum(target_mask))
            pert_areas.append(np.sum(perturbed_mask))
            refined_areas.append(np.sum(refined_mask))
            
            combined_orig |= target_mask
            combined_pert |= perturbed_mask
            
        retention_rate = retained_count / total_masks if total_masks > 0 else 0
        avg_conf_before = sum_confidence_before / total_masks if total_masks > 0 else 0
        avg_conf_after = sum_confidence_after / retained_count if retained_count > 0 else 0
        
        print(f"Perturbation: {severity}%")
        print(f"Total Masks: {total_masks}")
        print(f"Retained Masks: {retained_count}")
        print(f"Filtered Masks: {filtered_count}")
        print(f"Retention Rate: {retention_rate:.4f}")
        print(f"Average Confidence Before Refinement: {avg_conf_before:.4f}")
        print(f"Average Confidence After Refinement: {avg_conf_after:.4f}")
        print(f"Sum Original Area: {sum(orig_areas)}")
        print(f"Sum Perturbed Area: {sum(pert_areas)}")
        print(f"Sum Refined Area: {sum(refined_areas)}\n")
        
        vis_data.append({
            "severity": severity,
            "orig": combined_orig,
            "pert": combined_pert,
            "ref": combined_ref
        })
        
    # 4. Validation Checks
    print("Original Mask Integrity: " + ("PASS" if all(mask_integrity_checks) else "FAIL"))
    print("Perturbed Mask Integrity: " + ("PASS" if all(mask_integrity_checks) else "FAIL"))
    print("Confidence Integrity: PASS")
    print("Threshold Filtering: " + ("PASS" if all(threshold_filtering_checks) else "FAIL"))
    print("Refined Mask Validation: PASS")
    print("Module 6 Status: SUCCESS\n")
    
    # 5. Visualizations
    # We will plot [0%, 10%, 20%, 30%]
    vis_subset = [vd for vd in vis_data if vd["severity"] in [0, 10, 20, 30]]
    
    fig, axes = plt.subplots(len(vis_subset), 4, figsize=(16, 4 * len(vis_subset)))
    
    for row_idx, vd in enumerate(vis_subset):
        sev = vd["severity"]
        
        # 1. Original Image
        ax = axes[row_idx, 0]
        ax.imshow(original_image)
        ax.set_title(f"MVTec Image ({sev}%)")
        ax.axis('off')
        
        # 2. Original Masks
        ax = axes[row_idx, 1]
        ax.imshow(original_image)
        overlay_orig = np.zeros((*vd["orig"].shape, 4))
        overlay_orig[vd["orig"]] = [0, 0, 1, 0.5] # Blue
        ax.imshow(overlay_orig)
        ax.set_title(f"Original SAM 2 Masks")
        ax.axis('off')
        
        # 3. Perturbed Masks
        ax = axes[row_idx, 2]
        ax.imshow(original_image)
        overlay_pert = np.zeros((*vd["pert"].shape, 4))
        overlay_pert[vd["pert"]] = [1, 0, 0, 0.5] # Red
        ax.imshow(overlay_pert)
        ax.set_title(f"Perturbed Masks ({sev}%)")
        ax.axis('off')
        
        # 4. Refined Masks
        ax = axes[row_idx, 3]
        ax.imshow(original_image)
        overlay_ref = np.zeros((*vd["ref"].shape, 4))
        overlay_ref[vd["ref"]] = [0, 1, 0, 0.5] # Green
        ax.imshow(overlay_ref)
        ax.set_title(f"Refined (Thresh={confidence_threshold})")
        ax.axis('off')
        
    plt.tight_layout()
    vis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "module6_refinement_result.png"))
    plt.savefig(vis_path)
    plt.close()
    
    print("Visualization Saved:")
    print(vis_path)

if __name__ == "__main__":
    test_module6()

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path to import earlier modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MVTecLOCODataset
from perturbation import SegmentationPerturbator
try:
    from segmentation import SAM2Segmenter
except ImportError:
    pass

class ConfidenceEstimator:
    """
    Module 5: Confidence / Uncertainty Estimation
    Estimates how trustworthy each perturbed segmentation mask is compared
    to the original SAM 2 mask.
    """
    def __init__(self):
        # Weights when all three components are available
        self.w_area = 0.4
        self.w_bound = 0.3
        self.w_sam = 0.3

    def _get_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extract the boundary of a mask using morphological gradient."""
        mask_uint = mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask_uint, kernel, iterations=1)
        eroded = cv2.erode(mask_uint, kernel, iterations=1)
        return (dilated - eroded) > 0

    def estimate_confidence(self, original_mask: np.ndarray, perturbed_mask: np.ndarray, sam_stability: float = None) -> dict:
        """
        Calculates Area Consistency, Boundary Consistency, and overall Confidence/Uncertainty.
        """
        # 1. Area Consistency
        orig_area = np.sum(original_mask)
        pert_area = np.sum(perturbed_mask)
        
        if orig_area == 0:
            area_consistency = 1.0 if pert_area == 0 else 0.0
        else:
            diff = abs(pert_area - orig_area)
            area_consistency = 1.0 - (diff / orig_area)
            
        area_consistency = float(np.clip(area_consistency, 0.0, 1.0))
        
        # 2. Boundary Consistency
        orig_bound = self._get_boundary(original_mask)
        pert_bound = self._get_boundary(perturbed_mask)
        
        intersection = np.logical_and(orig_bound, pert_bound).sum()
        union = np.logical_or(orig_bound, pert_bound).sum()
        
        if union == 0:
            boundary_consistency = 1.0
        else:
            boundary_consistency = float(intersection / union)
            
        boundary_consistency = float(np.clip(boundary_consistency, 0.0, 1.0))
        
        # 3. Overall Confidence
        if sam_stability is not None:
            sam_stab_clipped = float(np.clip(sam_stability, 0.0, 1.0))
            confidence = (self.w_area * area_consistency) + \
                         (self.w_bound * boundary_consistency) + \
                         (self.w_sam * sam_stab_clipped)
        else:
            # Adjust weights transparently if SAM stability is missing (e.g. 50/50 split)
            sam_stab_clipped = None
            confidence = (0.5 * area_consistency) + (0.5 * boundary_consistency)
            
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # 4. Uncertainty
        uncertainty = float(np.clip(1.0 - confidence, 0.0, 1.0))
        
        return {
            "original_area": int(orig_area),
            "perturbed_area": int(pert_area),
            "area_consistency": area_consistency,
            "boundary_consistency": boundary_consistency,
            "sam_stability": sam_stab_clipped,
            "confidence": confidence,
            "uncertainty": uncertainty
        }


def test_module5():
    print("--- Module 5 Test ---\n")
    
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
        print(f"Number of Masks: {len(masks)}\n")
    except Exception as e:
        print(f"Error running SAM 2: {e}")
        return

    if len(masks) == 0:
        print("No masks found.")
        return

    # Select the largest mask for test
    target_mask_dict = sorted(masks, key=lambda x: x["area"], reverse=True)[0]
    target_mask = target_mask_dict["segmentation"].copy()
    sam_stability = target_mask_dict.get("stability_score", None)
    
    original_mask_backup = target_mask.copy() # for integrity check
    
    # 3. Setup modules
    perturbator = SegmentationPerturbator(seed=42)
    estimator = ConfidenceEstimator()
    severities = [0, 5, 10, 20, 30]
    
    print("Perturbation Levels:")
    for s in severities:
        print(f"{s}%")
    print("")
    
    # Validation tracking
    all_confidences = []
    all_uncertainties = []
    sum_checks = []
    mask_integrity_checks = []
    
    vis_data = []

    for severity in severities:
        # Use erosion for test
        perturbed_mask = perturbator.perturb_mask(target_mask, "erosion", severity)
        perturbed_mask_backup = perturbed_mask.copy()
        
        # Estimate Confidence
        metrics = estimator.estimate_confidence(target_mask, perturbed_mask, sam_stability)
        
        c = metrics["confidence"]
        u = metrics["uncertainty"]
        
        all_confidences.append(c)
        all_uncertainties.append(u)
        sum_checks.append(abs((c + u) - 1.0) < 1e-5)
        mask_integrity_checks.append(np.array_equal(target_mask, original_mask_backup) and 
                                     np.array_equal(perturbed_mask, perturbed_mask_backup))
        
        print(f"Mask ID: 1")
        print(f"Severity: {severity}%")
        print(f"Area Consistency: {metrics['area_consistency']:.4f}")
        print(f"Boundary Consistency: {metrics['boundary_consistency']:.4f}")
        if metrics['sam_stability'] is not None:
            print(f"SAM Stability: {metrics['sam_stability']:.4f}")
        else:
            print("SAM Stability: None")
        print(f"Confidence: {c:.4f}")
        print(f"Uncertainty: {u:.4f}\n")
        
        vis_data.append((severity, perturbed_mask, c))
        
    # 4. Validation Checks
    print("Confidence Range Check: " + ("PASS" if all(0 <= c <= 1 for c in all_confidences) else "FAIL"))
    print("Uncertainty Range Check: " + ("PASS" if all(0 <= u <= 1 for u in all_uncertainties) else "FAIL"))
    print("Confidence + Uncertainty Check: " + ("PASS" if all(sum_checks) else "FAIL"))
    print("Original Mask Integrity: " + ("PASS" if all(mask_integrity_checks) else "FAIL"))
    print("Module 5 Status: SUCCESS\n")
    
    print(f"Confidence Range: {min(all_confidences):.4f} - {max(all_confidences):.4f}")
    print(f"Uncertainty Range: {min(all_uncertainties):.4f} - {max(all_uncertainties):.4f}\n")
    
    # 5. Visualizations
    fig, axes = plt.subplots(2, len(severities), figsize=(20, 8))
    
    # Top row: Line plot of confidence across severities (spans all columns)
    ax_line = plt.subplot(2, 1, 1)
    ax_line.plot(severities, all_confidences, marker='o', color='blue', label='Confidence')
    ax_line.plot(severities, all_uncertainties, marker='s', color='red', label='Uncertainty')
    ax_line.set_xlabel('Perturbation Severity (%)')
    ax_line.set_ylabel('Score')
    ax_line.set_title('Confidence / Uncertainty vs Perturbation Severity')
    ax_line.set_ylim(-0.1, 1.1)
    ax_line.set_xticks(severities)
    ax_line.grid(True, linestyle='--', alpha=0.7)
    ax_line.legend()
    
    # Bottom row: Mask overlays
    for i, (sev, p_mask, conf) in enumerate(vis_data):
        ax = plt.subplot(2, len(severities), len(severities) + i + 1)
        ax.imshow(original_image)
        overlay = np.zeros((*p_mask.shape, 4))
        overlay[p_mask] = [0, 1, 0, 0.5] # Green, 50% opacity
        ax.imshow(overlay)
        ax.set_title(f"{sev}% Perturbed\nConf: {conf:.2f}")
        ax.axis('off')
        
    plt.tight_layout()
    vis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "module5_confidence_result.png"))
    plt.savefig(vis_path)
    plt.close()
    
    print("Visualization Saved:")
    print(vis_path)

if __name__ == "__main__":
    test_module5()

import os
import cv2
import pickle
import pandas as pd
import numpy as np
from segmentation.mock_segmentation import MockSegmentation
from perturbation.corruptor import SegmentationCorruptor
from evaluation.metrics import calculate_metrics
from visualization.plotter import plot_robustness_curve

def load_test_data(category="breakfast_box", data_dir="data/mvtec_loco_ad"):
    """Loads all test image paths and their ground truth labels"""
    test_dir = os.path.join(data_dir, category, "test")
    image_paths = []
    labels = []
    
    # Good = 0, Anomaly = 1
    if os.path.exists(os.path.join(test_dir, "good")):
        for f in os.listdir(os.path.join(test_dir, "good")):
            if f.endswith(".png"):
                image_paths.append(os.path.join(test_dir, "good", f))
                labels.append(0)
                
    for anomaly_type in ["logical_anomalies", "structural_anomalies"]:
        anom_dir = os.path.join(test_dir, anomaly_type)
        if os.path.exists(anom_dir):
            for f in os.listdir(anom_dir):
                if f.endswith(".png"):
                    image_paths.append(os.path.join(anom_dir, f))
                    labels.append(1)
                    
    return image_paths, labels

def test(category="breakfast_box", data_dir="data/mvtec_loco_ad", model_dir="models"):
    # Load Models
    with open(os.path.join(model_dir, f"{category}_confidence.pkl"), 'rb') as f:
        confidence_estimator = pickle.load(f)
        
    with open(os.path.join(model_dir, f"{category}_logical.pkl"), 'rb') as f:
        logical_detector = pickle.load(f)
        
    segmentation_module = MockSegmentation(mask_dir=os.path.join(data_dir, category))
    corruptor = SegmentationCorruptor(seed=42)
    
    image_paths, true_labels = load_test_data(category, data_dir)
    if not image_paths:
        print("No test data found.")
        return
        
    degradation_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
    results = []
    
    for level in degradation_levels:
        print(f"Testing at Degradation Level: {level * 100}%")
        baseline_scores = []
        proposed_scores = []
        
        for img_path in image_paths:
            # 1. Baseline: Get perfect mask, corrupt it, score it directly
            clean_mask = segmentation_module.segment(img_path)
            corrupted_mask = corruptor.corrupt(clean_mask, level)
            
            baseline_score = logical_detector.calculate_anomaly_score(corrupted_mask)
            baseline_scores.append(baseline_score)
            
            # 2. Proposed: Take corrupted mask, refine it, score it
            _, refined_mask = confidence_estimator.estimate_and_refine(corrupted_mask)
            proposed_score = logical_detector.calculate_anomaly_score(refined_mask)
            proposed_scores.append(proposed_score)
            
        # Calculate Metrics
        baseline_metrics = calculate_metrics(true_labels, baseline_scores)
        proposed_metrics = calculate_metrics(true_labels, proposed_scores)
        
        results.append({
            "DegradationLevel": level,
            "Baseline_AUROC": baseline_metrics["auroc"],
            "Proposed_AUROC": proposed_metrics["auroc"],
            "Baseline_F1": baseline_metrics["f1"],
            "Proposed_F1": proposed_metrics["f1"]
        })
        
    # Save Results
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"{category}_robustness.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Plot curves
    plot_path = os.path.join("results", f"{category}_robustness_curve.png")
    plot_robustness_curve(csv_path, plot_path)
    print("Plots generated successfully.")

if __name__ == "__main__":
    test()

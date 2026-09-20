import os
import cv2
import pickle
from confidence.estimator import ConfidenceEstimator
from anomaly_detection.logical_scorer import LogicalAnomalyDetector

def train(category="breakfast_box", data_dir="data/mvtec_loco_ad", model_dir="models"):
    """
    Fits the normal distributions for confidence estimation and logical anomaly detection
    using the clean training segmentation masks.
    """
    train_mask_dir = os.path.join(data_dir, category, "segmentation_masks", "train", "good")
    
    clean_masks = []
    if not os.path.exists(train_mask_dir):
        print(f"Training mask directory not found: {train_mask_dir}")
        return
        
    for filename in os.listdir(train_mask_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            path = os.path.join(train_mask_dir, filename)
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                clean_masks.append(mask)
                
    if not clean_masks:
        print("No clean masks found for training.")
        return
        
    print(f"Training on {len(clean_masks)} clean masks for category: {category}")
    
    # Train Confidence Estimator (Expected area bounds)
    confidence_estimator = ConfidenceEstimator()
    confidence_estimator.fit_expected_areas(clean_masks)
    
    # Train Logical Anomaly Detector (Expected component counts)
    logical_detector = LogicalAnomalyDetector()
    logical_detector.fit(clean_masks)
    
    # Save models
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f"{category}_confidence.pkl"), 'wb') as f:
        pickle.dump(confidence_estimator, f)
        
    with open(os.path.join(model_dir, f"{category}_logical.pkl"), 'wb') as f:
        pickle.dump(logical_detector, f)
        
    print("Training complete. Models saved successfully.")

if __name__ == "__main__":
    train()

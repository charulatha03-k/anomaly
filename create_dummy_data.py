import os
import cv2
import numpy as np

def create_dummy_dataset(base_dir="data/mvtec_loco_ad", category="breakfast_box"):
    """
    Creates a dummy dataset mimicking the MVTec LOCO AD folder structure.
    Generates synthetic images with simple shapes (components) and their corresponding segmentation masks.
    """
    # Create directory structure
    dirs = [
        f"{base_dir}/{category}/train/good",
        f"{base_dir}/{category}/test/good",
        f"{base_dir}/{category}/test/logical_anomalies",
        f"{base_dir}/{category}/test/structural_anomalies",
        f"{base_dir}/{category}/ground_truth/logical_anomalies",
        f"{base_dir}/{category}/ground_truth/structural_anomalies",
        f"{base_dir}/{category}/segmentation_masks/train/good",
        f"{base_dir}/{category}/segmentation_masks/test/good",
        f"{base_dir}/{category}/segmentation_masks/test/logical_anomalies",
        f"{base_dir}/{category}/segmentation_masks/test/structural_anomalies"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    def generate_image_and_mask(image_path, mask_path, is_anomaly=False, anomaly_type=None):
        # Create a blank image (e.g., a dark background like a conveyor belt)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 50
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        # We simulate "components" as circles and rectangles.
        # Normal configuration: 2 circles (class 1) and 1 rectangle (class 2)
        
        num_circles = 2
        num_rects = 1
        
        if is_anomaly and anomaly_type == "logical":
            # Quantity anomaly: 3 circles instead of 2
            num_circles = 3
            
        # Draw circles (component type 1)
        # Class 1 mask will have pixel value 1
        for i in range(num_circles):
            # Normal position roughly around center left/right
            if i == 0:
                center = (70, 128)
            elif i == 1:
                center = (180, 128)
            else:
                center = (128, 50) # Extra component
                
            if is_anomaly and anomaly_type == "logical" and num_circles == 2:
                # Spatial anomaly: circle moved
                center = (128, 200)
                
            cv2.circle(img, center, 30, (255, 0, 0), -1) # Blue circle
            cv2.circle(mask, center, 30, 1, -1) # Mask value 1
            
        # Draw rectangle (component type 2)
        # Class 2 mask will have pixel value 2
        for i in range(num_rects):
            top_left = (100, 180)
            bottom_right = (150, 230)
            
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), -1) # Green rect
            cv2.rectangle(mask, top_left, bottom_right, 2, -1) # Mask value 2
            
        if is_anomaly and anomaly_type == "structural":
            # Add a scratch (white line) over the background
            cv2.line(img, (20, 20), (100, 100), (255, 255, 255), 3)
            
        cv2.imwrite(image_path, img)
        cv2.imwrite(mask_path, mask) # Saving mask as a grayscale image (0, 1, 2)
        
    # Generate Train Good (5 images)
    for i in range(5):
        generate_image_and_mask(
            f"{base_dir}/{category}/train/good/{i:03d}.png",
            f"{base_dir}/{category}/segmentation_masks/train/good/{i:03d}.png"
        )
        
    # Generate Test Good (2 images)
    for i in range(2):
        generate_image_and_mask(
            f"{base_dir}/{category}/test/good/{i:03d}.png",
            f"{base_dir}/{category}/segmentation_masks/test/good/{i:03d}.png"
        )
        
    # Generate Test Logical Anomalies (2 images)
    for i in range(2):
        generate_image_and_mask(
            f"{base_dir}/{category}/test/logical_anomalies/{i:03d}.png",
            f"{base_dir}/{category}/segmentation_masks/test/logical_anomalies/{i:03d}.png",
            is_anomaly=True, anomaly_type="logical"
        )
        
    # Generate Test Structural Anomalies (2 images)
    for i in range(2):
        generate_image_and_mask(
            f"{base_dir}/{category}/test/structural_anomalies/{i:03d}.png",
            f"{base_dir}/{category}/segmentation_masks/test/structural_anomalies/{i:03d}.png",
            is_anomaly=True, anomaly_type="structural"
        )
        
    print(f"Dummy dataset created successfully for category: {category}")

if __name__ == "__main__":
    create_dummy_dataset()

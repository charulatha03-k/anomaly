import os
import glob
import cv2
from torch.utils.data import Dataset

class MVTecLOCODataset(Dataset):
    """
    PyTorch Dataset for MVTec LOCO AD Dataset.
    Supports loading normal and anomalous samples, with their associated
    ground truth masks (where available).
    """
    def __init__(self, root_dir, category='breakfast_box', split='train'):
        """
        Args:
            root_dir (str): Root directory of the MVTec LOCO AD dataset.
            category (str): The dataset category (e.g., 'breakfast_box').
            split (str): 'train', 'validation', or 'test'.
        """
        self.root_dir = root_dir
        self.category = category
        self.split = split
        
        self.category_dir = os.path.join(self.root_dir, self.category)
        if not os.path.exists(self.category_dir):
            raise FileNotFoundError(f"Category directory not found: {self.category_dir}. "
                                    f"Please ensure the MVTec LOCO AD dataset is downloaded and extracted.")

        self.samples = self._load_dataset_paths()

    def _load_dataset_paths(self):
        samples = []
        split_dir = os.path.join(self.category_dir, self.split)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Scan for all subdirectories (e.g., 'good', 'logical_anomalies', 'structural_anomalies')
        anomaly_types = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for anomaly_type in anomaly_types:
            img_dir = os.path.join(split_dir, anomaly_type)
            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
            
            for img_path in img_paths:
                is_anomaly = (anomaly_type != 'good')
                label = 1 if is_anomaly else 0
                
                # Ground truth paths for MVTec LOCO AD
                # GT is only available for test set anomalies. 
                gt_path = None
                if self.split == 'test' and is_anomaly:
                    img_name = os.path.basename(img_path)
                    img_base = os.path.splitext(img_name)[0]
                    
                    direct_gt = os.path.join(self.category_dir, "ground_truth", anomaly_type, img_name)
                    # LOCO AD format: ground_truth/logical_anomalies/000/000.png
                    folder_gt = os.path.join(self.category_dir, "ground_truth", anomaly_type, img_base, "000.png")
                    
                    if os.path.exists(direct_gt):
                        gt_path = direct_gt
                    elif os.path.exists(folder_gt):
                        gt_path = folder_gt
                    else:
                        gt_path = None # Do not invent paths if it genuinely doesn't exist
                
                samples.append({
                    "image_path": img_path,
                    "ground_truth_path": gt_path,
                    "category": self.category,
                    "anomaly_type": anomaly_type,
                    "label": label
                })
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load image
        image_path = sample_info["image_path"]
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        return {
            "image": image,
            "category": sample_info["category"],
            "label": sample_info["label"],
            "anomaly_type": sample_info["anomaly_type"],
            "image_path": image_path,
            "ground_truth_path": sample_info["ground_truth_path"]
        }


def test_dataloader():
    print("Testing MVTec LOCO AD Dataset Module...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "mvtec_loco_ad"))
    
    print(f"Expected Dataset Root: {root_dir}")
    
    if not os.path.exists(root_dir):
        print("Root directory not found.")
        return

    # Dynamically discover categories
    categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"\nDiscovered Categories: {categories}")
    
    # Process only one category for full print output, but verify counts for all
    target_category = "breakfast_box" if "breakfast_box" in categories else categories[0]
    
    try:
        train_dataset = MVTecLOCODataset(root_dir=root_dir, category=target_category, split="train")
        val_dataset = MVTecLOCODataset(root_dir=root_dir, category=target_category, split="validation")
        test_dataset = MVTecLOCODataset(root_dir=root_dir, category=target_category, split="test")
        
        print(f"\n--- Category: {target_category} ---")
        print(f"Number of TRAIN samples found: {len(train_dataset)}")
        print(f"Number of VALIDATION samples found: {len(val_dataset)}")
        print(f"Number of TEST samples found: {len(test_dataset)}")
        
        print("\n--- Test Sample (Normal) Example ---")
        normal_samples = [s for s in test_dataset if s['label'] == 0]
        if len(normal_samples) > 0:
            sample = normal_samples[0]
            img_shape = sample['image'].shape if sample['image'] is not None else "None"
            print(f"Image shape: {img_shape}")
            print(f"Category: {sample['category']}")
            print(f"Label (0=Normal, 1=Anomaly): {sample['label']}")
            print(f"Anomaly Type: {sample['anomaly_type']}")
            print(f"Image Path: {sample['image_path']}")
            print(f"GT Path: {sample['ground_truth_path']}")
        else:
            print("No normal samples found in the test set.")
            
        print("\n--- Test Sample (Anomalous) Example ---")
        anomaly_samples = [s for s in test_dataset if s['label'] == 1]
        if len(anomaly_samples) > 0:
            sample = anomaly_samples[0]
            img_shape = sample['image'].shape if sample['image'] is not None else "None"
            print(f"Image shape: {img_shape}")
            print(f"Category: {sample['category']}")
            print(f"Label (0=Normal, 1=Anomaly): {sample['label']}")
            print(f"Anomaly Type: {sample['anomaly_type']}")
            print(f"Image Path: {sample['image_path']}")
            print(f"GT Path: {sample['ground_truth_path']}")
            if sample['ground_truth_path'] is not None:
                print(f"GT exists on disk: {os.path.exists(sample['ground_truth_path'])}")
            else:
                print("GT exists on disk: False (None provided)")
        else:
            print("No anomaly samples found in the test set.")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")

if __name__ == "__main__":
    test_dataloader()

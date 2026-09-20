import streamlit as st
import cv2
import os
import pickle
import pandas as pd
from PIL import Image
import numpy as np

from segmentation.mock_segmentation import MockSegmentation
from perturbation.corruptor import SegmentationCorruptor
from confidence.estimator import ConfidenceEstimator
from anomaly_detection.logical_scorer import LogicalAnomalyDetector

st.set_page_config(layout="wide", page_title="Anomaly Detection Robustness Analysis")

st.title("Industrial Image Anomaly Detection: Robustness Analysis")
st.markdown("""
This system evaluates the robustness of an industrial anomaly detection framework against segmentation errors.
It introduces a **Confidence-Aware Segmentation Refinement** module to filter out unreliable regions.
""")

# Sidebar settings
st.sidebar.header("Settings")
category = st.sidebar.selectbox("Product Category", ["breakfast_box"])
degradation_level = st.sidebar.slider("Segmentation Degradation Level (%)", 0, 100, 10, step=5) / 100.0

# Load models
model_dir = "models"
data_dir = "data/mvtec_loco_ad"

@st.cache_resource
def load_system_models(cat):
    try:
        with open(os.path.join(model_dir, f"{cat}_confidence.pkl"), 'rb') as f:
            conf_est = pickle.load(f)
        with open(os.path.join(model_dir, f"{cat}_logical.pkl"), 'rb') as f:
            log_det = pickle.load(f)
        return conf_est, log_det
    except Exception as e:
        return None, None

confidence_estimator, logical_detector = load_system_models(category)

if confidence_estimator is None:
    st.error("Models not found. Please run the training script first.")
    st.stop()

# Image selection
test_dir = os.path.join(data_dir, category, "test")
image_options = []
if os.path.exists(test_dir):
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.endswith(".png"):
                image_options.append(os.path.join(root, f))
                
if not image_options:
    st.warning("No test images found.")
    st.stop()
    
selected_image_path = st.selectbox("Select Test Image", image_options)

col1, col2, col3 = st.columns(3)

# 1. Original Image
img = cv2.imread(selected_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
col1.subheader("Original Image")
col1.image(img_rgb, use_column_width=True)

# 2. Corrupted Mask (Baseline)
seg_module = MockSegmentation(mask_dir=os.path.join(data_dir, category))
corruptor = SegmentationCorruptor()

clean_mask = seg_module.segment(selected_image_path)
corrupted_mask = corruptor.corrupt(clean_mask, degradation_level)

col2.subheader("Corrupted Segmentation (Baseline)")
# Convert mask to visual colors
def mask_to_color(m):
    color_m = np.zeros((*m.shape, 3), dtype=np.uint8)
    color_m[m == 1] = [255, 0, 0] # Blue
    color_m[m == 2] = [0, 255, 0] # Green
    return color_m

col2.image(mask_to_color(corrupted_mask), use_column_width=True)

# 3. Refined Mask (Proposed)
confidence_map, refined_mask = confidence_estimator.estimate_and_refine(corrupted_mask)

col3.subheader("Refined Segmentation (Proposed)")
col3.image(mask_to_color(refined_mask), use_column_width=True)

st.divider()

# Results Section
col_res1, col_res2, col_res3 = st.columns(3)

baseline_score = logical_detector.calculate_anomaly_score(corrupted_mask)
proposed_score = logical_detector.calculate_anomaly_score(refined_mask)

# Dummy threshold for demo (in real system, find via validation set)
threshold = 5.0 

col_res1.metric("Baseline Anomaly Score", f"{baseline_score:.2f}")
baseline_status = "ANOMALY" if baseline_score > threshold else "NORMAL"
col_res1.markdown(f"**Baseline Prediction:** :{'red' if baseline_status == 'ANOMALY' else 'green'}[{baseline_status}]")

col_res2.metric("Proposed Anomaly Score", f"{proposed_score:.2f}")
proposed_status = "ANOMALY" if proposed_score > threshold else "NORMAL"
col_res2.markdown(f"**Proposed Prediction:** :{'red' if proposed_status == 'ANOMALY' else 'green'}[{proposed_status}]")

# Show Confidence Map
col_res3.subheader("Confidence Map")
# Scale confidence 0-1 to 0-255 grayscale
conf_vis = (confidence_map * 255).astype(np.uint8)
col_res3.image(conf_vis, use_column_width=True, clamp=True)

st.divider()

# Robustness Graph
st.subheader("Robustness Evaluation")
graph_path = os.path.join("results", f"{category}_robustness_curve_auroc.png")
if os.path.exists(graph_path):
    st.image(graph_path)
else:
    st.info("Run test.py to generate robustness evaluation graphs.")

import streamlit as st
import os
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Product Anomaly Detection", layout="wide")

st.sidebar.title("Pipeline Modules")
modules = [
    "1. Dataset Loading",
    "2. Image Preprocessing",
    "3. SAM 2 Segmentation",
    "4. Controlled Perturbation",
    "5. Confidence & Uncertainty",
    "6. Mask Refinement",
    "7. Component Extraction",
    "8. Quantity Analysis",
    "9. Spatial Relationship Analysis",
    "10. Logical Anomaly Scoring",
    "11. Final Evaluation"
]
selected_module = st.sidebar.radio("Navigate through the project:", modules)

base_dir = os.path.dirname(os.path.abspath(__file__))

st.title("Enhanced Systematic Robustness Analysis for Product Image Anomaly Detection")
st.markdown("### " + selected_module)
st.markdown("---")

def display_image(filename, caption=""):
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.warning(f"Result image not found: {filename}")

if selected_module == "1. Dataset Loading":
    st.write("Module 1 loads the MVTec LOCO AD dataset, extracting the normal and anomalous images for the `breakfast_box` category.")
    st.write("Loaded Dataset: **MVTec LOCO AD**")
    st.write("Category: **breakfast_box**")
    st.write("Format: **PyTorch Dataset**")

elif selected_module == "2. Image Preprocessing":
    st.write("Module 2 prepares the raw images for the segmentation pipeline. It resizes the images while preserving aspect ratio and pads them to the required input size.")
    st.write("Target Dimensions: **(1280, 1600, 3)**")

elif selected_module == "3. SAM 2 Segmentation":
    st.write("Module 3 uses **SAM 2.1 Hiera Tiny** to generate high-quality segmentation masks of the product components.")
    st.write("SAM 2 was run on CPU to generate individual masks for each physical component.")
    display_image("module3_segmentation_result.png", "SAM 2 Base Segmentation Masks")

elif selected_module == "4. Controlled Perturbation":
    st.write("Module 4 introduces controlled segmentation degradations (e.g., erosion, dilation, boundary distortion) at varying severities (0%, 10%, 20%, 30%).")
    st.write("This simulates real-world failure cases where the segmentation model underperforms.")
    display_image("module4_perturbation_result.png", "Perturbed Segmentation Masks")

elif selected_module == "5. Confidence & Uncertainty":
    st.write("Module 5 calculates Confidence and Uncertainty scores for the generated masks by comparing original/perturbed properties and SAM 2 stability scores.")
    st.write("Features extracted: Area Consistency, Boundary Stability.")
    display_image("module5_confidence_result.png", "Confidence & Uncertainty Estimation")

elif selected_module == "6. Mask Refinement":
    st.write("Module 6 utilizes the confidence scores from Module 5 to filter out unreliable and severely degraded masks using a strict **0.50 threshold**.")
    st.write("This prevents false anomalous detections caused purely by bad segmentation.")
    display_image("module6_refinement_result.png", "Confidence-Aware Refined Masks")

elif selected_module == "7. Component Extraction":
    st.write("Module 7 translates the pixel-level refined masks into discrete component features, extracting Area, Bounding Boxes, and Centroids for logical analysis.")
    display_image("module7_component_extraction_result.png", "Extracted Components and Bounding Boxes")

elif selected_module == "8. Quantity Analysis":
    st.write("Module 8 analyzes the total count of valid components after refinement, forming the basis for quantity-based anomaly detection.")
    display_image("module8_quantity_analysis_result.png", "Component Quantity vs Degradation")

elif selected_module == "9. Spatial Relationship Analysis":
    st.write("Module 9 computes the pairwise spatial relationships between components (LEFT/RIGHT, ABOVE/BELOW, NEAR/FAR), building a structural graph of the product.")
    display_image("module9_spatial_relationship_result.png", "Spatial Relationship Graph")

elif selected_module == "10. Logical Anomaly Scoring":
    st.write("Module 10 fuses the Quantity and Spatial features to calculate the Final Logical Anomaly Score. It compares the Baseline (Naive) approach to our Proposed (Confidence-Aware) approach.")
    display_image("module10_logical_anomaly_result.png", "Baseline vs Proposed Logical Anomaly Score")

elif selected_module == "11. Final Evaluation":
    st.write("The Final Evaluation proves the effectiveness of the proposed Confidence-Aware pipeline in maintaining accurate anomaly detection under severe segmentation degradation.")
    
    csv_path = os.path.join(base_dir, "evaluation_results", "robustness_evaluation.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)
    
    eval_img = os.path.join("evaluation_results", "robustness_evaluation.png")
    display_image(eval_img, "Robustness Decay Comparison")

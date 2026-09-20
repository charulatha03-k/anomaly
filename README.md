# Enhanced Systematic Robustness Analysis for Product Image Anomaly Detection

This project is a B.Tech final-year industrial computer vision mini-project based on the reference paper:
*"A Unified Framework for Image Anomaly Detection via Reconstruction, Segmentation and Spatial Relationship Modeling"*.

## Project Objective
The original framework relies on segmentation outputs to detect logical anomalies (quantity and spatial relationships). However, segmentation errors (due to noise, illumination, etc.) can severely degrade anomaly detection performance.

This project introduces a **Confidence-Aware Segmentation Filtering** module to improve the robustness of the system against segmentation degradations.

## Architecture

1.  **Image Preprocessing & Segmentation:** Extracts component masks from the industrial product image.
2.  **Controlled Perturbation (Simulation):** Artificially introduces noise, morphological degradations, and false positives/negatives into the mask to simulate real-world errors.
3.  **Confidence/Uncertainty Estimation (Proposed):** Analyzes the corrupted mask using heuristics (expected component sizes, shapes) to identify unreliable regions.
4.  **Mask Refinement:** Filters out low-confidence regions from the segmentation mask.
5.  **Logical Anomaly Detection:** Extracts component quantities and spatial relationships and compares them against learned normal distributions.

## Setup Instructions

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Generate the dummy dataset for local testing:
    ```bash
    python create_dummy_data.py
    ```
3.  Train the expected distributions using clean masks:
    ```bash
    python train.py
    ```
4.  Run the robustness evaluation to generate CSV results and graphs:
    ```bash
    python test.py
    ```
5.  Launch the interactive interface:
    ```bash
    streamlit run app.py
    ```

## Project Structure
-   `data/`: Contains the MVTec LOCO AD dataset (or dummy dataset).
-   `models/`: Saved `pickle` files for confidence and logical scoring distributions.
-   `segmentation/`: Loads ground truth masks (mocking a perfect segmentation network).
-   `perturbation/`: Corrupts masks with noise and morphological operations.
-   `confidence/`: Assigns confidence scores and refines masks.
-   `anomaly_detection/`: Calculates logical anomaly scores.
-   `evaluation/`: Calculates AUROC and F1 metrics.
-   `visualization/`: Generates Matplotlib graphs.
-   `app.py`: Streamlit frontend.

## Research Contribution
"An enhanced systematic robustness analysis framework for industrial product image anomaly detection that evaluates the effect of controlled segmentation perturbations on logical anomaly detection and introduces confidence-aware segmentation refinement to reduce the influence of unreliable segmentation outputs."

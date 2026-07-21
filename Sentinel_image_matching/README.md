# Sentinel-2 Image Matching (Task 2)

This repository contains the solution for **Task 2: Computer Vision - Sentinel-2 Image Matching**. The goal of this project is to build an algorithm capable of robustly matching satellite images of the same geographic location taken across different seasons. 

Because classical hand-crafted descriptors (like SIFT or ORB) struggle with drastic visual changes between seasons (e.g., summer foliage vs. winter snow), this solution utilizes a deep learning approach combining **LoFTR (Local Feature TRansformer)** for dense feature extraction and **RANSAC** for geometric verification.

## 📂 Repository Structure

* `dataset_creation.ipynb` - Jupyter notebook explaining the dataset parsing, high-resolution patching, pseudo-label generation, and dataset export.
* `algorithm.py` - Core Python module containing the `FeatureMatcher` class (LoFTR backbone + RANSAC implementation).
* `inference.py` - Command-line interface (CLI) script for running single-pair inference or batch evaluation across the dataset.
* `demo.ipynb` - Jupyter notebook demonstrating the inference pipeline, visual evaluation, and aggregated quantitative metrics.
* `requirements.txt` - Required Python libraries and dependencies.

## Dataset Preparation

The dataset utilizes the **Deforestation in Ukraine from Sentinel-2 data** (sourced from Kaggle). This dataset provides excellent multi-temporal image pairs ideal for cross-season matching.

The `dataset_creation.ipynb` pipeline processes the raw data by:
1.  **Cross-Season Pairing:** Extracting geographic Tile IDs to pair images from the same location but at different dates.
2.  **Tiling/Patching:** Slicing large rasters into smaller, high-resolution patches to prevent memory bottlenecks while preserving quality.
3.  **Pseudo-Labeling:** Utilizing a pre-trained outdoor LoFTR model to find robust keypoints across seasons and filtering them via RANSAC.
4.  **Dataset Export:** Saving the resulting image patches and generating a `dataset_labels.csv` file containing the serialized keypoints and match coordinates.

## Core Algorithm (`algorithm.py`)

The matching algorithm is encapsulated in the `FeatureMatcher` class. The pipeline operates as follows:
* **Preprocessing:** Normalizes RGB images into float32 tensors and converts them to grayscale.
* **Deep Feature Extraction:** Uses Kornia's LoFTR implementation (`K.feature.LoFTR`) to densely extract and match features.
* **Confidence Filtering:** Applies a strict confidence threshold (default `0.8`) to filter out weak matches.
* **Geometric Verification:** Uses OpenCV's `findFundamentalMat` with RANSAC (`USAC_ACCURATE`) to separate true matches (inliers) from false positives.

## Inference & Evaluation

### CLI Usage (`inference.py`)

The inference script is built to handle both individual image pairs and automated batch evaluation across the entire dataset. 

**Single Pair Mode:**
Run the algorithm on two specific images and save the visualization.
```bash
python inference.py --img1 path/to/seasonA.png --img2 path/to/seasonB.png --out match.png
```

**Batch Evaluation Mode:**
Execute inference across all patch pairs defined in the dataset CSV.
```bash
python inference.py --csv path/to/dataset_labels.csv --dataset_dir path/to/images --out evaluation_results/
```

**Metrics Addressed**
During batch evaluation, the script computes critical performance metrics:
1. **Mean Inliers:** The average number of geometrically verified keypoints found per pair.
2. **Inlier Ratio (Matching Score):** The ratio of valid inliers against the total raw keypoints detected.
3. **Success Rate:** The percentage of image pairs that successfully yielded ≥15 robust inlier matches.

## Demo Notebook (demo.ipynb)
The final demo.ipynb serves as a comprehensive report for the model's capabilities, utilizing functions imported directly from inference.py. It covers:
1. **Single Pair Inspection:** Visualizing raw feature extraction and geometric inlier filtering.
2. **Full Dataset Evaluation:** Running the batch inference pipeline.
3. **Quantitative Analysis:** Displaying the Mean Inliers, Inlier Ratios, and Success Rate distributions.
4. **Qualitative Verification:** Showcasing visualizations of both the best and worst cross-season matching outcomes.
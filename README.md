
# Practical Assignment 1 – Image Processing and Analysis
**Project Title:** People Counting in a Beach Surveillance Scene
### Group 4:
- **Ahmad Osaid Majed**
- **Alejandro Rafael Bordón Duarte**

---

## 1. Objective

The aim of this project is to design and implement an image processing system capable of **detecting and counting people** in a sequence of still images captured by a fixed beach surveillance camera.
The system applies classical computer vision techniques for low- and mid-level image processing, with a focus on robustness against lighting changes, shadows, and background complexity.

---

## 2. Description

This work consists of the development of a complete image processing pipeline to automatically estimate the number of people present in each frame.
The project integrates several fundamental operations in digital image analysis, including:

- Image enhancement through intensity transformations and histogram equalization
- Noise reduction and spatial filtering (Gaussian, bilateral, median)
- Edge detection and morphological post-processing
- Segmentation and object counting
- Quantitative evaluation of detection and counting performance

The dataset comprises static RGB images taken throughout the day, exhibiting different illumination conditions and crowd densities.

---

## 3. Execution Environment

All experiments run in Conda env **`sub11761`** (Python 3.11).

```bash
conda activate sub11761
```

### Quickstart (reproducible)
```bash
# install deps if needed
pip install -r requirements.txt

# default: clears outputs and runs pipelines A, B (legacy), and C
python run_pipeline.py

# run a single pipeline (A, B, or C)
python run_pipeline.py --pipeline A
python run_pipeline.py --pipeline C
```

Outputs:
- Metrics: `data/outputs/metrics/counts_all.csv` and `summary_all.csv`.
- Visualizations: `data/outputs/viz/A` and `data/outputs/viz/C`.
- Pipeline C assets: `data/outputs/viz/c_background/` (median background) and `data/outputs/viz/c_roi/` (ROI/water/sand masks).

Data layout:
- Raw images: `data/raw/*.jpg`
- Annotations (points): `data/annotations/*_points.csv`

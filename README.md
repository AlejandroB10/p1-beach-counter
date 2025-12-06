
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

All experiments are conducted within the Conda environment **`sub11761`**, using **Python 3.11**.

### Required Python Packages

Only standard scientific and computer vision libraries are employed:

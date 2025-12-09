"""Edge detection functions."""

import cv2
import numpy as np


def canny_gray(img_bgr: np.ndarray, low: int = 50, high: int = 150, aperture: int = 3) -> np.ndarray:
    """Canny edge detection on grayscale with Gaussian blur."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return cv2.Canny(blur, low, high, apertureSize=aperture)


def canny_lab_multichannel(img_bgr: np.ndarray, low: int = 50, high: int = 150, aperture: int = 3) -> np.ndarray:
    """Canny on each LAB channel, combined with OR to capture chroma edges."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    edges = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for i in range(3):
        blur = cv2.GaussianBlur(lab[:, :, i], (5, 5), 1.0)
        edges = np.bitwise_or(edges, cv2.Canny(blur, low, high, apertureSize=aperture))
    return edges

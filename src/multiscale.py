"""Multi-scale and multi-channel processing utilities."""

import cv2
import numpy as np


def detect_dark_regions_value(img_bgr, otsu: bool = True, thresh: int = 60):
    """Detect dark regions on the HSV Value channel (Otsu or fixed threshold)."""
    value = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    blur = cv2.GaussianBlur(value, (5, 5), 1.0)
    if otsu:
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)
    return mask


def canny_parallel_channels(img_bgr, low=50, high=150, colorspace='BGR'):
    """Apply Canny on each channel of a colorspace, combine with OR."""
    if colorspace == 'HSV':
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    elif colorspace == 'LAB':
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    else:
        img = img_bgr

    edges = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for i in range(3):
        blur = cv2.GaussianBlur(img[:, :, i], (5, 5), 1.0)
        edges = cv2.bitwise_or(edges, cv2.Canny(blur, low, high))
    return edges

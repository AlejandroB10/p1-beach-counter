"""Multi-scale and multi-channel helpers."""

import cv2
import numpy as np


def detect_dark_regions_percentile(img_bgr, percentile: float = 35.0):
    """Detect regions darker than a given percentile of the image."""
    value = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    blur = cv2.GaussianBlur(value, (5, 5), 1.0)
    
    thresh_val = np.percentile(blur, percentile)
    _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
    return mask.astype(np.uint8)

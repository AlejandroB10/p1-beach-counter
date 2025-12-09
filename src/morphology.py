"""Morphological operations for binary masks."""

import cv2
import numpy as np


def close_open(mask: np.ndarray, k: int = 3, it_close: int = 1, it_open: int = 1) -> np.ndarray:
    """Closing followed by opening to fill holes and remove noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    result = mask.astype(np.uint8)

    for _ in range(it_close):
        result = cv2.erode(cv2.dilate(result, kernel), kernel)

    for _ in range(it_open):
        result = cv2.dilate(cv2.erode(result, kernel), kernel)

    return result

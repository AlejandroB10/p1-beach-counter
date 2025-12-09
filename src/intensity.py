"""Intensity and contrast enhancement functions."""

import cv2
import numpy as np
from typing import Tuple


def clahe_on_L(img_bgr: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, np.ndarray]:
    """Apply CLAHE on the L channel in LAB space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    L_enhanced = clahe.apply(L)
    enhanced = cv2.cvtColor(cv2.merge([L_enhanced, a, b]), cv2.COLOR_LAB2BGR)
    return enhanced, L_enhanced

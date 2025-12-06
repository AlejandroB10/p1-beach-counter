"""Morphological operations for cleaning up binary masks."""
import cv2
import numpy as np


def close_open(mask: np.ndarray, k: int = 3, it_close: int = 1, it_open: int = 1) -> np.ndarray:
    """Apply morphological closing followed by opening.
    
    Closing fills small holes and gaps, opening removes small noise.
    Uses elliptical structuring element.
    
    Args:
        mask: Binary mask (uint8)
        k: Kernel size (k x k)
        it_close: Number of iterations for closing
        it_open: Number of iterations for opening
        
    Returns:
        Cleaned binary mask (uint8)
    """
    # Create elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    
    # Apply closing (dilation followed by erosion)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=it_close)
    
    # Apply opening (erosion followed by dilation)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=it_open)
    
    return mask_opened

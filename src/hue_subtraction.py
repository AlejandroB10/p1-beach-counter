"""HSV-based background subtraction utilities."""

import cv2
import numpy as np


def subtract_hue_channel(
    img_bgr: np.ndarray,
    bg_bgr: np.ndarray
) -> np.ndarray:
    """
    Subtract hue channel of background from image, keeping original S and V.
    
    Args:
        img_bgr: Current image in BGR format
        bg_bgr: Background image in BGR format
        
    Returns:
        Subtracted image in BGR format
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bg_hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
    
    # Subtract only hue channel
    h_sub = img_hsv[:, :, 0] - bg_hsv[:, :, 0]
    
    # Merge with original S and V channels
    subtracted_hsv = cv2.merge([
        h_sub,
        img_hsv[:, :, 1],   # original S
        img_hsv[:, :, 2]    # original V
    ])
    
    # Convert back to BGR
    subtracted_bgr = cv2.cvtColor(subtracted_hsv, cv2.COLOR_HSV2BGR)
    return subtracted_bgr


def otsu_threshold(
    img_bgr: np.ndarray,
    blur_kernel: int = 5,
    invert: bool = False
) -> tuple[np.ndarray, float]:
    """
    Apply Otsu's automatic thresholding.
    
    Args:
        img_bgr: Input image in BGR format
        blur_kernel: Gaussian blur kernel size (must be odd)
        invert: Whether to invert the thresholded result
        
    Returns:
        Tuple of (thresholded binary image, calculated threshold value)
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (blur_kernel, blur_kernel), 0)
    
    threshold_val, otsu_thresh = cv2.threshold(
        img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    if invert:
        otsu_thresh = 255 - otsu_thresh
    
    return otsu_thresh, threshold_val

"""Edge detection functions using Canny on different color spaces."""
import cv2
import numpy as np


def canny_gray(img_bgr: np.ndarray, low: int = 50, high: int = 150, aperture: int = 3) -> np.ndarray:
    """Apply Canny edge detection on grayscale conversion of BGR image.
    
    Args:
        img_bgr: Input BGR image
        low: Lower threshold for Canny
        high: Upper threshold for Canny
        aperture: Aperture size for Sobel operator
        
    Returns:
        Binary edge mask (uint8, values 0 or 255)
    """
    # Convert to grayscale (luma)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for stability
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Canny edge detection
    edges = cv2.Canny(gray_blur, low, high, apertureSize=aperture)
    
    return edges


def canny_L(l_channel: np.ndarray, low: int = 50, high: int = 150, aperture: int = 3) -> np.ndarray:
    """Apply Canny edge detection on L* channel from LAB.
    
    Args:
        l_channel: L* channel (grayscale, uint8)
        low: Lower threshold for Canny
        high: Upper threshold for Canny
        aperture: Aperture size for Sobel operator
        
    Returns:
        Binary edge mask (uint8, values 0 or 255)
    """
    # Apply Gaussian blur for stability
    l_blur = cv2.GaussianBlur(l_channel, (5, 5), 1.0)
    
    # Canny edge detection
    edges = cv2.Canny(l_blur, low, high, apertureSize=aperture)
    
    return edges


def canny_lab_fused(lab: np.ndarray, low: int = 50, high: int = 150, aperture: int = 3) -> np.ndarray:
    """Apply Canny on each LAB channel and fuse with bitwise OR.
    
    This detects both luminance and chromatic edges.
    
    Args:
        lab: LAB image (3 channels)
        low: Lower threshold for Canny
        high: Upper threshold for Canny
        aperture: Aperture size for Sobel operator
        
    Returns:
        Binary edge mask (uint8, values 0 or 255) - OR fusion of all channels
    """
    L, a, b = cv2.split(lab)
    
    # Apply Canny to each channel
    edges_L = cv2.Canny(cv2.GaussianBlur(L, (5, 5), 1.0), low, high, apertureSize=aperture)
    edges_a = cv2.Canny(cv2.GaussianBlur(a, (5, 5), 1.0), low, high, apertureSize=aperture)
    edges_b = cv2.Canny(cv2.GaussianBlur(b, (5, 5), 1.0), low, high, apertureSize=aperture)
    
    # Fuse with bitwise OR
    edges_fused = cv2.bitwise_or(edges_L, cv2.bitwise_or(edges_a, edges_b))
    
    return edges_fused

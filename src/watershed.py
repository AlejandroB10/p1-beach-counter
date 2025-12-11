"""Watershed segmentation and distance transform utilities."""

import cv2
import numpy as np
from typing import List, Tuple


Box = Tuple[int, int, int, int]  # x, y, w, h


def filter_components_by_geometry(
    markers: np.ndarray,
    min_size: int = 15,
    max_size: int = 2000,
    max_aspect_ratio: float = 5.0
) -> np.ndarray:
    """
    Filter connected components by size and aspect ratio.
    
    Args:
        markers: Labeled component image from connectedComponents
        min_size: Minimum component size in pixels
        max_size: Maximum component size in pixels
        max_aspect_ratio: Maximum allowed width/height ratio
        
    Returns:
        Mask with filtered components removed (set to 255)
    """
    labels = np.unique(markers)
    labels = labels[labels > 0]  # Skip background
    
    filtered_mask = np.zeros(markers.shape, dtype=np.uint8)
    
    for lbl in labels:
        component = (markers == lbl).astype(np.uint8)
        size = component.sum()
        
        # Filter by size
        if size < min_size or size > max_size:
            filtered_mask[markers == lbl] = 255
            continue
        
        # Find contours to compute aspect ratio
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            continue
        
        # Use minAreaRect for rotated components
        rect = cv2.minAreaRect(contours[0])
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
        
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            filtered_mask[markers == lbl] = 255
    
    return filtered_mask


def watershed_segmentation(
    binary_mask: np.ndarray,
    img_color: np.ndarray,
    dist_threshold_factor: float = 0.01,
    dilate_iterations: int = 2
) -> np.ndarray:
    """
    Apply watershed segmentation to separate touching objects.
    
    Args:
        binary_mask: Binary mask of objects (255 = object, 0 = background)
        img_color: Original color image for watershed
        dist_threshold_factor: Factor to multiply distance max for sure foreground
        dilate_iterations: Number of iterations for background dilation
        
    Returns:
        Watershed markers array where each object has unique label
    """
    # Distance transform
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Threshold to get sure foreground
    _, sure_fg = cv2.threshold(
        dist, dist_threshold_factor * dist.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)
    
    # Dilate to get sure background
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_mask, kernel, iterations=dilate_iterations)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 so background is not 0 (required for watershed)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(img_color, markers)
    
    return markers


def draw_watershed_boundaries(
    img_bgr: np.ndarray,
    markers: np.ndarray,
    boundary_color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """
    Draw watershed boundaries on image.
    
    Args:
        img_bgr: Original image
        markers: Watershed markers (boundaries marked with -1)
        boundary_color: BGR color for boundaries
        
    Returns:
        Image with boundaries drawn
    """
    result = img_bgr.copy()
    result[markers == -1] = boundary_color
    return result


def markers_to_boxes_centers(
    markers: np.ndarray,
    y_offset: int = 0
) -> Tuple[List[Box], List[Tuple[int, int]]]:
    """
    Convert watershed markers to bounding boxes and centers.
    
    Args:
        markers: Watershed markers array
        y_offset: Vertical offset to add to coordinates (for cropped images)
        
    Returns:
        Tuple of (boxes, centers) where boxes are (x, y, w, h) and centers are (cx, cy)
    """
    labels = np.unique(markers)
    # Filter out background (0), unknown (-1), and boundary (0, -1)
    labels = labels[(labels > 0)]
    
    boxes = []
    centers = []
    
    for lbl in labels:
        ys, xs = np.where(markers == lbl)
        if len(xs) == 0:
            continue
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        
        cx = int(xs.mean())
        cy = int(ys.mean()) + y_offset
        
        boxes.append((x_min, y_min + y_offset, w, h))
        centers.append((cx, cy))
    
    return boxes, centers

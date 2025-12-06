"""
Connected components analysis and visualization for people counting.
Enhanced with geometric filters based on 11761 course techniques.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional


def calculate_compactness(area: float, perimeter: float) -> float:
    """
    Calculate compactness (circularity) of a shape.
    Compactness = 4π * area / perimeter²
    
    Perfect circle = 1.0
    More irregular shapes < 1.0
    
    Args:
        area: Component area in pixels
        perimeter: Component perimeter in pixels
        
    Returns:
        Compactness value (0-1)
    """
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def connected_components(
    mask: np.ndarray, 
    min_area: int = 50, 
    max_area: Optional[int] = None,
    max_aspect: float = 4.0,
    min_compactness: Optional[float] = None,
    image_height: Optional[int] = None,
    min_vertical_position: Optional[float] = None,
    max_vertical_position: Optional[float] = None,
    image_width: Optional[int] = None,
    min_horizontal_position: Optional[float] = None,
    max_horizontal_position: Optional[float] = None
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int]]]:
    """
    Find connected components and filter by geometric properties.
    Enhanced with multiple filters to reduce false positives.

    Args:
        mask: Binary mask (uint8)
        min_area: Minimum area in pixels for valid component
        max_area: Maximum area in pixels (None = no limit)
        max_aspect: Maximum aspect ratio (width/height or height/width)
        min_compactness: Minimum compactness/circularity (None = no filter)
        image_height: Image height for vertical position filter
        min_vertical_position: Minimum Y position as fraction of height (None = no filter)
        max_vertical_position: Maximum Y position as fraction of height (None = no filter)
        image_width: Image width for horizontal position filter
        min_horizontal_position: Minimum X position as fraction of width (None = no filter)
        max_horizontal_position: Maximum X position as fraction of width (None = no filter)

    Returns:
        Tuple of (boxes, centers)
        - boxes: List of (x, y, w, h) bounding boxes
        - centers: List of (cx, cy) centroids
    """
    # Find connected components (8-connectivity)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )

    boxes = []
    centers = []

    # Skip label 0 (background)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        cx = int(centroids[i, 0])
        cy = int(centroids[i, 1])

        # Filter 1: Minimum area
        if area < min_area:
            continue

        # Filter 2: Maximum area (avoid very large blobs)
        if max_area is not None and area > max_area:
            continue

        # Filter 3: Aspect ratio (people are roughly square from above)
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if aspect > max_aspect:
            continue

        # Filter 4: Compactness (reject irregular/elongated shapes)
        if min_compactness is not None:
            # Extract component mask
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Find contour to calculate perimeter
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                perimeter = cv2.arcLength(contours[0], closed=True)
                compactness = calculate_compactness(area, perimeter)
                
                if compactness < min_compactness:
                    continue

        # Filter 5: Vertical position (exclude top region - sky/mountains)
        if min_vertical_position is not None and image_height is not None:
            relative_y = cy / image_height
            
            if relative_y < min_vertical_position:
                continue
        
        # Filter 6: Maximum vertical position (exclude bottom vegetation)
        if max_vertical_position is not None and image_height is not None:
            relative_y = cy / image_height
            
            if relative_y > max_vertical_position:
                continue
        
        # Filter 7: Minimum horizontal position (exclude left edges)
        if min_horizontal_position is not None and image_width is not None:
            relative_x = cx / image_width
            
            if relative_x < min_horizontal_position:
                continue
        
        # Filter 8: Maximum horizontal position (exclude right edges/buildings)
        if max_horizontal_position is not None and image_width is not None:
            relative_x = cx / image_width
            
            if relative_x > max_horizontal_position:
                continue

        # Valid component - passed all filters
        boxes.append((x, y, w, h))
        centers.append((cx, cy))

    return boxes, centers


def count_from_boxes(boxes: List[Tuple[int, int, int, int]]) -> int:
    """Count number of detections from bounding boxes.

    Args:
        boxes: List of (x, y, w, h) bounding boxes

    Returns:
        Count (number of boxes)
    """
    return len(boxes)


def draw_overlays(
    img_bgr: np.ndarray, 
    boxes: List[Tuple[int, int, int, int]], 
    centers: List[Tuple[int, int]]
) -> np.ndarray:
    """Draw bounding boxes and centroids on image.

    Args:
        img_bgr: Input BGR image
        boxes: List of (x, y, w, h) bounding boxes
        centers: List of (cx, cy) centroids

    Returns:
        Image with drawn overlays (copy)
    """
    overlay = img_bgr.copy()

    # Draw boxes in green
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw centroids in red
    for (cx, cy) in centers:
        cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)

    return overlay

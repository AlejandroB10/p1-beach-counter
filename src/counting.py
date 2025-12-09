"""Connected component detection and visualization."""

import cv2
import numpy as np
from typing import List, Tuple

Box = Tuple[int, int, int, int]  # (x, y, w, h)
Point = Tuple[int, int]  # (cx, cy)


def connected_components(
    mask: np.ndarray,
    min_area: int = 50,
    max_area: int = None,
    max_aspect: float = 4.0
) -> Tuple[List[Box], List[Point]]:
    """Find connected components filtered by area and aspect ratio using basic connectedComponents.

    This version avoids connectedComponentsWithStats to align with the simplest function seen in clase.
    Stats (bbox, area, centroid) are computed manually from the label map.
    """
    # Basic labeling (labels: 0 = background)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    boxes, centers = [], []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        area = len(xs)

        if area < min_area or (max_area and area > max_area):
            continue
        if max(w, h) / (min(w, h) + 1e-6) > max_aspect:
            continue

        cx = int(xs.mean())
        cy = int(ys.mean())

        boxes.append((x_min, y_min, w, h))
        centers.append((cx, cy))

    return boxes, centers


def draw_overlays(img_bgr: np.ndarray, boxes: List[Box], 
                  centers: List[Point]) -> np.ndarray:
    """Draw green boxes and red centroids on image.
    
    Args:
        img_bgr: Input image (H, W, 3)
        boxes: List of (x, y, w, h)
        centers: List of (cx, cy)
        
    Returns:
        Image with overlays
    """
    overlay = img_bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for cx, cy in centers:
        cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)
    return overlay


def count_segments(markers):
    labels = np.unique(markers)
    return len(labels[labels > 0])


def assign_components_to_df(markers, df):
    df['component_id'] = df.apply(
        lambda row: 0 if row['y'] - 420 < 0 else markers[row['y'] - 420, row['x']],
        axis=1
    )
    return df
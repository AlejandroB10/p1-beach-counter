"""Duplicate detection removal using spatial clustering."""

import numpy as np
from typing import List, Tuple

Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


def remove_duplicate_detections(
    centers: List[Point],
    boxes: List[Box],
    min_distance: float = 20.0,
    enabled: bool = True
) -> Tuple[List[Box], List[Point]]:
    """Remove detections closer than min_distance, keeping larger ones."""
    if not enabled:
        return list(boxes), list(centers)
    if not centers:
        return [], []

    areas = [w * h for _, _, w, h in boxes]
    keep = [True] * len(centers)

    for i in range(len(centers)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(centers)):
            if not keep[j]:
                continue

            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < min_distance:
                if areas[i] >= areas[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    filtered_boxes = [boxes[i] for i in range(len(boxes)) if keep[i]]
    filtered_centers = [centers[i] for i in range(len(centers)) if keep[i]]
    return filtered_boxes, filtered_centers

"""Connected components detection, filtering, and visualization."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.edges import canny_lab_multichannel
from src.io_utils import save_bgr, save_gray
from src.morphology import close_open
from src.roi import apply_roi_mask
from src.multiscale import detect_dark_regions_percentile

Box = Tuple[int, int, int, int]  # x, y, w, h
Point = Tuple[int, int]          # cx, cy


def connected_components(
    mask: np.ndarray,
    min_area: int = 50,
    max_area: int = 0,
    max_aspect: float = 4.0,
    min_solidity: float = 0.0,
    min_extent: float = 0.0,
    max_extent: float = 1.0
) -> Tuple[List[Box], List[Point]]:
    """Find connected components filtered by geometric properties."""
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

        # Basic filters
        if area < min_area or (max_area and area > max_area):
            continue
        if max(w, h) / (min(w, h) + 1e-6) > max_aspect:
            continue

        # Extent: fill ratio of the bounding box
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0
        if extent < min_extent or extent > max_extent:
            continue

        # Solidity: area / convex hull area
        if min_solidity > 0:
            component_mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                if solidity < min_solidity:
                    continue

        cx = int(xs.mean())
        cy = int(ys.mean())
        boxes.append((x_min, y_min, w, h))
        centers.append((cx, cy))

    return boxes, centers


def draw_overlays(
    img_bgr: np.ndarray,
    boxes: List[Box],
    centers: List[Point]
) -> np.ndarray:
    """Draw green bounding boxes and red centroids on image."""
    overlay = img_bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for cx, cy in centers:
        cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)
    return overlay


def filter_by_intensity(
    img_bgr: np.ndarray,
    boxes: List[Box],
    centers: List[Point],
    max_intensity: float
) -> Tuple[List[Box], List[Point]]:
    """Filter detections by mean intensity - keep only dark blobs."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    filtered_boxes = []
    filtered_centers = []
    for box, center in zip(boxes, centers):
        x, y, w, h = box
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            mean_intensity = roi.mean()
            if mean_intensity <= max_intensity:
                filtered_boxes.append(box)
                filtered_centers.append(center)
    return filtered_boxes, filtered_centers


def filter_by_background_diff(
    img_bgr: np.ndarray,
    bg_bgr: np.ndarray,
    boxes: List[Box],
    centers: List[Point],
    min_diff_threshold: float
) -> Tuple[List[Box], List[Point]]:
    """Filter detections by difference with background image."""
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_img, gray_bg)

    filtered_boxes = []
    filtered_centers = []
    for box, center in zip(boxes, centers):
        x, y, w, h = box
        roi_diff = diff[y:y+h, x:x+w]
        if roi_diff.size > 0:
            mean_diff = roi_diff.mean()
            if mean_diff >= min_diff_threshold:
                filtered_boxes.append(box)
                filtered_centers.append(center)
    return filtered_boxes, filtered_centers


def count_segments(markers: np.ndarray) -> int:
    """Count unique segment labels in watershed markers."""
    labels = np.unique(markers)
    return len(labels[labels > 0])


def assign_components_to_df(markers: np.ndarray, df):
    """Assign watershed component IDs to dataframe rows based on coordinates."""
    df['component_id'] = df.apply(
        lambda row: 0 if row['y'] - 420 < 0 else markers[row['y'] - 420, row['x']],
        axis=1
    )
    return df


def create_dark_mask(img_bgr: np.ndarray, params: dict) -> np.ndarray:
    """Create mask of dark regions where people are likely located."""
    dark_percentile = params.get('dark_percentile', 30.0)
    dark_mask = detect_dark_regions_percentile(img_bgr, percentile=dark_percentile)

    dark_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark_mask = cv2.dilate(dark_mask, dark_kernel, iterations=2)
    return dark_mask


def detect_in_zone(
    img_blur: np.ndarray,
    zone_mask: np.ndarray,
    dark_mask: np.ndarray,
    params: dict,
    zone_prefix: str
):
    """Detect people in a specific zone (water or sand)."""
    canny_low = params[f'{zone_prefix}_canny_low']
    canny_high = params[f'{zone_prefix}_canny_high']
    morph_kernel = params[f'{zone_prefix}_morph_kernel']
    close_iter = params[f'{zone_prefix}_close_iter']
    open_iter = params[f'{zone_prefix}_open_iter']
    min_area = params[f'{zone_prefix}_min_area']
    max_area = params[f'{zone_prefix}_max_area']
    max_aspect = params[f'{zone_prefix}_max_aspect']
    min_solidity = params.get(f'{zone_prefix}_min_solidity', 0.0)
    min_extent = params.get(f'{zone_prefix}_min_extent', 0.0)
    max_extent = params.get(f'{zone_prefix}_max_extent', 1.0)

    edges = canny_lab_multichannel(img_blur, canny_low, canny_high)
    edges = apply_roi_mask(edges, zone_mask)

    if dark_mask is not None:
        edges = cv2.bitwise_and(edges, dark_mask)

    mask = close_open(edges, morph_kernel, close_iter, open_iter)
    boxes, centers = connected_components(
        mask, min_area, max_area, max_aspect,
        min_solidity, min_extent, max_extent
    )
    return mask, boxes, centers


def save_results(
    results: dict,
    img_path: Path,
    out_dir: Path,
    pipeline_name: str = 'C'
) -> None:
    """Save pipeline visualization results to disk."""
    viz_dir = out_dir / "viz" / pipeline_name
    viz_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    save_gray(viz_dir / f"{stem}_mask.png", results['mask'])
    save_bgr(viz_dir / f"{stem}_overlay.png", results['overlay'])
    save_bgr(viz_dir / f"{stem}_enhanced.png", results['enhanced'])

"""ROI and zone masks for the beach scene."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src.io_utils import save_bgr, save_gray


def create_beach_roi_mask(img_shape: Tuple[int, int], exclude_top_ratio: float = 0.39, exclude_boats: bool = True) -> np.ndarray:
    """Mask excluding sky/mountains and optional boat area."""
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * exclude_top_ratio)
    mask[y_start:, :] = 255

    if exclude_boats:
        boat_polygon = np.array([
            [0, y_start],
            [int(w * 0.68), y_start],
            # [int(w * 0.72), int(h * 0.38)],
            [int(w * 0.92), int(h * 0.32)],
            [0, int(h * 0.42)],
        ], dtype=np.int32)
        cv2.fillPoly(mask, [boat_polygon], 0)

    return mask


def create_water_sand_masks(img_shape: Tuple[int, int], shoreline_points: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create separate masks for water and sand zones."""
    h, w = img_shape

    if shoreline_points is None:
        shoreline_points = np.array([
            [0, int(h * 0.56)],
            [int(w * 0.15), int(h * 0.54)],
            [int(w * 0.30), int(h * 0.52)],
            [int(w * 0.45), int(h * 0.50)],
            [int(w * 0.55), int(h * 0.49)],
            [int(w * 0.65), int(h * 0.47)],
            [int(w * 0.75), int(h * 0.45)],
            [int(w * 0.85), int(h * 0.38)],
            [int(w * 0.95), int(h * 0.36)],
            [w, int(h * 0.35)],
        ], dtype=np.int32)

    shoreline_points = shoreline_points.astype(np.int32)

    water_mask = np.zeros((h, w), dtype=np.uint8)
    water_poly = np.vstack([[[0, 0]], [[w, 0]], shoreline_points[::-1]])
    cv2.fillPoly(water_mask, [water_poly], 255)

    sand_mask = np.zeros((h, w), dtype=np.uint8)
    sand_poly = np.vstack([shoreline_points, [[w, h]], [[0, h]]])
    cv2.fillPoly(sand_mask, [sand_poly], 255)

    return water_mask, sand_mask


def apply_roi_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask to image or single channel."""
    return cv2.bitwise_and(img, img, mask=mask)


def visualize_roi_overlay(img_bgr: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.3) -> np.ndarray:
    """Overlay ROI mask on image with transparency and contour outline."""
    overlay = img_bgr.copy()
    colored = np.zeros_like(img_bgr)
    colored[mask > 0] = color
    cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0, overlay)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def create_and_save_masks(
    first_img: np.ndarray,
    params: dict,
    out_dir: Path,
    subdir: str = "roi"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create ROI/water/sand masks and save visualizations."""
    h, w = first_img.shape[:2]
    roi_mask = None
    water_mask, sand_mask = None, None

    if params.get('use_roi_beach', True):
        roi_mask = create_beach_roi_mask((h, w), params.get('roi_exclude_top_ratio', 0.39))

    if params.get('use_zones', True):
        water_mask, sand_mask = create_water_sand_masks((h, w))

    viz_dir = Path(out_dir) / "viz" / subdir
    viz_dir.mkdir(parents=True, exist_ok=True)
    if roi_mask is not None:
        save_bgr(viz_dir / "roi_overlay.png", visualize_roi_overlay(first_img, roi_mask))
        save_gray(viz_dir / "roi_mask.png", roi_mask)
    if water_mask is not None:
        save_gray(viz_dir / "water_mask.png", water_mask)
    if sand_mask is not None:
        save_gray(viz_dir / "sand_mask.png", sand_mask)

    return roi_mask, water_mask, sand_mask

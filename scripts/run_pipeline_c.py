import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure repository root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io_utils import read_bgr
from src.intensity import clahe_on_L
from src.edges import canny_lab_multichannel
from src.morphology import close_open
from src.counting import (
    connected_components,
    create_dark_mask,
    detect_in_zone,
    draw_overlays,
    filter_by_intensity,
    filter_by_background_diff,
)
from src.roi import apply_roi_mask

PARAMS = {
    # ROI settings
    'use_roi_beach': True,
    'use_zones': True,
    'roi_exclude_top_ratio': 0.39,

    # Dark region detection (people are darker than sand/water)
    'use_dark_regions': True,
    'dark_percentile': 17.0,

    # Post-filtering by intensity
    'filter_by_intensity': True,
    'max_mean_intensity': 125,

    # Background difference filter (removes static elements)
    'use_bg_diff_filter': True,
    'bg_diff_threshold': 17,

    # General enhancement
    'clahe_clip': 2.0,
    'clahe_tiles': 8,
    'blur_ksize': 5,

    # Water zone (more sensitive - lower contrast in water)
    'water_enabled': True,
    'water_canny_low': 45,
    'water_canny_high': 120,
    'water_morph_kernel': 4,
    'water_close_iter': 2,
    'water_open_iter': 1,
    'water_min_area': 90,
    'water_max_area': 1200,
    'water_max_aspect': 2.5,
    'water_min_solidity': 0.32,
    'water_min_extent': 0.18,
    'water_max_extent': 0.88,

    # Sand zone (stricter - more texture/clutter)
    'sand_canny_low': 60,
    'sand_canny_high': 120,
    'sand_morph_kernel': 5,
    'sand_close_iter': 2,
    'sand_open_iter': 1,
    'sand_min_area': 140,
    'sand_max_area': 1700,
    'sand_max_aspect': 2.2,
    'sand_min_solidity': 0.35,
    'sand_min_extent': 0.20,
    'sand_max_extent': 0.84,
}


def pipeline_c(
    img_bgr: np.ndarray,
    params: dict,
    roi_mask: np.ndarray,
    water_mask: np.ndarray,
    sand_mask: np.ndarray,
    bg_img: np.ndarray = None
) -> dict:
    """Pipeline C: CLAHE + LAB Canny + Dark Regions + Zone Detection.
        Steps:
        1. Enhance contrast (CLAHE on L) + blur
        2. Optional dark-region mask (people darker than sand/water)
        3. Zone-aware detection (water/sand) or fallback single-zone
        4. Post-filters (background diff, intensity)
        5. Build outputs
    """
    # Step 1: Enhance contrast (CLAHE on L) + blur
    img_enhanced, L_enhanced = clahe_on_L(
        img_bgr,
        params['clahe_clip'],
        (params['clahe_tiles'],) * 2
    )
    img_blur = cv2.GaussianBlur(
        img_enhanced,
        (params['blur_ksize'],) * 2,
        1.0
    )

    # Step 2: Optional dark-region mask (people darker than sand/water)
    dark_mask = None
    if params.get('use_dark_regions', False):
        dark_mask = create_dark_mask(img_bgr, params)

    all_boxes, all_centers = [], []
    water_count, sand_count = 0, 0

    # Step 3: Zone-aware detection (water/sand) or fallback single-zone
    if params.get('water_enabled', True) and water_mask is not None and sand_mask is not None:
        # Combine zone masks with ROI
        if roi_mask is not None:
            water_roi = cv2.bitwise_and(water_mask, roi_mask)
            sand_roi = cv2.bitwise_and(sand_mask, roi_mask)
        else:
            water_roi = water_mask
            sand_roi = sand_mask

        # Water zone detection
        mask_water, boxes_w, centers_w = detect_in_zone(
            img_blur, water_roi, dark_mask, params, 'water'
        )
        all_boxes.extend(boxes_w)
        all_centers.extend(centers_w)
        water_count = len(boxes_w)

        # Sand zone detection
        mask_sand, boxes_s, centers_s = detect_in_zone(
            img_blur, sand_roi, dark_mask, params, 'sand'
        )
        all_boxes.extend(boxes_s)
        all_centers.extend(centers_s)
        sand_count = len(boxes_s)

        mask = cv2.bitwise_or(mask_water, mask_sand)
    else:
        # Fallback: single-zone detection
        edges = canny_lab_multichannel(
            img_blur, params['sand_canny_low'], params['sand_canny_high']
        )
        if roi_mask is not None:
            edges = apply_roi_mask(edges, roi_mask)
        if dark_mask is not None:
            edges = cv2.bitwise_and(edges, dark_mask)

        mask = close_open(
            edges,
            params['sand_morph_kernel'],
            params['sand_close_iter'],
            params['sand_open_iter']
        )
        boxes, centers = connected_components(
            mask,
            params['sand_min_area'],
            params['sand_max_area'],
            params['sand_max_aspect'],
            params.get('sand_min_solidity', 0.0),
            params.get('sand_min_extent', 0.0),
            params.get('sand_max_extent', 1.0)
        )
        all_boxes = list(boxes)
        all_centers = list(centers)
        sand_count = len(boxes)

    # Step 4: Post-filters (background diff, intensity)
    if params.get('use_bg_diff_filter', False) and bg_img is not None:
        all_boxes, all_centers = filter_by_background_diff(
            img_bgr, bg_img, all_boxes, all_centers,
            params.get('bg_diff_threshold', 15)
        )

    # Post-filtering by intensity
    if params.get('filter_by_intensity', False):
        all_boxes, all_centers = filter_by_intensity(
            img_bgr, all_boxes, all_centers,
            params.get('max_mean_intensity', 150)
        )

    # Step 5: Build outputs
    return {
        'enhanced': img_enhanced,
        'L_enhanced': L_enhanced,
        'mask': mask,
        'overlay': draw_overlays(img_bgr, all_boxes, all_centers),
        'boxes': all_boxes,
        'centers': all_centers,
        'water_count': water_count,
        'sand_count': sand_count,
    }


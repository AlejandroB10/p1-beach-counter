import sys
from pathlib import Path

# Ensure src on path if executed directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.hue_subtraction import subtract_hue_channel, otsu_threshold
from src.watershed import (
    filter_components_by_geometry,
    watershed_segmentation,
    markers_to_boxes_centers
)
from src.counting import draw_overlays


# Pipeline B parameters (optimized)
PARAMS = {
    'crop_y': 420,                    # Vertical crop position (remove top portion)
    'blur_kernel': 5,                 # Gaussian blur kernel for Otsu
    'min_component_size': 15,         # Minimum component size in pixels
    'max_component_size': 2000,       # Maximum component size in pixels
    'max_aspect_ratio': 5.0,          # Maximum aspect ratio for components
    'watershed_dist_factor': 0.0005,  # Distance transform threshold factor
    'watershed_dilate_iter': 2,       # Background dilation iterations
}


def pipeline_b(img_bgr, params, extra_ctx):
    """
    Pipeline B: HSV Hue Subtraction + Otsu + Watershed.
    
    Steps:
    1. Crop image at specified y-coordinate
    2. Subtract hue channel from background image
    3. Apply Otsu's automatic thresholding
    4. Filter connected components by size and aspect ratio
    5. Apply watershed segmentation to separate touching objects
    6. Extract bounding boxes and centers (with y-offset correction)
    
    Args:
        img_bgr: Input image in BGR format
        params: Pipeline parameters dictionary
        extra_ctx: Extra context containing 'bg' (background image)
    """
    crop_y = params['crop_y']
    bg_img = extra_ctx.get('bg')
    
    if bg_img is None:
        raise ValueError("Pipeline B requires background image in extra_ctx['bg']")
    
    # Step 1: Crop image and background
    img_cropped = img_bgr[crop_y:, :]
    bg_cropped = bg_img[crop_y:, :]
    
    # Step 2: Subtract hue channel
    subtracted = subtract_hue_channel(img_cropped, bg_cropped)
    
    # Step 3: Apply Otsu thresholding (inverted for distance transform)
    thresholded, threshold_val = otsu_threshold(
        subtracted,
        blur_kernel=params['blur_kernel'],
        invert=True
    )
    
    # Step 4: Initial connected components for filtering
    num_labels, markers = cv2.connectedComponents(thresholded)
    
    # Filter by size and aspect ratio
    filtered_mask = filter_components_by_geometry(
        markers,
        min_size=params['min_component_size'],
        max_size=params['max_component_size'],
        max_aspect_ratio=params['max_aspect_ratio']
    )
    
    # Remove filtered components from thresholded image
    thresholded[filtered_mask == 255] = 0
    
    # Step 5: Apply watershed segmentation
    markers = watershed_segmentation(
        thresholded,
        img_cropped,
        dist_threshold_factor=params['watershed_dist_factor'],
        dilate_iterations=params['watershed_dilate_iter']
    )
    
    # Step 6: Extract boxes and centers with y-offset correction
    boxes, centers = markers_to_boxes_centers(markers, y_offset=crop_y)
    
    # Create visualization overlay on original (uncropped) image
    overlay = draw_overlays(img_bgr, boxes, centers)
    
    # For visualization: show cropped mask and subtracted image
    # Create full-size versions by padding with black
    mask_full = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    mask_full[crop_y:, :] = thresholded
    
    subtracted_full = np.zeros_like(img_bgr)
    subtracted_full[crop_y:, :] = subtracted
    
    return {
        'edges': mask_full,           # Use mask as "edges" for consistency
        'mask': mask_full,
        'overlay': overlay,
        'enhanced': subtracted_full,  # Hue-subtracted image
        'boxes': boxes,
        'centers': centers,
    }


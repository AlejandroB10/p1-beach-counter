import sys
from pathlib import Path

# Ensure src on path if executed directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edges import canny_gray
from src.morphology import close_open
from src.counting import connected_components, draw_overlays


# Pipeline A parameters (optimized)
PARAMS = {
    'canny_low': 80,              # Lower threshold for Canny edge detection
    'canny_high': 200,            # Upper threshold for Canny edge detection
    'canny_aperture': 3,          # Sobel kernel size
    'morph_kernel': 5,            # Morphological kernel size
    'morph_close_iter': 2,        # Closing iterations (fill gaps)
    'morph_open_iter': 3,         # Opening iterations (remove noise)
    'cc_min_area': 300,           # Minimum component area in pixels
    'cc_max_aspect': 2.5,         # Maximum aspect ratio (width/height)
}


def pipeline_a(img_bgr, params):
    """
    Pipeline A: Grayscale + Canny + Morphology + Connected Components.
    
    Steps:
    1. Convert to grayscale
    2. Apply Canny edge detection
    3. Morphological closing (fill gaps) + opening (remove noise)
    4. Find connected components
    5. Filter by area and aspect ratio
    6. Count detections
    """
    # Step 1: Canny edge detection on grayscale
    edges = canny_gray(
        img_bgr,
        low=params['canny_low'],
        high=params['canny_high'],
        aperture=params['canny_aperture']
    )

    # Step 2: Morphological operations to clean up edges
    mask = close_open(
        edges,
        k=params['morph_kernel'],
        it_close=params['morph_close_iter'],
        it_open=params['morph_open_iter']
    )

    # Step 3: Find and filter connected components
    boxes, centers = connected_components(
        mask,
        min_area=params['cc_min_area'],
        max_aspect=params['cc_max_aspect']
    )

    # Step 4: Draw visualizations
    overlay = draw_overlays(img_bgr, boxes, centers)

    return {
        'edges': edges,
        'mask': mask,
        'overlay': overlay,
        'boxes': boxes,
        'centers': centers,
    }


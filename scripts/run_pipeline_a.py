"""
Simple script to run Pipeline A: Grayscale + Canny edge detection + Morphology + Connected Components.

This is a baseline people counter using classical computer vision techniques.
No deep learning, only OpenCV and NumPy.

Usage:
    python scripts/run_pipeline_a.py
    
Output:
    - Visualizations saved to: data/outputs/viz/A/
    - Counts saved to: data/outputs/metrics/counts_<timestamp>.csv
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from io_utils import list_images, read_bgr, save_bgr, save_gray
from edges import canny_gray
from morphology import close_open
from counting import connected_components, draw_overlays, count_from_boxes
from evaluation import load_annotations, get_gt_count


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


def pipeline_A(img_bgr, params):
    """
    Pipeline A: Grayscale + Canny + Morphology + Connected Components.
    
    Steps:
    1. Convert to grayscale
    2. Apply Canny edge detection
    3. Morphological closing (fill gaps) + opening (remove noise)
    4. Find connected components
    5. Filter by area and aspect ratio
    6. Count detections
    
    Args:
        img_bgr: Input BGR image
        params: Dictionary with pipeline parameters
        
    Returns:
        Dictionary with intermediate results and final count
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


def main():
    # Setup paths (relative to repository root)
    root_dir = Path(__file__).parent.parent
    raw_dir = root_dir / 'data' / 'raw'
    out_dir = root_dir / 'data' / 'outputs'
    annotations_dir = root_dir / 'data' / 'annotations'

    # List all images
    all_images = list_images(raw_dir)
    if len(all_images) == 0:
        print(f"Error: No images found in {raw_dir}")
        return 1

    # Process first 3 images (or all if less than 3)
    images = all_images[:3]
    print(f"Pipeline A: People Counter")
    print(f"Processing {len(images)} images")
    print(f"Parameters: {PARAMS}")
    print()

    # Load ground truth annotations if available
    gt_dict = {}
    annotation_files = list(annotations_dir.glob('*_points.csv'))
    if annotation_files:
        gt_dict = load_annotations(annotation_files[0])
        print(f"Loaded annotations from: {annotation_files[0].name}\n")

    # Create output directories
    viz_dir = out_dir / "viz" / "A"
    metrics_dir = out_dir / "metrics"
    viz_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Prepare CSV for results
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    metrics_file = metrics_dir / f"counts_A_{timestamp}.csv"
    csv_rows = []

    # Process each image
    for img_path in images:
        img_name = img_path.name
        img_stem = img_path.stem
        print(f"Processing: {img_name}")

        # Read image
        img_bgr = read_bgr(img_path)

        # Run Pipeline A
        results = pipeline_A(img_bgr, PARAMS)

        # Save visualizations
        save_gray(viz_dir / f"{img_stem}_edges.png", results['edges'])
        save_gray(viz_dir / f"{img_stem}_mask.png", results['mask'])
        save_bgr(viz_dir / f"{img_stem}_overlay.png", results['overlay'])

        # Get count
        count = count_from_boxes(results['boxes'])
        gt_count = get_gt_count(img_name, gt_dict)

        # Print result
        print(f"  Detected: {count} people", end='')
        if gt_count is not None:
            residual = count - gt_count
            print(f" (Ground truth: {gt_count}, Residual: {residual:+d})")
        else:
            print()

        # Store for CSV
        row = {
            'filename': img_name,
            'count': count,
        }
        if gt_count is not None:
            row['gt_count'] = gt_count
            row['residual'] = count - gt_count
        csv_rows.append(row)

        print()

    # Write CSV
    fieldnames = ['filename', 'count']
    if any('gt_count' in row for row in csv_rows):
        fieldnames.extend(['gt_count', 'residual'])

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    # Print summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Image':<25} {'Count':<10} {'GT':<10} {'Residual':<10}")
    print("-"*60)
    for row in csv_rows:
        img = row['filename']
        count = row['count']
        gt = row.get('gt_count', '-')
        residual = row.get('residual', '-')
        if residual != '-':
            residual = f"{residual:+d}"
        print(f"{img:<25} {count:<10} {gt:<10} {residual:<10}")
    print("="*60)
    print(f"\nVisualizations saved to: {viz_dir}/")
    print(f"Metrics saved to: {metrics_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

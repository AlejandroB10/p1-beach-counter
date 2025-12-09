#!/usr/bin/env python3
"""Pipeline C: Zone-aware LAB+Canny counter (former Pipeline B).

Runs on all images in data/raw by default, generates overlays/masks and metrics.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Ensure repository root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import hsv_hue_subtract
from src.io_utils import list_images, read_bgr, save_bgr, save_gray
from src.intensity import clahe_on_L
from src.edges import canny_lab_multichannel
from src.morphology import close_open
from src.counting import connected_components, draw_overlays
from src.evaluation import (
    load_annotations,
    load_points,
    compute_mae,
    compute_mse,
    match_points,
    precision_recall_f1,
)
from src.roi import (
    create_beach_roi_mask,
    apply_roi_mask,
    visualize_roi_overlay,
    create_water_sand_masks,
)
from src.clustering import remove_duplicate_detections


PARAMS = {
    'use_roi_beach': True,
    'use_zones': True,
    'roi_exclude_top_ratio': 0.35,

    # Pipeline C - General
    'C_clahe_clip': 2.5,
    'C_clahe_tiles': 8,
    'C_blur_ksize': 5,

    # Pipeline C - Water zone (sensible)
    'C_water_canny_low': 48,
    'C_water_canny_high': 130,
    'C_water_morph_kernel': 4,
    'C_water_close_iter': 2,
    'C_water_open_iter': 1,
    'C_water_min_area': 90,
    'C_water_max_area': 1400,
    'C_water_max_aspect': 2.5,

    # Pipeline C - Sand zone (estricto)
    'C_sand_canny_low': 100,
    'C_sand_canny_high': 200,
    'C_sand_morph_kernel': 6,
    'C_sand_close_iter': 3,
    'C_sand_open_iter': 2,
    'C_sand_min_area': 200,
    'C_sand_max_area': 1800,
    'C_sand_max_aspect': 2.0,

    # Pipeline C - Duplicate removal
    'C_duplicate_dist': 22.0,
    'C_duplicate_enabled': True,

    # Pipeline C - Fallback (sin zonas)
    'C_canny_low': 75,
    'C_canny_high': 185,
    'C_morph_kernel': 6,
    'C_morph_close_iter': 2,
    'C_morph_open_iter': 2,
    'C_cc_min_area': 150,
    'C_cc_max_area': 1900,
    'C_cc_max_aspect': 2.4,
}


def pipeline_C(img_bgr, params, roi_mask, water_mask, sand_mask):
    """Pipeline C: CLAHE + LAB Canny + Zone Detection + Duplicate Removal"""
    img_enhanced, L_enhanced = clahe_on_L(
        img_bgr, params['C_clahe_clip'], (params['C_clahe_tiles'],) * 2
    )
    img_blur = cv2.GaussianBlur(img_enhanced, (params['C_blur_ksize'],) * 2, 1.0)

    all_boxes, all_centers = [], []

    if water_mask is not None and sand_mask is not None:
        if roi_mask is not None:
            water_roi = cv2.bitwise_and(water_mask, roi_mask)
            sand_roi = cv2.bitwise_and(sand_mask, roi_mask)
        else:
            water_roi = water_mask
            sand_roi = sand_mask

        # Agua (más sensible)
        edges_water = canny_lab_multichannel(img_blur, params['C_water_canny_low'], params['C_water_canny_high'])
        edges_water = apply_roi_mask(edges_water, water_roi)
        mask_water = close_open(edges_water, params['C_water_morph_kernel'],
                                params['C_water_close_iter'], params['C_water_open_iter'])
        boxes_w, centers_w = connected_components(
            mask_water,
            params['C_water_min_area'],
            params['C_water_max_area'],
            params['C_water_max_aspect'],
        )
        all_boxes.extend(boxes_w)
        all_centers.extend(centers_w)
        water_count = len(boxes_w)

        # Arena (más estricto)
        edges_sand = canny_lab_multichannel(img_blur, params['C_sand_canny_low'], params['C_sand_canny_high'])
        edges_sand = apply_roi_mask(edges_sand, sand_roi)
        mask_sand = close_open(edges_sand, params['C_sand_morph_kernel'],
                               params['C_sand_close_iter'], params['C_sand_open_iter'])
        boxes_s, centers_s = connected_components(
            mask_sand,
            params['C_sand_min_area'],
            params['C_sand_max_area'],
            params['C_sand_max_aspect'],
        )
        all_boxes.extend(boxes_s)
        all_centers.extend(centers_s)
        sand_count = len(boxes_s)

        mask = cv2.bitwise_or(mask_water, mask_sand)
    else:
        # Sin zonas
        edges = canny_lab_multichannel(img_blur, params['C_canny_low'], params['C_canny_high'])
        if roi_mask is not None:
            edges = apply_roi_mask(edges, roi_mask)
        mask = close_open(edges, params['C_morph_kernel'],
                          params['C_morph_close_iter'], params['C_morph_open_iter'])
        boxes, centers = connected_components(
            mask,
            params['C_cc_min_area'],
            params['C_cc_max_area'],
            params['C_cc_max_aspect'],
        )
        all_boxes = list(boxes)
        all_centers = list(centers)
        water_count = 0
        sand_count = len(boxes)

    # Deduplicar detecciones cercanas
    all_boxes, all_centers = remove_duplicate_detections(
        all_centers, all_boxes, params['C_duplicate_dist'], enabled=params['C_duplicate_enabled']
    )

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


PIPELINES = {'C': pipeline_C}


def run_pipeline(name, img_bgr, img_path, out_dir, params, roi_mask, water_mask, sand_mask):
    results = PIPELINES[name](img_bgr, params, roi_mask, water_mask, sand_mask)

    viz_dir = out_dir / "viz" / name
    viz_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    save_gray(viz_dir / f"{stem}_mask.png", results['mask'])
    save_bgr(viz_dir / f"{stem}_overlay.png", results['overlay'])
    save_bgr(viz_dir / f"{stem}_enhanced.png", results['enhanced'])

    return len(results['boxes']), results


def main():
    raw_dir = Path("data/raw")
    out_dir = Path("data/outputs")
    pipeline_names = ['C']
    use_roi = PARAMS['use_roi_beach']
    use_zones = PARAMS['use_zones']
    bg_image = None

    images = list_images(raw_dir)
    if not images:
        print(f"No images in {raw_dir}")
        return 1

    print("Pipeline C: Zone-aware LAB+Canny")
    print(f"Processing {len(images)} images | ROI: {use_roi} | Zones: {use_zones}\n")

    gt_dict = {}
    ann_dir = Path('data/annotations')
    gt_points = {}
    if ann_dir.exists():
        point_files = sorted(ann_dir.glob('*_points.csv'))
        ann_files = sorted(ann_dir.glob('labels_*.csv'))
        if point_files:
            gt_points = load_points(point_files[0])
            gt_dict = {k: len(v) for k, v in gt_points.items()}
        elif ann_files:
            gt_dict = load_annotations(ann_files[0])

    first_img = read_bgr(images[0])
    bg_img = read_bgr(bg_image) if bg_image else None
    h, w = first_img.shape[:2]

    roi_mask = create_beach_roi_mask((h, w), PARAMS['roi_exclude_top_ratio']) if use_roi else None
    water_mask, sand_mask = create_water_sand_masks((h, w)) if use_zones else (None, None)

    if roi_mask is not None:
        roi_dir = out_dir / "viz" / "roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        save_bgr(roi_dir / "roi_overlay.png", visualize_roi_overlay(first_img, roi_mask))
        save_gray(roi_dir / "roi_mask.png", roi_mask)
        if water_mask is not None:
            save_gray(roi_dir / "water_mask.png", water_mask)
            save_gray(roi_dir / "sand_mask.png", sand_mask)

    csv_rows = []
    mae_data = {p: [] for p in pipeline_names}
    person_totals = {p: {'tp': 0, 'fp': 0, 'fn': 0} for p in pipeline_names}

    for img_path in images:
        name = img_path.name
        img_bgr = read_bgr(img_path)
        if bg_img is not None:
            img_bgr = hsv_hue_subtract(img_bgr, bg_img)
        gt = gt_dict.get(name)
        gt_pts = gt_points.get(name, [])

        for p in pipeline_names:
            count, results = run_pipeline(p, img_bgr, img_path, out_dir, PARAMS, roi_mask, water_mask, sand_mask)
            res = f"(gt={gt}, diff={count-gt:+d})" if gt else ""
            extra = f" | water={results['water_count']} sand={results['sand_count']}" if water_mask is not None and sand_mask is not None else ""
            print(f"  {name} [{p}]: {count} {res}{extra}")

            row = {'filename': name, 'pipeline': p, 'count': count}
            if gt is not None:
                row['gt_count'] = gt
                row['residual'] = count - gt
                mae_data[p].append((count, gt))
                if gt_pts:
                    tp, fp, fn = match_points(results['centers'], gt_pts, radius=20.0)
                    person_totals[p]['tp'] += tp
                    person_totals[p]['fp'] += fp
                    person_totals[p]['fn'] += fn
                    prec_i, rec_i, f1_i = precision_recall_f1(tp, fp, fn)
                    row.update({'tp': tp, 'fp': fp, 'fn': fn,
                                'precision': round(prec_i, 3),
                                'recall': round(rec_i, 3),
                                'f1': round(f1_i, 3)})
            csv_rows.append(row)

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "counts_all.csv"

    summary_rows = []
    for p in pipeline_names:
        preds = [pred for pred, _ in mae_data[p]]
        gts = [gt for _, gt in mae_data[p]]
        mae = compute_mae(preds, gts) if preds else 0.0
        mse = compute_mse(preds, gts) if preds else 0.0
        tp = person_totals[p]['tp']
        fp = person_totals[p]['fp']
        fn = person_totals[p]['fn']
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        summary_rows.append({
            'filename': 'SUMMARY',
            'pipeline': p,
            'count': '',
            'gt_count': '',
            'residual': '',
            'images_with_gt': len(preds),
            'mae': round(mae, 3),
            'mse': round(mse, 3),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(prec, 3),
            'recall': round(rec, 3),
            'f1': round(f1, 3),
        })

    # Single CSV with per-image rows + summary rows, consistent headers.
    # Merge with existing counts_all.csv (replace pipeline C entries).
    fields = ['filename', 'pipeline', 'count', 'gt_count', 'residual',
              'tp', 'fp', 'fn', 'precision', 'recall', 'f1',
              'images_with_gt', 'mae', 'mse']
    existing = []
    if metrics_file.exists():
        with open(metrics_file, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get('pipeline') != 'C':
                    existing.append(r)

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + csv_rows + summary_rows)

    print(f"\nSaved (counts + summary): {metrics_file}")
    print("Summary:")
    for row in summary_rows:
        print(f"  {row}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

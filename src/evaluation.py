"""Evaluation utilities for image-level and person-level metrics."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

Point = Tuple[int, int]


def load_annotations(annotations_file: Path) -> Dict[str, int]:
    """Load ground truth counts from a CSV where each row is a person."""
    if not annotations_file.exists():
        return {}

    counts = {}
    with open(annotations_file, "r") as f:
        for row in csv.DictReader(f):
            filename = row.get("image_name") or row.get("filename")
            if filename:
                counts[filename] = counts.get(filename, 0) + 1
    return counts


def load_points(annotations_file: Path) -> Dict[str, List[Point]]:
    """Load ground truth person points (x,y) per image from CSV."""
    if not annotations_file.exists():
        return {}
    points: Dict[str, List[Point]] = {}
    with open(annotations_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                x, y = int(row[1]), int(row[2])
            except ValueError:
                continue
            filename = row[3]
            points.setdefault(filename, []).append((x, y))
    return points


def compute_mse(predictions: List[int], ground_truth: List[int]) -> float:
    """Mean Squared Error for image-level counting."""
    if len(predictions) != len(ground_truth) or not predictions:
        return 0.0
    return sum((p - g) ** 2 for p, g in zip(predictions, ground_truth)) / len(predictions)


def match_points(pred: List[Point], gt: List[Point], radius: float = 20.0) -> Tuple[int, int, int]:
    """Greedy matching between predicted centers and GT points within a radius."""
    if not gt:
        return 0, len(pred), 0
    if not pred:
        return 0, 0, len(gt)

    gt_used = [False] * len(gt)
    tp = 0
    for px, py in pred:
        best_idx = -1
        best_dist2 = radius * radius
        for i, (gx, gy) in enumerate(gt):
            if gt_used[i]:
                continue
            dx = px - gx
            dy = py - gy
            d2 = dx * dx + dy * dy
            if d2 <= best_dist2:
                best_dist2 = d2
                best_idx = i
        if best_idx >= 0:
            gt_used[best_idx] = True
            tp += 1

    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1-score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def init_run(
    annotations_dir: Path,
    out_dir: Path,
    viz_subdir: str
):
    """Common setup: load GT, create viz/metrics dirs, init accumulators."""
    gt_dict, gt_points = load_ground_truth(annotations_dir)
    source = next(iter(annotations_dir.glob('*_points.csv')), None) or next(iter(annotations_dir.glob('labels_*.csv')), None)

    viz_dir = out_dir / "viz" / viz_subdir
    metrics_dir = out_dir / "metrics"
    viz_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    mse_data = []
    totals = {'tp': 0, 'fp': 0, 'fn': 0}
    return gt_dict, gt_points, viz_dir, metrics_dir, csv_rows, mse_data, totals, source


def collect_metrics_row(
    img_name: str,
    pipeline: str,
    count: int,
    centers: List[Point],
    gt_dict: Dict[str, int],
    gt_points: Dict[str, List[Point]],
    totals: Dict[str, int],
    mse_data: List[Tuple[int, int]],
    radius: float = 20.0
) -> Tuple[dict, str, str]:
    """Build metrics row, update totals/MSE, and return progress and GT display."""
    row = {'filename': img_name, 'pipeline': pipeline, 'count': count}
    progress = ""
    gt_display = ""
    gt = gt_dict.get(img_name)
    if gt is not None:
        residual = count - gt
        row['gt_count'] = gt
        row['residual'] = residual
        mse_data.append((count, gt))
        progress = f" (gt={gt}, diff={residual:+d})"
        gt_display = f"(gt={gt}, diff={residual:+d})"
        gt_pts = gt_points.get(img_name, [])
        if gt_pts:
            tp, fp, fn = match_points(centers, gt_pts, radius=radius)
            totals['tp'] += tp
            totals['fp'] += fp
            totals['fn'] += fn
            prec_i, rec_i, f1_i = precision_recall_f1(tp, fp, fn)
            row.update({
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': round(prec_i, 3),
                'recall': round(rec_i, 3),
                'f1': round(f1_i, 3)
            })
    return row, progress, gt_display


def load_ground_truth(ann_dir: Path) -> Tuple[Dict[str, int], Dict[str, List[Point]]]:
    """Load ground truth counts and points if available."""
    gt_dict: Dict[str, int] = {}
    gt_points: Dict[str, List[Point]] = {}
    if not ann_dir.exists():
        return gt_dict, gt_points

    point_files = sorted(ann_dir.glob('*_points.csv'))
    ann_files = sorted(ann_dir.glob('labels_*.csv'))
    if point_files:
        gt_points = load_points(point_files[0])
        gt_dict = {k: len(v) for k, v in gt_points.items()}
    elif ann_files:
        gt_dict = load_annotations(ann_files[0])
    return gt_dict, gt_points


def finalize_metrics(
    pipeline: str,
    metrics_dir: Path,
    csv_rows: List[Dict],
    mse_data: List[Tuple[int, int]],
    tp: int,
    fp: int,
    fn: int,
    fields: List[str]
) -> Dict[str, float]:
    """Write counts_all.csv and summary_all.csv, merging with existing data."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "counts_all.csv"
    summary_file = metrics_dir / "summary_all.csv"

    # Merge counts
    existing = []
    if metrics_file.exists():
        with open(metrics_file, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get('pipeline') != pipeline and r.get('filename') != 'SUMMARY':
                    existing.append({k: v for k, v in r.items() if k in fields})

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + csv_rows)

    # Summary
    prec, rec, f1 = precision_recall_f1(tp, fp, fn)
    mse = compute_mse([x[0] for x in mse_data], [x[1] for x in mse_data]) if mse_data else 0.0
    summary_fields = ['pipeline', 'images_with_gt', 'mse', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']
    summary_row = {
        'pipeline': pipeline,
        'images_with_gt': len(mse_data),
        'mse': round(mse, 3),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(prec, 3),
        'recall': round(rec, 3),
        'f1': round(f1, 3),
    }

    existing_summary = []
    if summary_file.exists():
        with open(summary_file, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get('pipeline') != pipeline:
                    existing_summary.append(r)

    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(existing_summary + [summary_row])

    return {
        'metrics_file': metrics_file,
        'summary_file': summary_file,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'mse': mse,
    }

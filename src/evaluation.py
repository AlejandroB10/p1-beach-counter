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


def compute_mae(predictions: List[int], ground_truth: List[int]) -> float:
    """Mean Absolute Error for image-level counting."""
    if len(predictions) != len(ground_truth) or not predictions:
        return 0.0
    return sum(abs(p - g) for p, g in zip(predictions, ground_truth)) / len(predictions)


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

"""Evaluation utilities for counting accuracy."""
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv


def count_from_boxes(boxes: List[Tuple[int, int, int, int]]) -> int:
    """Count number of detections from bounding boxes.
    
    Args:
        boxes: List of (x, y, w, h) bounding boxes
        
    Returns:
        Count (number of boxes)
    """
    return len(boxes)


def load_annotations(annotations_file: Path) -> Dict[str, int]:
    """Load ground truth annotations from CSV file.
    
    Expected CSV format: Has 'image_name' column (one row per person)
    Returns count per filename.
    
    Args:
        annotations_file: Path to CSV file with annotations
        
    Returns:
        Dictionary mapping filename to count
    """
    if not annotations_file.exists():
        return {}
    
    counts: Dict[str, int] = {}
    
    with open(annotations_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Support both 'filename' and 'image_name' columns
            filename = row.get('image_name') or row.get('filename')
            if filename:
                counts[filename] = counts.get(filename, 0) + 1
    
    return counts


def compute_mse(predictions: List[int], ground_truth: List[int]) -> float:
    """Compute Mean Squared Error between predictions and ground truth.
    
    Args:
        predictions: List of predicted counts
        ground_truth: List of ground truth counts
        
    Returns:
        MSE value
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    squared_errors = [(pred - gt) ** 2 for pred, gt in zip(predictions, ground_truth)]
    return sum(squared_errors) / len(squared_errors)


def get_gt_count(filename: str, gt_dict: Dict[str, int]) -> Optional[int]:
    """Get ground truth count for a filename.
    
    Args:
        filename: Image filename (basename only, e.g., "1660366800.jpg")
        gt_dict: Dictionary from load_annotations
        
    Returns:
        Count if available, None otherwise
    """
    return gt_dict.get(filename)

"""Preprocessing functions for background subtraction and thresholding."""

from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from src.io_utils import save_bgr


def hsv_hue_subtract(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Subtract background using HSV hue channel difference."""
    img_h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg_h = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)

    h_sub = img_h[:, :, 0] - bg_h[:, :, 0]

    merged = cv2.merge([h_sub, img_h[:, :, 1], img_h[:, :, 2]])
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def threshold_images(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Otsu and adaptive thresholding to grayscale image."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, otsu = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        7,
        5
    )

    return otsu, adaptive


def compute_median_background(
    image_paths: List[Union[str, Path]],
    max_images: int = 10
) -> np.ndarray:
    """Compute median background from multiple images."""
    imgs = []
    for p in image_paths[:max_images]:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)
    if not imgs:
        return None
    stack = np.stack(imgs, axis=0)
    median_bg = np.median(stack, axis=0).astype(np.uint8)
    return median_bg


def compute_and_save_background(
    image_paths: List[Union[str, Path]],
    out_dir: Path,
    max_images: int = 10,
    subdir: str = "background"
) -> np.ndarray:
    """Compute median background and save it; returns the image or None."""
    bg_img = compute_median_background(image_paths, max_images=max_images)
    if bg_img is None:
        return None
    bg_dir = Path(out_dir) / "viz" / subdir
    bg_dir.mkdir(parents=True, exist_ok=True)
    save_bgr(bg_dir / "median_background.png", bg_img)
    return bg_img

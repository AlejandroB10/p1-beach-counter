"""File I/O utilities for image handling."""

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np


def list_images(directory: Union[str, Path]) -> List[Path]:
    """List all .jpg images in directory, sorted alphabetically by name."""
    return sorted(Path(directory).glob("*.jpg"))


def read_bgr(path: Union[str, Path]) -> np.ndarray:
    """Read image in BGR format."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img


def save_bgr(path: Union[str, Path], img: np.ndarray) -> None:
    """Save BGR image to file, creating directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def save_gray(path: Union[str, Path], img: np.ndarray) -> None:
    """Save grayscale image to file, creating directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def load_and_crop(path: Union[str, Path], crop_y: int = 420) -> np.ndarray:
    """Load an image and crop from a given y-offset downward."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img[crop_y:, :]

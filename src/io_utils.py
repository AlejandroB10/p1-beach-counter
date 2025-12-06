"""I/O utilities for reading/writing images and managing directories."""
from pathlib import Path
from typing import List, Union
import cv2
import numpy as np


def list_images(raw_dir: Union[str, Path]) -> List[Path]:
    """List all .jpg images in raw_dir, sorted alphabetically by name.
    
    Args:
        raw_dir: Path to directory containing images
        
    Returns:
        List of Path objects, sorted by filename
    """
    raw_path = Path(raw_dir)
    images = sorted(raw_path.glob("*.jpg"))
    return images


def read_bgr(path: Union[str, Path]) -> np.ndarray:
    """Read an image in BGR color format.
    
    Args:
        path: Path to image file
        
    Returns:
        BGR image as numpy array
        
    Raises:
        FileNotFoundError: If image cannot be read
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def ensure_dir(path: Union[str, Path]) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def save_bgr(path: Union[str, Path], img: np.ndarray) -> None:
    """Save a BGR color image to disk.
    
    Args:
        path: Destination path
        img: BGR image array
    """
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


def save_gray(path: Union[str, Path], img: np.ndarray) -> None:
    """Save a grayscale image to disk.
    
    Args:
        path: Destination path
        img: Grayscale image array (single channel or 2D)
    """
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)

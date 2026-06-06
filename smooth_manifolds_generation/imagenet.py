"""
ImageNet dataset utilities.

Provides image loading, label reading, and global parameter initialisation.
Replaces the MATLAB functions: init_imagenet, read_imagenet_labels,
read_imagenet_thumbnails, read_imagenet_training_size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ImageNetState:
    """
    Replaces MATLAB global variables for ImageNet configuration.

    Attributes
    ----------
    image_size : int
        Target output image size (pixels, square).
    frame_size : int
        Frame size used for warping (padded input).
    object_size : int
        Canonical object diameter in pixels.
    data_dir : Path
        Root directory of the ImageNet dataset.
    categories : List[str]
        WordNet synset IDs (e.g. 'n01440764').
    labels : Dict[str, str]
        Mapping from synset ID to human-readable label.
    n_features : int
        Number of network features (set after convnet initialisation).
    """

    image_size: int = 64
    frame_size: int = 80
    object_size: int = 40
    data_dir: Path = Path(".")
    categories: List[str] = None
    labels: Dict[str, str] = None
    n_features: int = 0

    def __post_init__(self) -> None:
        if self.categories is None:
            self.categories = []
        if self.labels is None:
            self.labels = {}


_DEFAULT_STATE: Optional[ImageNetState] = None


def init_imagenet(
    data_dir: str,
    image_size: int = 64,
    frame_size: Optional[int] = None,
    object_size: Optional[int] = None,
) -> ImageNetState:
    """
    Initialise the ImageNet dataset state.

    Parameters
    ----------
    data_dir : str
        Root of the ImageNet dataset (must contain 'labels.txt' and
        per-synset subdirectories).
    image_size : int
        Output image size (default 64).
    frame_size : int, optional
        Defaults to image_size * 1.25.
    object_size : int, optional
        Defaults to image_size * 0.625.

    Returns
    -------
    ImageNetState
    """
    global _DEFAULT_STATE
    data_path = Path(data_dir)
    frame_size = frame_size or int(image_size * 1.25)
    object_size = object_size or int(image_size * 0.625)

    state = ImageNetState(
        image_size=image_size,
        frame_size=frame_size,
        object_size=object_size,
        data_dir=data_path,
    )

    labels_file = data_path / "labels.txt"
    if labels_file.exists():
        state.labels = _read_labels(labels_file)
        state.categories = list(state.labels.keys())

    _DEFAULT_STATE = state
    return state


def get_state() -> ImageNetState:
    """Return the globally initialised ImageNet state."""
    if _DEFAULT_STATE is None:
        raise RuntimeError("Call init_imagenet() before using ImageNet utilities.")
    return _DEFAULT_STATE


def read_imagenet_labels(labels_file: str) -> Dict[str, str]:
    """
    Read ImageNet category labels from a text file.

    Each line: '<synset_id> <label_text>'.
    """
    return _read_labels(Path(labels_file))


def _read_labels(path: Path) -> Dict[str, str]:
    labels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                labels[parts[0]] = parts[1]
    return labels


def read_imagenet_thumbnails(
    data_dir: str,
    categories: List[str],
    image_size: int = 64,
    max_per_category: int = 1300,
) -> Dict[str, np.ndarray]:
    """
    Load thumbnail images for a list of ImageNet categories.

    Parameters
    ----------
    data_dir : str
        Root ImageNet directory.
    categories : list of str
        Synset IDs to load.
    image_size : int
        Resize to this square size.
    max_per_category : int
        Maximum images per category.

    Returns
    -------
    dict mapping synset_id -> np.ndarray of shape (N, H, W, 3) uint8
    """
    from PIL import Image

    root = Path(data_dir)
    result: Dict[str, np.ndarray] = {}
    for syn in categories:
        syn_dir = root / syn
        if not syn_dir.is_dir():
            continue
        imgs = []
        for img_path in sorted(syn_dir.iterdir())[:max_per_category]:
            try:
                img = Image.open(img_path).convert("RGB").resize(
                    (image_size, image_size), Image.BICUBIC
                )
                imgs.append(np.array(img, dtype=np.uint8))
            except Exception:
                continue
        if imgs:
            result[syn] = np.stack(imgs)
    return result


def read_imagenet_training_size(
    data_dir: str, categories: List[str]
) -> Dict[str, int]:
    """Return the number of training images per category."""
    root = Path(data_dir)
    sizes: Dict[str, int] = {}
    for syn in categories:
        syn_dir = root / syn
        if syn_dir.is_dir():
            sizes[syn] = sum(1 for _ in syn_dir.iterdir())
        else:
            sizes[syn] = 0
    return sizes


def choose_template_images(
    data_dir: str,
    categories: List[str],
    n_objects: int,
    image_size: int = 64,
    seed: int = 0,
) -> Tuple[List[int], List[str]]:
    """
    Select a diverse set of template images (one per category).

    Corresponds to ``choose_imagenet_template_images.m``.
    """
    rng = np.random.default_rng(seed)
    selected_categories = rng.choice(categories, size=min(n_objects, len(categories)), replace=False).tolist()
    indices = [categories.index(c) for c in selected_categories]
    return indices, selected_categories

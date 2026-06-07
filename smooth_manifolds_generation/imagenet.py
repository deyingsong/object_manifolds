"""
ImageNet dataset utilities loaded from the pre-built thumbnail archive.

Mirrors the MATLAB functions in ``smooth_manifolds_generation/``:

  init_imagenet.m                  → ``init_imagenet()``
  read_imagenet_training_size.m    → ``read_imagenet_training_size()``
  read_imagenet_labels.m           → ``read_imagenet_labels()``
  read_imagenet_thumbnails.m       → ``read_imagenet_thumbnails(image_id)``
  choose_imagenet_template_images.m→ ``choose_imagenet_template_images()``
  sample_indices_one_per_category.m→ ``sample_indices_one_per_category()``

The thumbnail archive ``imagenet_all_thumbnails_<N>px.mat`` is a MATLAB v7.3
(HDF5) file with two datasets:

  thumbnails : uint8, shape (N_IMAGES, 3, FRAME_SIZE, FRAME_SIZE)
               Raw padded images at the large *frame* resolution.
               (h5py reverses MATLAB column-major dims, so MATLAB's
                (FRAME_SIZE, FRAME_SIZE, 3, N_IMAGES) → Python (N,3,H,W))
  labels     : HDF5 object-reference array of shape (N_IMAGES, 1)
               Each reference resolves to a uint16 array spelling the
               ImageNet synset ID (e.g. b'n01440764').

MATLAB geometry (from init_imagenet.m for the 64-px setting):

  IMAGENET_OBJECT_SIZE  = 48          object square in pixels
  SURROUND_FACTOR       = 3
  IMAGENET_FRAME_SIZE   = 144         input thumbnail size (48 × 3)
  IMAGENET_FRAME        = 8           border added on each side
  IMAGENET_IMAGE_SIZE   = 64          output image size (48 + 2×8)
  IMAGENET_FRAME_LIMITS = [41, 104]   world limits of the output view
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Global state (mirrors MATLAB globals in init_imagenet.m)
# ---------------------------------------------------------------------------

@dataclass
class ImageNetState:
    """
    Replaces MATLAB global variables set by ``init_imagenet.m``.

    Attributes
    ----------
    image_size : int
        Final output image size in pixels (``IMAGENET_IMAGE_SIZE``).
    frame_size : int
        Padded input size: ``object_size × surround_factor``
        (``IMAGENET_FRAME_SIZE``).
    object_size : int
        Canonical object diameter (``IMAGENET_OBJECT_SIZE``).
    frame : int
        Border width on each side (``IMAGENET_FRAME``).
    surround_factor : float
        ``frame_size / object_size`` (``SURROUND_FACTOR``).
    frame_limits : Tuple[float, float]
        World-coordinate limits of the output view in the input frame
        (``IMAGENET_FRAME_LIMITS``).
    thumbnails_path : Path
        Full path to the ``.mat`` archive.
    n_features : int
        Network feature count; set after ``generate_convnet_model_metadata``.
    """

    image_size: int = 64
    frame_size: int = 144
    object_size: int = 48
    frame: int = 8
    surround_factor: float = 3.0
    frame_limits: Tuple[float, float] = (41.0, 104.0)
    thumbnails_path: Path = Path("smooth_manifolds_generation/imagenet_all_thumbnails_64px.mat")
    n_features: int = 0


# Module-level singletons (equivalent to MATLAB globals)
_STATE: Optional[ImageNetState] = None
_THUMBNAILS: Optional[np.ndarray] = None   # shape (N, 3, H, W) uint8, cached
_LABELS: Optional[List[str]] = None        # list of synset strings, cached


# ---------------------------------------------------------------------------
# Initialisation  (init_imagenet.m)
# ---------------------------------------------------------------------------

def init_imagenet(
    thumbnails_path: Optional[str] = None,
    image_size: int = 64,
) -> ImageNetState:
    """
    Initialise the ImageNet state from the thumbnail archive.

    Corresponds to ``init_imagenet.m``.  Call once before any other function
    in this module.

    Parameters
    ----------
    thumbnails_path : str, optional
        Path to ``imagenet_all_thumbnails_<N>px.mat``.  Defaults to
        ``smooth_manifolds_generation/imagenet_all_thumbnails_64px.mat``
        relative to the current working directory.
    image_size : int
        Desired output image size (64, 128, …).  Controls which archive is
        loaded and sets all derived geometry.

    Returns
    -------
    ImageNetState
    """
    global _STATE, _THUMBNAILS, _LABELS

    # ---- geometry matching init_imagenet.m --------------------------------
    if image_size == 64:
        object_size, frame, surround_factor = 48, 8, 3
    elif image_size == 128:
        object_size, frame, surround_factor = 64, 32, 3
    elif image_size == 32:
        object_size, frame, surround_factor = 24, 4, 2
    elif image_size == 50:
        object_size, frame, surround_factor = 40, 5, 2
    else:
        # Generic fallback: keep MATLAB's ratio
        object_size = int(round(image_size * 0.75))
        frame = (image_size - object_size) // 2
        surround_factor = 3

    frame_size = object_size * surround_factor
    # IMAGENET_FRAME_LIMITS = [1 + (frame_size - image_size)/2,
    #                          frame_size/2 + image_size/2]
    lo = 1.0 + (frame_size - image_size) / 2.0
    hi = frame_size / 2.0 + image_size / 2.0

    if thumbnails_path is None:
        thumbnails_path = (
            f"smooth_manifolds_generation/imagenet_all_thumbnails_{image_size}px.mat"
        )

    _STATE = ImageNetState(
        image_size=image_size,
        frame_size=int(frame_size),
        object_size=object_size,
        frame=frame,
        surround_factor=float(surround_factor),
        frame_limits=(lo, hi),
        thumbnails_path=Path(thumbnails_path),
    )
    # Reset caches so the new archive is loaded on next access
    _THUMBNAILS = None
    _LABELS = None

    print(
        f"Objects are {object_size} x {object_size} "
        f"in the middle of {image_size} x {image_size} image"
    )
    return _STATE


def get_state() -> ImageNetState:
    """Return the module-level ``ImageNetState``; raises if not initialised."""
    if _STATE is None:
        raise RuntimeError("Call init_imagenet() before using ImageNet utilities.")
    return _STATE


# ---------------------------------------------------------------------------
# Internal HDF5 helpers
# ---------------------------------------------------------------------------

def _load_archive() -> None:
    """Lazy-load thumbnails and labels from the ``.mat`` HDF5 file."""
    global _THUMBNAILS, _LABELS
    if _THUMBNAILS is not None:
        return                 # already loaded

    import h5py

    state = get_state()
    path = state.thumbnails_path
    if not path.exists():
        raise FileNotFoundError(
            f"Thumbnail archive not found: {path}\n"
            "Download imagenet_all_thumbnails_64px.mat from "
            "https://doi.org/10.6084/m9.figshare.11494314 "
            f"and save it at {path}"
        )

    print(f"Loading ImageNet thumbnails from {path} …")
    with h5py.File(path, "r") as f:
        # thumbnails: MATLAB stores as (H, W, C, N) column-major.
        # h5py reverses dims → (N, C, H, W), dtype uint8.
        _THUMBNAILS = f["thumbnails"][:]

        # labels: MATLAB cell array stored as HDF5 object-reference array.
        # Shape is (N_IMAGES, 1); each ref resolves to a uint16 array
        # whose values are the ASCII codes of the synset string.
        ref_array = f["labels"][:]          # (N_IMAGES, 1) dtype object
        labels: List[str] = []
        for ref in ref_array.flat:
            char_data = f[ref][:]           # 1-D uint16 array
            labels.append("".join(chr(int(c)) for c in char_data))
        _LABELS = labels

    assert _THUMBNAILS.ndim == 4, "Unexpected thumbnail shape"


# ---------------------------------------------------------------------------
# Public API mirroring the MATLAB functions
# ---------------------------------------------------------------------------

def read_imagenet_training_size() -> int:
    """
    Return the number of images in the thumbnail archive.

    Corresponds to ``read_imagenet_training_size.m``.
    """
    _load_archive()
    return int(_THUMBNAILS.shape[0])


def read_imagenet_labels() -> List[str]:
    """
    Return the list of synset-ID labels, one per training image (1-based
    index maps to ``labels[index - 1]``).

    Corresponds to ``read_imagenet_labels.m``.
    """
    _load_archive()
    return _LABELS


def read_imagenet_thumbnails(image_id: int) -> np.ndarray:
    """
    Return the thumbnail for a single image.

    Corresponds to ``read_imagenet_thumbnails.m``.

    Parameters
    ----------
    image_id : int
        1-based image index (matches MATLAB convention).

    Returns
    -------
    np.ndarray
        Shape ``(3, FRAME_SIZE, FRAME_SIZE)``, dtype uint8.
        The caller (``calc_imagenet_warp``) casts to ``float32`` and
        divides by 255 before passing to the network.
    """
    _load_archive()
    n = _THUMBNAILS.shape[0]
    if not (1 <= image_id <= n):
        raise IndexError(f"image_id={image_id} out of range [1, {n}]")
    return _THUMBNAILS[image_id - 1]       # (3, H, W) uint8


# ---------------------------------------------------------------------------
# Image selection  (choose_imagenet_template_images.m,
#                   sample_indices_one_per_category.m)
# ---------------------------------------------------------------------------

def sample_indices_one_per_category(
    n_train: int,
    n_objects: int,
    rng: Optional[np.random.Generator] = None,
    blacklisted: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample ``n_objects`` image indices, at most one per synset category.

    Corresponds to ``sample_indices_one_per_category.m``.

    Parameters
    ----------
    n_train : int
        Total number of training images.
    n_objects : int
        How many objects to select.
    rng : np.random.Generator, optional
        RNG to use.  Created from the global numpy state if None.
    blacklisted : np.ndarray of int, optional
        1-based indices to exclude.

    Returns
    -------
    np.ndarray of int
        1-based image indices, shape ``(n_objects,)``.
    """
    _load_archive()
    if rng is None:
        rng = np.random.default_rng()

    labels = _LABELS
    blacklist_set = set(blacklisted.tolist()) if blacklisted is not None else set()

    perm = rng.permutation(n_train) + 1     # 1-based shuffle
    used_categories: set = set()
    selected: List[int] = []
    skipped = 0

    for idx in perm:
        if int(idx) in blacklist_set:
            skipped += 1
            continue
        cat = labels[idx - 1]
        if cat in used_categories:
            skipped += 1
            continue
        used_categories.add(cat)
        selected.append(int(idx))
        if len(selected) == n_objects:
            break

    if len(selected) < n_objects:
        raise RuntimeError(
            f"Could only select {len(selected)}/{n_objects} objects "
            f"with unique categories (skipped {skipped})."
        )
    print(f"Skipped {skipped} images to choose {len(selected)} images with unique categories")
    return np.array(selected, dtype=int)


def _sample_indices_blacklisted(
    n_train: int,
    n_objects: int,
    blacklisted: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Helper used by choose_imagenet_template_images for random_seed > 0."""
    return sample_indices_one_per_category(
        n_train, n_objects, rng=rng, blacklisted=blacklisted
    )


def choose_imagenet_template_images(
    n_objects: int,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Select a reproducible, non-overlapping set of template images.

    Corresponds to ``choose_imagenet_template_images.m``.

    Uses ``sample_indices_one_per_category`` with a fixed seed strategy that
    mirrors the MATLAB implementation:

    * ``n_objects == 128`` → ``base_seed = 1, random_seed = 1``
    * ``n_objects == 32/16`` → ``base_seed = 0, random_seed = 4``
    * other → ``base_seed = 0, random_seed = 0``

    Passing an explicit ``random_seed`` overrides the defaults.

    Parameters
    ----------
    n_objects : int
    random_seed : int, optional

    Returns
    -------
    np.ndarray of int
        1-based image indices, shape ``(n_objects,)``.
    """
    n_train = read_imagenet_training_size()

    # Determine seeds following the MATLAB default logic
    if random_seed is None:
        if n_objects == 128:
            base_seed, random_seed = 1, 1
        elif n_objects in (32, 16):
            base_seed, random_seed = 0, 4
        else:
            base_seed, random_seed = 0, 0
    else:
        base_seed = 1 if n_objects == 128 else 0

    if random_seed == base_seed:
        rng = np.random.default_rng(random_seed)
        return sample_indices_one_per_category(n_train, n_objects, rng=rng)

    # Build up blacklist by replaying all previous seeds
    blacklisted = np.array([], dtype=int)
    for seed in range(base_seed, random_seed):
        rng = np.random.default_rng(seed)
        if seed == base_seed:
            prev = sample_indices_one_per_category(n_train, n_objects, rng=rng)
        else:
            prev = _sample_indices_blacklisted(n_train, n_objects, blacklisted, rng)
        blacklisted = np.concatenate([blacklisted, prev])

    rng = np.random.default_rng(random_seed)
    return _sample_indices_blacklisted(n_train, n_objects, blacklisted, rng)

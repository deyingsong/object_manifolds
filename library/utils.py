"""
Utility functions shared across the library.
"""

import socket
import warnings
from typing import Optional

import numpy as np


def assert_warn(condition: bool, message: str, *args) -> None:
    """Print a warning when condition is False (non-fatal assertion)."""
    if not condition:
        formatted = message % args if args else message
        warnings.warn(f"Warning: {formatted}", stacklevel=2)


def hostname() -> str:
    """Return the machine's hostname."""
    try:
        return socket.gethostname()
    except Exception:
        return ""


def sample_indices(N: int, K: int, R: int = 1) -> np.ndarray:
    """
    Sample R independent sets of K indices from {0, ..., N-1} without replacement.

    Parameters
    ----------
    N : int
        Pool size.
    K : int
        Subset size.
    R : int
        Number of subsets.

    Returns
    -------
    np.ndarray
        Shape (R, K), dtype int.
    """
    rng = np.random.default_rng()
    samples = np.zeros((R, K), dtype=int)
    for r in range(R):
        samples[r] = rng.choice(N, size=K, replace=False)
    return samples


def sample_random_labels(n_objects: int, labeling_type: int = 0) -> np.ndarray:
    """
    Sample binary labels ±1 for *n_objects* objects.

    Parameters
    ----------
    n_objects : int
        Number of objects.
    labeling_type : int
        0 – IID Bernoulli ±1
        1 – balanced (equal +1/-1 counts)
        2 – sparse (one −1, rest +1)

    Returns
    -------
    np.ndarray
        Shape (n_objects,) with values ±1.
    """
    rng = np.random.default_rng()
    if labeling_type == 2:
        y = np.ones(n_objects, dtype=float)
        y[rng.integers(n_objects)] = -1.0
    elif labeling_type == 1:
        y = np.ones(n_objects, dtype=float)
        half = round(n_objects / 2)
        idx = rng.choice(n_objects, size=half, replace=False)
        y[idx] = -1.0
        assert abs(y.sum()) <= 1, "Non-balanced labeling"
    else:
        y = rng.integers(0, 2, size=n_objects) * 2.0 - 1.0
    return y

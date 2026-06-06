"""
Sampling utilities module.

Provides utilities for sampling indices and random labels.
"""

from typing import Tuple
import numpy as np


def sample_indices(N: int, K: int, R: int = 1) -> np.ndarray:
    """
    Sample R samples of K indices from N values without replacement.
    
    Parameters
    ----------
    N : int
        Total number of values
    K : int
        Number of indices to sample per sample
    R : int, optional
        Number of samples to draw (default: 1)
    
    Returns
    -------
    np.ndarray
        Array of shape (R, K) containing sampled indices
    """
    samples = np.zeros((R, K), dtype=int)
    
    for r in range(R):
        samples[r, :] = np.random.choice(N, K, replace=False)
    
    return samples


def sample_random_labels(N_OBJECTS: int, random_labeling_type: int = 1) -> np.ndarray:
    """
    Sample random labels for objects using specified labeling scheme.
    
    Parameters
    ----------
    N_OBJECTS : int
        Number of objects
    random_labeling_type : int, optional
        Type of labeling scheme (default: 1)
        - 0: IID random labels (binary)
        - 1: Balanced labels (approximately half +1, half -1)
        - 2: Sparse labels (one -1, rest +1)
    
    Returns
    -------
    np.ndarray
        Array of shape (N_OBJECTS,) with labels in {-1, +1}
    
    Raises
    ------
    AssertionError
        If balanced labeling is not balanced within tolerance
    """
    if random_labeling_type == 2:  # Sparse labeling
        y = np.ones(N_OBJECTS, dtype=int)
        y[np.random.randint(N_OBJECTS)] = -1
    
    elif random_labeling_type == 1:  # Balanced labeling
        y = np.ones(N_OBJECTS, dtype=int)
        n_negative = N_OBJECTS // 2
        negative_indices = np.random.choice(N_OBJECTS, n_negative, replace=False)
        y[negative_indices] = -1
        assert np.abs(np.sum(y)) <= 1, f"Non-balanced labeling: sum={np.sum(y)}"
    
    else:  # IID labeling (random_labeling_type == 0)
        y = 2 * np.random.randint(0, 2, N_OBJECTS) - 1
    
    return y

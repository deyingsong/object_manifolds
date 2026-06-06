"""
Modified Gram-Schmidt orthogonalization.

Reference:
  Z. Wen and W. Yin
  A feasible method for optimization with orthogonality constraints
"""

import numpy as np


def gram_schmidt(V: np.ndarray) -> np.ndarray:
    """
    Modified Gram-Schmidt orthogonalization.

    Parameters
    ----------
    V : np.ndarray
        Matrix of shape (n, k) whose columns are to be orthonormalized.

    Returns
    -------
    np.ndarray
        Orthonormal matrix of shape (n, k).
    """
    n, k = V.shape
    V = V.copy().astype(float)
    for j in range(k):
        for i in range(j):
            proj = np.dot(V[:, i], V[:, j]) / np.dot(V[:, i], V[:, i])
            V[:, j] -= proj * V[:, i]
        norm = np.linalg.norm(V[:, j])
        if norm > 0:
            V[:, j] /= norm
    return V

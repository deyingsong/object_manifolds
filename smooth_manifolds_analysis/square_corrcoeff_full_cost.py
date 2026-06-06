"""
Square correlation coefficient cost function module.

Computes cost and gradient for square correlation minimization.
Based on square_corrcoeff_full_cost.m
"""

from typing import Tuple, Optional
import numpy as np


def square_corrcoeff_full_cost(V: np.ndarray, X: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """
    Compute cost and gradient of square correlation coefficient.
    
    Parameters
    ----------
    V : np.ndarray
        Basis vectors of shape (N, K) where N is the data dimension and K is the rank
    X : np.ndarray
        Data matrix of shape (P, N) where P is the number of samples
    
    Returns
    -------
    tuple
        (cost, gradient) where:
        - cost: float, the computed cost
        - gradient: np.ndarray of shape (N, K), the gradient with respect to V
    
    Notes
    -----
    This assumes V is on the Stiefel manifold (V'*V = I).
    
    The cost measures the square correlations between residual vectors after
    projecting out the subspace spanned by V.
    """
    P, N = X.shape
    K = V.shape[1]
    
    assert V.shape == (N, K), f"V shape {V.shape} does not match (N={N}, K={K})"
    
    # Compute correlation matrix
    C = X @ X.T  # [P x P]
    c = X @ V    # [P x K]
    c0 = np.diag(C) - np.sum(c ** 2, axis=1)  # [P]
    
    # Compute cost
    denom = np.outer(c0, c0)  # [P x P]
    numerator = C - c @ c.T  # [P x P]
    Fmn = numerator ** 2 / denom  # [P x P]
    cost = np.sum(Fmn) / 2
    
    # Compute gradient if needed
    gradient = None
    gradient = _compute_gradient(V, X, C, c, c0, Fmn, numerator, denom)
    
    return cost, gradient


def _compute_gradient(V: np.ndarray, X: np.ndarray, C: np.ndarray, c: np.ndarray,
                     c0: np.ndarray, Fmn: np.ndarray, numerator: np.ndarray,
                     denom: np.ndarray) -> np.ndarray:
    """
    Compute gradient of the cost function.
    
    Parameters
    ----------
    V : np.ndarray
        Basis vectors
    X : np.ndarray
        Data matrix
    C : np.ndarray
        Correlation matrix (X @ X.T)
    c : np.ndarray
        Projections (X @ V)
    c0 : np.ndarray
        Residual norms
    Fmn : np.ndarray
        Cost matrix
    numerator : np.ndarray
        Numerator of cost function
    denom : np.ndarray
        Denominator of cost function
    
    Returns
    -------
    np.ndarray
        Gradient of shape (N, K)
    """
    P, N = X.shape
    K = V.shape[1]
    
    # Reshape for broadcasting
    X1 = X[:, np.newaxis, :]  # [P, 1, N]
    X2 = X[np.newaxis, :, :]  # [1, P, N]
    C1 = c[:, np.newaxis, np.newaxis, :]  # [P, 1, 1, K]
    C2 = c[np.newaxis, :, np.newaxis, :]  # [1, P, 1, K]
    c0_1 = c0[:, np.newaxis, np.newaxis, np.newaxis]  # [P, 1, 1, 1]
    c0_2 = c0[np.newaxis, :, np.newaxis, np.newaxis]  # [1, P, 1, 1]
    denom_sq = denom ** 2  # [P, P]
    
    # Compute gradient components
    Gmni = np.zeros((P, P, N, K))
    
    # First term
    term1 = (numerator / denom[:, :, np.newaxis, np.newaxis]) * C1 * X1
    Gmni -= term1
    
    # Second term
    term2 = (numerator / denom[:, :, np.newaxis, np.newaxis]) * C2 * X2
    Gmni -= term2
    
    # Third term
    ratio = (Fmn / denom_sq)[:, :, np.newaxis, np.newaxis]
    term3 = ratio * c0_1 * C2 * X1
    Gmni += term3
    
    # Fourth term
    term4 = ratio * c0_2 * C1 * X2
    Gmni += term4
    
    # Sum over P dimensions
    gradient = np.sum(np.sum(Gmni, axis=0), axis=0)  # [N, K]
    
    return gradient

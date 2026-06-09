"""
Data preprocessing and dimensionality reduction for tuning-function arrays.

Tuning functions have shape (N_NEURONS, N_SAMPLES, N_OBJECTS).

Classes
-------
TuningFunctionPreprocessor
    8 global preprocessing schemes (z-normalisation, whitening, center
    decorrelation, etc.).
LowDimensionalManifold
    SVD-based low-rank approximation of each manifold, with optional
    center-subspace projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import assert_warn


class TuningFunctionPreprocessor:
    """
    Apply one of 8 global preprocessing schemes to a tuning-function array.

    Mode 0 – identity (no preprocessing)
    Mode 1 – per-neuron z-score (mean-zero, unit variance across all conditions)
    Mode 2 – whitening (full covariance whitening)
    Mode 3 – orthogonalise manifold centers (decorrelate)
    Mode 4/5 – partial center decorrelation (preserving norms)
    Mode 6 – project into center span, decorrelated
    Mode 7 – project into center span, without decorrelation
    Mode 8 – project into center span with norm scaling
    """

    def __init__(self, mode: int = 0) -> None:
        if mode not in range(9):
            raise ValueError(f"Unknown preprocessing mode: {mode}")
        self.mode = mode

    def __call__(self, tf: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        tf : np.ndarray
            Tuning function of shape (N_NEURONS, N_SAMPLES, N_OBJECTS).

        Returns
        -------
        np.ndarray
            Preprocessed tuning function (same shape unless mode 6-8).
        """
        return self.apply(tf)

    def apply(self, tf: np.ndarray) -> np.ndarray:
        n_neurons, n_samples, n_objects = tf.shape
        N_total = n_samples * n_objects

        if self.mode == 0:
            return tf

        if self.mode == 1:
            flat = tf.reshape(n_neurons, N_total)
            mean_ = np.nanmean(flat, axis=1, keepdims=True)
            flat_c = flat - mean_
            std_ = np.nanstd(flat_c, ddof=1, axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            flat_normed = flat_c / std_
            mean_normed = mean_ / std_
            # Check: mean of squares of the zero-mean normalised data == (N-1)/N
            assert_warn(
                np.linalg.norm(
                    np.nanmean(flat_normed**2, axis=1) - (N_total - 1) / N_total,
                    np.inf,
                ) < 1e-8,
                "Mode-1 normalisation failed",
            )
            result = flat_normed + mean_normed
            return result.reshape(n_neurons, n_samples, n_objects)

        if self.mode == 2:
            flat = tf.reshape(n_neurons, N_total)
            flat_c = flat - np.nanmean(flat, axis=1, keepdims=True)
            U, S, _ = np.linalg.svd(flat_c, full_matrices=False)
            S_mat = np.diag(S)
            result = np.linalg.solve(S_mat, U.T @ flat_c) * np.sqrt(N_total - 1)
            assert_warn(
                np.linalg.norm(np.nanmean(result**2, axis=1) - (N_total - 1) / N_total, np.inf) < 1e-10,
                "Mode-2 normalisation failed",
            )
            return result.reshape(n_neurons, n_samples, n_objects)

        if self.mode == 3:
            Xc = np.nanmean(tf, axis=1).reshape(n_neurons, n_objects)
            _, cS, cV = np.linalg.svd(Xc, full_matrices=False)
            I = np.eye(n_neurons)
            W = (I - Xc @ np.linalg.pinv(Xc.T @ Xc) @ Xc.T
                 + Xc @ cV.T @ np.diag(np.diag(np.linalg.matrix_power(np.diag(cS), -3))) @ cV @ Xc.T)
            assert not np.any(np.isnan(W))
            result = (W @ tf.reshape(n_neurons, N_total)).reshape(n_neurons, n_samples, n_objects)
            centers = np.nanmean(result, axis=1)
            Cc = centers.T @ centers
            assert_warn(
                np.linalg.norm(Cc - np.eye(n_objects), np.inf) < 1e-8,
                "Center decorrelation error: %.3e", np.linalg.norm(Cc - np.eye(n_objects), np.inf),
            )
            return result

        if self.mode in (4, 5):
            Xc = np.nanmean(tf, axis=1).reshape(n_neurons, n_objects)
            cU, cS, cV = np.linalg.svd(Xc, full_matrices=False)
            Var = np.diag(np.diag(Xc.T @ Xc))
            I = np.eye(n_neurons)
            W = (I - Xc @ np.linalg.pinv(Xc.T @ Xc) @ Xc.T
                 + cU @ np.sqrt(Var) @ cV @ np.diag(cS**-2) @ cV.T @ Xc.T)
            assert not np.any(np.isnan(W))
            result = (W @ tf.reshape(n_neurons, N_total)).reshape(n_neurons, n_samples, n_objects)
            if self.mode == 5:
                gm = np.nanmean(result.reshape(n_neurons, N_total), axis=1, keepdims=True)
                result = result - gm.reshape(n_neurons, 1, 1)
            return result

        if self.mode == 6:
            Xc = np.nanmean(tf, axis=1).reshape(n_neurons, n_objects)
            cU, cS, _ = np.linalg.svd(Xc, full_matrices=False)
            W = np.diag(1.0 / cS) @ cU.T
            result = (W @ tf.reshape(n_neurons, N_total)).reshape(n_objects, n_samples, n_objects)
            return result

        if self.mode == 7:
            Xc = np.nanmean(tf, axis=1).reshape(n_neurons, n_objects)
            cU, _, _ = np.linalg.svd(Xc, full_matrices=False)
            W = cU.T
            result = (W @ tf.reshape(n_neurons, N_total)).reshape(n_objects, n_samples, n_objects)
            gm = np.nanmean(result.reshape(n_objects, N_total), axis=1, keepdims=True)
            result = result - gm.reshape(n_objects, 1, 1)
            return result

        if self.mode == 8:
            Xc = np.nanmean(tf, axis=1).reshape(n_neurons, n_objects)
            Var = np.diag(np.diag(Xc.T @ Xc))
            cU, cS, cV = np.linalg.svd(Xc, full_matrices=False)
            W = np.sqrt(Var) @ cV @ np.diag(1.0 / cS) @ cU.T
            result = (W @ tf.reshape(n_neurons, N_total)).reshape(n_objects, n_samples, n_objects)
            gm = np.nanmean(result.reshape(n_objects, N_total), axis=1, keepdims=True)
            result = result - gm.reshape(n_objects, 1, 1)
            return result

        return tf


class LowDimensionalManifold:
    """
    Project a tuning-function array into a low-rank subspace.

    Optionally preserves common center correlations by using a pre-computed
    Stiefel basis.
    """

    def __init__(
        self,
        target_dim: Optional[int] = None,
        local_preprocessing: int = 0,
        center_norm_factor: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        target_dim : int, optional
            Desired reduced dimension.  Defaults to min(N_NEURONS, N_SAMPLES) - 1.
        local_preprocessing : int
            0 – raw data (only SVD reduction)
            1 – orthogonalise centers
            2 – random centers
            3 – permuted manifold
        center_norm_factor : float
            Multiplicative factor applied to manifold centers.
        """
        self.target_dim = target_dim
        self.local_preprocessing = local_preprocessing
        self.center_norm_factor = center_norm_factor

    def reduce(self, tf: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        tf : np.ndarray  (N_NEURONS, N_SAMPLES, N_OBJECTS)

        Returns
        -------
        np.ndarray  (D, N_SAMPLES, N_OBJECTS) where D <= min(N, M).
        """
        n_neurons, n_samples, n_objects = tf.shape
        D = self.target_dim or min(n_neurons, n_samples) - 1
        rng = np.random.default_rng()

        result = np.empty((D, n_samples, n_objects))
        for obj_i in range(n_objects):
            X = tf[:, :, obj_i]                          # (N, M)
            center = np.mean(X, axis=1, keepdims=True)
            Xc = X - center

            U, _, _ = np.linalg.svd(Xc, full_matrices=False)
            P = U[:, :D]                                  # top D PCs

            if self.local_preprocessing == 1:
                # Orthogonalise center
                c = center.ravel()
                c /= np.linalg.norm(c) + 1e-30
                P = P - np.outer(c, c @ P)
                P, _ = np.linalg.qr(P, mode="reduced")
                P = P[:, :D]
            elif self.local_preprocessing == 2:
                center = rng.standard_normal(center.shape)
                center /= np.linalg.norm(center)
            elif self.local_preprocessing == 3:
                perm = rng.permutation(n_samples)
                X = X[:, perm]

            result[:, :, obj_i] = P.T @ (
                X * self.center_norm_factor + center * (1 - self.center_norm_factor)
            )
        return result

    def reduce_with_basis(
        self,
        tf: np.ndarray,
        V_basis: np.ndarray,
        target_dim: Optional[int] = None,
    ) -> np.ndarray:
        """
        Project manifolds into the null-space of a pre-computed basis V_basis,
        then optionally further reduce dimension.

        Parameters
        ----------
        tf : np.ndarray  (N, M, P)
        V_basis : np.ndarray  (N, K)  – columns span the common subspace to remove.
        target_dim : int, optional

        Returns
        -------
        np.ndarray
        """
        n_neurons, n_samples, n_objects = tf.shape
        D = target_dim or self.target_dim or min(n_neurons, n_samples) - 1

        # Remove common subspace from each manifold's center
        centers = np.mean(tf, axis=1)                    # (N, P)
        centers_proj = V_basis @ (V_basis.T @ centers)
        residual_centers = centers - centers_proj        # (N, P)
        # Rescale by center_norm_factor
        new_centers = residual_centers * self.center_norm_factor

        result_list = []
        for obj_i in range(n_objects):
            X = tf[:, :, obj_i]
            center_old = centers[:, obj_i:obj_i + 1]
            # Rebuild manifold with modified center
            X_new = X - center_old + new_centers[:, obj_i:obj_i + 1]
            # Null-space projection
            X_ns = X_new - V_basis @ (V_basis.T @ X_new)
            result_list.append(X_ns)

        stacked = np.stack(result_list, axis=2)          # (N, M, P)
        return stacked

"""
Manifold geometry estimation.

Three methods for computing the mean field theory (MFT) manifold properties:
  - ManifoldPropertiesIterative  (calc_manifold_properties)
  - ManifoldPropertiesLS         (calc_manifold_properties2, no center-axis corrs)
  - ManifoldPropertiesLSCorr     (calc_manifold_properties3, with center-axis corrs)

All methods return a :class:`ManifoldGeometry` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .theory import theory_alpha0_cached
from .low_rank import ConstrainedLeastSquares
from .utils import assert_warn


@dataclass
class ManifoldGeometry:
    """
    Geometric properties of a single manifold.

    Attributes
    ----------
    mean_half_width1 : float
        Average maximal projection (first estimate, before convergence).
    mean_argmax_norm1 : float
        Average squared norm of argmax point (first estimate).
    mean_half_width2 : float
        Converged mean half-width (R_M).
    mean_argmax_norm2 : float
        Converged mean squared argmax norm.
    effective_dimension : float
        D_M = mean(w^2 / s^2).
    effective_dimension2 : float
        Alternative D_M estimate.
    alphac_hat : float
        Predicted classification capacity 1/alpha_c.
    """

    mean_half_width1: float = np.nan
    mean_argmax_norm1: float = np.nan
    mean_half_width2: float = np.nan
    mean_argmax_norm2: float = np.nan
    effective_dimension: float = np.nan
    effective_dimension2: float = np.nan
    alphac_hat: float = np.nan
    alphac_hat2: float = np.nan


class ManifoldPropertiesIterative:
    """
    Compute manifold geometry via an iterative mean-field method.

    Corresponds to ``calc_manifold_properties.m``.
    """

    def __init__(self, n_random_projections: int = 1000, kappa: float = 0.0) -> None:
        self.n_random_projections = n_random_projections
        self.kappa = kappa

    def compute(self, cF: np.ndarray, center_norm: float) -> ManifoldGeometry:
        """
        Parameters
        ----------
        cF : np.ndarray
            Tuning function of shape (N_NEURONS, N_SAMPLES).
        center_norm : float
            Norm used to scale the data and kappa.

        Returns
        -------
        ManifoldGeometry
        """
        assert center_norm > 0, "Norm must be positive"
        TOLERANCE = 0.05 / center_norm
        MAX_ITER = 100

        n_neurons = cF.shape[0]
        cF = cF / center_norm
        kappa = self.kappa / center_norm

        rng = np.random.default_rng()
        w = rng.standard_normal((self.n_random_projections, n_neurons))
        w_norm = np.sqrt(np.sum(w**2, axis=1))

        wF = w @ cF
        argmax = np.argmax(wF, axis=1)
        max_proj = wF[np.arange(self.n_random_projections), argmax]
        s0 = cF[:, argmax].T                        # (R, N)
        s0_norm_sq = np.sum(s0**2, axis=1)
        ws0 = np.sum(w * s0, axis=1)

        assert_warn(
            np.max(np.abs(ws0 - max_proj)) < 1e-5,
            "%.1e", np.max(np.abs(ws0 - max_proj)),
        )
        geom = ManifoldGeometry(
            mean_half_width1=float(np.mean(ws0)),
            mean_argmax_norm1=float(np.mean(s0_norm_sq)),
        )

        # Iterative convergence
        eta = 0.1
        error = 1.0
        it = 0
        while error > TOLERANCE and it < MAX_ITER:
            it += 1
            z0 = (ws0 + kappa) / (1.0 + s0_norm_sq)
            dw = w - z0[:, None] * s0
            wF_dw = dw @ cF
            argmax_new = np.argmax(wF_dw, axis=1)
            max_proj_dw = wF_dw[np.arange(self.n_random_projections), argmax_new]
            s0_proj = np.sum(dw * s0, axis=1)
            error_vec = max_proj_dw - s0_proj
            error = float(np.max(error_vec))
            if it > 1 and np.mean(error_vec) > np.mean(error_vec):
                eta *= 0.8
            if error > TOLERANCE:
                s1 = cF[:, argmax_new].T
                s0 = (1 - eta) * s0 + eta * s1
                s0_norm_sq = np.sum(s0**2, axis=1)
                ws0 = np.sum(w * s0, axis=1)

        geom.mean_half_width2 = float(np.mean(ws0))
        geom.mean_argmax_norm2 = float(np.mean(s0_norm_sq))
        geom.effective_dimension = float(np.mean(ws0**2 / s0_norm_sq))
        geom.effective_dimension2 = float(
            n_neurons * np.mean(ws0 / (w_norm * np.sqrt(s0_norm_sq))) ** 2
        )
        alpha_vals = theory_alpha0_cached(ws0 + kappa) / (1.0 + s0_norm_sq)
        geom.alphac_hat = float(1.0 / np.mean(1.0 / alpha_vals))
        return geom


class ManifoldPropertiesLS:
    """
    Compute manifold geometry via least-squares (no center-axis correlations).

    Corresponds to ``calc_manifold_properties2.m``.
    """

    def __init__(self, n_random_projections: int = 1000, kappa: float = 0.0) -> None:
        self.n_random_projections = n_random_projections
        self.kappa = kappa

    def compute(
        self, tuning_function: np.ndarray, center_norm: float
    ) -> ManifoldGeometry:
        """
        Parameters
        ----------
        tuning_function : np.ndarray
            Shape (N_NEURONS, N_SAMPLES).
        center_norm : float
            Scalar center norm for scaling.

        Returns
        -------
        ManifoldGeometry
        """
        assert np.isscalar(center_norm)
        n_neurons, n_samples = tuning_function.shape
        min_n = min(n_neurons, n_samples)
        D = min_n - 1

        # Subtract center and project onto low-dimensional subspace
        center = np.mean(tuning_function, axis=1, keepdims=True)
        cF = tuning_function - center
        Q, _ = np.linalg.qr(cF, mode="reduced")
        Q = Q[:, :D]
        F = np.vstack([Q.T @ tuning_function, np.ones((1, n_samples)) * center_norm])

        TOLERANCE = 0.05 / center_norm
        F = F / center_norm
        kappa_scaled = self.kappa / center_norm

        rng = np.random.default_rng()
        T = rng.standard_normal((self.n_random_projections, min_n))
        t_norm = np.sqrt(np.sum(T**2, axis=1))

        argmax = np.argmax(T @ F, axis=1)
        S = F[:, argmax].T
        theory_center = np.mean(S, axis=0)
        dS = S - theory_center
        dS_norm = np.sqrt(np.sum(dS**2, axis=1))
        TdS = np.sum(T * dS, axis=1)

        geom = ManifoldGeometry(
            mean_half_width1=float(np.mean(TdS)),
            mean_argmax_norm1=float(np.mean(dS_norm**2)),
        )

        # Interior / boundary partition
        interior_mask = np.sum(T * S, axis=1) + kappa_scaled >= 0
        center_vec = np.zeros(min_n)
        center_vec[-1] = 1.0

        V = T.copy()
        Lambda = np.full(self.n_random_projections, np.nan)

        if np.any(interior_mask):
            solver = ConstrainedLeastSquares(tolerance=TOLERANCE)
            VI = solver.solve_batch(T[interior_mask], F, kappa_scaled)
            V[interior_mask] = VI
            lam = (T[interior_mask] - VI) @ center_vec
            Lambda[interior_mask] = lam
            S[interior_mask] = (T[interior_mask] - VI) / lam[:, None]
            assert np.max(np.abs(S[:, -1] - 1)) < 1e-10

        Delta = np.sum((T - V)**2, axis=1)
        theory_center = np.mean(S, axis=0)
        dS = S - theory_center
        dS_norm = np.sqrt(np.sum(dS**2, axis=1))
        TdS = np.sum(T * dS, axis=1)

        geom.mean_half_width2 = float(np.mean(TdS))
        geom.mean_argmax_norm2 = float(np.mean(dS_norm**2))
        geom.effective_dimension = float(np.mean(TdS**2 / dS_norm**2))
        geom.effective_dimension2 = float(D * np.mean(TdS / (t_norm * dS_norm)) ** 2)
        alpha_vals = theory_alpha0_cached(TdS - kappa_scaled) / (1.0 + dS_norm**2)
        geom.alphac_hat = float(1.0 / np.mean(1.0 / alpha_vals))
        geom.alphac_hat2 = float(1.0 / np.mean(Delta))
        return geom


class ManifoldPropertiesLSCorr:
    """
    Compute manifold geometry via least-squares including center-axis correlations.

    Corresponds to ``calc_manifold_properties3.m``.
    """

    def __init__(self, n_random_projections: int = 1000, kappa: float = 0.0) -> None:
        self.n_random_projections = n_random_projections
        self.kappa = kappa

    def compute(
        self, tuning_function: np.ndarray, center_norm: float
    ) -> ManifoldGeometry:
        """
        Parameters
        ----------
        tuning_function : np.ndarray
            Shape (N_NEURONS, N_SAMPLES).
        center_norm : float

        Returns
        -------
        ManifoldGeometry
        """
        # Augmented data with center direction included
        n_neurons, n_samples = tuning_function.shape
        min_n = min(n_neurons, n_samples)

        center = np.mean(tuning_function, axis=1, keepdims=True)
        cF = tuning_function - center
        Q, _ = np.linalg.qr(cF, mode="reduced")
        Q = Q[:, :min_n]

        F_aug = np.vstack([
            Q.T @ tuning_function,
            np.ones((1, n_samples)) * center_norm,
        ])

        TOLERANCE = 0.05 / center_norm
        F_aug = F_aug / center_norm
        kappa_scaled = self.kappa / center_norm

        rng = np.random.default_rng()
        T = rng.standard_normal((self.n_random_projections, min_n + 1))
        t_norm = np.sqrt(np.sum(T**2, axis=1))

        argmax = np.argmax(T @ F_aug, axis=1)
        S = F_aug[:, argmax].T
        theory_center = np.mean(S, axis=0)
        dS = S - theory_center
        dS_norm = np.sqrt(np.sum(dS**2, axis=1))
        TdS = np.sum(T * dS, axis=1)

        geom = ManifoldGeometry(
            mean_half_width1=float(np.mean(TdS)),
            mean_argmax_norm1=float(np.mean(dS_norm**2)),
        )

        interior_mask = np.sum(T * S, axis=1) + kappa_scaled >= 0
        center_vec = np.zeros(min_n + 1)
        center_vec[-1] = 1.0

        V = T.copy()
        Lambda = np.full(self.n_random_projections, np.nan)

        if np.any(interior_mask):
            solver = ConstrainedLeastSquares(tolerance=TOLERANCE)
            VI = solver.solve_batch(T[interior_mask], F_aug, kappa_scaled)
            V[interior_mask] = VI
            lam = (T[interior_mask] - VI) @ center_vec
            Lambda[interior_mask] = lam
            S[interior_mask] = (T[interior_mask] - VI) / lam[:, None]

        Delta = np.sum((T - V)**2, axis=1)
        theory_center = np.mean(S, axis=0)
        dS = S - theory_center
        dS_norm = np.sqrt(np.sum(dS**2, axis=1))
        TdS = np.sum(T * dS, axis=1)

        geom.mean_half_width2 = float(np.mean(TdS))
        geom.mean_argmax_norm2 = float(np.mean(dS_norm**2))
        geom.effective_dimension = float(np.mean(TdS**2 / dS_norm**2))
        geom.effective_dimension2 = float(
            min_n * np.mean(TdS / (t_norm * dS_norm)) ** 2
        )
        alpha_vals = theory_alpha0_cached(TdS - kappa_scaled) / (1.0 + dS_norm**2)
        geom.alphac_hat = float(1.0 / np.mean(1.0 / alpha_vals))
        geom.alphac_hat2 = float(1.0 / np.mean(Delta))
        return geom

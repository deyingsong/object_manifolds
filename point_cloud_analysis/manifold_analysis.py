"""
Mean-field theory capacity analysis for general point-cloud manifolds.

Takes into account correlations between manifold centers via a factor-analysis
decomposition.

Primary entry point: :class:`ManifoldStableAnalysisCorr`
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
from scipy.integrate import dblquad

# Allow importing FOptM from package root
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from FOptM import opt_stiefel_gbb, StiefelOptions
from library.low_rank import SquareCorrCoeffCost
from library.cplex_interface import CplexOptions, cplexoptimset, cplexlsqlin


@dataclass
class ManifoldAnalysisOutput:
    """Results of manifold_stable_analysis_corr."""

    a_Mfull_vec: np.ndarray
    """Per-manifold full capacity (a_Mfull)."""
    a_M_vec: np.ndarray
    """Per-manifold geometric capacity (a_M via alphaB)."""
    R_M_vec: np.ndarray
    """Per-manifold mean radius R_M."""
    D_M_vec: np.ndarray
    """Per-manifold effective dimension D_M."""
    res_coeff0: float
    """Residual correlation coefficient before low-rank removal."""
    K0: int
    """Optimal low-rank K."""


@dataclass
class AnalysisOptions:
    """Options for ManifoldStableAnalysisCorr."""

    kappa: float = 0.0
    n_t: int = 1000
    flag_NbyM: bool = False
    center_scale: float = 1.0


class ManifoldStableAnalysisCorr:
    """
    Compute MFT capacity for a collection of point-cloud manifolds,
    accounting for center-center correlations.

    Parameters
    ----------
    options : AnalysisOptions

    Usage
    -----
    >>> opts = AnalysisOptions(kappa=0.0, n_t=1000)
    >>> analyser = ManifoldStableAnalysisCorr(opts)
    >>> output = analyser.analyse(manifolds)
    where ``manifolds`` is a list of (M_i, N) numpy arrays.
    """

    def __init__(self, options: AnalysisOptions) -> None:
        self.opts = options

    def analyse(self, XtotT: List[np.ndarray]) -> ManifoldAnalysisOutput:
        """
        Parameters
        ----------
        XtotT : list of np.ndarray
            Each element has shape (M_i, N) if flag_NbyM=False (default),
            or (N, M_i) if flag_NbyM=True.
            M_i = number of samples in the i-th manifold, N = feature dimension.

        Returns
        -------
        ManifoldAnalysisOutput
        """
        kappa = self.opts.kappa
        n_t = self.opts.n_t
        center_scale = self.opts.center_scale

        P = len(XtotT)
        # Transpose to (N, M_i) if needed
        Xtot = []
        for ii in range(P):
            Xi = XtotT[ii]
            if not self.opts.flag_NbyM:
                Xi = Xi.T
            Xtot.append(Xi.copy())

        N = Xtot[0].shape[0]
        M_vec = np.array([Xi.shape[1] for Xi in Xtot])

        # Global mean
        Xori = np.hstack(Xtot)
        X0 = np.mean(Xori, axis=1, keepdims=True)

        # Centre each manifold
        centers = np.zeros((N, P))
        centers_old = np.zeros((N, P))
        Xtot0 = []
        for pp in range(P):
            Xp_c = Xtot[pp] - X0
            c_old = np.mean(Xp_c, axis=1)
            centers_old[:, pp] = c_old
            centers[:, pp] = c_old * center_scale
            Xp_new = Xp_c - c_old[:, None] + centers[:, pp, None]
            Xtot0.append(Xp_new)

        # Factor analysis: find low-rank center structure
        K0, V11, res_coeff0 = self._run_factor_analysis(centers, max_iterations=20000)
        print(f"Optimal K: {K0}.")

        # Null-space projection
        XtotInput = []
        Xr0_ns_norm = np.zeros(P)
        for ii in range(P):
            M = M_vec[ii]
            Xr = Xtot0[ii]
            Xr_ns = Xr - V11 @ (V11.T @ Xr)
            Xr0_ns = np.mean(Xr_ns, axis=1)
            Xr0_ns_norm[ii] = np.linalg.norm(Xr0_ns)
            Xrr_ns = (Xr_ns - Xr0_ns[:, None]) / Xr0_ns_norm[ii]
            XtotInput.append(Xrr_ns)

        # Per-manifold analysis
        a_Mfull_vec = np.zeros(P)
        a_M_vec = np.zeros(P)
        R_M_vec = np.zeros(P)
        D_M_vec = np.zeros(P)

        for ii in range(P):
            sD1 = self._make_d1_data(XtotInput[ii])
            a_Mfull, a_M, R_M, D_M = self._each_manifold_analysis_D1(sD1, kappa, n_t)
            R_M_vec[ii] = R_M
            D_M_vec[ii] = D_M
            a_M_vec[ii] = a_M
            a_Mfull_vec[ii] = a_Mfull
            print(
                f"{ii+1} th manifold: D_M={D_M:.2f}, R_M={R_M:.2f}, "
                f"a_M={a_M:.2f}, norm={Xr0_ns_norm[ii]:.3f}"
            )

        print(
            f"Average of {P} manifolds: <D_M>={np.mean(D_M_vec):.2f}, "
            f"<R_M>={np.mean(R_M_vec):.2f}, "
            f"1/<1/a_M>={1.0/np.mean(1.0/a_M_vec):.2f}"
        )

        return ManifoldAnalysisOutput(
            a_Mfull_vec=a_Mfull_vec,
            a_M_vec=a_M_vec,
            R_M_vec=R_M_vec,
            D_M_vec=D_M_vec,
            res_coeff0=res_coeff0,
            K0=K0,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_factor_analysis(
        self, centers: np.ndarray, max_iterations: int = 20000
    ):
        """Run the factor-analysis optimisation to find the optimal K."""
        N_NEURONS, N_OBJECTS = centers.shape
        _, S_svd, _ = np.linalg.svd(centers - np.mean(centers, axis=1, keepdims=True))
        ll = S_svd
        cumvar = np.cumsum(ll**2) / np.sum(ll**2)
        maxK = int(np.where(cumvar >= 0.95)[0][0] if np.any(cumvar >= 0.95) else len(ll)) + 10

        # Basis
        global_mean = np.mean(centers.T, axis=0)
        Xb = centers.T - global_mean
        xbnorm = np.sqrt(np.sum(Xb**2, axis=1))
        Q, _ = np.linalg.qr(Xb.T, mode="reduced")
        N_NEURONS_red = N_OBJECTS - 1
        Proj = Q[:, :N_NEURONS_red]
        X = Xb @ Proj

        X0 = X.copy()
        xnorm0 = np.sqrt(np.sum(X0**2, axis=1))
        C0 = X0 @ X0.T / (np.outer(xnorm0, xnorm0) + 1e-30)
        res_coeff0 = float(
            (np.sum(np.abs(C0)) - N_OBJECTS) / N_OBJECTS / (N_OBJECTS - 1)
        )

        opts = StiefelOptions(
            record=0, mxitr=max_iterations,
            gtol=1e-6, xtol=1e-6, ftol=1e-8,
        )
        cost_fn = SquareCorrCoeffCost.vectorised

        rng = np.random.default_rng()
        V1 = None
        V1_mat: Dict[int, np.ndarray] = {}
        res_coeff = np.zeros(maxK + 1)
        best_k = 0
        best_V1 = None

        for ik in range(min(maxK, N_NEURONS_red)):
            k = ik + 1
            s = rng.standard_normal((N_OBJECTS, 1))
            init = (X @ s).ravel()
            V0 = np.column_stack([init] + ([V1] if V1 is not None else []))
            V0, _ = np.linalg.qr(V0, mode="reduced")
            V0 = V0[:, :k]

            V1_tmp, out = opt_stiefel_gbb(V0, cost_fn, opts, X)
            V1_mat[k] = V1_tmp
            X0 = X - (X @ V1_tmp) @ V1_tmp.T
            xnorm = np.sqrt(np.sum(X0**2, axis=1))
            C0 = X0 @ X0.T / (np.outer(xnorm, xnorm) + 1e-30)
            cost = float(
                (np.sum(np.abs(C0)) - N_OBJECTS) / N_OBJECTS / (N_OBJECTS - 1)
            )
            res_coeff[ik] = cost
            print(f" K={k} mean={cost:.3f} ({out.itr} iterations)")
            V1 = V1_tmp
            if ik > 3 and all(res_coeff[ik - j] > res_coeff[ik - j - 1] for j in range(3)):
                print("Optimal K0 found.")
                best_k = np.argmin(res_coeff[:ik + 1]) + 1
                break
        else:
            best_k = int(np.argmin(res_coeff[:ik + 1])) + 1

        best_V1 = V1_mat.get(best_k, np.zeros((N_NEURONS_red, 1)))
        V11 = Proj @ best_V1
        return best_k, V11, res_coeff0

    @staticmethod
    def _make_d1_data(S_r: np.ndarray) -> np.ndarray:
        """Reduce to D+1 dimensional representation."""
        D, m = S_r.shape
        if D > m:
            Q, _ = np.linalg.qr(S_r, mode="reduced")
            S_r = Q.T @ S_r
            D, m = S_r.shape

        sD = S_r.copy()
        sc = 1.0
        sD1_0 = np.vstack([sD, np.full((1, m), sc)])
        sD1 = sD1_0 / sc
        return sD1

    @staticmethod
    def _each_manifold_analysis_D1(
        sD1: np.ndarray, kappa: float, n_t: int
    ):
        """Analyse one manifold in D+1 dimensions."""
        D1, m = sD1.shape
        D = D1 - 1
        sc = 1.0
        c_hat = np.zeros(D1)
        c_hat[-1] = 1.0

        rng = np.random.default_rng()
        t_vec = rng.standard_normal((D1, n_t))

        ss, gg = ManifoldStableAnalysisCorr._maxproj(t_vec, sD1, sc)

        s_all = np.zeros((D1, n_t))
        v_f_all = np.zeros((D1, n_t))

        for jj in range(n_t):
            if gg[jj] + kappa < 0:
                v_f = t_vec[:, jj]
                s_f = ss[:, jj]
            else:
                v_f, s_f = ManifoldStableAnalysisCorr._compute_v_allpt(
                    t_vec[:, jj], ss[:, jj], 1e-8, kappa, sD1
                )
            s_all[:, jj] = s_f
            v_f_all[:, jj] = v_f

        # Geometry
        s0 = np.mean(s_all, axis=1)
        ds0 = s_all - s0[:, None]
        ds = ds0[:D, :] / (s_all[D:, :] + 1e-30)

        R_M = float(np.sqrt(np.mean(np.sum(ds**2, axis=0))))
        tD = t_vec[:D, :]
        sD_vec = s_all[:D, :]
        t_hat = tD / (np.sqrt(np.sum(tD**2, axis=0)) + 1e-30)
        s_hat = sD_vec / (np.sqrt(np.sum(sD_vec**2, axis=0)) + 1e-30)
        D_M = float(D * np.mean(np.sum(t_hat * s_hat, axis=0))**2)

        a_M = float(ManifoldStableAnalysisCorr._alpha_B(kappa, R_M, D_M))
        dists = np.sum((v_f_all - t_vec)**2, axis=0)
        a_Mfull = float(
            1.0 / np.mean(
                np.maximum(np.sum(t_vec * s_all, axis=0) + kappa, 0.0)**2
                / (np.sum(s_all**2, axis=0) + 1e-30)
            )
        )
        return a_Mfull, a_M, R_M, D_M

    @staticmethod
    def _maxproj(t_vec: np.ndarray, sD1: np.ndarray, sc: float):
        D1, n_t = t_vec.shape
        D = D1 - 1
        m = sD1.shape[1]
        scores = t_vec[:D, :].T @ sD1[:D, :]   # (n_t, m)
        imax = np.argmax(scores, axis=1)
        gt = np.max(scores, axis=1)
        s0 = np.zeros((D1, n_t))
        for j in range(n_t):
            s0[:D, j] = sD1[:D, imax[j]]
            s0[D, j] = sc
            gt[j] = t_vec[:, j] @ s0[:, j]
        return s0, gt

    @staticmethod
    def _compute_v_allpt(
        tt: np.ndarray, sDi: np.ndarray, eps: float, kappa: float, sD1: np.ndarray
    ):
        """
        Find v* = argmin ||v-t||^2  s.t.  sD1' v >= -kappa.

        Solved via IBM ILOG CPLEX (cplexlsqlin)

        Formulation::

            min  0.5 ||I*v - tt||^2
            s.t. -sD1.T @ v <= kappa * ones(m)   (i.e. sD1.T @ v >= -kappa)
        """
        D1, m = sD1.shape

        # Aineq v <= bineq  ⟺  -sD1.T v <= kappa*1  ⟺  sD1.T v >= -kappa
        Aineq = -sD1.T                              # (m, D1)
        bineq = kappa * np.ones(m)

        C_id = np.eye(D1)
        options = cplexoptimset(Display="off")

        v_new, _, _, flag, _, _ = cplexlsqlin(
            C_id, tt, Aineq, bineq, None, None, None, None, options
        )

        if flag >= 0 and np.all(np.isfinite(v_new)):
            v_f = v_new
        else:
            v_f = tt.copy()

        lam = float(np.sum(np.maximum(sD1.T @ v_f + kappa, 0)))
        if lam > 1e-10:
            s_f = (tt - v_f) / lam
        else:
            s_f = sDi.copy()
        return v_f, s_f

    @staticmethod
    def _alpha_B(kappa: float, radius: float, d: float) -> float:
        """Analytical capacity formula via 2D numerical integration."""
        R = radius
        k = kappa
        L = 50.0

        def p_d(r, d_):
            return (2**(1 - d_/2) * r**(d_-1) * np.exp(-0.5*r**2)
                    / max(float(np.math.gamma(d_/2)), 1e-300))

        def A_func(k_, R_, r, t):
            term1 = (((R_*r - t + k_)**2 / (R_**2 + 1))
                     * ((t - (k_ - r/R_)) > 0)
                     * ((k_ + R_*r - t) > 0))
            term2 = ((t - k_)**2 + r**2) * ((k_ - r/R_ - t) > 0)
            return term1 + term2

        def integrand(t, r):
            phi = np.exp(-0.5*t**2) / np.sqrt(2*np.pi)
            return A_func(k, R, r, t) * phi * p_d(r, d)

        try:
            val, _ = dblquad(integrand, 0, L, -L, L,
                             epsabs=1e-4, epsrel=1e-4)
            return 1.0 / val if val > 0 else np.inf
        except Exception:
            return np.nan

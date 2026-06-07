"""
Low-rank structure analysis of manifold centers.

Classes
-------
SquareCorrCoeffCost
    Objective and gradient for minimizing off-diagonal squared correlation
    coefficients (used with OptStiefelGBB).
ConstrainedLeastSquares
    Solve  min 0.5 ||Cv-t||^2  s.t.  Sv <= b  via a cutting-plane loop.
    Uses IBM ILOG CPLEX (cplexlsqlin) to exactly match the MATLAB original.
OptimalLowRankStructure
    Find the minimum rank K such that residual inter-manifold correlations
    are minimised.
"""

from __future__ import annotations

import sys
import os
import socket
from typing import Optional, Tuple

import numpy as np

# Allow importing FOptM from the package root
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from FOptM import opt_stiefel_gbb, StiefelOptions
from .utils import assert_warn
from .cplex_interface import CplexOptions, cplexoptimset, cplexlsqlin


def _is_cluster_node() -> bool:
    try:
        name = socket.gethostname()
        return not name or "brain" in name or "ielsc" in name
    except Exception:
        return False


class SquareCorrCoeffCost:
    """
    Cost function for minimising off-diagonal squared correlation coefficients.

    Objective:
      F(V) = 0.5 * sum_{m≠n} [C_mn^2 / (c0_m * c0_n)]

    where  C = X X',  c = X V,  c0_m = C_mm - ||c_m||^2.
    """

    @staticmethod
    def __call__(V: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Parameters
        ----------
        V : np.ndarray  (N, K)  – orthonormal basis (X is (P, N))
        X : np.ndarray  (P, N)

        Returns
        -------
        cost : float
        gradient : np.ndarray  (N, K)
        """
        return SquareCorrCoeffCost.compute(V, X)

    @staticmethod
    def compute(
        V: np.ndarray, X: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        P, N = X.shape
        K = V.shape[1]
        assert V.shape == (N, K)

        C = X @ X.T                        # (P, P)
        c = X @ V                           # (P, K)
        c0 = np.diag(C) - np.sum(c**2, axis=1)   # (P,)
        outer_c0 = np.outer(c0, c0)        # (P, P)
        Fmn = (C - c @ c.T)**2 / outer_c0  # (P, P)
        cost = float(np.sum(Fmn)) / 2.0

        # Gradient
        X1 = X[np.newaxis, :, :]           # (1, P, N)
        X2 = X[:, np.newaxis, :]           # (P, 1, N)
        ratio = (C - c @ c.T) / outer_c0   # (P, P)
        ratio2 = (C - c @ c.T)**2 / outer_c0**2  # (P, P)

        # sum over m,n: derivative w.r.t. V through c = XV
        # dF/dV = X' * dF/dc  (shape N x K)
        # Uses broadcasting for efficiency
        G = np.zeros_like(V)
        for ki in range(K):
            c_ki = c[:, ki]              # (P,)
            # d(c*c'.)[m,n] / d V_{.,k} involves c_m X_m + c_n X_n
            term1 = -2.0 * (ratio * c_ki[np.newaxis, :]) @ X   # (P, N)
            term2 = ratio2 * (c0[np.newaxis, :] * c_ki[np.newaxis, :]) @ X  # (P, N) - approx
            G[:, ki] = np.sum(term1 + term2, axis=0)
        return cost, G

    @staticmethod
    def vectorised(V: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """Fully vectorised version (higher memory, but avoids K loop)."""
        P, N = X.shape
        K = V.shape[1]
        assert V.shape == (N, K)

        C = X @ X.T
        c = X @ V
        c0 = np.diag(C) - np.sum(c**2, axis=1)
        outer_c0 = np.outer(c0, c0)
        resid = C - c @ c.T
        Fmn = resid**2 / outer_c0
        cost = float(np.sum(Fmn)) / 2.0

        # Shape convention for Gmni: (m=P, n=P, i=N, k=K)
        X1 = X[:, np.newaxis, :, np.newaxis]     # (P, 1, N, 1)  — m-th row
        X2 = X[np.newaxis, :, :, np.newaxis]     # (1, P, N, 1)  — n-th row
        C1 = c[:, np.newaxis, np.newaxis, :]     # (P, 1, 1, K)  — c[m, k]
        C2 = c[np.newaxis, :, np.newaxis, :]     # (1, P, 1, K)  — c[n, k]
        ratio = (resid / outer_c0)[:, :, np.newaxis, np.newaxis]    # (P, P, 1, 1)
        ratio2 = (resid**2 / outer_c0**2)[:, :, np.newaxis, np.newaxis]  # (P, P, 1, 1)
        c0_col = c0[:, np.newaxis, np.newaxis, np.newaxis]  # (P, 1, 1, 1)
        c0_row = c0[np.newaxis, :, np.newaxis, np.newaxis]  # (1, P, 1, 1)

        # Each term broadcasts to (P, P, N, K); sum over (m, n) → (N, K)
        Gmni = (-ratio * (C1 * X1)           # (P, P, N, K)
                - ratio * (C2 * X2)           # (P, P, N, K)
                + ratio2 * (c0_col * C2 * X1) # (P, P, N, K)
                + ratio2 * (c0_row * C1 * X2)) # (P, P, N, K)
        gradient = Gmni.sum(axis=(0, 1)).reshape(N, K)
        return cost, gradient


class ConstrainedLeastSquares:
    """
    Solve  min 0.5 ||Cv-t||^2  subject to  Sv <= b  via a cutting-plane loop.

    Directly mirrors ``calc_constrainted_least_square_cutting_plane.m``, which
    called IBM ILOG CPLEX ``cplexlsqlin`` in an active-set (cutting-plane) loop.
    Each iteration adds the most-violated constraint until the violation drops
    below ``tolerance`` or ``max_samples`` constraints have been added.

    MATLAB original::

        while violation > TOLERANCE && n_indices <= MAX_SAMPLES
            [v, ~, ~, flag, output, lambda] = cplexlsqlin(
                C, t, S(I(1:n_indices),:), b(I(1:n_indices)), [], [], [], [], options);
            beta = lambda.ineqlin;
            ...
            [violation, i] = max(S*v - b);
            I(n_indices) = i;
        end
    """

    def __init__(self, tolerance: float = 1e-3, max_samples: int = 1000) -> None:
        self.tolerance = tolerance
        self.max_samples = max_samples

    def _make_options(self) -> CplexOptions:
        opts = cplexoptimset(Display="off")
        if _is_cluster_node():
            opts.threads = 1
        return opts

    def solve(
        self,
        C: np.ndarray,
        t: np.ndarray,
        S: np.ndarray,
        b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        C : (N, N)
        t : (N,)
        S : (M, N)
        b : (M,)

        Returns
        -------
        v    : (N,) solution vector
        beta : (M,) Lagrange multipliers (non-zero for active constraints)
        """
        N = C.shape[0]
        M = S.shape[0]
        assert C.shape == (N, N), f"C must be ({N},{N}), got {C.shape}"
        assert t.shape == (N,),   f"t must be ({N},)"
        assert b.shape == (M,),   f"b must be ({M},)"
        assert S.shape == (M, N), f"S must be ({M},{N})"

        INITIAL_SAMPLES = min(5, M)
        rng = np.random.default_rng()
        # Initialise active-set index array (mirrors MATLAB I)
        I = np.zeros(self.max_samples, dtype=int)
        n_indices = INITIAL_SAMPLES
        I[:n_indices] = rng.choice(M, size=INITIAL_SAMPLES, replace=False)

        options = self._make_options()
        v = None
        beta_active = np.array([])
        violation = np.inf

        while violation > self.tolerance and n_indices <= self.max_samples:
            active = I[:n_indices]
            Sa = S[active]
            ba = b[active]

            v_new, _, _, flag, output, lam = cplexlsqlin(
                C, t, Sa, ba, None, None, None, None, options
            )

            beta_active = lam.ineqlin

            if flag != 1:
                print(output)
                print(
                    f"Warning: Solution not found "
                    f"(flag={flag}, {output.get('message','')})"
                )
                break

            v = v_new
            violations = S @ v - b
            violation = float(np.max(violations))
            worst = int(np.argmax(violations))

            if n_indices < self.max_samples:
                I[n_indices] = worst
            n_indices += 1

        if n_indices > self.max_samples:
            print(
                f"Warning: violation of {violation:.3f} after "
                f"{self.max_samples} iterations"
            )

        if v is None:
            v = np.zeros(N)

        # Build full beta vector (zeros for inactive constraints)
        beta = np.zeros(M)
        if len(beta_active) > 0:
            active_final = I[:min(n_indices - 1, self.max_samples)]
            for k, idx in enumerate(active_final):
                if k < len(beta_active):
                    beta[idx] = beta_active[k]

        return v, beta

    def solve_batch(
        self,
        T: np.ndarray,
        F: np.ndarray,
        kappa: float,
    ) -> np.ndarray:
        """
        Solve the manifold projection problem for a batch of random directions.

        For each row t in T, find:
          v* = argmin ||v - t||^2  s.t.  F' v >= kappa  (all columns of F)

        Formulated as ``cplexlsqlin(I, t_i, -F', -kappa * ones(M))`` per row.

        Parameters
        ----------
        T : (n_t, D)  random directions (interior points)
        F : (D, M)    manifold samples
        kappa : float

        Returns
        -------
        V : (n_t, D)  projected directions
        """
        n_t, D = T.shape
        M = F.shape[1]
        assert F.shape[0] == D

        # Aineq x <= bineq  <=>  -F'x <= -kappa
        Aineq = -F.T                                     # (M, D)
        bineq = -kappa * np.ones(M)
        C_id = np.eye(D)                                 # min ||Iv - t||^2

        options = self._make_options()
        V = np.empty_like(T)

        for i in range(n_t):
            t_i = T[i]
            v, _, _, flag, _, _ = cplexlsqlin(
                C_id, t_i, Aineq, bineq, None, None, None, None, options
            )
            V[i] = v if (flag >= 0 and np.all(np.isfinite(v))) else t_i

        return V


class OptimalLowRankStructure:
    """
    Find the minimal rank K that minimises inter-manifold correlations.

    For each K from 0 to MAX_K, a Stiefel manifold optimisation removes
    the top-K common directions from the manifold centers.  The optimal K
    is the one that minimises the mean squared (or absolute) off-diagonal
    correlation coefficient of the residual centers.

    Corresponds to ``optimal_low_rank_structure2.m``.
    """

    def __init__(
        self,
        verbose: int = 1,
        minimize_square: bool = True,
        n_repeats: int = 1,
        max_iterations: int = 10000,
        early_termination: bool = True,
    ) -> None:
        self.verbose = verbose
        self.minimize_square = minimize_square
        self.n_repeats = n_repeats
        self.max_iterations = max_iterations
        self.early_termination = early_termination

    def compute(
        self, X: np.ndarray, max_k: Optional[int] = None
    ) -> Tuple[
        Optional[np.ndarray],
        np.ndarray,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Parameters
        ----------
        X : np.ndarray  (N, P)
            Manifold centers; N is ambient dimension, P is number of manifolds.
        max_k : int, optional
            Maximum rank to test.  Defaults to ceil(P/2).

        Returns
        -------
        Vopt, Xopt, Kopt,
        residual_centers_norm  (MAX_K+1, P),
        mean_square_corrcoef   (MAX_K+1,),
        mean_abs_corrcoef      (MAX_K+1,),
        mean_square_corr       (MAX_K+1,),
        mean_abs_corr          (MAX_K+1,)
        """
        N, P = X.shape
        if max_k is None:
            max_k = int(np.ceil(P / 2))

        # Result containers
        mean_sq_corr = np.full(max_k + 1, np.nan)
        mean_abs_corr = np.full(max_k + 1, np.nan)
        mean_sq_corrcoef = np.full(max_k + 1, np.nan)
        mean_abs_corrcoef = np.full(max_k + 1, np.nan)
        residual_norms = np.full((max_k + 1, P), np.nan)

        # Reduce ambient dimension if N > P-1
        if N > P - 1:
            Q, _ = np.linalg.qr(X, mode="reduced")
            Q = Q[:, :P - 1]
            Xq = Q.T @ X
        else:
            Xq = X.copy()
            Q = np.eye(N)

        opts = StiefelOptions(
            record=0, mxitr=self.max_iterations,
            gtol=1e-6, xtol=1e-6, ftol=1e-8,
        )
        cost_fn = SquareCorrCoeffCost.vectorised
        rng = np.random.default_rng()

        best_cost = np.inf
        Vopt: Optional[np.ndarray] = None
        Xopt: np.ndarray = Q @ Xq
        Kopt = 0
        V = None   # accumulated basis vectors

        for ik in range(max_k + 1):
            k = ik
            if self.early_termination and k > Kopt + 3:
                if self.verbose:
                    print(f"Early termination Kopt={Kopt}")
                break

            import time
            t0 = time.time()

            if k == 0:
                V_k = None
                Xk = Xq.copy()
            else:
                best_stability = 0.0
                best_V_local = None
                for _ in range(self.n_repeats):
                    s = rng.standard_normal((Xq.shape[1], 1))
                    init_cols = [Xq @ s]
                    if V is not None:
                        init_cols.append(V)
                    V0 = np.hstack(init_cols)
                    V0, _ = np.linalg.qr(V0, mode="reduced")
                    V0 = V0[:, :k]
                    assert V0.shape == (Xq.shape[0], k)

                    V1, out = opt_stiefel_gbb(V0, cost_fn, opts, Xq.T)
                    assert_warn(
                        out.itr < self.max_iterations,
                        "Max iterations reached at k=%d (%s)", k, out.msg,
                    )
                    Xk_tmp = Xq - V1 @ (V1.T @ Xq)
                    stability = float(
                        np.min(np.sqrt(np.sum(Xk_tmp**2, axis=0))
                               / np.sqrt(np.sum(Xq**2, axis=0)))
                    )
                    if stability > best_stability:
                        best_stability = stability
                        best_V_local = V1

                V = best_V_local
                V_k = V
                Xk = Xq - V @ (V.T @ Xq)

            # Compute correlation metrics
            Xk_norm = np.sqrt(np.sum(Xk**2, axis=0))
            residual_norms[ik] = Xk_norm

            Ck = Xk.T @ Xk
            diag_Ck = np.diag(np.diag(Ck))
            sq_off = (Ck - diag_Ck)**2
            mean_sq_corr[ik] = np.sum(sq_off) / (P - 1) / P
            abs_off = np.abs(Ck - diag_Ck)
            mean_abs_corr[ik] = np.sum(abs_off) / (P - 1) / P

            outer_norm = Xk_norm[:, None] * Xk_norm[None, :]
            Ck0 = Ck / (outer_norm + 1e-30)
            diag_Ck0 = np.diag(np.diag(Ck0))
            sq_off0 = (Ck0 - diag_Ck0)**2
            mean_sq_corrcoef[ik] = np.sum(sq_off0) / (P - 1) / P
            abs_off0 = np.abs(Ck0 - diag_Ck0)
            mean_abs_corrcoef[ik] = np.sum(abs_off0) / (P - 1) / P

            if self.verbose >= 1:
                print(
                    f"k={k} <square>={mean_sq_corrcoef[ik]:.4f} "
                    f"<abs>={mean_abs_corrcoef[ik]:.3f} "
                    f"(took {time.time()-t0:.1f}s)"
                )

            current_cost = mean_sq_corrcoef[ik] if self.minimize_square else mean_abs_corrcoef[ik]
            if current_cost < best_cost:
                best_cost = current_cost
                Vopt = None if V_k is None else Q @ V_k
                Xopt = Q @ Xk
                Kopt = k

        return Vopt, Xopt, Kopt, residual_norms, mean_sq_corrcoef, mean_abs_corrcoef, mean_sq_corr, mean_abs_corr

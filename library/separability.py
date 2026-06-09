"""
Linear separability tests via SVM (max-margin classification).

Uses IBM ILOG CPLEX QP solver (cplexqp) to exactly match the original

Classes
-------
LinearSeparabilitySVM
    Test whether a labelled dataset is linearly separable.
LinearSeparabilityGeneralizationSVM
    Same test but with generalisation evaluation on held-out samples.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .utils import assert_warn
from .cplex_interface import CplexOptions, cplexoptimset, cplexqp


def _is_cluster_node() -> bool:
    """Return True when running on ELSC cluster nodes (use single thread)."""
    try:
        name = socket.gethostname()
        return not name or "brain" in name or "ielsc" in name
    except Exception:
        return False


@dataclass
class SVMResult:
    """Output of a linear separability test."""

    separable: bool = False
    w: Optional[np.ndarray] = None
    margin: float = np.nan
    flag: int = -1
    n_support_vectors: int = 0
    sv_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    lagrange_multipliers: Optional[np.ndarray] = None


class LinearSeparabilitySVM:
    """
    Test linear separability using the SVM primal or dual QP.

    The primal minimises  0.5 ||w||^2  s.t.  y_i(w'x_i + b) >= 1.
    Uses IBM ILOG CPLEX via the cplexqp Python API.
    """

    def __init__(
        self,
        tolerance: float = 1e-10,
        solve_dual: bool = False,
        max_iterations: int = 0,
    ) -> None:
        self.tolerance = tolerance
        self.solve_dual = solve_dual
        self.max_iterations = max_iterations

    def _make_options(self) -> CplexOptions:
        opts = cplexoptimset(
            Display="off",
            feasibility_tolerance=self.tolerance,
            optimality_tolerance=self.tolerance,
        )
        if _is_cluster_node():
            opts.threads = 1
        if self.max_iterations > 0:
            opts.max_iterations = self.max_iterations
        return opts

    def check(self, X: np.ndarray, y: np.ndarray) -> SVMResult:
        """
        Parameters
        ----------
        X : np.ndarray  (N, M) – feature matrix (N features, M samples)
        y : np.ndarray  (M,) or (1, M) – binary labels ±1

        Returns
        -------
        SVMResult
        """
        y = np.asarray(y, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == len(y), "X must be (N, M)"
        assert np.all(np.abs(y) == 1), "y must be ±1"

        # Remove samples with non-finite values
        finite_mask = np.all(np.isfinite(X), axis=0)
        X, y = X[:, finite_mask], y[finite_mask]
        N, M = X.shape

        result = SVMResult()

        if self.solve_dual:
            result = self._solve_dual(X, y, N, M)
        else:
            result = self._solve_primal(X, y, N, M)

        if result.w is None or not np.all(np.isfinite(result.w)):
            result.flag = -1
            return result

        Xb = np.vstack([X, np.ones((1, M))])
        pred = result.w @ Xb
        result.separable = bool(np.all(np.sign(pred) == y))
        wnorm = float(np.linalg.norm(result.w[:N]))
        if wnorm == 0:
            result.margin = np.nan
        else:
            result.margin = float(np.min(pred * y)) / wnorm
        return result

    def _solve_primal(self, X: np.ndarray, y: np.ndarray, N: int, M: int) -> SVMResult:
        """
        Primal SVM: min 0.5 x'Hx + f'x  s.t.  Aineq x <= bineq
        """
        Xb = np.vstack([X, np.ones((1, M))])            # (N+1, M)
        Xy = Xb * y[np.newaxis, :]                       # (N+1, M)

        H = np.eye(N + 1)
        H[N, N] = 0.0
        f = np.zeros(N + 1)
        Aineq = -Xy.T                                    # (M, N+1)
        bineq = -np.ones(M)

        options = self._make_options()

        result = SVMResult()
        try:
            w, L, flag, output = cplexqp(
                H, f, Aineq, bineq, None, None, None, None, None, options
            )
        except Exception as err:
            msg = str(err)
            if "1256" in msg or "singular" in msg.lower():
                print("Warning: Basis singular")
            else:
                import warnings
                warnings.warn(f"cplex failed: {msg}")
            result.flag = -100
            return result

        result.flag = flag

        if w is not None and np.all(np.isfinite(w)):
            assert_warn(
                flag <= 0 or L >= 0,
                "Got a negative result L=%1.1f (flag=%d)", L, flag,
            )
            Xw = Xy.T @ w
            assert_warn(
                flag < 0 or flag == 5 or np.all(Xw - 1 >= -1e-4),
                "Violation of kkt conditions: %1.1e (flag=%d, |w|=%1.1e)",
                float(np.min(Xw - 1)), flag, float(np.linalg.norm(w)),
            )
            result.sv_indices = np.where(Xw - 1 < 1e-3)[0]
            result.n_support_vectors = len(result.sv_indices)
            result.w = w

        return result

    def _solve_dual(self, X: np.ndarray, y: np.ndarray, N: int, M: int) -> SVMResult:
        """
        Dual SVM: max sum(a) - 0.5 a' (Xy Xy') a  s.t.  y'a = 0,  a >= 0.
        """
        Xy = X * y[np.newaxis, :]                        # (N, M)
        H = Xy.T @ Xy                                    # (M, M)
        f = -np.ones(M)
        lb = np.zeros(M)
        Aeq = y.reshape(1, M)
        beq = np.zeros(1)

        options = self._make_options()
        # Non-convex H (indefinite kernel) → first-order solution only
        options.optimality_target = 2

        result = SVMResult()

        w, L, flag, output = cplexqp(
            H, f, None, None, Aeq, beq, lb, None, None, options
        )
        L = -L  

        if flag == -999:   # 'Unknown status' equivalent
            flag = 0

        result.flag = flag

        if flag >= 0:
            a = w
            assert_warn(L >= 0, "L=%1.1f flag=%d", float(L), flag)
            assert np.all(a >= 0), "Negative kkt coefficients found"
            assert abs(y @ a) < 1e-1, (
                f"Bias condition does not hold: {abs(y@a):.1e} (flag={flag})"
            )
            result.lagrange_multipliers = a
            result.sv_indices = np.where(a > np.max(a) * 1e-3)[0]
            result.n_support_vectors = len(result.sv_indices)

            w_primal = Xy @ a
            Xw = X.T @ w_primal
            b = float(np.mean(Xw[result.sv_indices] - y[result.sv_indices]))
            result.w = np.r_[w_primal, -b]

        return result


class LinearSeparabilityGeneralizationSVM:
    """
    Test linear separability with a generalisation check on a random subset.
    """

    def __init__(
        self,
        tolerance: float = 1e-10,
        solve_dual: bool = False,
        max_iterations: int = 0,
        max_samples: int = 100,
    ) -> None:
        self.tolerance = tolerance
        self.solve_dual = solve_dual
        self.max_iterations = max_iterations
        self.max_samples = max_samples
        self._base = LinearSeparabilitySVM(tolerance, solve_dual, max_iterations)

    def check(
        self,
        Xs: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray], float, int, int, np.ndarray]:
        """
        Parameters
        ----------
        Xs : np.ndarray  (N_NEURONS, N_SAMPLES, N_OBJECTS)
        y  : np.ndarray  (N_OBJECTS,)  binary labels ±1

        Returns
        -------
        separable, w, margin, samples_used, n_sv, sv_indices
        """
        y = np.asarray(y, dtype=float).ravel()
        n_neurons, n_samples, n_objects = Xs.shape
        assert len(y) == n_objects

        rng = np.random.default_rng()
        sample_idx = rng.choice(n_samples, size=min(self.max_samples, n_samples), replace=False)
        samples_used = len(sample_idx)

        Xs_sub = Xs[:, sample_idx, :]           # (N, samples_used, P)
        Y_rep = np.tile(y, samples_used)         # (samples_used * P,)
        X_flat = Xs_sub.reshape(n_neurons, -1)   # (N, samples_used*P)

        result = self._base.check(X_flat, Y_rep)
        sv_indices = result.sv_indices
        return (
            result.separable,
            result.w,
            result.margin,
            samples_used,
            result.n_support_vectors,
            sv_indices,
        )

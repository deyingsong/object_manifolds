"""
Linear separability tests via SVM (max-margin classification).

Replaces the CPLEX-based MATLAB functions with scipy.optimize QP solvers.

Classes
-------
LinearSeparabilitySVM
    Test whether a labelled dataset is linearly separable.
LinearSeparabilityGeneralizationSVM
    Same test but with generalisation evaluation on held-out samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .utils import assert_warn


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

    Corresponds to ``check_linear_seperability_svm_cplexqp.m``.

    The primal minimises  0.5 ||w||^2  s.t.  y_i(w'x_i + b) >= 1.
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
        """Primal SVM: min 0.5||w||^2  s.t.  y_i(w'x_i+b)>=1."""
        Xb = np.vstack([X, np.ones((1, M))])
        Xy = Xb * y[np.newaxis, :]        # (N+1, M)

        # Use scipy SLSQP: min 0.5||w[:-1]||^2  s.t. Xy' w >= 1
        w0 = np.zeros(N + 1)
        obj = lambda w: (0.5 * np.dot(w[:N], w[:N]), np.r_[w[:N], 0.0])
        constraints = {
            "type": "ineq",
            "fun": lambda w: Xy.T @ w - 1.0,
            "jac": lambda w: Xy.T,
        }
        opts = {"ftol": self.tolerance**2, "disp": False}
        if self.max_iterations > 0:
            opts["maxiter"] = self.max_iterations
        res = minimize(obj, w0, jac=True, constraints=constraints,
                       method="SLSQP", options=opts)

        result = SVMResult(flag=0 if res.success else -1)
        if res.success or np.all(np.isfinite(res.x)):
            w = res.x
            Xw = Xy.T @ w
            result.sv_indices = np.where(Xw - 1 < 1e-3)[0]
            result.n_support_vectors = len(result.sv_indices)
            result.w = w
        return result

    def _solve_dual(self, X: np.ndarray, y: np.ndarray, N: int, M: int) -> SVMResult:
        """Dual SVM: max sum(a) - 0.5 a'Qa  s.t. a>=0, ya=0."""
        from scipy.optimize import minimize

        Xy = X * y[np.newaxis, :]
        Q = Xy.T @ Xy                          # (M, M)

        # min 0.5 a'Qa - sum(a)  s.t. ya=0, a>=0
        obj = lambda a: (0.5 * a @ Q @ a - a.sum(), Q @ a - 1.0)
        constraints = {"type": "eq", "fun": lambda a: y @ a, "jac": lambda a: y}
        bounds = [(0.0, None)] * M
        a0 = np.zeros(M)
        res = minimize(obj, a0, jac=True, method="SLSQP",
                       constraints=constraints, bounds=bounds,
                       options={"ftol": self.tolerance**2, "disp": False})

        result = SVMResult(flag=0 if res.success else -1)
        if res.success:
            a = res.x
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

    Corresponds to ``check_linear_seperability_generalization_svm_cplexqp.m``.
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

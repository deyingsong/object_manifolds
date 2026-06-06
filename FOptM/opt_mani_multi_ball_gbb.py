"""
Line-search algorithm for optimization on the unit-sphere manifold.

Solves:  min f(X),  s.t. ||X_i||_2 = 1,  X in R^{n x p}

Each column of X is constrained to lie on a unit sphere.

Reference:
  Z. Wen and W. Yin
  A feasible method for optimization with orthogonality constraints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np


@dataclass
class MultiBallOptions:
    """Configuration for the multi-ball GBB optimizer."""

    xtol: float = 1e-6
    ftol: float = 1e-12
    gtol: float = 1e-6
    rho: float = 1e-4
    eta: float = 0.2
    gamma: float = 0.85
    tau: float = 1e-3
    stpeps: float = 1e-10
    M: int = 10
    nt: int = 5
    mxitr: int = 1000
    record: int = 0

    def validate(self) -> None:
        for name, lo, hi, default in [
            ("xtol", 0, 1, 1e-6),
            ("ftol", 0, 1, 1e-12),
            ("gtol", 0, 1, 1e-6),
            ("rho", 0, 1, 1e-4),
            ("eta", 0, 1, 0.2),
            ("gamma", 0, 1, 0.85),
            ("tau", 0, 1e3, 1e-3),
            ("nt", 0, 100, 5),
            ("mxitr", 0, 2**20, 1000),
        ]:
            v = getattr(self, name)
            if not (lo <= v <= hi):
                setattr(self, name, default)


@dataclass
class MultiBallResult:
    """Output of the multi-ball GBB optimizer."""

    x: np.ndarray
    g: np.ndarray
    fval: float
    nrmG: float
    itr: int
    nfe: int
    feasi: float
    msg: str = "exceed max iteration"


class OptManiMultiBallGBB:
    """
    Gradient-based optimizer on the multi-sphere manifold.

    Each column of X is restricted to the unit sphere.
    """

    def __init__(self, opts: Optional[MultiBallOptions] = None) -> None:
        self.opts = opts or MultiBallOptions()
        self.opts.validate()

    def optimize(
        self,
        x0: np.ndarray,
        fun: Callable[..., Tuple[float, np.ndarray]],
        *args: Any,
    ) -> MultiBallResult:
        """
        Run the optimizer.

        Parameters
        ----------
        x0 : np.ndarray
            Initial point of shape (n, p); each column will be normalized.
        fun : callable
            Objective returning (f, g).  Signature: fun(x, *args).
        *args :
            Extra arguments forwarded to ``fun``.

        Returns
        -------
        MultiBallResult
        """
        opts = self.opts
        xtol, ftol, gtol = opts.xtol, opts.ftol, opts.gtol
        rho, eta, gamma = opts.rho, opts.eta, opts.gamma

        x = x0.copy().astype(float)
        n, p = x.shape

        # Normalize each column
        nrmx = np.sum(x**2, axis=0)
        if np.linalg.norm(nrmx - 1) > 1e-8:
            x /= np.sqrt(nrmx)

        f, g = fun(x, *args)
        nfe = 1

        xtg = np.sum(x * g, axis=0)
        gg = np.sum(g * g, axis=0)
        xx = np.sum(x * x, axis=0)
        xxgg = xx * gg
        dtX = xtg * x - g
        nrmG = np.linalg.norm(dtX, "fro")

        Q = 1.0
        Cval = f
        tau = opts.tau
        crit = np.ones((opts.nt, 3))

        msg = "exceed max iteration"

        if opts.record >= 1:
            print("------- Gradient Method with Line search -------")
            print(f"{'Iter':>4} {'tau':>10} {'f(X)':>12} {'nrmG':>10}")

        for itr in range(1, opts.mxitr + 1):
            xp, fp, gp, dtXP = x.copy(), f, g.copy(), dtX.copy()

            nls = 1
            deriv = rho * nrmG**2

            while True:
                tau2 = tau / 2.0
                beta = 1.0 + (tau2**2) * (-xtg**2 + xxgg)
                a1 = ((1 + tau2 * xtg)**2 - (tau2**2) * xxgg) / beta
                a2 = -tau * xx / beta
                x = a1 * xp + a2 * gp

                f, g = fun(x, *args)
                nfe += 1

                if f <= Cval - tau * deriv or nls >= 5:
                    break
                tau *= eta
                nls += 1

            xtg = np.sum(x * g, axis=0)
            gg = np.sum(g * g, axis=0)
            xx = np.sum(x * x, axis=0)
            xxgg = xx * gg
            dtX = xtg * x - g
            nrmG = np.linalg.norm(dtX, "fro")

            s = x - xp
            XDiff = np.linalg.norm(s, "fro") / np.sqrt(n)
            FDiff = abs(fp - f) / (abs(fp) + 1)

            if opts.record >= 1:
                print(f"{itr:4d}  {tau:.2e}  {f:.6e}  {nrmG:.2e}  {XDiff:.2e}  {FDiff:.2e}")

            idx = (itr - 1) % opts.nt
            crit[idx] = [nrmG, XDiff, FDiff]
            window = min(opts.nt, itr)
            mcrit = crit[:window].mean(axis=0)

            converged = (
                (XDiff < xtol and FDiff < ftol)
                or nrmG < gtol
                or (mcrit[1] < 10 * xtol and mcrit[2] < 10 * ftol)
            )
            if converged:
                msg = "converge"
                break

            y = dtX - dtXP
            sy = abs(float(np.sum(s * y)))
            tau = opts.tau
            if sy > 0:
                tau = (float(np.sum(s * s)) / sy if itr % 2 == 0
                       else sy / float(np.sum(y * y)))
                tau = float(np.clip(tau, 1e-20, 1e20))

            Qp = Q
            Q = gamma * Qp + 1
            Cval = (gamma * Qp * Cval + f) / Q

        feasi = float(np.linalg.norm(np.sum(x**2, axis=0) - 1))
        if feasi > 1e-14:
            nrmx = np.sum(x**2, axis=0)
            x /= np.sqrt(nrmx)
            f, g = fun(x, *args)
            nfe += 1
            feasi = float(np.linalg.norm(np.sum(x**2, axis=0) - 1))

        return MultiBallResult(x=x, g=g, fval=f, nrmG=float(nrmG), itr=itr, nfe=nfe, feasi=feasi, msg=msg)


def opt_mani_multi_ball_gbb(
    x0: np.ndarray,
    fun: Callable,
    opts: Optional[MultiBallOptions] = None,
    *args: Any,
) -> Tuple[np.ndarray, np.ndarray, MultiBallResult]:
    """
    Functional interface to OptManiMultiBallGBB.

    Returns
    -------
    x : np.ndarray
        Optimal unit-sphere columns.
    g : np.ndarray
        Gradient at the solution.
    out : MultiBallResult
        Solver output information.
    """
    solver = OptManiMultiBallGBB(opts)
    result = solver.optimize(x0, fun, *args)
    return result.x, result.g, result

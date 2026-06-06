"""
Curvilinear search algorithm for optimization on the Stiefel manifold.

Solves:  min F(X),  s.t. X'X = I_k,  X in R^{n x k}

Reference:
  Z. Wen and W. Yin
  A feasible method for optimization with orthogonality constraints
  Mathematical Programming, 2013.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Any

import numpy as np
from scipy.linalg import solve

from .gram_schmidt import gram_schmidt


@dataclass
class StiefelOptions:
    """Configuration for the Stiefel GBB optimizer."""

    xtol: float = 1e-6
    gtol: float = 1e-6
    ftol: float = 1e-12
    rho: float = 1e-4
    eta: float = 0.2
    gamma: float = 0.85
    tau: float = 1e-3
    stpeps: float = 1e-10
    nt: int = 5
    mxitr: int = 1000
    record: int = 0
    proj_g: int = 1
    iscomplex: bool = False

    def _clamp(self, name: str, lo: float, hi: float, default: float) -> None:
        v = getattr(self, name)
        if not (lo <= v <= hi):
            setattr(self, name, default)

    def validate(self) -> None:
        self._clamp("xtol", 0, 1, 1e-6)
        self._clamp("gtol", 0, 1, 1e-6)
        self._clamp("ftol", 0, 1, 1e-12)
        self._clamp("rho", 0, 1, 1e-4)
        self._clamp("eta", 0, 1, 0.2)
        self._clamp("gamma", 0, 1, 0.85)
        self._clamp("tau", 0, 1e3, 1e-3)
        self._clamp("nt", 0, 100, 5)
        self._clamp("mxitr", 0, 2**20, 1000)
        if self.proj_g not in (1, 2):
            self.proj_g = 1


@dataclass
class StiefelResult:
    """Output of the Stiefel GBB optimizer."""

    X: np.ndarray
    fval: float
    nrmG: float
    itr: int
    nfe: int
    feasi: float
    msg: str = "exceed max iteration"


class OptStiefelGBB:
    """
    Gradient-based optimizer on the Stiefel manifold.

    Minimizes F(X) subject to X'X = I_k using a curvilinear search.
    """

    def __init__(self, opts: Optional[StiefelOptions] = None) -> None:
        self.opts = opts or StiefelOptions()
        self.opts.validate()

    def optimize(
        self,
        X0: np.ndarray,
        fun: Callable[..., Tuple[float, np.ndarray]],
        *args: Any,
    ) -> StiefelResult:
        """
        Run the optimizer.

        Parameters
        ----------
        X0 : np.ndarray
            Initial point of shape (n, k) with X0'X0 ≈ I_k.
        fun : callable
            Objective returning (F, G) where F is scalar and G is the gradient
            of shape (n, k).  Signature: fun(X, *args) -> (float, np.ndarray).
        *args :
            Extra arguments forwarded to ``fun``.

        Returns
        -------
        StiefelResult
        """
        opts = self.opts
        xtol, gtol, ftol = opts.xtol, opts.gtol, opts.ftol
        rho, eta, gamma = opts.rho, opts.eta, opts.gamma

        X = X0.copy().astype(complex if opts.iscomplex else float)
        n, k = X.shape
        invH = k >= n / 2
        eye2k = np.eye(2 * k) if not invH else None

        # Initial evaluation
        F, G = fun(X, *args)
        nfe = 1
        GX = G.T @ X

        def _update_UV(G, X):
            if opts.proj_g == 2:
                GB = G - 0.5 * X @ (X.T @ G)
                U = np.hstack([GB, X])
                V = np.hstack([X, -GB])
            else:
                U = np.hstack([G, X])
                V = np.hstack([X, -G])
            VU = V.T @ U
            VX = V.T @ X
            return U, V, VU, VX

        if invH:
            GXT = G @ X.T
            H = 0.5 * (GXT - GXT.T)
            RX = H @ X
        else:
            U, V, VU, VX = _update_UV(G, X)

        dtX = G - X @ GX
        nrmG = np.linalg.norm(dtX, "fro")

        Q = 1.0
        Cval = F
        tau = opts.tau
        crit = np.ones((opts.nt, 3))

        msg = "exceed max iteration"

        if opts.record >= 1:
            print(f"{'Iter':>4} {'tau':>8} {'F(X)':>12} {'nrmG':>10} {'XDiff':>10}")

        for itr in range(1, opts.mxitr + 1):
            XP, FP, dtXP = X.copy(), F, dtX.copy()

            nls = 1
            deriv = rho * nrmG**2

            while True:
                if invH:
                    A = np.eye(n) + tau * H
                    X = np.linalg.solve(A, XP - tau * RX)
                else:
                    aa = np.linalg.solve(eye2k + 0.5 * tau * VU, VX)
                    X = XP - U @ (tau * aa)

                F, G = fun(X, *args)
                nfe += 1

                if F <= Cval - tau * deriv or nls >= 5:
                    break
                tau *= eta
                nls += 1

            GX = G.T @ X
            if invH:
                GXT = G @ X.T
                H = 0.5 * (GXT - GXT.T)
                RX = H @ X
            else:
                U, V, VU, VX = _update_UV(G, X)

            dtX = G - X @ GX
            nrmG = np.linalg.norm(dtX, "fro")

            S = X - XP
            XDiff = np.linalg.norm(S, "fro") / np.sqrt(n)
            tau = opts.tau
            FDiff = abs(FP - F) / (abs(FP) + 1)

            if opts.iscomplex:
                Y = dtX - dtXP
                SY = abs(np.sum(np.conj(S) * Y))
                tau = (np.sum(np.conj(S) * S).real / SY if itr % 2 == 0
                       else SY / np.sum(np.conj(Y) * Y).real)
            else:
                Y = dtX - dtXP
                SY = abs(np.sum(S * Y))
                if SY > 0:
                    tau = (np.sum(S * S) / SY if itr % 2 == 0
                           else SY / np.sum(Y * Y))

            tau = float(np.clip(tau, 1e-20, 1e20))

            if opts.record >= 1:
                print(f"{itr:4d}  {tau:.2e}  {F:.4e}  {nrmG:.2e}  {XDiff:.2e}")

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
                if itr <= 2:
                    ftol *= 0.1
                    xtol *= 0.1
                    gtol *= 0.1
                else:
                    msg = "converge"
                    break

            Qp = Q
            Q = gamma * Qp + 1
            Cval = (gamma * Qp * Cval + F) / Q

        feasi = np.linalg.norm(X.T @ X - np.eye(k), "fro")
        if feasi > 1e-13:
            X = gram_schmidt(X.real)
            F, G = fun(X, *args)
            nfe += 1
            feasi = np.linalg.norm(X.T @ X - np.eye(k), "fro")

        return StiefelResult(X=X, fval=F, nrmG=nrmG, itr=itr, nfe=nfe, feasi=feasi, msg=msg)


def opt_stiefel_gbb(
    X0: np.ndarray,
    fun: Callable,
    opts: Optional[StiefelOptions] = None,
    *args: Any,
) -> Tuple[np.ndarray, StiefelResult]:
    """
    Functional interface to OptStiefelGBB.

    Parameters
    ----------
    X0 : np.ndarray
        Initial Stiefel point (n, k).
    fun : callable
        Objective function returning (F, G).
    opts : StiefelOptions, optional
        Solver options.
    *args :
        Extra arguments forwarded to ``fun``.

    Returns
    -------
    X : np.ndarray
        Optimal Stiefel point.
    out : StiefelResult
        Optimizer output information.
    """
    solver = OptStiefelGBB(opts)
    result = solver.optimize(X0, fun, *args)
    return result.X, result

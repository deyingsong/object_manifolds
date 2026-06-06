"""
Demo: Leading eigenvalue computation via OptStiefelGBB.

Solves:  min -0.5 Tr(X'AX)  s.t. X'X = I
which yields the leading k eigenvectors of A.
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from FOptM import StiefelOptions, opt_stiefel_gbb


def _eig_objective(X: np.ndarray, A: np.ndarray):
    G = -(A @ X)
    F = 0.5 * float(np.sum(G * X))
    return F, G


def run_demo(n: int = 500, k: int = 6, seed: int = 2010) -> None:
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((n, n))
    A = A.T @ A                 # symmetric positive-definite

    # Reference: numpy eigvalsh
    t0 = time.time()
    eigvals = np.linalg.eigvalsh(A)
    feig = float(np.sum(np.sort(eigvals)[::-1][:k]))
    t_np = time.time() - t0
    print(f"numpy: obj={feig:.6e}, cpu={t_np:.3f}s")

    X0, _ = np.linalg.qr(rng.standard_normal((n, k)))
    opts = StiefelOptions(mxitr=1000, xtol=1e-5, gtol=1e-5, ftol=1e-8, tau=1e-3)

    t0 = time.time()
    X, out = opt_stiefel_gbb(X0, _eig_objective, opts, A)
    t_ours = time.time() - t0

    fval = -2.0 * out.fval
    err = (feig - fval) / (abs(feig) + 1)
    feasi = np.linalg.norm(X.T @ X - np.eye(k), "fro")
    print(
        f"ours:  obj={fval:.6e}, cpu={t_ours:.3f}s, "
        f"nfe={out.nfe}, itr={out.itr}, |XT*X-I|={feasi:.2e}"
    )
    print(f"relative diff: {err:.2e}")


if __name__ == "__main__":
    run_demo()

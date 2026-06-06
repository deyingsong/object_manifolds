"""
Demo: Max-cut SDP relaxation solved via OptManiMultiBallGBB.

Solves:
  max Tr(C*X),  s.t. X_ii = 1,  X psd
via the low-rank model:
  max Tr(C*V'*V),  s.t. ||V_i|| = 1
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from FOptM import MultiBallOptions, opt_mani_multi_ball_gbb


def maxcut_quad(V: np.ndarray, C: np.ndarray):
    """Maxcut objective and gradient."""
    g = 2.0 * (V @ C)
    f = float(np.sum(g * V)) / 2.0
    return f, g


def run_demo(seed: int = 2010) -> None:
    rng = np.random.default_rng(seed)

    # Tiny random graph (torusg3-8 style would require loading a .mat file)
    n = 50
    A = rng.standard_normal((n, n))
    C = -(A + A.T) / 2          # symmetric negative weight matrix

    p = max(1, min(int(round(np.sqrt(2 * n) / 2)), 20))

    x0 = rng.standard_normal((p, n))
    x0 /= np.sqrt(np.sum(x0**2, axis=0))

    opts = MultiBallOptions(mxitr=600, gtol=1e-5, xtol=1e-5, ftol=1e-8, tau=1e-3)

    x, g, out = opt_mani_multi_ball_gbb(x0, maxcut_quad, opts, C)

    obj = -float(out.fval)
    print(
        f"n={n}, p={p}, f={obj:.4e}, itr={out.itr}, "
        f"nfe={out.nfe}, feasi={out.feasi:.2e}, ||Hx||={out.nrmG:.2e}"
    )


if __name__ == "__main__":
    run_demo()

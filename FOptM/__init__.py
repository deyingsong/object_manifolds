"""
FOptM – Feasible Optimization on Manifolds.

Implements Riemannian gradient methods for:
  1. Stiefel manifold  (X'X = I)   ->  OptStiefelGBB
  2. Multi-ball manifold (||X_i||=1) -> OptManiMultiBallGBB

Reference:
  Z. Wen and W. Yin, "A feasible method for optimization with
  orthogonality constraints," Mathematical Programming, 2013.
"""

from .gram_schmidt import gram_schmidt
from .opt_stiefel_gbb import (
    OptStiefelGBB,
    StiefelOptions,
    StiefelResult,
    opt_stiefel_gbb,
)
from .opt_mani_multi_ball_gbb import (
    OptManiMultiBallGBB,
    MultiBallOptions,
    MultiBallResult,
    opt_mani_multi_ball_gbb,
)

__all__ = [
    "gram_schmidt",
    "OptStiefelGBB",
    "StiefelOptions",
    "StiefelResult",
    "opt_stiefel_gbb",
    "OptManiMultiBallGBB",
    "MultiBallOptions",
    "MultiBallResult",
    "opt_mani_multi_ball_gbb",
]

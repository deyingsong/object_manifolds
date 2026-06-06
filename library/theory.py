"""
Theoretical capacity formula for point manifolds.

theory_alpha0(kappa) computes:
    1 / integral_{-kappa}^{inf} phi(t) * (t + kappa)^2 dt

where phi is the standard normal density.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy import integrate

from .utils import assert_warn


class TheoryAlpha0:
    """
    Vectorised computation of alpha_0(kappa) with a pre-built lookup table.

    The first call to :meth:`compute` builds the cache; subsequent calls
    interpolate from it.
    """

    KAPPA_MIN: float = -50.0
    KAPPA_MAX: float = 100.0
    KAPPA_STEP: float = 0.01

    _cached_kappas: np.ndarray | None = None
    _cached_alphas: np.ndarray | None = None

    @classmethod
    def _build_cache(cls) -> None:
        if cls._cached_kappas is not None:
            return
        print("Creating alpha0 cache …", flush=True)
        kappas = np.arange(cls.KAPPA_MIN, cls.KAPPA_MAX + cls.KAPPA_STEP / 2, cls.KAPPA_STEP)
        alphas = np.array([cls._integral(k) for k in kappas])
        cls._cached_kappas = kappas
        cls._cached_alphas = alphas
        print(f"Cache ready ({len(kappas)} points)")

    @staticmethod
    def _integral(kappa: float) -> float:
        """Direct numerical integration for a single kappa."""
        if np.isnan(kappa):
            return np.nan
        if kappa > 100:
            return float(kappa) ** -2
        phi = lambda t: np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)
        integrand = lambda t: phi(t) * (t + kappa) ** 2
        I, _ = integrate.quad(integrand, -kappa, np.inf)
        return 1.0 / I if I > 0 else np.inf

    @classmethod
    def compute(cls, kappa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Return alpha_0(kappa) for scalar or array kappa.

        For kappa > 100 the asymptotic approximation kappa^{-2} is used.
        """
        cls._build_cache()
        scalar = np.isscalar(kappa)
        kappa = np.asarray(kappa, dtype=float)

        result = np.full_like(kappa, np.nan)
        large = kappa > 100
        result[large] = kappa[large] ** (-2)

        rest = ~large & np.isfinite(kappa)
        if np.any(rest):
            result[rest] = np.interp(
                kappa[rest],
                cls._cached_kappas,
                cls._cached_alphas,
                left=np.inf,
                right=np.inf,
            )

        assert_warn(
            bool(np.all(np.isfinite(result) | np.isnan(result))),
            "Infinite values. Kappa range [%.3e, %.3e]",
            float(np.nanmin(kappa)),
            float(np.nanmax(kappa)),
        )
        assert_warn(
            bool(np.all((result > 0) | np.isnan(result))),
            "Negative values. Kappa range [%.3e, %.3e]",
            float(np.nanmin(kappa)),
            float(np.nanmax(kappa)),
        )
        return float(result) if scalar else result

    @classmethod
    def clear_cache(cls) -> None:
        """Discard the pre-built lookup table."""
        cls._cached_kappas = None
        cls._cached_alphas = None


def theory_alpha0(kappa: float) -> float:
    """Direct (uncached) computation of alpha_0(kappa)."""
    return TheoryAlpha0._integral(kappa)


def theory_alpha0_cached(kappa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Cached computation of alpha_0(kappa), MATLAB-compatible interface."""
    return TheoryAlpha0.compute(kappa)

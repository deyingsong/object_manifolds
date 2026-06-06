"""
Theoretical alpha0 computation module.

Computes theoretical alpha0 values using numerical integration and caching.
Based on theory_alpha0.m and theory_alpha0_cached.m
"""

from typing import Union, Optional
from functools import lru_cache
import logging

import numpy as np
from scipy import integrate

logger = logging.getLogger(__name__)


class TheoreticalAlpha0:
    """Computes theoretical alpha0 values with caching."""
    
    # Class-level cache
    _cached_kappas = None
    _cached_alphas = None
    _cache_lock = None
    
    # Cache parameters
    KAPPA_MIN = -50
    KAPPA_MAX = 100
    KAPPA_STEP = 0.01
    
    @classmethod
    def _initialize_cache(cls) -> None:
        """Initialize the cache if not already done."""
        if cls._cached_kappas is not None:
            return
        
        cls._cached_kappas = np.arange(
            cls.KAPPA_MIN,
            cls.KAPPA_MAX + cls.KAPPA_STEP,
            cls.KAPPA_STEP
        )
        cls._cached_alphas = np.zeros_like(cls._cached_kappas)
        
        logger.info(f"Creating alpha0 cache ({len(cls._cached_kappas)} points)...")
        
        for i, kappa in enumerate(cls._cached_kappas):
            cls._cached_alphas[i] = cls._compute_alpha0_integral(kappa)
        
        logger.info("Alpha0 cache created")
    
    @staticmethod
    def _density_function(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Standard normal probability density function."""
        return np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi)
    
    @staticmethod
    def _compute_alpha0_integral(kappa: float) -> float:
        """
        Compute alpha0 for a single kappa value using numerical integration.
        
        Parameters
        ----------
        kappa : float
            The kappa parameter
        
        Returns
        -------
        float
            Computed alpha0 value
        """
        if np.isnan(kappa):
            return np.nan
        
        # Use asymptotic approximation for large kappa
        if kappa > 100:
            return kappa ** (-2)
        
        # Define integrand
        def integrand(t):
            return TheoreticalAlpha0._density_function(t) * (t + kappa) ** 2
        
        # Integrate from -kappa to infinity
        try:
            I, _ = integrate.quad(integrand, -kappa, np.inf)
            alpha = 1.0 / I if I > 0 else np.inf
            return alpha
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Integration failed for kappa={kappa}: {e}")
            return np.nan
    
    @classmethod
    def compute(cls, kappa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute alpha0 for given kappa value(s) using caching.
        
        Parameters
        ----------
        kappa : float or np.ndarray
            The kappa parameter(s)
        
        Returns
        -------
        float or np.ndarray
            Computed alpha0 value(s)
        
        Notes
        -----
        - For kappa > 100, uses asymptotic approximation kappa^-2
        - For other values, uses cached interpolation or direct integration
        - Returns np.nan for np.nan input
        """
        # Initialize cache on first use
        cls._initialize_cache()
        
        # Handle scalar input
        if np.isscalar(kappa):
            if np.isnan(kappa):
                return np.nan
            
            if kappa > 100:
                return kappa ** (-2)
            
            # Use interpolation from cache
            alpha = np.interp(
                kappa,
                cls._cached_kappas,
                cls._cached_alphas,
                left=np.inf,
                right=np.inf
            )
            return alpha
        
        # Handle array input
        kappa = np.asarray(kappa)
        result = np.full_like(kappa, np.nan, dtype=float)
        
        # Asymptotic values for large kappa
        large_mask = kappa > 100
        result[large_mask] = kappa[large_mask] ** (-2)
        
        # Interpolate for other values
        small_mask = ~large_mask & np.isfinite(kappa)
        if np.any(small_mask):
            result[small_mask] = np.interp(
                kappa[small_mask],
                cls._cached_kappas,
                cls._cached_alphas,
                left=np.inf,
                right=np.inf
            )
        
        return result
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the computation cache."""
        cls._cached_kappas = None
        cls._cached_alphas = None
        logger.info("Alpha0 cache cleared")


def theory_alpha0(kappa: float) -> float:
    """
    Compute theoretical alpha0 value (direct computation without caching).
    
    Parameters
    ----------
    kappa : float
        The kappa parameter
    
    Returns
    -------
    float
        Computed alpha0 value
    
    Notes
    -----
    Use TheoreticalAlpha0.compute() for cached computation with better performance.
    """
    if np.isnan(kappa):
        return np.nan
    
    if kappa > 100:
        return kappa ** (-2)
    
    return TheoreticalAlpha0._compute_alpha0_integral(kappa)


def theory_alpha0_cached(kappa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute theoretical alpha0 value with caching (MATLAB-compatible interface).
    
    Parameters
    ----------
    kappa : float or np.ndarray
        The kappa parameter(s)
    
    Returns
    -------
    float or np.ndarray
        Computed alpha0 value(s)
    """
    return TheoreticalAlpha0.compute(kappa)

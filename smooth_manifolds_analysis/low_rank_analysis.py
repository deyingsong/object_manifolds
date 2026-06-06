"""
Low-rank correlation analysis for DNN manifolds.

Corresponds to ``check_convnet_covariance_low_rank_approx_optimal_K.m``.

For each layer and transformation direction, compute the optimal low-rank
subspace that minimises residual inter-manifold correlations.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from library import OptimalLowRankStructure
from smooth_manifolds_generation.network import load_network_metadata


@dataclass
class LowRankAnalysisConfig:
    """Configuration for the low-rank correlation analysis."""

    verbose: int = 1
    minimize_square: bool = True
    n_repeats: int = 1
    max_iterations: int = 10000
    early_termination: bool = True


@dataclass
class LowRankLayerResults:
    """Per-layer results of optimal-K analysis."""

    K_opt: int = 0
    V_opt: Optional[np.ndarray] = None
    X_opt: Optional[np.ndarray] = None
    residual_norms: Optional[np.ndarray] = None
    mean_square_corrcoef: Optional[np.ndarray] = None
    mean_abs_corrcoef: Optional[np.ndarray] = None
    mean_square_corr: Optional[np.ndarray] = None
    mean_abs_corr: Optional[np.ndarray] = None


class CovarianceLowRankAnalysis:
    """
    Find the optimal low-rank common subspace of manifold centers for
    each network layer and transformation direction.

    Corresponds to ``check_convnet_covariance_low_rank_approx_optimal_K.m``.
    """

    def __init__(
        self,
        config: LowRankAnalysisConfig,
        network_type: int,
        layers_grouping_level: int = 0,
        epoch: Optional[int] = None,
    ) -> None:
        self.config = config
        self.network_type = network_type
        self.metadata = load_network_metadata(
            network_type, layers_grouping_level, epoch
        )

    def run(
        self,
        tuning_functions: Dict[str, Dict[str, np.ndarray]],
        run_directions: Optional[List[str]] = None,
        run_layers: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, LowRankLayerResults]]:
        """
        Parameters
        ----------
        tuning_functions : {direction_name: {layer_name: np.ndarray(N, M, P)}}
        run_directions : list of str
        run_layers : list of str

        Returns
        -------
        dict {direction_name: {layer_name: LowRankLayerResults}}
        """
        cfg = self.config
        if run_layers is None:
            run_layers = [self.metadata.layer_names[i]
                          for i in self.metadata.enabled_layers]

        analyser = OptimalLowRankStructure(
            verbose=cfg.verbose,
            minimize_square=cfg.minimize_square,
            n_repeats=cfg.n_repeats,
            max_iterations=cfg.max_iterations,
            early_termination=cfg.early_termination,
        )

        results: Dict[str, Dict[str, LowRankLayerResults]] = {}
        direction_keys = run_directions or list(tuning_functions.keys())

        for direction in direction_keys:
            results[direction] = {}
            for layer_name in run_layers:
                tf = tuning_functions.get(direction, {}).get(layer_name)
                if tf is None:
                    continue
                assert tf.ndim == 3, "Tuning function must be (N, M, P)"

                # Centers: mean over samples
                centers = np.mean(tf, axis=1)          # (N, P)
                # Remove global mean
                centers -= np.mean(centers, axis=1, keepdims=True)
                # Keep only non-zero neurons
                nz = np.any(centers != 0, axis=1)
                if not np.any(nz):
                    continue
                centers = centers[nz]

                print(f"Analysing {direction} {layer_name} "
                      f"(N={centers.shape[0]}, P={centers.shape[1]})")

                Vopt, Xopt, Kopt, res_norms, sq_cc, abs_cc, sq_c, abs_c = (
                    analyser.compute(centers)
                )
                results[direction][layer_name] = LowRankLayerResults(
                    K_opt=Kopt,
                    V_opt=Vopt,
                    X_opt=Xopt,
                    residual_norms=res_norms,
                    mean_square_corrcoef=sq_cc,
                    mean_abs_corrcoef=abs_cc,
                    mean_square_corr=sq_c,
                    mean_abs_corr=abs_c,
                )

        return results

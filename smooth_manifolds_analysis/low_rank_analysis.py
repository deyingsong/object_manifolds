"""
Low-rank correlation and manifold geometry analysis for DNN manifolds.

Corresponds to ``check_convnet_covariance_low_rank_approx_optimal_K.m``.

For each layer and transformation direction:
  1. Compute manifold centers (mean over samples).
  2. Find the optimal low-rank common subspace via Stiefel optimisation.
  3. Compute per-manifold geometry (radius, dimension) and theoretical capacity.

Typical usage
-------------
    from smooth_manifolds_analysis.low_rank_analysis import (
        CovarianceLowRankAnalysis, LowRankAnalysisConfig,
    )

    cfg = LowRankAnalysisConfig(n_objects=128, range_factor=0.5,
                                n_samples=15, max_k=5)
    analysis = CovarianceLowRankAnalysis(cfg)

    # tuning_function: output of OneDimensionalManifoldGenerator.collect()
    #   {layer_name → (N_DIRECTIONS, N_OBJECTS, N_SAMPLES, N_FEATURES)}
    results = analysis.run(tuning_function, layer_number=20)
    print(results.theory_capacity)   # (N_DIRECTIONS,)
    print(results.mean_radius)       # (N_DIRECTIONS,)
    print(results.mean_dimension)    # (N_DIRECTIONS,)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from library import OptimalLowRankStructure
from library.manifold_properties import ManifoldPropertiesLS
from smooth_manifolds_generation.network import load_network_metadata
from smooth_manifolds_analysis.capacity_analysis import DIRECTION_NAMES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LowRankAnalysisConfig:
    """
    All parameters for low-rank covariance analysis in a single config object.

    Mirrors the script arguments of
    ``check_convnet_covariance_low_rank_approx_optimal_K.m``.

    Parameters
    ----------
    n_objects : int
        Number of object classes (P in MATLAB).
    n_samples : int
        Samples per manifold (M in MATLAB).
    range_factor : float
        Transform magnitude used during generation.
    max_k : int
        Maximum rank to test in the low-rank optimisation.
    network_type : int
        1=AlexNet, 3=ResNet50, 5=VGG16.
    layers_grouping_level : int
        0 = all layers individually.
    epoch : int, optional
        Training epoch (None = fully trained).
    verbose : int
        0=silent, 1=progress per K.
    minimize_square : bool
        Minimise mean squared correlation (True) or absolute (False).
    n_repeats : int
        Random restarts for Stiefel optimisation.
    max_iterations : int
        Stiefel solver iteration limit.
    early_termination : bool
        Stop Stiefel search when cost stops decreasing.
    n_random_projections : int
        Monte-Carlo directions for manifold geometry.
    kappa : float
        Margin for the manifold geometry computation.
    """

    n_objects: int = 128
    n_samples: int = 15
    range_factor: float = 0.5
    n_transform_dims: int = 2
    max_k: int = 5
    network_type: int = 1
    layers_grouping_level: int = 0
    epoch: Optional[int] = None
    # --- Stiefel optimiser hyper-parameters ---
    verbose: int = 1
    minimize_square: bool = True
    n_repeats: int = 1
    max_iterations: int = 10000
    early_termination: bool = True
    # --- manifold geometry hyper-parameters ---
    n_random_projections: int = 1000
    kappa: float = 0.0


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class LowRankResults:
    """
    Low-rank and geometry results for all directions of one network layer.

    Attributes
    ----------
    layer_name : str
    theory_capacity : np.ndarray  (N_DIRECTIONS,)
        Theoretical capacity alpha_c_hat per direction, averaged over manifolds.
    mean_radius : np.ndarray  (N_DIRECTIONS,)
        Mean manifold half-width R_M per direction.
    mean_dimension : np.ndarray  (N_DIRECTIONS,)
        Mean effective manifold dimension D_M per direction.
    K_opt : np.ndarray  (N_DIRECTIONS,)
        Optimal low-rank subspace rank per direction.
    mean_square_corrcoef : list of np.ndarray
        Mean squared correlation-coefficient curve vs K, per direction.
    mean_abs_corrcoef : list of np.ndarray
        Mean absolute correlation-coefficient curve vs K, per direction.
    residual_norms : list of np.ndarray  (MAX_K+1, N_OBJECTS)
        Residual center norm curve vs K, per direction.
    """

    layer_name: str = ""
    theory_capacity: np.ndarray = field(
        default_factory=lambda: np.full(len(DIRECTION_NAMES), np.nan)
    )
    mean_radius: np.ndarray = field(
        default_factory=lambda: np.full(len(DIRECTION_NAMES), np.nan)
    )
    mean_dimension: np.ndarray = field(
        default_factory=lambda: np.full(len(DIRECTION_NAMES), np.nan)
    )
    K_opt: np.ndarray = field(
        default_factory=lambda: np.full(len(DIRECTION_NAMES), np.nan)
    )
    mean_square_corrcoef: List[np.ndarray] = field(default_factory=list)
    mean_abs_corrcoef: List[np.ndarray] = field(default_factory=list)
    residual_norms: List[np.ndarray] = field(default_factory=list)

    def __repr__(self) -> str:
        n = len(self.theory_capacity)
        lines = [f"LowRankResults(layer='{self.layer_name}')"]
        for i in range(min(n, len(DIRECTION_NAMES))):
            lines.append(
                f"  {DIRECTION_NAMES[i]:18s}: "
                f"capacity={self.theory_capacity[i]:.4f}  "
                f"radius={self.mean_radius[i]:.4f}  "
                f"dim={self.mean_dimension[i]:.2f}  "
                f"K_opt={self.K_opt[i]:.0f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis class
# ---------------------------------------------------------------------------

class CovarianceLowRankAnalysis:
    """
    Optimal low-rank decomposition and manifold geometry analysis.

    Corresponds to ``check_convnet_covariance_low_rank_approx_optimal_K.m``.

    Parameters
    ----------
    config : LowRankAnalysisConfig
        Fully-specified experiment and analysis configuration.
    """

    N_DIRECTIONS = 7

    def __init__(self, config: LowRankAnalysisConfig) -> None:
        self.config = config
        self.metadata = load_network_metadata(
            config.network_type,
            config.layers_grouping_level,
            config.epoch,
        )

    def run(
        self,
        tuning_function: Dict[str, np.ndarray],
        layer_number: Optional[int] = None,
        layer_name: Optional[str] = None,
        run_directions: Optional[List[int]] = None,
    ) -> LowRankResults:
        """
        Compute optimal low-rank structure and manifold geometry for one layer.

        Parameters
        ----------
        tuning_function : dict
            Output of ``OneDimensionalManifoldGenerator.collect()``:
            ``{layer_name → np.ndarray (N_DIRECTIONS, N_OBJECTS, N_SAMPLES, N_FEATURES)}``.
        layer_number : int, optional
            1-based MATLAB-convention index (pixel=1, convnet layers 2…N+1).
        layer_name : str, optional
            Direct layer name; takes precedence over ``layer_number``.
        run_directions : list of int, optional
            0-based direction indices (default: all 7).

        Returns
        -------
        LowRankResults
        """
        # --- select layer --------------------------------------------------
        available = list(tuning_function.keys())
        if not available:
            raise ValueError("tuning_function dict is empty.")

        selected = self._select_layer(available, layer_number, layer_name)
        tf_full = tuning_function[selected]   # (N_DIR, N_OBJ, N_SMP, N_FEAT)

        if tf_full.ndim != 4:
            raise ValueError(
                f"Expected tuning_function['{selected}'] with 4 dims "
                f"(N_DIR, N_OBJ, N_SMP, N_FEAT), got {tf_full.shape}."
            )

        n_dir_avail, n_obj, n_smp, n_feat = tf_full.shape

        if run_directions is None:
            run_directions = list(range(min(self.N_DIRECTIONS, n_dir_avail)))

        n_dirs = len(run_directions)

        # --- build analysers -----------------------------------------------
        cfg = self.config
        lr_analyser = OptimalLowRankStructure(
            verbose=cfg.verbose,
            minimize_square=cfg.minimize_square,
            n_repeats=cfg.n_repeats,
            max_iterations=cfg.max_iterations,
            early_termination=cfg.early_termination,
        )
        geom_analyser = ManifoldPropertiesLS(
            n_random_projections=cfg.n_random_projections,
            kappa=cfg.kappa,
        )

        # --- result containers ---------------------------------------------
        theory_cap = np.full(n_dirs, np.nan)
        mean_radius = np.full(n_dirs, np.nan)
        mean_dim    = np.full(n_dirs, np.nan)
        K_opt_arr   = np.full(n_dirs, np.nan)
        sq_cc_list:    List[np.ndarray] = []
        abs_cc_list:   List[np.ndarray] = []
        res_norm_list: List[np.ndarray] = []

        for result_idx, d in enumerate(run_directions):
            dir_name = (DIRECTION_NAMES[d] if d < len(DIRECTION_NAMES)
                        else f"direction_{d}")
            if cfg.verbose:
                print(f"  [{result_idx+1}/{n_dirs}] "
                      f"direction={dir_name}  layer={selected}")

            # tf_dir: (N_OBJ, N_SMP, N_FEAT)
            tf_dir = tf_full[d].astype(np.float64)
            # → (N_FEAT, N_SMP, N_OBJ)
            tf_nfp = tf_dir.transpose(2, 1, 0)

            # Remove zero-variance neurons
            mean_sq = np.nanmean(tf_nfp ** 2, axis=(1, 2))
            nz = mean_sq > 0
            tf_nfp = tf_nfp[nz]                  # (N_nz, N_SMP, N_OBJ)

            if tf_nfp.shape[0] == 0:
                if cfg.verbose:
                    print("    Skipping: all-zero features.")
                sq_cc_list.append(np.array([]))
                abs_cc_list.append(np.array([]))
                res_norm_list.append(np.array([]))
                continue

            n_nz, _, n_obj_used = tf_nfp.shape

            # Centers: (N_nz, N_OBJ)
            centers = np.mean(tf_nfp, axis=1)
            centers -= np.mean(centers, axis=1, keepdims=True)  # remove global mean

            # ---- low-rank structure ----------------------------------------
            Vopt, Xopt, Kopt, res_norms, sq_cc, abs_cc, sq_c, abs_c = (
                lr_analyser.compute(centers, max_k=cfg.max_k)
            )
            K_opt_arr[result_idx] = Kopt
            sq_cc_list.append(sq_cc)
            abs_cc_list.append(abs_cc)
            res_norm_list.append(res_norms)

            # ---- manifold geometry -----------------------------------------
            # Use tf_nfp for geometry: per-object (N_nz, N_SMP)
            cap_vals:   List[float] = []
            radius_vals: List[float] = []
            dim_vals:   List[float] = []

            for p in range(n_obj_used):
                tf_p = tf_nfp[:, :, p]            # (N_nz, N_SMP)
                center_p = np.mean(tf_p, axis=1)  # (N_nz,)
                center_norm = float(np.linalg.norm(center_p))
                if center_norm < 1e-12:
                    continue
                try:
                    geom = geom_analyser.compute(tf_p, center_norm)
                    if np.isfinite(geom.alphac_hat):
                        cap_vals.append(geom.alphac_hat)
                    if np.isfinite(geom.mean_half_width2):
                        radius_vals.append(geom.mean_half_width2)
                    if np.isfinite(geom.effective_dimension):
                        dim_vals.append(geom.effective_dimension)
                except Exception as exc:
                    if cfg.verbose:
                        print(f"    object {p}: geometry failed ({exc})")

            if cap_vals:
                theory_cap[result_idx] = float(np.mean(cap_vals))
            if radius_vals:
                mean_radius[result_idx] = float(np.mean(radius_vals))
            if dim_vals:
                mean_dim[result_idx] = float(np.mean(dim_vals))

            if cfg.verbose:
                print(f"    K_opt={Kopt}  "
                      f"theory_capacity={theory_cap[result_idx]:.4f}  "
                      f"radius={mean_radius[result_idx]:.4f}  "
                      f"dim={mean_dim[result_idx]:.2f}")

        return LowRankResults(
            layer_name=selected,
            theory_capacity=theory_cap,
            mean_radius=mean_radius,
            mean_dimension=mean_dim,
            K_opt=K_opt_arr,
            mean_square_corrcoef=sq_cc_list,
            mean_abs_corrcoef=abs_cc_list,
            residual_norms=res_norm_list,
        )

    def run_all_layers(
        self,
        tuning_function: Dict[str, np.ndarray],
        run_directions: Optional[List[int]] = None,
    ) -> Dict[str, LowRankResults]:
        """
        Run analysis for every layer present in ``tuning_function``.

        Returns
        -------
        dict  {layer_name → LowRankResults}
        """
        return {
            ln: self.run(tuning_function, layer_name=ln,
                         run_directions=run_directions)
            for ln in tuning_function
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_layer(
        self,
        available: List[str],
        layer_number: Optional[int],
        layer_name: Optional[str],
    ) -> str:
        if layer_name is not None:
            if layer_name in available:
                return layer_name
            raise KeyError(
                f"layer_name='{layer_name}' not found. Available: {available}"
            )

        if layer_number is not None:
            meta_names = self.metadata.layer_names
            # MATLAB convention: pixel=1, convnet layers 2..N+1
            conv_idx = layer_number - 2
            if 0 <= conv_idx < len(meta_names):
                candidate = meta_names[conv_idx]
                if candidate in available:
                    return candidate
            idx = max(0, min(layer_number - 1, len(available) - 1))
            chosen = available[idx]
            if self.config.verbose:
                print(
                    f"  layer_number={layer_number} → using '{chosen}' "
                    f"(available: {available})"
                )
            return chosen

        return available[0]

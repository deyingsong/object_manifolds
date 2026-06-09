"""
High-level capacity analysis for DNN manifolds.

Typical usage
-------------
    from smooth_manifolds_analysis.capacity_analysis import (
        CapacityAnalysisConfig, OneDimensionalCapacityAnalysis,
    )

    cfg = CapacityAnalysisConfig(n_objects=128, range_factor=0.5, n_samples=15)
    analysis = OneDimensionalCapacityAnalysis(cfg)

    # tuning_function: output of OneDimensionalManifoldGenerator.collect()
    #   {layer_name → (N_DIRECTIONS, N_OBJECTS, N_SAMPLES, N_FEATURES)}
    results = analysis.run(tuning_function, layer_number=20)
    print(results)   # LayerCapacityResults per direction
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from library import BinaryDichotomiesCapacity, CapacityResults
from smooth_manifolds_generation.network import load_network_metadata


DIRECTION_NAMES = [
    "x-translation", "y-translation",
    "x-scale",        "y-scale",
    "x-shear",        "y-shear",
    "rotation",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CapacityAnalysisConfig:
    """
    All parameters for capacity analysis in a single config object.

    Parameters
    ----------
    n_objects : int
        Number of object classes (P in MATLAB).
    n_samples : int
        Number of samples per manifold (M in MATLAB).
    range_factor : float
        Transform magnitude factor used during generation.
    network_type : int
        NetworkType enum value (1=AlexNet, 3=ResNet50, 5=VGG16).
    layers_grouping_level : int
        0 = all individual layers, 1/2 = grouped (see load_network_metadata).
    epoch : int, optional
        Training epoch (None = fully trained model).
    expected_precision : float
        Target standard error on the separability fraction (``EXPECTED_PRECISION``).
    max_samples : int
        Maximum number of neuron-count samples to test.
    precision : int
        Rounding precision for the binary search.
    features_type : int
        0 = subsample, 1 = first-N (PCA-like), 2 = random projections.
    random_labeling_type : int
        0 = iid binary, 1 = balanced, 2 = sparse.
    global_preprocessing : int
        0 = none, 1 = z-normalise, 2 = whiten, 3 = centre decorrelation.
    local_preprocessing : int
        0 = none, 1 = orthogonalise centres, 2 = random centres, …
    verbose : bool
    """

    n_objects: int = 128
    n_samples: int = 15
    range_factor: float = 0.5
    n_transform_dims: int = 2
    network_type: int = 1
    layers_grouping_level: int = 0
    epoch: Optional[int] = None
    # --- analysis hyper-parameters (match MATLAB defaults) ---
    expected_precision: float = 0.05
    max_samples: int = 100
    precision: int = 1
    features_type: int = 2
    random_labeling_type: int = 1
    global_preprocessing: int = 0
    local_preprocessing: int = 0
    verbose: bool = True


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class LayerCapacityResults:
    """
    Capacity results for all transformation directions of one network layer.

    Attributes
    ----------
    layer_name : str
    capacity_alpha_c : np.ndarray  shape (N_DIRECTIONS,)
        alpha_c = N_OBJECTS / N_c  where N_c is the critical neuron count.
    separability : np.ndarray  shape (N_DIRECTIONS, N_samples)
        Fraction of separable dichotomies at each neuron-count sample.
    n_neuron_samples : np.ndarray  shape (N_DIRECTIONS, N_samples)
        Neuron counts at which separability was tested.
    n_support_vectors : np.ndarray  shape (N_DIRECTIONS, N_samples, N_OBJECTS)
        Number of support vectors per test point.
    """

    layer_name: str = ""
    capacity_alpha_c: np.ndarray = field(
        default_factory=lambda: np.full(len(DIRECTION_NAMES), np.nan)
    )
    separability: np.ndarray = field(default_factory=lambda: np.array([]))
    n_neuron_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    n_support_vectors: np.ndarray = field(default_factory=lambda: np.array([]))

    def __repr__(self) -> str:
        ac = self.capacity_alpha_c
        parts = [
            f"  {DIRECTION_NAMES[i]:18s}: alpha_c={ac[i]:.4f}"
            for i in range(len(ac))
            if i < len(DIRECTION_NAMES)
        ]
        return (
            f"LayerCapacityResults(layer='{self.layer_name}')\n"
            + "\n".join(parts)
        )


# ---------------------------------------------------------------------------
# One-dimensional capacity analysis
# ---------------------------------------------------------------------------

class OneDimensionalCapacityAnalysis:
    """
    Estimate classification capacity across transformation directions for
    a single network layer using pre-generated 1-D manifold data.

    Parameters
    ----------
    config : CapacityAnalysisConfig
        Fully-specified experiment and analysis configuration.
    """

    N_DIRECTIONS = 7

    def __init__(self, config: CapacityAnalysisConfig) -> None:
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
    ) -> LayerCapacityResults:
        """
        Compute capacity for all transformation directions on one layer.

        Parameters
        ----------
        tuning_function : dict
            Output of ``OneDimensionalManifoldGenerator.collect()``:
            ``{layer_name → np.ndarray (N_DIRECTIONS, N_OBJECTS, N_SAMPLES, N_FEATURES)}``.
        layer_number : int, optional
            1-based layer index in MATLAB convention (pixel layer = 1, first
            convnet layer = 2, …).  Selects the layer to analyse.
            If the exact layer is not in the dict, the nearest available one
            is used.
        layer_name : str, optional
            Direct layer name override; takes precedence over ``layer_number``.
        run_directions : list of int, optional
            0-based direction indices to process (default: all 7).

        Returns
        -------
        LayerCapacityResults
        """
        # --- select layer ---------------------------------------------------
        available = list(tuning_function.keys())
        if not available:
            raise ValueError("tuning_function dict is empty.")

        selected = self._select_layer(available, layer_number, layer_name)
        tf_full = tuning_function[selected]   # (N_DIR, N_OBJ, N_SMP, N_FEAT)

        if tf_full.ndim != 4:
            raise ValueError(
                f"Expected tuning_function['{selected}'] to have 4 dimensions "
                f"(N_DIR, N_OBJ, N_SMP, N_FEAT), got shape {tf_full.shape}."
            )

        n_dir_avail, n_obj, n_smp, n_feat = tf_full.shape

        if run_directions is None:
            run_directions = list(range(min(self.N_DIRECTIONS, n_dir_avail)))

        # --- build capacity estimator ---------------------------------------
        cfg = self.config
        estimator = BinaryDichotomiesCapacity(
            expected_precision=cfg.expected_precision,
            verbose=cfg.verbose,
            random_labeling_type=cfg.random_labeling_type,
            precision=cfg.precision,
            max_samples=cfg.max_samples,
            global_preprocessing=cfg.global_preprocessing,
            features_type=cfg.features_type,
        )

        # --- result containers ----------------------------------------------
        n_result_dirs = len(run_directions)
        alpha_c = np.full(n_result_dirs, np.nan)
        sep_list = []
        nsamp_list = []
        nsv_list = []

        for result_idx, d in enumerate(run_directions):
            dir_name = (DIRECTION_NAMES[d] if d < len(DIRECTION_NAMES)
                        else f"direction_{d}")
            if cfg.verbose:
                print(f"  [{result_idx+1}/{n_result_dirs}] "
                      f"direction={dir_name}  layer={selected}")

            # Extract (N_FEAT, N_SMP, N_OBJ) for this direction
            # tf_full[d] shape: (N_OBJ, N_SMP, N_FEAT)
            tf_dir = tf_full[d]                   # (N_OBJ, N_SMP, N_FEAT)
            tf_nmp = tf_dir.transpose(2, 1, 0)    # (N_FEAT, N_SMP, N_OBJ)

            # Remove zero-variance neurons (mirrors MATLAB's nzIndices logic)
            mean_sq = np.nanmean(tf_nmp ** 2, axis=(1, 2))
            nz = mean_sq > 0
            tf_nmp = tf_nmp[nz]

            if tf_nmp.shape[0] == 0:
                if cfg.verbose:
                    print(f"    Skipping: all-zero features.")
                sep_list.append(np.array([]))
                nsamp_list.append(np.array([]))
                nsv_list.append(np.array([]))
                continue

            cap_res: CapacityResults = estimator.estimate(tf_nmp)

            if cap_res.Nc and cap_res.Nc > 0:
                alpha_c[result_idx] = n_obj / cap_res.Nc
            sep_list.append(cap_res.separability_results)
            nsamp_list.append(cap_res.n_neuron_samples_used)
            nsv_list.append(cap_res.n_support_vectors)

        return LayerCapacityResults(
            layer_name=selected,
            capacity_alpha_c=alpha_c,
            separability=_pad_stack(sep_list),
            n_neuron_samples=_pad_stack(nsamp_list),
            n_support_vectors=_pad_stack(nsv_list),
        )

    def run_all_layers(
        self,
        tuning_function: Dict[str, np.ndarray],
        run_directions: Optional[List[int]] = None,
    ) -> Dict[str, LayerCapacityResults]:
        """
        Run capacity analysis for every layer present in ``tuning_function``.

        Returns
        -------
        dict  {layer_name → LayerCapacityResults}
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
        """Return the layer name to analyse."""
        if layer_name is not None:
            if layer_name in available:
                return layer_name
            raise KeyError(
                f"layer_name='{layer_name}' not found. Available: {available}"
            )

        if layer_number is not None:
            meta_names = self.metadata.layer_names     # ordered convnet layers
            # Try to find by position in metadata first
            conv_idx = layer_number - 2  # skip pixel layer (MATLAB layer 1)
            if 0 <= conv_idx < len(meta_names):
                candidate = meta_names[conv_idx]
                if candidate in available:
                    return candidate
            # Fall back: nearest by index in available
            idx = max(0, min(layer_number - 1, len(available) - 1))
            chosen = available[idx]
            if self.config.verbose:
                print(
                    f"  layer_number={layer_number} → using '{chosen}' "
                    f"(available: {available})"
                )
            return chosen

        # No selection: return the first available layer
        return available[0]


# ---------------------------------------------------------------------------
# Random-change capacity analysis
# ---------------------------------------------------------------------------

class RandomChangeCapacityAnalysis:
    """
    Estimate classification capacity for random-transform manifolds.

    Parameters
    ----------
    config : CapacityAnalysisConfig
    degrees_of_freedom : int
        Number of random transform directions (2 for 2-D manifolds).
    """

    def __init__(
        self,
        config: CapacityAnalysisConfig,
        degrees_of_freedom: int = 2,
    ) -> None:
        self.config = config
        self.dof = degrees_of_freedom
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
    ) -> LayerCapacityResults:
        """
        Parameters
        ----------
        tuning_function : dict
            Output of ``RandomManifoldGenerator.collect()``.
            Accepts two formats:
            - ``(N_OBJECTS, N_SAMPLES, N_FEATURES)``  — single realization
            - ``(N_TRANSFORM_DIMS, N_OBJECTS, N_SAMPLES, N_FEATURES)`` — multi-realization

            For the 4-D case each transform realization is analysed independently
            and the results (capacity, separability) are averaged across dims.

        Returns
        -------
        LayerCapacityResults  (one entry; capacity averaged over transform dims)
        """
        available = list(tuning_function.keys())
        selected = self._select_layer(available, layer_number, layer_name)
        tf_full = tuning_function[selected]

        # Normalise to 4-D: (N_XDIMS, N_OBJ, N_SMP, N_FEAT)
        if tf_full.ndim == 3:
            tf_full = tf_full[np.newaxis]   # treat as single realization
        n_xdims, n_obj, n_smp, n_feat = tf_full.shape

        cfg = self.config
        estimator = BinaryDichotomiesCapacity(
            expected_precision=cfg.expected_precision,
            verbose=cfg.verbose,
            random_labeling_type=cfg.random_labeling_type,
            precision=cfg.precision,
            max_samples=cfg.max_samples,
            global_preprocessing=cfg.global_preprocessing,
            features_type=cfg.features_type,
        )

        alpha_c_list: List[float] = []
        sep_list = []
        nsamp_list = []
        nsv_list = []

        for xd in range(n_xdims):
            if cfg.verbose and n_xdims > 1:
                print(f"  transform_realization {xd+1}/{n_xdims}  layer={selected}")

            tf_xd = tf_full[xd]                    # (N_OBJ, N_SMP, N_FEAT)
            tf_nmp = tf_xd.transpose(2, 1, 0)      # (N_FEAT, N_SMP, N_OBJ)

            mean_sq = np.nanmean(tf_nmp ** 2, axis=(1, 2))
            tf_nmp = tf_nmp[mean_sq > 0]

            if tf_nmp.shape[0] == 0:
                continue

            cap_res = estimator.estimate(tf_nmp)
            if cap_res.Nc and cap_res.Nc > 0:
                alpha_c_list.append(n_obj / cap_res.Nc)
            sep_list.append(cap_res.separability_results)
            nsamp_list.append(cap_res.n_neuron_samples_used)
            nsv_list.append(cap_res.n_support_vectors)

        mean_alpha = np.array([float(np.mean(alpha_c_list))]) if alpha_c_list else np.array([np.nan])

        return LayerCapacityResults(
            layer_name=selected,
            capacity_alpha_c=mean_alpha,
            separability=_pad_stack(sep_list),
            n_neuron_samples=_pad_stack(nsamp_list),
            n_support_vectors=_pad_stack(nsv_list),
        )

    def _select_layer(
        self,
        available: List[str],
        layer_number: Optional[int],
        layer_name: Optional[str],
    ) -> str:
        if layer_name is not None and layer_name in available:
            return layer_name
        if layer_number is not None:
            idx = max(0, min(layer_number - 1, len(available) - 1))
            return available[idx]
        return available[0]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pad_stack(arrays: List[np.ndarray]) -> np.ndarray:
    """Stack arrays of potentially different lengths along axis 0, padding with NaN."""
    if not arrays or all(a.size == 0 for a in arrays):
        return np.array([])
    max_len = max(a.size for a in arrays)
    out = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        if a.size > 0:
            out[i, :a.size] = a.ravel()
    return out

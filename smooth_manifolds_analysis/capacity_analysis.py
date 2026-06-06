"""
High-level capacity analysis for DNN manifolds.

Corresponds to:
  - check_convnet_capacity_one_dimensional_change2.m
  - check_convnet_capacity_random_change2.m

Both functions iterate over network layers and transformation directions,
loading pre-generated tuning functions, then calling the capacity estimator.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Resolve the package root so both library and smooth_manifolds_generation
# are importable from the analysis module.
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from library import BinaryDichotomiesCapacity, CapacityResults
from smooth_manifolds_generation.network import load_network_metadata


DIRECTION_NAMES = [
    "x-translation", "y-translation",
    "x-scale", "y-scale",
    "x-shear", "y-shear",
    "rotation",
]


@dataclass
class CapacityAnalysisConfig:
    """
    Configuration shared by both 1-D and random-change capacity analyses.
    """

    expected_precision: float = 0.05
    max_samples: int = 100
    precision: int = 1
    features_type: int = 2       # 2 = random projections
    random_labeling_type: int = 1  # 1 = balanced
    global_preprocessing: int = 0
    local_preprocessing: int = 0
    verbose: bool = True


@dataclass
class LayerCapacityResults:
    """Per-layer capacity results for one transformation direction."""

    capacity: float = np.nan
    separability: np.ndarray = field(default_factory=lambda: np.array([]))
    n_neuron_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    n_support_vectors: np.ndarray = field(default_factory=lambda: np.array([]))


class OneDimensionalCapacityAnalysis:
    """
    Estimate classification capacity across network layers and
    transformation directions using pre-generated 1-D manifold data.

    Corresponds to ``check_convnet_capacity_one_dimensional_change2.m``.

    Parameters
    ----------
    config : CapacityAnalysisConfig
    network_type : int
    n_objects : int
    n_samples : int
    range_factor : float
    layers_grouping_level : int
    """

    def __init__(
        self,
        config: CapacityAnalysisConfig,
        network_type: int,
        n_objects: int,
        n_samples: int,
        range_factor: float = 1.0,
        layers_grouping_level: int = 0,
        epoch: Optional[int] = None,
    ) -> None:
        self.config = config
        self.network_type = network_type
        self.n_objects = n_objects
        self.n_samples = n_samples
        self.range_factor = range_factor
        self.epoch = epoch

        self.metadata = load_network_metadata(
            network_type, layers_grouping_level, epoch
        )

    def run(
        self,
        tuning_functions: Dict[str, Dict[str, np.ndarray]],
        run_directions: Optional[List[int]] = None,
        run_layers: Optional[List[str]] = None,
    ) -> Dict[int, Dict[str, LayerCapacityResults]]:
        """
        Run capacity analysis.

        Parameters
        ----------
        tuning_functions : dict
            {direction_name: {layer_name: np.ndarray(N, M, P)}}
        run_directions : list of int, optional
            0-based direction indices to process.  Defaults to all.
        run_layers : list of str, optional
            Layer names to process.  Defaults to all enabled layers.

        Returns
        -------
        dict
            {direction_idx: {layer_name: LayerCapacityResults}}
        """
        cfg = self.config
        if run_directions is None:
            run_directions = list(range(len(DIRECTION_NAMES)))
        if run_layers is None:
            run_layers = [self.metadata.layer_names[i]
                          for i in self.metadata.enabled_layers]

        estimator = BinaryDichotomiesCapacity(
            expected_precision=cfg.expected_precision,
            verbose=cfg.verbose,
            random_labeling_type=cfg.random_labeling_type,
            precision=cfg.precision,
            max_samples=cfg.max_samples,
            global_preprocessing=cfg.global_preprocessing,
            features_type=cfg.features_type,
        )

        results: Dict[int, Dict[str, LayerCapacityResults]] = {}

        for d in run_directions:
            direction_name = DIRECTION_NAMES[d]
            results[d] = {}

            for layer_name in run_layers:
                if direction_name not in tuning_functions:
                    continue
                if layer_name not in tuning_functions[direction_name]:
                    continue

                tf = tuning_functions[direction_name][layer_name]
                assert tf.ndim == 3, "Tuning function must be (N, M, P)"

                if cfg.verbose:
                    print(f"Working on {direction_name} {layer_name}")

                cap_res: CapacityResults = estimator.estimate(tf)
                results[d][layer_name] = LayerCapacityResults(
                    capacity=cap_res.Nc / tf.shape[2] if cap_res.Nc else np.nan,
                    separability=cap_res.separability_results,
                    n_neuron_samples=cap_res.n_neuron_samples_used,
                    n_support_vectors=cap_res.n_support_vectors,
                )

        return results


class RandomChangeCapacityAnalysis:
    """
    Estimate classification capacity across network layers for random-change
    manifolds (two random affine transformation directions per manifold).

    Corresponds to ``check_convnet_capacity_random_change2.m``.
    """

    def __init__(
        self,
        config: CapacityAnalysisConfig,
        network_type: int,
        n_objects: int,
        n_samples: int,
        range_factor: float = 1.0,
        degrees_of_freedom: int = 2,
        layers_grouping_level: int = 0,
        epoch: Optional[int] = None,
    ) -> None:
        self.config = config
        self.network_type = network_type
        self.n_objects = n_objects
        self.n_samples = n_samples
        self.range_factor = range_factor
        self.dof = degrees_of_freedom
        self.epoch = epoch

        self.metadata = load_network_metadata(
            network_type, layers_grouping_level, epoch
        )

    def run(
        self,
        tuning_functions: Dict[int, Dict[str, np.ndarray]],
        run_directions: Optional[List[int]] = None,
        run_layers: Optional[List[str]] = None,
    ) -> Dict[int, Dict[str, LayerCapacityResults]]:
        """
        Parameters
        ----------
        tuning_functions : {dof_idx: {layer_name: np.ndarray(N, M, P)}}
        run_directions : list of int  (0-based, 0 or 1)
        run_layers : list of str
        """
        cfg = self.config
        if run_directions is None:
            run_directions = list(range(self.dof))
        if run_layers is None:
            run_layers = [self.metadata.layer_names[i]
                          for i in self.metadata.enabled_layers]

        estimator = BinaryDichotomiesCapacity(
            expected_precision=cfg.expected_precision,
            verbose=cfg.verbose,
            random_labeling_type=cfg.random_labeling_type,
            precision=cfg.precision,
            max_samples=cfg.max_samples,
            global_preprocessing=cfg.global_preprocessing,
            features_type=cfg.features_type,
        )

        results: Dict[int, Dict[str, LayerCapacityResults]] = {}

        for d in run_directions:
            results[d] = {}
            for layer_name in run_layers:
                if d not in tuning_functions:
                    continue
                if layer_name not in tuning_functions[d]:
                    continue

                tf = tuning_functions[d][layer_name]
                assert tf.ndim == 3

                if cfg.verbose:
                    print(f"Working on DOF={d} {layer_name}")

                cap_res = estimator.estimate(tf)
                results[d][layer_name] = LayerCapacityResults(
                    capacity=cap_res.Nc / tf.shape[2] if cap_res.Nc else np.nan,
                    separability=cap_res.separability_results,
                    n_neuron_samples=cap_res.n_neuron_samples_used,
                    n_support_vectors=cap_res.n_support_vectors,
                )

        return results

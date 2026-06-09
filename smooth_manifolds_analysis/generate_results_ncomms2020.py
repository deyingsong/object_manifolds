"""
generate_results_ncomms2020.py
==============================

Reproduces all experiments from:
  Chung, Lee & Abbott, "Classification and Geometry of General Perceptual Manifolds",
  Nature Communications 2020.

Structure
---------
Part 1 – 1-D manifolds
    § 1.1  Standard capacity  (AlexNet / ResNet50 / VGG16, 9 range factors)
    § 1.2  Preprocessing variants  (random centers, manifolds, axes, permuted)
    § 1.3  Untrained AlexNet (epoch=0)
    § 1.4  Data-manipulation controls  (vary D_clip and center_norm_factor)
    § 1.5  Sample-count sweep  (vary M used in analysis)
    § 1.6  Neuron-count sweep  (vary N in geometry analysis)
    § 1.7  Object-count sweep  (vary P)
    § 1.8  Sphere control manifolds

Part 2 – Randomly-sampled (2-D) manifolds
    § 2.1  Standard capacity  (AlexNet / ResNet50 / VGG16)
    § 2.2  Preprocessing variants
    § 2.3  Low-rank geometry (AlexNet / ResNet50 / VGG16)
    § 2.4  Untrained AlexNet (epoch=0)

Usage
-----
Run the whole paper::

    python generate_results_ncomms2020.py

Or import and call individual sections::

    from smooth_manifolds_analysis.generate_results_ncomms2020 import (
        run_1d_standard_capacity, run_2d_standard_capacity,
    )
    run_1d_standard_capacity(network_type=1, output_dir="results/")
"""

from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from smooth_manifolds_generation.generation import (
    GenerationConfig, OneDimensionalManifoldGenerator, RandomManifoldGenerator,
)
from smooth_manifolds_generation.imagenet import init_imagenet
from smooth_manifolds_generation.network import NetworkType, load_network_metadata
from smooth_manifolds_analysis.capacity_analysis import (
    CapacityAnalysisConfig, OneDimensionalCapacityAnalysis,
    RandomChangeCapacityAnalysis, LayerCapacityResults, DIRECTION_NAMES,
)
from smooth_manifolds_analysis.low_rank_analysis import (
    LowRankAnalysisConfig, CovarianceLowRankAnalysis, LowRankResults,
)
from library import (
    BinaryDichotomiesCapacity, TuningFunctionPreprocessor, LowDimensionalManifold,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# ---------------------------------------------------------------------------
# Paper parameter grids  (Table 1 / Figure parameters from the MATLAB script)
# ---------------------------------------------------------------------------

# 1-D manifolds
RANGE_FACTORS_1D = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
SAMPLES_1D       = [5,     9,    15,  27,  51,  101, 201, 201,  201]

# 2-D / random manifolds
RANGE_FACTORS_2D = RANGE_FACTORS_1D
SAMPLES_2D       = [51,   101,  201, 401, 801, 1601, 3201, 3201, 3201]

# Network type codes → (n_batches for P=128, n_batches for P=64)
# AlexNet (1): N_LAYERS=19, ResNet50 (3): N_LAYERS=20, VGG16 (5): N_LAYERS=19
# MATLAB used distributed jobs (N_LAYERS × N_DIRECTIONS run_ids per analysis).
# Python runs all layers in one collect() call.
N_BATCHES_1D = 4   # objects per batch = 128 / 4 = 32
N_BATCHES_2D = 4   # objects per batch = 64  / 4 = 16

# Local-preprocessing variant codes → output suffix (mirrors MATLAB naming)
LOCAL_PREPROCESSING_NAMES = {
    0: "",
    1: "_orth",
    2: "_centers_random",
    3: "_manifold_random",
    5: "_axes_random",
    7: "_permute_random",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _results_path(
    output_dir: str,
    kind: str,              # "1d_capacity", "1d_lowrank", "2d_capacity", "2d_lowrank"
    network_type: int,
    range_factor: float,
    n_objects: int,
    n_samples: int,
    local_preprocessing: int = 0,
    epoch: Optional[int] = None,
    D_clip: Optional[int] = None,
    center_norm_factor: float = 1.0,
    samples_jump: int = 1,
    n_neurons: int = 0,
    p_objects: int = 0,
) -> Path:
    """Return the .npz path for a given set of experiment parameters."""
    nt  = NetworkType(network_type)
    tag = f"{range_factor:.1f}" if range_factor >= 0.1 else f"{range_factor:f}"
    suffix = LOCAL_PREPROCESSING_NAMES.get(local_preprocessing, f"_lp{local_preprocessing}")
    if epoch is not None:
        suffix += f"_epoch{epoch}"
    if D_clip is not None:
        suffix += f"_D{D_clip}"
    if center_norm_factor != 1.0:
        suffix += f"_f{center_norm_factor:.2f}"
    if samples_jump > 1:
        suffix += f"_s{samples_jump}"
    if n_neurons > 0:
        suffix += f"_N{n_neurons}"
    if p_objects > 0:
        suffix += f"_P{p_objects}"

    fname = (
        f"{kind}_{nt.name.lower()}_range{tag}"
        f"_P{n_objects}_M{n_samples}{suffix}.npz"
    )
    return Path(output_dir) / fname


def _save_capacity_results(
    path: Path,
    results_by_layer: Dict[str, LayerCapacityResults],
) -> None:
    """Save a {layer → LayerCapacityResults} dict to a single .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    layer_names = list(results_by_layer.keys())
    arrays["layer_names"] = np.array(layer_names)
    for ln, res in results_by_layer.items():
        safe = ln.replace("-", "_").replace(" ", "_")
        arrays[f"{safe}_capacity_alpha_c"]  = res.capacity_alpha_c
        if res.separability.size:
            arrays[f"{safe}_separability"]     = res.separability
        if res.n_neuron_samples.size:
            arrays[f"{safe}_n_neuron_samples"]  = res.n_neuron_samples
    np.savez_compressed(path, **arrays)
    log.info("Saved capacity results → %s", path)


def _save_lowrank_results(
    path: Path,
    results_by_layer: Dict[str, LowRankResults],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    layer_names = list(results_by_layer.keys())
    arrays["layer_names"] = np.array(layer_names)
    for ln, res in results_by_layer.items():
        safe = ln.replace("-", "_").replace(" ", "_")
        arrays[f"{safe}_theory_capacity"] = res.theory_capacity
        arrays[f"{safe}_mean_radius"]     = res.mean_radius
        arrays[f"{safe}_mean_dimension"]  = res.mean_dimension
        arrays[f"{safe}_K_opt"]           = res.K_opt
    np.savez_compressed(path, **arrays)
    log.info("Saved low-rank results → %s", path)


def load_capacity_results(path: Path) -> Dict[str, LayerCapacityResults]:
    """Load a .npz written by _save_capacity_results."""
    data = np.load(path, allow_pickle=True)
    layer_names = list(data["layer_names"])
    out = {}
    for ln in layer_names:
        safe = ln.replace("-", "_").replace(" ", "_")
        res = LayerCapacityResults(layer_name=ln)
        res.capacity_alpha_c = data[f"{safe}_capacity_alpha_c"]
        if f"{safe}_separability" in data:
            res.separability = data[f"{safe}_separability"]
        if f"{safe}_n_neuron_samples" in data:
            res.n_neuron_samples = data[f"{safe}_n_neuron_samples"]
        out[ln] = res
    return out


def load_lowrank_results(path: Path) -> Dict[str, LowRankResults]:
    data = np.load(path, allow_pickle=True)
    layer_names = list(data["layer_names"])
    out = {}
    for ln in layer_names:
        safe = ln.replace("-", "_").replace(" ", "_")
        res = LowRankResults(layer_name=ln)
        res.theory_capacity = data[f"{safe}_theory_capacity"]
        res.mean_radius     = data[f"{safe}_mean_radius"]
        res.mean_dimension  = data[f"{safe}_mean_dimension"]
        res.K_opt           = data[f"{safe}_K_opt"]
        out[ln] = res
    return out


def _apply_local_preprocessing(
    tf: np.ndarray,
    local_preprocessing: int,
    low_rank_results: Optional[Dict[str, LowRankResults]] = None,
    layer_name: str = "",
    direction_idx: int = 0,
    D_clip: Optional[int] = None,
    center_norm_factor: float = 1.0,
) -> np.ndarray:
    """
    Apply local preprocessing to a single-direction tuning function.

    Parameters
    ----------
    tf : (N_FEAT, N_SAMPLES, N_OBJECTS) — already transposed
    local_preprocessing :
        0  no preprocessing
        1  orthogonalise centers
        2  random centers        (requires nothing extra)
        3  permuted manifold     (requires nothing extra)
        5  random axes           (equivalent to random manifold)
        7  permute center+axes   (equivalent to permuted manifold)
    low_rank_results : optional
        Needed only if local_preprocessing != 0 to obtain V_k.
    """
    if local_preprocessing == 0 and D_clip is None and center_norm_factor == 1.0:
        return tf

    # Get Vk from low-rank results if available
    Vk: Optional[np.ndarray] = None
    if low_rank_results is not None and layer_name in low_rank_results:
        lr = low_rank_results[layer_name]
        # K_opt per direction
        k_opt = int(lr.K_opt[direction_idx]) if direction_idx < len(lr.K_opt) else 0
        # Retrieve basis vectors if they were saved (optional — not always stored)
        # If not available, fall back to no-basis processing
        # NOTE: to use Vk you'd need to re-run LowRankAnalysis with save_basis=True

    lp_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        5: 2,   # random axes ≈ random manifold
        7: 3,   # permute center+axes ≈ permuted manifold
    }
    pp_code = lp_map.get(local_preprocessing, 0)
    ldm = LowDimensionalManifold(
        target_dim=D_clip,
        local_preprocessing=pp_code,
        center_norm_factor=center_norm_factor,
    )
    if Vk is not None:
        return ldm.reduce_with_basis(tf, Vk)
    else:
        return ldm.reduce(tf)


# ---------------------------------------------------------------------------
# § 1.1  1-D standard capacity
# ---------------------------------------------------------------------------

def run_1d_standard_capacity(
    network_type: int = NetworkType.ALEXNET,
    n_objects: int = 128,
    n_batches: int = N_BATCHES_1D,
    range_factors: List[float] = None,
    samples_list: List[int] = None,
    output_dir: str = "results",
    epoch: Optional[int] = None,
    local_preprocessing: int = 0,
    D_clip: Optional[int] = None,
    center_norm_factor: float = 1.0,
    samples_jump: int = 1,
    low_rank_results_by_rf: Optional[Dict[float, Dict[str, LowRankResults]]] = None,
    device: str = "cpu",
    skip_existing: bool = True,
) -> Dict[float, Dict[str, LayerCapacityResults]]:
    """
    Numerically calculate capacity for all (range_factor, layer) combinations.

    Mirrors the inner loop of the MATLAB ``generate_results_ncomms2020.m``
    Part 1 (1-D manifolds) calling ``check_convnet_capacity_one_dimensional_change2``.

    Parameters
    ----------
    network_type : int
        NetworkType enum value (1=AlexNet, 3=ResNet50, 5=VGG16).
    n_objects : int
        P – number of object manifolds.
    n_batches : int
        Number of generation batches (divides n_objects).
    range_factors : list of float, optional
        Transform magnitudes. Default: ``RANGE_FACTORS_1D``.
    samples_list : list of int, optional
        N_SAMPLES per range_factor. Default: ``SAMPLES_1D``.
    output_dir : str
        Root directory for saved .npz files.
    epoch : int, optional
        For untrained network (0 = random init).
    local_preprocessing : int
        0=standard, 2=random centers, 3=random manifold, 5=random axes, 7=permuted.
    D_clip : int, optional
        Clip effective dimension of manifolds.
    center_norm_factor : float
        Multiplicative factor on manifold centers.
    samples_jump : int
        Sub-sample every Kth sample (1=use all).
    low_rank_results_by_rf : dict, optional
        Pre-computed low-rank results keyed by range_factor.
        Required only if local_preprocessing != 0.
    device : str
        Torch device ('cpu' or 'cuda').
    skip_existing : bool
        Skip (range_factor, network) combinations whose result file exists.

    Returns
    -------
    dict  {range_factor → {layer_name → LayerCapacityResults}}
    """
    if range_factors is None:
        range_factors = RANGE_FACTORS_1D
    if samples_list is None:
        samples_list = SAMPLES_1D
    assert len(range_factors) == len(samples_list), \
        "range_factors and samples_list must have the same length"

    all_results: Dict[float, Dict[str, LayerCapacityResults]] = {}
    imagenet_state = init_imagenet()

    for rf, n_smp in zip(range_factors, samples_list):
        out_path = _results_path(
            output_dir, "1d_capacity", network_type,
            rf, n_objects, n_smp,
            local_preprocessing=local_preprocessing,
            epoch=epoch,
            D_clip=D_clip,
            center_norm_factor=center_norm_factor,
            samples_jump=samples_jump,
        )

        if skip_existing and out_path.exists():
            log.info("Loading existing: %s", out_path)
            all_results[rf] = load_capacity_results(out_path)
            continue

        log.info("=== range_factor=%.3f  n_samples=%d  network=%s ===",
                 rf, n_smp, NetworkType(network_type).name)

        # ---- Step 1: Generate tuning functions --------------------------
        gen_cfg = GenerationConfig(
            network_type=network_type,
            range_factor=rf,
            n_objects=n_objects,
            n_samples=n_smp,
            n_batches=n_batches,
            output_dir=output_dir,
            device=device,
            epoch=epoch,
        )
        generator = OneDimensionalManifoldGenerator(gen_cfg, imagenet_state)
        total_runs = generator.N_DIRECTIONS * n_batches
        for run_id in range(1, total_runs + 1):
            generator.generate(run_id)
        tuning_function = generator.collect()
        # tuning_function: {layer_name → (N_DIR, N_OBJ, N_SMP, N_FEAT)}

        # ---- Step 2: Apply preprocessing --------------------------------
        if local_preprocessing != 0 or D_clip is not None or center_norm_factor != 1.0:
            lr_for_rf = (low_rank_results_by_rf or {}).get(rf, {})
            tuning_function = _preprocess_1d(
                tuning_function, local_preprocessing,
                lr_for_rf, D_clip, center_norm_factor,
            )

        # ---- Step 3: Sub-sample M if needed -----------------------------
        if samples_jump > 1:
            tuning_function = {
                ln: tf[:, :, ::samples_jump, :]
                for ln, tf in tuning_function.items()
            }

        # ---- Step 4: Capacity analysis ----------------------------------
        ana_cfg = CapacityAnalysisConfig(
            n_objects=n_objects,
            n_samples=n_smp,
            range_factor=rf,
            network_type=network_type,
            epoch=epoch,
            verbose=False,
        )
        analysis = OneDimensionalCapacityAnalysis(ana_cfg)
        results_by_layer = analysis.run_all_layers(tuning_function)
        all_results[rf] = results_by_layer

        _save_capacity_results(out_path, results_by_layer)

    return all_results


def _preprocess_1d(
    tuning_function: Dict[str, np.ndarray],
    local_preprocessing: int,
    low_rank_results: Dict[str, LowRankResults],
    D_clip: Optional[int],
    center_norm_factor: float,
) -> Dict[str, np.ndarray]:
    """
    Apply local preprocessing to all layers and directions of a 1-D
    tuning function dict.

    Returns a new dict of the same structure.
    """
    lp_map = {0: 0, 1: 1, 2: 2, 3: 3, 5: 2, 7: 3}
    pp_code = lp_map.get(local_preprocessing, 0)
    out: Dict[str, np.ndarray] = {}
    for ln, tf in tuning_function.items():
        # tf: (N_DIR, N_OBJ, N_SMP, N_FEAT)
        n_dir, n_obj, n_smp, n_feat = tf.shape
        processed_dirs = []
        for d in range(n_dir):
            tf_d = tf[d].transpose(2, 1, 0)   # → (N_FEAT, N_SMP, N_OBJ)
            # Get Vk if available
            Vk = None
            if ln in low_rank_results:
                lr = low_rank_results[ln]
                k_opt = int(lr.K_opt[d]) if d < len(lr.K_opt) else 0
                # Vk would need to be stored in LowRankResults.V_opt_by_dir[d]
                # Currently not stored — use without Vk fallback
            ldm = LowDimensionalManifold(
                target_dim=D_clip,
                local_preprocessing=pp_code,
                center_norm_factor=center_norm_factor,
            )
            if Vk is not None:
                tf_d = ldm.reduce_with_basis(tf_d, Vk)
            else:
                tf_d = ldm.reduce(tf_d)         # (D', N_SMP, N_OBJ)
            # Transpose back → (N_OBJ, N_SMP, D')
            processed_dirs.append(tf_d.transpose(2, 1, 0))
        # Stack → (N_DIR, N_OBJ, N_SMP, D')
        out[ln] = np.stack(processed_dirs, axis=0)
    return out


# ---------------------------------------------------------------------------
# § 1.2  1-D preprocessing variants
# ---------------------------------------------------------------------------

def run_1d_preprocessing_variants(
    network_type: int = NetworkType.ALEXNET,
    n_objects: int = 128,
    n_batches: int = N_BATCHES_1D,
    output_dir: str = "results",
    device: str = "cpu",
    skip_existing: bool = True,
) -> None:
    """
    Run all four preprocessing variants from the paper:
      - random centers (local_preprocessing=2)
      - random manifolds (local_preprocessing=3)
      - random axes (local_preprocessing=5)
      - permuted (local_preprocessing=7)

    Requires standard tuning functions to already be generated.
    Low-rank results are needed for full preprocessing but a no-basis
    fallback is used if not available.
    """
    for lp in [2, 3, 5, 7]:
        log.info("--- Preprocessing variant: %s ---",
                 LOCAL_PREPROCESSING_NAMES[lp])
        run_1d_standard_capacity(
            network_type=network_type,
            n_objects=n_objects,
            n_batches=n_batches,
            output_dir=output_dir,
            local_preprocessing=lp,
            device=device,
            skip_existing=skip_existing,
        )


# ---------------------------------------------------------------------------
# § 1.3  Untrained network (epoch=0)
# ---------------------------------------------------------------------------

def run_1d_untrained(
    network_type: int = NetworkType.ALEXNET,
    output_dir: str = "results",
    device: str = "cpu",
) -> None:
    """Capacity for untrained (randomly initialised) AlexNet (epoch=0)."""
    # Use the largest range_factor / sample count only (index 8 in MATLAB)
    run_1d_standard_capacity(
        network_type=network_type,
        n_objects=128,
        range_factors=[RANGE_FACTORS_1D[-1]],
        samples_list=[SAMPLES_1D[-1]],
        output_dir=output_dir,
        epoch=0,
        device=device,
    )


# ---------------------------------------------------------------------------
# § 1.4  Data-manipulation controls (vary D_clip and center_norm_factor)
# ---------------------------------------------------------------------------

def run_1d_dimension_manipulation(
    output_dir: str = "results",
    device: str = "cpu",
) -> None:
    """
    Vary D (intrinsic dimension clip) and center_norm_factor for the
    maximum range_factor (16.0) to disentangle R and D effects.

    Mirrors the "Data manipulation" block in the MATLAB script.
    """
    rf     = RANGE_FACTORS_1D[-1]    # 16.0
    n_smp  = SAMPLES_1D[-1]          # 201

    # Vary center_norm_factor (D=nan in MATLAB = no clipping)
    for f in [0.90, 0.95, 1.00, 1.05, 1.10]:
        run_1d_standard_capacity(
            network_type=NetworkType.ALEXNET,
            range_factors=[rf], samples_list=[n_smp],
            output_dir=output_dir,
            center_norm_factor=f,
            device=device,
        )

    # Vary D_clip (center_norm_factor=1)
    for D in [10, 20, 30, 40, 50]:
        run_1d_standard_capacity(
            network_type=NetworkType.ALEXNET,
            range_factors=[rf], samples_list=[n_smp],
            output_dir=output_dir,
            D_clip=D,
            device=device,
        )


# ---------------------------------------------------------------------------
# § 1.5  Sample-count sweep
# ---------------------------------------------------------------------------

def run_1d_sample_sweep(
    output_dir: str = "results",
    device: str = "cpu",
) -> None:
    """
    Analyse capacity using sub-sets of samples (vary M used in analysis)
    for the maximum range_factor.
    """
    rf    = RANGE_FACTORS_1D[-1]
    n_smp = SAMPLES_1D[-1]
    # sample_jump=1 → use all M; =2 → use M/2; =200 → use 1 sample
    SAMPLE_JUMPS = [2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200]
    for sj in SAMPLE_JUMPS:
        run_1d_standard_capacity(
            network_type=NetworkType.ALEXNET,
            range_factors=[rf], samples_list=[n_smp],
            output_dir=output_dir,
            samples_jump=sj,
            device=device,
        )


# ---------------------------------------------------------------------------
# § 1.6  Neuron-count sweep
# ---------------------------------------------------------------------------

def run_1d_neuron_sweep(
    output_dir: str = "results",
    device: str = "cpu",
) -> None:
    """
    Vary N (number of neurons used in the geometry analysis).

    This is handled at the CovarianceLowRankAnalysis level since that
    is where neuron sub-sampling occurs.
    """
    rf    = RANGE_FACTORS_1D[-1]
    n_smp = SAMPLES_1D[-1]
    for n_neurons in [128, 256, 512, 1024, 2048, 3072]:
        log.info("=== Neuron sweep  N=%d ===", n_neurons)
        gen_cfg = GenerationConfig(
            network_type=NetworkType.ALEXNET,
            range_factor=rf, n_objects=128, n_samples=n_smp,
            n_batches=N_BATCHES_1D, output_dir=output_dir, device=device,
        )
        imagenet_state = init_imagenet()
        generator = OneDimensionalManifoldGenerator(gen_cfg, imagenet_state)
        # Assume generation already done; just collect
        try:
            tuning_function = generator.collect()
        except FileNotFoundError:
            log.warning("Tuning function not found — run run_1d_standard_capacity first.")
            continue

        lr_cfg = LowRankAnalysisConfig(
            n_objects=128, n_samples=n_smp, range_factor=rf,
            max_k=5, network_type=NetworkType.ALEXNET,
            verbose=0, n_random_projections=n_neurons,
        )
        analysis = CovarianceLowRankAnalysis(lr_cfg)
        results  = analysis.run_all_layers(tuning_function)

        out_path = _results_path(
            output_dir, "1d_lowrank", NetworkType.ALEXNET,
            rf, 128, n_smp, n_neurons=n_neurons,
        )
        _save_lowrank_results(out_path, results)


# ---------------------------------------------------------------------------
# § 1.7  Object-count sweep
# ---------------------------------------------------------------------------

def run_1d_object_sweep(
    output_dir: str = "results",
    device: str = "cpu",
) -> None:
    """
    Vary P (number of object manifolds) used in capacity analysis.
    Generation uses P=128; analysis sub-selects P objects.
    """
    for P_analysis in [32, 64, 96, 128]:
        log.info("=== Object sweep  P=%d ===", P_analysis)
        run_1d_standard_capacity(
            network_type=NetworkType.ALEXNET,
            n_objects=P_analysis,
            output_dir=output_dir,
            device=device,
        )


# ---------------------------------------------------------------------------
# § 1.8  Sphere control manifolds
# ---------------------------------------------------------------------------

def run_1d_sphere_controls(
    output_dir: str = "results",
    device: str = "cpu",
) -> None:
    """
    Capacity for sphere manifolds with the same geometry as 1-D AlexNet manifolds.

    Corresponds to ``check_convnet_capacity_one_dimensional_change2_spheres.m``.
    Requires pre-computed low-rank results (geometry) for each range_factor.

    Note: Sphere generation is not yet fully automated; this function sets up
    the analysis configurations. Sphere manifolds are constructed from the
    manifold geometry (R, D) estimated by CovarianceLowRankAnalysis.
    """
    for rf, n_smp in zip(RANGE_FACTORS_1D, SAMPLES_1D):
        log.info("=== Sphere control  range_factor=%.3f ===", rf)
        out_path_lr = _results_path(
            output_dir, "1d_lowrank", NetworkType.ALEXNET,
            rf, 128, n_smp,
        )
        if not out_path_lr.exists():
            log.warning("Low-rank results not found — run run_1d_lowrank first.")
            continue

        lr_by_layer = load_lowrank_results(out_path_lr)
        # Generate sphere capacity for each layer using estimated R, D
        _run_sphere_capacity_from_geometry(
            lr_by_layer=lr_by_layer,
            n_objects=128,
            n_samples=n_smp,
            range_factor=rf,
            randomise_centers=False,
            output_dir=output_dir,
        )
        _run_sphere_capacity_from_geometry(
            lr_by_layer=lr_by_layer,
            n_objects=128,
            n_samples=n_smp,
            range_factor=rf,
            randomise_centers=True,
            output_dir=output_dir,
        )


def _run_sphere_capacity_from_geometry(
    lr_by_layer: Dict[str, LowRankResults],
    n_objects: int,
    n_samples: int,
    range_factor: float,
    randomise_centers: bool,
    output_dir: str,
) -> Dict[str, LayerCapacityResults]:
    """
    Generate sphere manifolds from estimated geometry (R, D) and compute capacity.

    For each layer × direction: synthesise P Gaussian sphere manifolds
    with radius R_M and dimension D_M, then run BinaryDichotomiesCapacity.
    """
    from library import BinaryDichotomiesCapacity, CapacityResults

    results_by_layer: Dict[str, LayerCapacityResults] = {}
    estimator = BinaryDichotomiesCapacity(
        expected_precision=0.05, verbose=False,
        random_labeling_type=1, precision=1,
        max_samples=100, features_type=2,
    )
    rng = np.random.default_rng(42)

    for ln, lr in lr_by_layer.items():
        n_dirs = len(lr.theory_capacity)
        alpha_c = np.full(n_dirs, np.nan)

        for d in range(n_dirs):
            R = float(lr.mean_radius[d]) if np.isfinite(lr.mean_radius[d]) else 0.05
            D = max(1, int(round(float(lr.mean_dimension[d])))) if np.isfinite(lr.mean_dimension[d]) else 5
            if R <= 0:
                continue

            # Synthesise sphere manifolds: N_FEAT random axes, each object
            # has a sphere of radius R in D dimensions
            N_FEAT = 512
            center_norms = rng.standard_normal((N_FEAT, n_objects))
            center_norms /= np.linalg.norm(center_norms, axis=0, keepdims=True)
            center_norms *= rng.standard_normal(n_objects) * 10   # random center norms
            if randomise_centers:
                perm = rng.permutation(n_objects)
                center_norms = center_norms[:, perm]

            # Sphere samples: random D-dim unit vectors scaled by R
            axes = rng.standard_normal((N_FEAT, D, n_objects))
            axes /= np.linalg.norm(axes, axis=0, keepdims=True) + 1e-12
            sph_pts = rng.standard_normal((D, n_samples, n_objects))
            sph_pts /= np.linalg.norm(sph_pts, axis=0, keepdims=True) + 1e-12

            # tf: (N_FEAT, N_SAMPLES, N_OBJECTS)
            tf_sphere = (
                center_norms[:, np.newaxis, :]       # (N_FEAT, 1, N_OBJECTS)
                + R * np.einsum("ndo,dmo->nmo", axes, sph_pts)
            )
            cap_res = estimator.estimate(tf_sphere)
            if cap_res.Nc and cap_res.Nc > 0:
                alpha_c[d] = n_objects / cap_res.Nc

        results_by_layer[ln] = LayerCapacityResults(
            layer_name=ln,
            capacity_alpha_c=alpha_c,
        )

    tag = "sphere_rand" if randomise_centers else "sphere"
    rf_tag = f"{range_factor:.1f}" if range_factor >= 0.1 else f"{range_factor:f}"
    out_path = Path(output_dir) / (
        f"1d_capacity_alexnet_range{rf_tag}_P{n_objects}_M{n_samples}_{tag}.npz"
    )
    _save_capacity_results(out_path, results_by_layer)
    return results_by_layer


# ---------------------------------------------------------------------------
# § 2.1  2-D standard capacity
# ---------------------------------------------------------------------------

def run_2d_standard_capacity(
    network_type: int = NetworkType.ALEXNET,
    n_objects: int = 64,
    n_transform_dims: int = 2,
    n_batches: int = N_BATCHES_2D,
    range_factors: List[float] = None,
    samples_list: List[int] = None,
    output_dir: str = "results",
    epoch: Optional[int] = None,
    local_preprocessing: int = 0,
    device: str = "cpu",
    skip_existing: bool = True,
) -> Dict[float, Dict[str, LayerCapacityResults]]:
    """
    Numerically calculate capacity for 2-D (randomly-sampled) manifolds.

    Mirrors ``check_convnet_capacity_random_change2`` calls in the MATLAB script.

    Parameters
    ----------
    network_type : int
    n_objects : int
        P = 64 in the paper.
    n_transform_dims : int
        Degrees of freedom (2 in the paper).
    n_batches : int
    range_factors : list of float, optional
    samples_list : list of int, optional

    Returns
    -------
    dict  {range_factor → {layer_name → LayerCapacityResults}}
    """
    if range_factors is None:
        range_factors = RANGE_FACTORS_2D
    if samples_list is None:
        samples_list = SAMPLES_2D
    assert len(range_factors) == len(samples_list)

    all_results: Dict[float, Dict[str, LayerCapacityResults]] = {}
    imagenet_state = init_imagenet()

    for rf, n_smp in zip(range_factors, samples_list):
        out_path = _results_path(
            output_dir, "2d_capacity", network_type,
            rf, n_objects, n_smp,
            local_preprocessing=local_preprocessing,
            epoch=epoch,
        )
        if skip_existing and out_path.exists():
            log.info("Loading existing: %s", out_path)
            all_results[rf] = load_capacity_results(out_path)
            continue

        log.info("=== 2D  range_factor=%.3f  n_samples=%d  network=%s ===",
                 rf, n_smp, NetworkType(network_type).name)

        # ---- Step 1: Generate --------------------------------------------
        gen_cfg = GenerationConfig(
            network_type=network_type,
            range_factor=rf,
            n_objects=n_objects,
            n_samples=n_smp,
            n_batches=n_batches,
            n_transform_dims=n_transform_dims,
            output_dir=output_dir,
            device=device,
        )
        generator = RandomManifoldGenerator(gen_cfg, imagenet_state)
        total_runs = n_batches * n_transform_dims
        for run_id in range(1, total_runs + 1):
            generator.generate(run_id)
        tuning_function = generator.collect()
        # tuning_function: {layer → (N_TRANSFORM_DIMS, N_OBJ, N_SMP, N_FEAT)}

        # ---- Step 2: Apply preprocessing if needed ----------------------
        if local_preprocessing != 0:
            lp_map = {2: 2, 3: 3, 5: 2, 7: 3}
            pp_code = lp_map.get(local_preprocessing, 0)
            processed: Dict[str, np.ndarray] = {}
            for ln, tf in tuning_function.items():
                # tf: (N_XDIMS, N_OBJ, N_SMP, N_FEAT)
                dims = []
                for xd in range(tf.shape[0]):
                    tf_xd = tf[xd].transpose(2, 1, 0)   # (N_FEAT, N_SMP, N_OBJ)
                    tf_xd = LowDimensionalManifold(
                        local_preprocessing=pp_code
                    ).reduce(tf_xd)
                    dims.append(tf_xd.transpose(2, 1, 0))
                processed[ln] = np.stack(dims, axis=0)
            tuning_function = processed

        # ---- Step 3: Capacity analysis -----------------------------------
        ana_cfg = CapacityAnalysisConfig(
            n_objects=n_objects,
            n_samples=n_smp,
            range_factor=rf,
            n_transform_dims=n_transform_dims,
            network_type=network_type,
            epoch=epoch,
            verbose=False,
        )
        analysis  = RandomChangeCapacityAnalysis(ana_cfg)
        # Run each layer
        results_by_layer: Dict[str, LayerCapacityResults] = {}
        for ln, tf in tuning_function.items():
            results_by_layer[ln] = analysis.run({ln: tf}, layer_name=ln)
        all_results[rf] = results_by_layer

        _save_capacity_results(out_path, results_by_layer)

    return all_results


# ---------------------------------------------------------------------------
# § 2.2  2-D preprocessing variants
# ---------------------------------------------------------------------------

def run_2d_preprocessing_variants(
    network_type: int = NetworkType.ALEXNET,
    n_objects: int = 64,
    output_dir: str = "results",
    device: str = "cpu",
) -> None:
    """Run all four preprocessing control variants for 2-D manifolds."""
    for lp in [2, 3, 5, 7]:
        log.info("--- 2D preprocessing variant: %s ---",
                 LOCAL_PREPROCESSING_NAMES[lp])
        run_2d_standard_capacity(
            network_type=network_type,
            n_objects=n_objects,
            output_dir=output_dir,
            local_preprocessing=lp,
            device=device,
        )


# ---------------------------------------------------------------------------
# § 2.3  Low-rank geometry for 2-D manifolds
# ---------------------------------------------------------------------------

def run_2d_lowrank(
    network_type: int = NetworkType.ALEXNET,
    n_objects: int = 64,
    n_transform_dims: int = 2,
    n_batches: int = N_BATCHES_2D,
    range_factors: List[float] = None,
    samples_list: List[int] = None,
    max_k: int = 5,
    output_dir: str = "results",
    epoch: Optional[int] = None,
    device: str = "cpu",
    skip_existing: bool = True,
) -> Dict[float, Dict[str, LowRankResults]]:
    """
    Compute low-rank geometry (theory_capacity, mean_radius, mean_dimension)
    for 2-D randomly-sampled manifolds.

    Mirrors ``check_convnet_covariance_low_rank_approx_optimal_K`` calls.

    Returns
    -------
    dict  {range_factor → {layer_name → LowRankResults}}
    """
    if range_factors is None:
        range_factors = RANGE_FACTORS_2D
    if samples_list is None:
        samples_list = SAMPLES_2D
    assert len(range_factors) == len(samples_list)

    all_results: Dict[float, Dict[str, LowRankResults]] = {}
    imagenet_state = init_imagenet()

    for rf, n_smp in zip(range_factors, samples_list):
        out_path = _results_path(
            output_dir, "2d_lowrank", network_type,
            rf, n_objects, n_smp, epoch=epoch,
        )
        if skip_existing and out_path.exists():
            log.info("Loading existing: %s", out_path)
            all_results[rf] = load_lowrank_results(out_path)
            continue

        log.info("=== 2D low-rank  range_factor=%.3f  network=%s ===",
                 rf, NetworkType(network_type).name)

        # Collect pre-generated tuning functions
        gen_cfg = GenerationConfig(
            network_type=network_type,
            range_factor=rf,
            n_objects=n_objects,
            n_samples=n_smp,
            n_batches=n_batches,
            n_transform_dims=n_transform_dims,
            output_dir=output_dir,
        )
        generator = RandomManifoldGenerator(gen_cfg, imagenet_state)
        try:
            tuning_function = generator.collect()
        except FileNotFoundError:
            log.warning("Tuning function not found — run run_2d_standard_capacity first.")
            continue

        lr_cfg = LowRankAnalysisConfig(
            n_objects=n_objects,
            n_samples=n_smp,
            range_factor=rf,
            n_transform_dims=n_transform_dims,
            max_k=max_k,
            network_type=network_type,
            epoch=epoch,
            verbose=0,
        )
        analysis = CovarianceLowRankAnalysis(lr_cfg)
        results_by_layer = analysis.run_all_layers(tuning_function)
        all_results[rf] = results_by_layer

        _save_lowrank_results(out_path, results_by_layer)

    return all_results


# ---------------------------------------------------------------------------
# § 1-D low-rank geometry (needed for preprocessing controls)
# ---------------------------------------------------------------------------

def run_1d_lowrank(
    network_type: int = NetworkType.ALEXNET,
    n_objects: int = 128,
    n_batches: int = N_BATCHES_1D,
    range_factors: List[float] = None,
    samples_list: List[int] = None,
    max_k: int = 5,
    output_dir: str = "results",
    epoch: Optional[int] = None,
    D_clip: Optional[int] = None,
    center_norm_factor: float = 1.0,
    device: str = "cpu",
    skip_existing: bool = True,
) -> Dict[float, Dict[str, LowRankResults]]:
    """
    Compute low-rank geometry for 1-D manifolds.

    Must be run before preprocessing-variant capacity analyses.

    Returns
    -------
    dict  {range_factor → {layer_name → LowRankResults}}
    """
    if range_factors is None:
        range_factors = RANGE_FACTORS_1D
    if samples_list is None:
        samples_list = SAMPLES_1D
    assert len(range_factors) == len(samples_list)

    all_results: Dict[float, Dict[str, LowRankResults]] = {}
    imagenet_state = init_imagenet()

    for rf, n_smp in zip(range_factors, samples_list):
        out_path = _results_path(
            output_dir, "1d_lowrank", network_type,
            rf, n_objects, n_smp, epoch=epoch,
            D_clip=D_clip, center_norm_factor=center_norm_factor,
        )
        if skip_existing and out_path.exists():
            log.info("Loading existing: %s", out_path)
            all_results[rf] = load_lowrank_results(out_path)
            continue

        log.info("=== 1D low-rank  range_factor=%.3f  network=%s ===",
                 rf, NetworkType(network_type).name)

        gen_cfg = GenerationConfig(
            network_type=network_type,
            range_factor=rf,
            n_objects=n_objects,
            n_samples=n_smp,
            n_batches=n_batches,
            output_dir=output_dir,
        )
        generator = OneDimensionalManifoldGenerator(gen_cfg, imagenet_state)
        try:
            tuning_function = generator.collect()
        except FileNotFoundError:
            log.warning("Tuning function not found — run run_1d_standard_capacity first.")
            continue

        if D_clip is not None or center_norm_factor != 1.0:
            tuning_function = _preprocess_1d(
                tuning_function, 0, {}, D_clip, center_norm_factor
            )

        lr_cfg = LowRankAnalysisConfig(
            n_objects=n_objects,
            n_samples=n_smp,
            range_factor=rf,
            max_k=max_k,
            network_type=network_type,
            epoch=epoch,
            verbose=0,
        )
        analysis = CovarianceLowRankAnalysis(lr_cfg)
        results_by_layer = analysis.run_all_layers(tuning_function)
        all_results[rf] = results_by_layer

        _save_lowrank_results(out_path, results_by_layer)

    return all_results


# ---------------------------------------------------------------------------
# § 2.4  Untrained 2-D network
# ---------------------------------------------------------------------------

def run_2d_untrained(
    network_type: int = NetworkType.ALEXNET,
    output_dir: str = "results",
    device: str = "cpu",
) -> None:
    """Capacity and geometry for untrained (epoch=0) network on 2-D manifolds."""
    run_2d_standard_capacity(
        network_type=network_type,
        n_objects=64,
        range_factors=[RANGE_FACTORS_2D[-1]],
        samples_list=[SAMPLES_2D[-1]],
        output_dir=output_dir,
        epoch=0,
        device=device,
    )
    run_2d_lowrank(
        network_type=network_type,
        n_objects=64,
        range_factors=[RANGE_FACTORS_2D[-1]],
        samples_list=[SAMPLES_2D[-1]],
        output_dir=output_dir,
        epoch=0,
        device=device,
    )


# ---------------------------------------------------------------------------
# Top-level runners
# ---------------------------------------------------------------------------

def run_all_1d(
    output_dir: str = "results",
    device: str = "cpu",
    networks: List[int] = None,
) -> None:
    """
    Run the complete 1-D manifold analysis pipeline from the paper.

    Order (matches MATLAB script):
      1. Standard capacity — AlexNet, ResNet50, VGG16
      2. Preprocessing variants — AlexNet only
      3. Untrained AlexNet
      4. Data-manipulation controls (D, center_norm_factor)
      5. Sample-count sweep
      6. Neuron-count sweep
      7. Object-count sweep
      8. Sphere controls

    Tip: run on a compute cluster, one range_factor at a time, using
    ``run_1d_standard_capacity(range_factors=[rf], ...)``.
    """
    if networks is None:
        networks = [NetworkType.ALEXNET, NetworkType.RESNET50, NetworkType.VGG16]

    # § 1.1 Standard capacity
    for nt in networks:
        run_1d_standard_capacity(network_type=nt, output_dir=output_dir, device=device)

    # § 1.3 Untrained
    run_1d_untrained(output_dir=output_dir, device=device)

    # § 1-D low-rank (prerequisite for preprocessing)
    run_1d_lowrank(output_dir=output_dir, device=device)

    # § 1.2 Preprocessing variants
    run_1d_preprocessing_variants(output_dir=output_dir, device=device)

    # § 1.4 Data-manipulation controls
    run_1d_dimension_manipulation(output_dir=output_dir, device=device)

    # § 1.5 Sample sweep
    run_1d_sample_sweep(output_dir=output_dir, device=device)

    # § 1.6 Neuron sweep
    run_1d_neuron_sweep(output_dir=output_dir, device=device)

    # § 1.7 Object sweep
    run_1d_object_sweep(output_dir=output_dir, device=device)

    # § 1.8 Sphere controls
    run_1d_sphere_controls(output_dir=output_dir, device=device)


def run_all_2d(
    output_dir: str = "results",
    device: str = "cpu",
    networks: List[int] = None,
) -> None:
    """
    Run the complete 2-D (random) manifold analysis pipeline from the paper.

    Order:
      1. Standard capacity — AlexNet, ResNet50, VGG16
      2. Preprocessing variants — AlexNet
      3. Low-rank geometry — AlexNet, ResNet50, VGG16
      4. Untrained AlexNet
    """
    if networks is None:
        networks = [NetworkType.ALEXNET, NetworkType.RESNET50, NetworkType.VGG16]

    # § 2.1 Standard capacity
    for nt in networks:
        run_2d_standard_capacity(network_type=nt, output_dir=output_dir, device=device)

    # § 2.2 Preprocessing variants
    run_2d_preprocessing_variants(output_dir=output_dir, device=device)

    # § 2.3 Low-rank geometry
    for nt in networks:
        run_2d_lowrank(network_type=nt, output_dir=output_dir, device=device)

    # § 2.4 Untrained
    run_2d_untrained(output_dir=output_dir, device=device)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reproduce NComms 2020 DNN manifold results."
    )
    parser.add_argument("--part", choices=["1d", "2d", "all"], default="all",
                        help="Which part to run (1d, 2d, or all)")
    parser.add_argument("--network", type=int, default=None,
                        help="Network type (1=AlexNet, 3=ResNet50, 5=VGG16). "
                             "Default: all three.")
    parser.add_argument("--output-dir", default="results",
                        help="Root directory for output files")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (cpu / cuda)")
    parser.add_argument("--range-factor", type=float, default=None,
                        help="Run only a single range_factor value")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override n_samples (requires --range-factor)")
    args = parser.parse_args()

    networks = [args.network] if args.network else None

    if args.range_factor is not None:
        # Single-factor mode
        n_smp = args.n_samples or SAMPLES_1D[RANGE_FACTORS_1D.index(args.range_factor)]
        log.info("Single range_factor=%.3f  n_samples=%d", args.range_factor, n_smp)
        if args.part in ("1d", "all"):
            run_1d_standard_capacity(
                network_type=args.network or NetworkType.ALEXNET,
                range_factors=[args.range_factor],
                samples_list=[n_smp],
                output_dir=args.output_dir,
                device=args.device,
            )
        if args.part in ("2d", "all"):
            run_2d_standard_capacity(
                network_type=args.network or NetworkType.ALEXNET,
                range_factors=[args.range_factor],
                samples_list=[SAMPLES_2D[RANGE_FACTORS_2D.index(args.range_factor)]],
                output_dir=args.output_dir,
                device=args.device,
            )
    else:
        if args.part == "1d":
            run_all_1d(output_dir=args.output_dir, device=args.device, networks=networks)
        elif args.part == "2d":
            run_all_2d(output_dir=args.output_dir, device=args.device, networks=networks)
        else:
            run_all_1d(output_dir=args.output_dir, device=args.device, networks=networks)
            run_all_2d(output_dir=args.output_dir, device=args.device, networks=networks)

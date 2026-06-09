"""
Classification capacity estimation.

Two approaches are provided:

1. **Direct (empirical)**  –  binary search for the minimum number of neurons N*
   such that > 50% of random binary dichotomies are linearly separable.

2. **Theoretical (MFT)**  –  use manifold geometry estimates to predict capacity
   via the mean-field theory formula.

Classes
-------
CapacityResults
    Container for capacity estimation outputs.
BinaryDichotomiesCapacity
    Direct binary-search capacity estimator.
HierarchicalCapacity
    Theoretical MFT capacity from manifold properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from .utils import sample_indices, sample_random_labels, assert_warn
from .separability import LinearSeparabilitySVM, LinearSeparabilityGeneralizationSVM
from .preprocessing import TuningFunctionPreprocessor
from .theory import theory_alpha0_cached
from .manifold_properties import ManifoldPropertiesIterative, ManifoldGeometry


@dataclass
class CapacityResults:
    """Output of the capacity estimator."""

    Nc: Optional[int] = None
    separability_results: np.ndarray = field(default_factory=lambda: np.array([]))
    Ns: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    n_neuron_samples_used: np.ndarray = field(default_factory=lambda: np.array([]))
    n_support_vectors: np.ndarray = field(default_factory=lambda: np.array([]))


class BinaryDichotomiesCapacity:
    """
    Find the critical neuron count N_c such that >= 50% of random binary
    dichotomies become linearly separable when N = N_c features are used.

    Algorithm: first scan in ``jumps`` to find an upper bound, then binary
    search to precision ``precision``.
    """

    def __init__(
        self,
        expected_precision: float = 0.05,
        verbose: bool = False,
        random_labeling_type: int = 0,
        precision: int = 1,
        max_samples: int = 0,
        global_preprocessing: int = 0,
        features_type: int = 0,
        jumps: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        expected_precision : float
            Target standard error for the separability fraction (default 0.05).
        verbose : bool
        random_labeling_type : int  0=IID, 1=balanced, 2=sparse
        precision : int
            Binary-search stopping criterion on N.
        max_samples : int
            If > 0 use generalisation SVM with this many samples.
        global_preprocessing : int
            Global preprocessing mode (see TuningFunctionPreprocessor).
        features_type : int
            0=sub-sample, 1=first-n, 2=random-projections.
        jumps : int, optional
            Step size for the initial scan.  Auto-determined if None.
        """
        self.expected_precision = expected_precision
        self.verbose = verbose
        self.random_labeling_type = random_labeling_type
        self.precision = precision
        self.max_samples = max_samples
        self.global_preprocessing = global_preprocessing
        self.features_type = features_type
        self.jumps = jumps

    def estimate(self, XsAll: np.ndarray) -> CapacityResults:
        """
        Parameters
        ----------
        XsAll : np.ndarray  (N_NEURONS, N_SAMPLES, N_OBJECTS)

        Returns
        -------
        CapacityResults
        """
        assert XsAll.ndim == 3, "Data must be (N_NEURONS, N_SAMPLES, N_OBJECTS)"
        n_neurons, n_samples, n_objects = XsAll.shape

        jumps = self.jumps
        if jumps is None:
            if n_neurons < 100:
                jumps = 10
            elif n_neurons < 1024:
                jumps = 100 if n_neurons % 100 == 0 else 128
            else:
                jumps = 500 if n_neurons % 500 == 0 else 512

        if self.verbose:
            print(f" {n_neurons} neurons {n_samples} conditions {n_objects} objects")

        separability = np.full(n_neurons, np.nan)
        n_samples_used = np.full(n_neurons, np.nan)
        n_sv = np.full((n_neurons, n_objects), np.nan)

        sampler = _DichotomiesSampler(
            expected_precision=self.expected_precision,
            random_labeling_type=self.random_labeling_type,
            max_samples=self.max_samples,
            global_preprocessing=self.global_preprocessing,
            features_type=self.features_type,
        )

        # Initial scan
        J = list(dict.fromkeys(
            list(range(jumps, n_neurons - n_neurons % jumps + 1, jumps)) + [n_neurons]
        ))
        for n in J:
            if np.isnan(separability[n - 1]):
                sep, nsv = sampler.evaluate(XsAll, n)
                n_samples_used[n - 1] = int(np.sum(np.isfinite(sep)))
                separability[n - 1] = float(np.nanmean(sep))
                n_sv[n - 1] = nsv
            if self.verbose:
                print(f"  N={n} {separability[n-1]:.2f} separable ({int(n_samples_used[n-1])} samples)")
            if separability[n - 1] == 1.0:
                break

        max_N = n
        zeros = np.where(separability == 0.0)[0]
        min_N = int(zeros[-1] + 1) if len(zeros) else 0

        # Binary search
        n1, n2 = None, None
        while True:
            min_val, min_idx = float(np.nanmin(separability)), int(np.nanargmin(separability))
            max_val, max_idx = float(np.nanmax(separability)), int(np.nanargmax(separability))

            if max_N - min_N <= self.precision:
                break

            n = int(np.ceil((max_N + min_N) / 2))
            if n == n1 or n == n2:
                break

            if np.isnan(separability[n - 1]):
                sep, nsv = sampler.evaluate(XsAll, n)
                n_samples_used[n - 1] = int(np.sum(np.isfinite(sep)))
                separability[n - 1] = float(np.nanmean(sep))
                n_sv[n - 1] = nsv
            if self.verbose:
                print(f"  N={n} [{min_N}-{max_N}] {separability[n-1]:.2f} separable")

            if separability[n - 1] > 0.5:
                max_N = n
            else:
                min_N = n
            n2, n1 = n1, n

        crit_idx = np.where(separability >= 0.5)[0]
        Nc = int(crit_idx[0] + 1) if len(crit_idx) else None
        if self.verbose:
            if Nc:
                print(f" Critical N={Nc} alpha={n_objects / Nc:.2f}")
            else:
                print(" Critical N not found")

        Ns = np.where(np.isfinite(separability))[0] + 1
        return CapacityResults(
            Nc=Nc,
            separability_results=separability,
            Ns=Ns,
            n_neuron_samples_used=n_samples_used,
            n_support_vectors=n_sv,
        )


class _DichotomiesSampler:
    """
    Internal helper: evaluate separability for a given N.
    """

    def __init__(
        self,
        expected_precision: float = 0.05,
        random_labeling_type: int = 0,
        max_samples: int = 0,
        global_preprocessing: int = 0,
        features_type: int = 0,
    ) -> None:
        self.expected_precision = expected_precision
        self.random_labeling_type = random_labeling_type
        self.max_samples = max_samples
        self.global_preprocessing = global_preprocessing
        self.features_type = features_type

    def evaluate(self, XsAll: np.ndarray, n: int):
        """Return (separability array, nsv_per_cluster array)."""
        n_neurons, n_samples, n_objects = XsAll.shape
        ep = self.expected_precision
        MIN_REPS = int(np.ceil(1 / ep))
        N_REPS = int(np.ceil((0.5 / ep) ** 2))
        N_DICHOTOMIES = 1
        tolerance = 1e-10
        max_iter = 1000

        preprocessor = TuningFunctionPreprocessor(self.global_preprocessing)
        rng = np.random.default_rng()

        if self.features_type == 0:
            features_used = sample_indices(n_neurons, n, N_REPS)
        elif self.features_type == 1:
            features_used = np.tile(np.arange(n), (1, 1))
        else:
            features_used = None   # random projections

        svm = LinearSeparabilitySVM(tolerance=tolerance, max_iterations=max_iter)
        gen_svm = LinearSeparabilityGeneralizationSVM(
            tolerance=tolerance, max_iterations=max_iter, max_samples=self.max_samples
        ) if self.max_samples > 0 else None

        separability = np.full((N_REPS, N_DICHOTOMIES), np.nan)
        nsv_out = np.zeros(n_objects)

        for r in range(N_REPS):
            if features_used is None:
                R_mat = rng.standard_normal((n_neurons, n)) / np.sqrt(n_neurons)
                Xr = R_mat.T @ XsAll.reshape(n_neurons, n_samples * n_objects)
                Xs = preprocessor(Xr.reshape(n, n_samples, n_objects))
            else:
                idx = features_used[r]
                Xs = preprocessor(XsAll[idx])

            X_flat = Xs.reshape(n, n_samples * n_objects)

            for i in range(N_DICHOTOMIES):
                y = sample_random_labels(n_objects, self.random_labeling_type)
                Y = np.tile(y, n_samples)

                if gen_svm is not None:
                    sep_flag, _, margin, used, _, sv_idx = gen_svm.check(Xs, y)
                    nsv_tmp = np.zeros(n_samples * n_objects)
                    nsv_tmp[sv_idx] = 1
                    nsv_out += nsv_tmp.reshape(n_samples, n_objects).sum(axis=0)
                    separability[r, i] = float(margin > 0)
                else:
                    result = svm.check(X_flat, Y)
                    nsv_tmp = np.zeros(n_samples * n_objects)
                    nsv_tmp[result.sv_indices] = 1
                    nsv_out += nsv_tmp.reshape(n_samples, n_objects).sum(axis=0)
                    separability[r, i] = float(result.margin > 0)

            # Early stopping if precision is met
            n_done = int(np.sum(np.isfinite(separability)))
            if n_done >= MIN_REPS:
                p = float(np.nansum(separability == 1) / n_done)
                q = 1.0 - p
                std_mean = np.sqrt(p * q / n_done)
                if std_mean <= ep:
                    break

        return separability, nsv_out


class HierarchicalCapacity:
    """
    Theoretical MFT capacity estimation from manifold geometry.

    For each manifold, compute its geometry (R_M, D_M, kappa_M) and combine
    via the mean-field theory capacity formula.
    """

    def __init__(
        self,
        n_random_projections: int = 1000,
        kappa: float = 0.0,
        n_repeats: int = 1,
    ) -> None:
        self.n_random_projections = n_random_projections
        self.kappa = kappa
        self.n_repeats = n_repeats

    def estimate(self, XsAll: np.ndarray) -> dict:
        """
        Parameters
        ----------
        XsAll : np.ndarray  (N_NEURONS, N_SAMPLES, N_OBJECTS)

        Returns
        -------
        dict with keys: 'alphac_hat', 'geom_list', 'mean_radius',
                        'mean_dimension', 'center_norms'
        """
        n_neurons, n_samples, n_objects = XsAll.shape
        geom_list: List[ManifoldGeometry] = []
        center_norms = np.zeros(n_objects)

        estimator = ManifoldPropertiesIterative(
            n_random_projections=self.n_random_projections,
            kappa=self.kappa,
        )

        for obj_i in range(n_objects):
            cF = XsAll[:, :, obj_i]
            center = np.mean(cF, axis=1, keepdims=True)
            cn = float(np.linalg.norm(center))
            center_norms[obj_i] = cn if cn > 0 else 1.0
            geom = estimator.compute(cF - center, center_norms[obj_i])
            geom_list.append(geom)

        alphac_values = np.array([g.alphac_hat for g in geom_list])
        alphac_hat = float(1.0 / np.mean(1.0 / alphac_values)) if np.all(alphac_values > 0) else np.nan

        return {
            "alphac_hat": alphac_hat,
            "geom_list": geom_list,
            "mean_radius": float(np.mean([g.mean_half_width2 for g in geom_list])),
            "mean_dimension": float(np.mean([g.effective_dimension for g in geom_list])),
            "center_norms": center_norms,
        }

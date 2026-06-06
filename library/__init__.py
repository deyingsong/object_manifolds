"""
library – core analysis functions for object-manifold geometry and capacity.

Modules
-------
utils           assert_warn, hostname, sample_indices, sample_random_labels
theory          theory_alpha0, theory_alpha0_cached, TheoryAlpha0
manifold_properties  ManifoldPropertiesIterative, ManifoldPropertiesLS,
                     ManifoldPropertiesLSCorr, ManifoldGeometry
low_rank        SquareCorrCoeffCost, ConstrainedLeastSquares,
                OptimalLowRankStructure
separability    LinearSeparabilitySVM, LinearSeparabilityGeneralizationSVM,
                SVMResult
preprocessing   TuningFunctionPreprocessor, LowDimensionalManifold
capacity        BinaryDichotomiesCapacity, HierarchicalCapacity, CapacityResults
"""

from .utils import assert_warn, hostname, sample_indices, sample_random_labels
from .theory import TheoryAlpha0, theory_alpha0, theory_alpha0_cached
from .manifold_properties import (
    ManifoldGeometry,
    ManifoldPropertiesIterative,
    ManifoldPropertiesLS,
    ManifoldPropertiesLSCorr,
)
from .low_rank import (
    SquareCorrCoeffCost,
    ConstrainedLeastSquares,
    OptimalLowRankStructure,
)
from .separability import (
    SVMResult,
    LinearSeparabilitySVM,
    LinearSeparabilityGeneralizationSVM,
)
from .preprocessing import TuningFunctionPreprocessor, LowDimensionalManifold
from .capacity import BinaryDichotomiesCapacity, HierarchicalCapacity, CapacityResults

__all__ = [
    "assert_warn", "hostname", "sample_indices", "sample_random_labels",
    "TheoryAlpha0", "theory_alpha0", "theory_alpha0_cached",
    "ManifoldGeometry",
    "ManifoldPropertiesIterative", "ManifoldPropertiesLS", "ManifoldPropertiesLSCorr",
    "SquareCorrCoeffCost", "ConstrainedLeastSquares", "OptimalLowRankStructure",
    "SVMResult", "LinearSeparabilitySVM", "LinearSeparabilityGeneralizationSVM",
    "TuningFunctionPreprocessor", "LowDimensionalManifold",
    "BinaryDichotomiesCapacity", "HierarchicalCapacity", "CapacityResults",
]

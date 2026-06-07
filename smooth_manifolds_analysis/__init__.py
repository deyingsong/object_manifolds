"""
smooth_manifolds_analysis – high-level capacity and geometry analysis for DNN manifolds.

Imports the core library (library/) and adds:
  - capacity_analysis.py   OneDimensionalCapacityAnalysis, RandomChangeCapacityAnalysis
  - low_rank_analysis.py   CovarianceLowRankAnalysis

All analysis utilities from the library are re-exported for convenience.
"""

# Re-export everything from the shared library
from library import (
    assert_warn, hostname, sample_indices, sample_random_labels,
    TheoryAlpha0, theory_alpha0, theory_alpha0_cached,
    ManifoldGeometry,
    ManifoldPropertiesIterative, ManifoldPropertiesLS, ManifoldPropertiesLSCorr,
    SquareCorrCoeffCost, ConstrainedLeastSquares, OptimalLowRankStructure,
    SVMResult, LinearSeparabilitySVM, LinearSeparabilityGeneralizationSVM,
    TuningFunctionPreprocessor, LowDimensionalManifold,
    BinaryDichotomiesCapacity, HierarchicalCapacity, CapacityResults,
)

from .capacity_analysis import (
    CapacityAnalysisConfig,
    LayerCapacityResults,
    OneDimensionalCapacityAnalysis,
    RandomChangeCapacityAnalysis,
    DIRECTION_NAMES,
)
from .low_rank_analysis import (
    LowRankAnalysisConfig,
    LowRankResults,
    CovarianceLowRankAnalysis,
)

__all__ = [
    # utils
    "assert_warn", "hostname", "sample_indices", "sample_random_labels",
    # theory
    "TheoryAlpha0", "theory_alpha0", "theory_alpha0_cached",
    # geometry
    "ManifoldGeometry",
    "ManifoldPropertiesIterative", "ManifoldPropertiesLS", "ManifoldPropertiesLSCorr",
    # low rank
    "SquareCorrCoeffCost", "ConstrainedLeastSquares", "OptimalLowRankStructure",
    # separability
    "SVMResult", "LinearSeparabilitySVM", "LinearSeparabilityGeneralizationSVM",
    # preprocessing
    "TuningFunctionPreprocessor", "LowDimensionalManifold",
    # capacity
    "BinaryDichotomiesCapacity", "HierarchicalCapacity", "CapacityResults",
    # analysis
    "CapacityAnalysisConfig", "LayerCapacityResults",
    "OneDimensionalCapacityAnalysis", "RandomChangeCapacityAnalysis", "DIRECTION_NAMES",
    "LowRankAnalysisConfig", "LowRankResults", "CovarianceLowRankAnalysis",
]

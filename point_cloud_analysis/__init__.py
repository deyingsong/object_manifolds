"""
point_cloud_analysis – MFT capacity for general point-cloud manifolds.

Accounts for center-center correlations via a factor-analysis step.
"""

from .manifold_analysis import (
    ManifoldStableAnalysisCorr,
    ManifoldAnalysisOutput,
    AnalysisOptions,
)

__all__ = [
    "ManifoldStableAnalysisCorr",
    "ManifoldAnalysisOutput",
    "AnalysisOptions",
]

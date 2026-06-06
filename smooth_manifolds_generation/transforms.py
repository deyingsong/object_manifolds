"""
Affine transformation utilities for generating image manifolds.

Classes
-------
ImageNetConfig
    Global image/frame size configuration (replaces MATLAB globals).
AffineTransform
    Dataclass wrapping a 3x3 affine matrix.
TransformType
    Enum for the 7 transformation directions.
AffineTransformFactory
    Build 1-D or random 2-D affine transforms from scalar parameters.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np


@dataclass
class ImageNetConfig:
    """Global image geometry parameters."""

    image_size: int = 64
    frame_size: int = 80
    object_size: int = 40

    @property
    def input_scale(self) -> float:
        return self.frame_size / self.object_size

    @property
    def output_scale(self) -> float:
        return self.image_size / self.object_size


class TransformType(enum.IntEnum):
    TRANSLATE_X = 1
    TRANSLATE_Y = 2
    SCALE_X = 3
    SCALE_Y = 4
    SHEAR_X = 5
    SHEAR_Y = 6
    ROTATION = 7


@dataclass
class AffineTransform:
    """Affine 2-D transformation represented as a 3x3 homogeneous matrix.

    Convention (matches MATLAB affine2d / imwarp):
        [x', y', 1] = [x, y, 1] @ T
    so T has shape (3, 3) with T[:, 2] == [0, 0, 1].
    """

    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        assert self.matrix.shape == (3, 3)
        np.testing.assert_allclose(self.matrix[:, 2], [0, 0, 1], atol=1e-10)

    @property
    def T(self) -> np.ndarray:
        return self.matrix

    def magnitude(self) -> float:
        """Affine transform magnitude (pixel RMS displacement at corners)."""
        corners = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
        v = self.matrix[:2, :2].ravel() - np.array([1, 0, 0, 1])
        tx, ty = self.matrix[2, 0], self.matrix[2, 1]
        disp_x = corners @ v[:2] + tx
        disp_y = corners @ v[2:] + ty
        return float(np.sqrt(np.mean(disp_x**2) + np.mean(disp_y**2)))


class AffineTransformFactory:
    """
    Build 1-D affine transforms for a single degree of freedom,
    or random 2-D transforms.
    """

    def __init__(self, config: ImageNetConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Parameter ranges (based on image size)
    # ------------------------------------------------------------------

    def _get_ranges(self, scale_factor: float, n_samples: int) -> dict:
        s = scale_factor
        sz = self.config.image_size
        if sz == 64:
            return {
                TransformType.TRANSLATE_X: np.linspace(-1, 1, n_samples) * s,
                TransformType.TRANSLATE_Y: np.linspace(-1, 1, n_samples) * s,
                TransformType.SCALE_X: np.exp(np.linspace(-0.018, 0.018, n_samples) * s),
                TransformType.SCALE_Y: np.exp(np.linspace(-0.018, 0.018, n_samples) * s),
                TransformType.SHEAR_X: np.linspace(-1, 1, n_samples) / 52 * s,
                TransformType.SHEAR_Y: np.linspace(-1, 1, n_samples) / 52 * s,
                TransformType.ROTATION: np.linspace(-1.5 * np.pi / 360, 1.5 * np.pi / 360, n_samples) * s,
            }
        elif sz == 224:
            return {
                TransformType.TRANSLATE_X: np.linspace(-1, 1, n_samples) * s,
                TransformType.TRANSLATE_Y: np.linspace(-1, 1, n_samples) * s,
                TransformType.SCALE_X: np.exp(np.linspace(-0.01, 0.01, n_samples) * 0.58 * s),
                TransformType.SCALE_Y: np.exp(np.linspace(-0.01, 0.01, n_samples) * 0.52 * s),
                TransformType.SHEAR_X: np.linspace(-1, 1, n_samples) / 175 * s,
                TransformType.SHEAR_Y: np.linspace(-1, 1, n_samples) / 175 * s,
                TransformType.ROTATION: np.linspace(-np.pi / 360, np.pi / 360, n_samples) * 0.45 * s,
            }
        else:  # 32, 128 and default
            return {
                TransformType.TRANSLATE_X: np.linspace(-1, 1, n_samples) * s,
                TransformType.TRANSLATE_Y: np.linspace(-1, 1, n_samples) * s,
                TransformType.SCALE_X: np.exp(np.linspace(-0.018, 0.018, n_samples) * s),
                TransformType.SCALE_Y: np.exp(np.linspace(-0.018, 0.018, n_samples) * s),
                TransformType.SHEAR_X: np.linspace(-1, 1, n_samples) / 52 * s,
                TransformType.SHEAR_Y: np.linspace(-1, 1, n_samples) / 52 * s,
                TransformType.ROTATION: np.linspace(-1.5 * np.pi / 360, 1.5 * np.pi / 360, n_samples) * s,
            }

    def create_1d(
        self, range_factor: float, param_id: int, j: int, n_samples: int
    ) -> AffineTransform:
        """
        Create a 1-D affine transform at sample index j.

        Parameters
        ----------
        range_factor : float
        param_id : int  (1-7 matching TransformType)
        j : int  zero-based sample index
        n_samples : int
        """
        scale_factor = range_factor / (self.config.object_size / 2.0)
        ranges = self._get_ranges(scale_factor, n_samples)
        t_type = TransformType(param_id)
        param_val = float(ranges[t_type][j])
        return self._make_from_type(t_type, param_val)

    def create_random(
        self,
        range_factor: float,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> AffineTransform:
        """Create a random 2-D affine transform from two random directions."""
        if rng is None:
            rng = np.random.default_rng()
        scale_factor = range_factor / (self.config.object_size / 2.0)
        ranges = self._get_ranges(scale_factor, n_samples)
        # Pick two random transformation types and magnitudes
        types = list(TransformType)
        t1, t2 = rng.choice(len(types), size=2, replace=False)
        t_type1, t_type2 = types[t1], types[t2]
        v1 = float(rng.choice(ranges[t_type1]))
        v2 = float(rng.choice(ranges[t_type2]))
        M1 = self._make_from_type(t_type1, v1).matrix
        M2 = self._make_from_type(t_type2, v2).matrix
        return AffineTransform(M1 @ M2)

    @staticmethod
    def _make_from_type(t_type: TransformType, value: float) -> AffineTransform:
        M = np.eye(3)
        if t_type == TransformType.TRANSLATE_X:
            M[2, 0] = value
        elif t_type == TransformType.TRANSLATE_Y:
            M[2, 1] = value
        elif t_type == TransformType.SCALE_X:
            M[0, 0] = value
        elif t_type == TransformType.SCALE_Y:
            M[1, 1] = value
        elif t_type == TransformType.SHEAR_X:
            M[1, 0] = value
        elif t_type == TransformType.SHEAR_Y:
            M[0, 1] = value
        elif t_type == TransformType.ROTATION:
            c, s = np.cos(value), np.sin(value)
            M[0, 0], M[0, 1] = c, s
            M[1, 0], M[1, 1] = -s, c
        return AffineTransform(M)

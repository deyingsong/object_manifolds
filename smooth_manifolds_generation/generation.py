"""
Manifold generation: forward-pass images through a DNN under affine transforms.

Two generation modes:
  1. One-dimensional change  –  one transformation type, sweeping a single
     parameter  (generate_convnet_one_dimensional_change).
  2. Random change  –  two random transformation types applied jointly
     (generate_convnet_random_change / generate_convnet_random_change2).

Output tuning-function shape:
  (N_DIRECTIONS, N_OBJECTS, N_SAMPLES, N_NEURONS)
which is then transposed to (N_NEURONS, N_SAMPLES, N_OBJECTS) per manifold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .transforms import AffineTransformFactory, ImageNetConfig, TransformType
from .network import ConvNetExtractor, load_network_metadata, NetworkType
from .imagenet import ImageNetState


@dataclass
class GenerationConfig:
    """Common configuration for manifold generation."""

    n_objects: int = 100
    n_samples: int = 11
    range_factor: float = 1.0
    network_type: int = 1
    layers_grouping_level: int = 0
    random_seed: int = 0
    output_dir: str = "."
    device: str = "cpu"


class OneDimensionalManifoldGenerator:
    """
    Generate DNN response manifolds by sweeping one affine-transform parameter.

    Corresponds to ``generate_convnet_one_dimensional_change.m``.

    The output tuning function has shape:
      (N_DIRECTIONS, N_OBJECTS, N_SAMPLES, N_NEURONS)
    for N_DIRECTIONS = 7 transformation types.
    """

    DIRECTION_NAMES = [
        "x-translation", "y-translation",
        "x-scale", "y-scale",
        "x-shear", "y-shear",
        "rotation",
    ]

    def __init__(
        self,
        config: GenerationConfig,
        imagenet_state: ImageNetState,
    ) -> None:
        self.config = config
        self.imagenet = imagenet_state
        self.img_config = ImageNetConfig(
            image_size=imagenet_state.image_size,
            frame_size=imagenet_state.frame_size,
            object_size=imagenet_state.object_size,
        )
        self.factory = AffineTransformFactory(self.img_config)

    def generate(
        self,
        template_images: np.ndarray,
        layer_name: str,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        template_images : np.ndarray  (N_OBJECTS, H, W, 3)  uint8
        layer_name : str

        Returns
        -------
        np.ndarray  (N_DIRECTIONS, N_OBJECTS, N_SAMPLES, N_FEATURES)
        """
        n_dirs = len(self.DIRECTION_NAMES)
        n_objects, H, W, C = template_images.shape
        n_samples = self.config.n_samples

        # Determine feature dimension by a trial forward pass
        with ConvNetExtractor(
            self.config.network_type,
            layer_names=[layer_name],
            device=self.config.device,
        ) as extractor:
            trial = extractor.extract(template_images[:1])
            n_features = trial.get(layer_name, trial.get("output")).shape[1]

            tuning = np.zeros(
                (n_dirs, n_objects, n_samples, n_features), dtype=np.float32
            )

            for d, direction_name in enumerate(self.DIRECTION_NAMES):
                param_id = d + 1
                for j in range(n_samples):
                    transform = self.factory.create_1d(
                        self.config.range_factor, param_id, j, n_samples
                    )
                    warped = self._warp_images(template_images, transform)
                    feats = extractor.extract(warped)
                    feat = feats.get(layer_name, feats.get("output"))
                    tuning[d, :, j, :] = feat

        return tuning

    def _warp_images(
        self, images: np.ndarray, transform
    ) -> np.ndarray:
        """Apply an affine transform to a batch of images."""
        from PIL import Image

        out_size = self.img_config.image_size
        M = transform.matrix
        warped = np.zeros_like(images)
        for i, img in enumerate(images):
            pil = Image.fromarray(img)
            # affine coefficients for PIL: (a,b,c,d,e,f) meaning
            # x_src = a*x + b*y + c,  y_src = d*x + e*y + f
            a, b, c = M[0, 0], M[1, 0], M[2, 0]
            d_c, e, f = M[0, 1], M[1, 1], M[2, 1]
            pil_warped = pil.transform(
                (out_size, out_size),
                Image.AFFINE,
                (a, b, c, d_c, e, f),
                resample=Image.BICUBIC,
            )
            warped[i] = np.array(pil_warped)
        return warped


class RandomManifoldGenerator:
    """
    Generate DNN response manifolds with random pairs of affine transforms.

    Corresponds to ``generate_convnet_random_change2.m``.

    Output shape: (N_DIRECTIONS=2, N_OBJECTS, N_SAMPLES, N_FEATURES)
    where direction 0 = first transform, direction 1 = second transform.
    """

    def __init__(
        self,
        config: GenerationConfig,
        imagenet_state: ImageNetState,
        degrees_of_freedom: int = 2,
    ) -> None:
        self.config = config
        self.imagenet = imagenet_state
        self.dof = degrees_of_freedom
        self.img_config = ImageNetConfig(
            image_size=imagenet_state.image_size,
            frame_size=imagenet_state.frame_size,
            object_size=imagenet_state.object_size,
        )
        self.factory = AffineTransformFactory(self.img_config)

    def generate(
        self,
        template_images: np.ndarray,
        layer_name: str,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        template_images : np.ndarray  (N_OBJECTS, H, W, 3)
        layer_name : str

        Returns
        -------
        np.ndarray  (2, N_OBJECTS, N_SAMPLES, N_FEATURES)
        """
        rng = np.random.default_rng(self.config.random_seed)
        n_objects = len(template_images)
        n_samples = self.config.n_samples

        with ConvNetExtractor(
            self.config.network_type,
            layer_names=[layer_name],
            device=self.config.device,
        ) as extractor:
            trial = extractor.extract(template_images[:1])
            n_features = trial.get(layer_name, trial.get("output")).shape[1]

            tuning = np.zeros(
                (self.dof, n_objects, n_samples, n_features), dtype=np.float32
            )

            for d in range(self.dof):
                for j in range(n_samples):
                    transform = self.factory.create_random(
                        self.config.range_factor, n_samples, rng
                    )
                    warped = self._warp_images(template_images, transform)
                    feats = extractor.extract(warped)
                    feat = feats.get(layer_name, feats.get("output"))
                    tuning[d, :, j, :] = feat

        return tuning

    def _warp_images(self, images: np.ndarray, transform) -> np.ndarray:
        from PIL import Image
        out_size = self.img_config.image_size
        M = transform.matrix
        warped = np.zeros_like(images)
        for i, img in enumerate(images):
            pil = Image.fromarray(img)
            a, b, c = M[0, 0], M[1, 0], M[2, 0]
            d_c, e, f = M[0, 1], M[1, 1], M[2, 1]
            pil_warped = pil.transform(
                (out_size, out_size),
                Image.AFFINE,
                (a, b, c, d_c, e, f),
                resample=Image.BICUBIC,
            )
            warped[i] = np.array(pil_warped)
        return warped

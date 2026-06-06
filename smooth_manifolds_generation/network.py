"""
Neural network initialisation and metadata.

Replaces the MATLAB MatConvNet utilities with PyTorch equivalents.

Classes
-------
NetworkType
    Enum for supported architectures.
NetworkMetadata
    Layer names, grouping, and feature dimensions.
ConvNetExtractor
    PyTorch-based forward-pass with intermediate layer extraction.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class NetworkType(enum.IntEnum):
    ALEXNET = 1
    GOOGLENET = 2
    RESNET50 = 3
    RESNET18 = 4
    VGG16 = 5
    VGG_FACE = 6


@dataclass
class NetworkMetadata:
    """
    Metadata describing which layers to extract features from.

    Corresponds to the output of ``load_network_metadata.m``.
    """

    network_name: str
    n_layers: int
    layer_names: List[str]
    n_features: List[int]
    enabled_layers: List[int] = field(default_factory=list)
    grouping_level: int = 0
    epoch: Optional[int] = None


# ---------------------------------------------------------------------------
# Predefined metadata for common architectures
# ---------------------------------------------------------------------------

def _alexnet_metadata(grouping_level: int = 0) -> NetworkMetadata:
    if grouping_level == 0:
        names = ["conv1", "relu1", "pool1",
                 "conv2", "relu2", "pool2",
                 "conv3", "relu3",
                 "conv4", "relu4",
                 "conv5", "relu5", "pool5",
                 "fc6", "relu6",
                 "fc7", "relu7",
                 "fc8", "prob"]
        features = [96 * 27 * 27, 96 * 27 * 27, 96 * 13 * 13,
                    256 * 13 * 13, 256 * 13 * 13, 256 * 6 * 6,
                    384 * 6 * 6, 384 * 6 * 6,
                    384 * 6 * 6, 384 * 6 * 6,
                    256 * 6 * 6, 256 * 6 * 6, 256 * 6 * 6,
                    4096, 4096, 4096, 4096, 1000, 1000]
    else:
        names = ["pool1", "pool2", "relu3", "relu4", "pool5", "relu6", "relu7", "prob"]
        features = [96 * 13 * 13, 256 * 6 * 6, 384 * 6 * 6,
                    384 * 6 * 6, 256 * 6 * 6, 4096, 4096, 1000]
    return NetworkMetadata(
        network_name="alexnet",
        n_layers=len(names),
        layer_names=names,
        n_features=features,
        enabled_layers=list(range(len(names))),
        grouping_level=grouping_level,
    )


def _resnet50_metadata(grouping_level: int = 2) -> NetworkMetadata:
    names = ["conv1", "pool1", "res2a", "res2b", "res2c",
             "res3a", "res3b", "res3c", "res3d",
             "res4a", "res4b", "res4c", "res4d", "res4e", "res4f",
             "res5a", "res5b", "res5c", "pool5", "fc1000"]
    features = [64 * 112 * 112, 64 * 56 * 56] + [256 * 56 * 56] * 3 + \
               [512 * 28 * 28] * 4 + [1024 * 14 * 14] * 6 + \
               [2048 * 7 * 7] * 3 + [2048, 1000]
    return NetworkMetadata(
        network_name="resnet50",
        n_layers=len(names),
        layer_names=names,
        n_features=features,
        enabled_layers=list(range(len(names))),
        grouping_level=grouping_level,
    )


_METADATA_REGISTRY: Dict[int, NetworkMetadata] = {}


def load_network_metadata(
    network_type: int,
    grouping_level: int = 0,
    epoch: Optional[int] = None,
    seed: int = 0,
) -> NetworkMetadata:
    """
    Return metadata for the given network type.

    Parameters
    ----------
    network_type : int  (see NetworkType)
    grouping_level : int
    epoch : int, optional
    seed : int  (random seed, not used by metadata)
    """
    key = (network_type, grouping_level)
    if key in _METADATA_REGISTRY:
        return _METADATA_REGISTRY[key]

    nt = NetworkType(network_type)
    if nt == NetworkType.ALEXNET:
        meta = _alexnet_metadata(grouping_level)
    elif nt in (NetworkType.RESNET50, NetworkType.RESNET18):
        meta = _resnet50_metadata(grouping_level)
    else:
        # Fallback: single-layer placeholder
        meta = NetworkMetadata(
            network_name=nt.name.lower(),
            n_layers=1,
            layer_names=["features"],
            n_features=[4096],
            enabled_layers=[0],
            grouping_level=grouping_level,
        )
    if epoch is not None:
        meta.epoch = epoch

    _METADATA_REGISTRY[key] = meta
    return meta


class ConvNetExtractor:
    """
    Extract intermediate feature representations from a PyTorch model.

    Corresponds to the forward-pass logic inside
    ``generate_convnet_one_dimensional_change.m`` and
    ``generate_convnet_random_change.m``.
    """

    def __init__(
        self,
        network_type: int,
        layer_names: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        network_type : int
        layer_names : list of str, optional
            Which layers to extract.  If None, extract the last layer.
        device : str
            'cpu' or 'cuda'.
        """
        self.network_type = NetworkType(network_type)
        self.device = device
        self._model = None
        self._hooks: Dict[str, np.ndarray] = {}
        self._handles = []
        self._layer_names = layer_names or []
        self._metadata = load_network_metadata(network_type)

    def _load_model(self) -> None:
        """Lazily load the PyTorch model."""
        import torch
        import torchvision.models as models

        nt = self.network_type
        if nt == NetworkType.ALEXNET:
            model = models.alexnet(pretrained=True)
        elif nt == NetworkType.RESNET50:
            model = models.resnet50(pretrained=True)
        elif nt == NetworkType.RESNET18:
            model = models.resnet18(pretrained=True)
        elif nt == NetworkType.VGG16:
            model = models.vgg16(pretrained=True)
        elif nt == NetworkType.GOOGLENET:
            model = models.googlenet(pretrained=True)
        else:
            raise ValueError(f"Unsupported network type: {nt}")
        model.eval().to(self.device)
        self._model = model

    def _register_hooks(self) -> None:
        """Register forward hooks for intermediate layers."""
        import torch.nn as nn

        def make_hook(name: str):
            def hook(module, inp, output):
                self._hooks[name] = output.detach().cpu().numpy()
            return hook

        for name, module in self._model.named_modules():
            if name in self._layer_names:
                self._handles.append(module.register_forward_hook(make_hook(name)))

    def extract(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward-pass and extract activations.

        Parameters
        ----------
        images : np.ndarray  (B, H, W, 3) uint8 or float32 in [0, 255]

        Returns
        -------
        dict mapping layer_name -> np.ndarray of shape (B, N_FEATURES)
        """
        import torch
        import torchvision.transforms.functional as TF

        if self._model is None:
            self._load_model()
            self._register_hooks()

        # Preprocess: convert to float, normalise for ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        imgs = images.astype(np.float32) / 255.0
        imgs = (imgs - mean) / std
        tensor = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).to(self.device)

        self._hooks.clear()
        with torch.no_grad():
            out = self._model(tensor)
            self._hooks["output"] = out.detach().cpu().numpy()

        return {k: v.reshape(len(images), -1) for k, v in self._hooks.items()}

    def close(self) -> None:
        """Remove forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> ConvNetExtractor:
        return self

    def __exit__(self, *args) -> None:
        self.close()

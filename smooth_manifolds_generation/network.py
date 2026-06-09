"""
Neural network initialisation and metadata.

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

    Images are expected as ``(B, H, W, 3)`` float32 in the [0, 255] range
    (i.e. the raw warped output of ``_calc_imagenet_warp``).  They are
    automatically resized to the model's canonical input resolution and
    normalised with ImageNet mean/std before the forward pass.

    Feature sub-selection mirrors the MATLAB ``layer_indices`` mechanism:
    when a layer has more than ``n_features`` units, ``n_features`` indices
    are drawn once (with ``feature_seed``) and kept fixed across all calls.
    """

    # ImageNet normalisation constants (PyTorch convention)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        network_type: int,
        layer_names: Optional[List[str]] = None,
        device: str = "cpu",
        n_features: int = 4096,
        feature_seed: int = 1,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        network_type : int
        layer_names : list of str, optional
            Which layers to extract.  If None, extracts the final output.
        device : str
        n_features : int
            Maximum features per layer (``N_HMAX_FEATURES`` in MATLAB).
        feature_seed : int
            RNG seed for reproducible random feature sub-selection.
        epoch : int, optional
            Training epoch checkpoint to load (None = fully trained, 0 = random init).
        """
        self.network_type = NetworkType(network_type)
        self.device = device
        self.n_features = n_features
        self.feature_seed = feature_seed
        self.epoch = epoch
        self._model = None
        self._hooks: Dict[str, np.ndarray] = {}
        self._handles: list = []
        self._layer_names: List[str] = layer_names or []
        self._metadata = load_network_metadata(network_type, epoch=epoch)
        self._feat_idx: Dict[str, Optional[np.ndarray]] = {}  # feature sub-selection

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily load the pretrained PyTorch model."""
        import torchvision.models as models

        weights_map = {
            NetworkType.ALEXNET:  ("alexnet",   "AlexNet_Weights.IMAGENET1K_V1"),
            NetworkType.RESNET50: ("resnet50",  "ResNet50_Weights.IMAGENET1K_V1"),
            NetworkType.RESNET18: ("resnet18",  "ResNet18_Weights.IMAGENET1K_V1"),
            NetworkType.VGG16:    ("vgg16",     "VGG16_Weights.IMAGENET1K_V1"),
            NetworkType.GOOGLENET:("googlenet", "GoogLeNet_Weights.IMAGENET1K_V1"),
        }
        nt = self.network_type
        if nt not in weights_map:
            raise ValueError(f"Unsupported network type: {nt}")
        fn_name, weights_attr = weights_map[nt]
        fn = getattr(models, fn_name)
        if self.epoch == 0:
            # Untrained (randomly initialised) network — no pretrained weights
            model = fn(weights=None)
        else:
            # Use new weights API (torchvision ≥ 0.13); fall back for older versions
            try:
                weights_cls = getattr(models, weights_attr.split(".")[0])
                weights = getattr(weights_cls, weights_attr.split(".")[1])
                model = fn(weights=weights)
            except AttributeError:
                import warnings
                warnings.warn(f"Falling back to pretrained=True for {fn_name}.")
                model = fn(pretrained=True)  # noqa: deprecated

        model.eval().to(self.device)
        self._model = model
        self._input_size = 224  # canonical input for all torchvision models above

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward hooks and compute feature-selection indices."""
        rng = np.random.default_rng(self.feature_seed)

        def make_hook(name: str):
            def hook(module, inp, output):
                arr = output.detach().cpu().numpy()
                self._hooks[name] = arr
            return hook

        for name, module in self._model.named_modules():
            if name in self._layer_names:
                self._handles.append(
                    module.register_forward_hook(make_hook(name))
                )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def extract(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward-pass and extract activations.

        Parameters
        ----------
        images : np.ndarray  (B, H, W, 3)  float32 in [0, 255]

        Returns
        -------
        dict mapping layer_name → np.ndarray of shape (B, n_feat)
        """
        import torch
        from PIL import Image as _PIL

        if self._model is None:
            self._load_model()
            self._register_hooks()

        B, H, W, C = images.shape

        # --- Resize → normalise → tensor --------------------------------
        target = self._input_size
        if H != target or W != target:
            resized = np.stack([
                np.array(
                    _PIL.fromarray(images[b].astype(np.uint8)).resize(
                        (target, target), _PIL.BICUBIC
                    ),
                    dtype=np.float32,
                )
                for b in range(B)
            ])
        else:
            resized = images.astype(np.float32)

        imgs = resized / 255.0
        imgs = (imgs - self._MEAN) / self._STD
        # (B, H, W, 3) → (B, 3, H, W)
        tensor = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).to(self.device)

        # --- Forward pass -----------------------------------------------
        self._hooks.clear()
        with torch.no_grad():
            out = self._model(tensor)
            self._hooks["output"] = out.detach().cpu().numpy()

        # --- Flatten + sub-select features ------------------------------
        result: Dict[str, np.ndarray] = {}
        for name, raw in self._hooks.items():
            flat = raw.reshape(B, -1)   # (B, total_features)
            total = flat.shape[1]
            if name not in self._feat_idx:
                if total > self.n_features:
                    rng = np.random.default_rng(self.feature_seed)
                    self._feat_idx[name] = rng.choice(total, size=self.n_features, replace=False)
                else:
                    self._feat_idx[name] = None  # use all
            idx = self._feat_idx[name]
            result[name] = flat[:, idx] if idx is not None else flat

        return result

    def close(self) -> None:
        """Remove forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> ConvNetExtractor:
        return self

    def __exit__(self, *args) -> None:
        self.close()

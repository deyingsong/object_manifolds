"""
smooth_manifolds_generation – generate DNN tuning-function manifolds.

Modules
-------
imagenet        ImageNetState, init_imagenet, read_imagenet_*
transforms      AffineTransform, AffineTransformFactory, TransformType,
                ImageNetConfig
network         ConvNetExtractor, NetworkMetadata, NetworkType,
                load_network_metadata
generation      OneDimensionalManifoldGenerator, RandomManifoldGenerator,
                GenerationConfig
"""

from .imagenet import (
    ImageNetState,
    init_imagenet,
    get_state,
    read_imagenet_labels,
    read_imagenet_thumbnails,
    read_imagenet_training_size,
    choose_imagenet_template_images,
    sample_indices_one_per_category,
)
from .transforms import (
    ImageNetConfig,
    AffineTransform,
    AffineTransformFactory,
    TransformType,
)
from .network import (
    NetworkType,
    NetworkMetadata,
    ConvNetExtractor,
    load_network_metadata,
)
from .generation import (
    GenerationConfig,
    OneDimensionalManifoldGenerator,
    RandomManifoldGenerator,
)

__all__ = [
    "ImageNetState", "init_imagenet", "get_state",
    "read_imagenet_labels", "read_imagenet_thumbnails",
    "read_imagenet_training_size",
    "choose_imagenet_template_images", "sample_indices_one_per_category",
    "ImageNetConfig", "AffineTransform", "AffineTransformFactory", "TransformType",
    "NetworkType", "NetworkMetadata", "ConvNetExtractor", "load_network_metadata",
    "GenerationConfig", "OneDimensionalManifoldGenerator", "RandomManifoldGenerator",
]

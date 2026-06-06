"""
Network metadata loader module.

Loads and manages network metadata including layer information and structure.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

import numpy as np
import scipy.io as sio

logger = logging.getLogger(__name__)


class NetworkMetadata:
    """Container for network metadata."""
    
    NETWORK_TYPES = {
        0: 'hmax',
        1: 'alexnet',
        2: 'googlenet',
        3: 'resnet50',
        4: 'resnet18',
        5: 'vgg16',
        6: 'vggface'
    }
    
    TYPE_MARKERS = {
        'full': ['o', '>', 'd', 's', '^', 'V', '<', '*', 'h', 'p', '^', 'p', '<', '^', '^', '<', '^'],
        'merged_relu': ['o', '>', 'd', 's', 's', 'V', '<', '*', 'h', 'h', '^', 'p', '<', '<', 'V', '<', '<']
    }
    
    TYPE_MARKER_NAMES = [
        'Input',
        'Max Pooling',
        'Average Pooling',
        'Convolution',
        'ReLU (after Conv)',
        'LRN',
        'Concat',
        'SoftMax',
        'FC',
        'ReLU (after FC)',
        'Gabor',
        'RBF',
        'Sum',
        'ReLU (after Sum)',
        'ReLU (after LRN)',
        'Downsample',
        'ReLU (after Downsample)'
    ]
    
    def __init__(self, network_type: int, layers_grouping_level: int = 0,
                 epoch: Optional[int] = None, seed: int = 0,
                 imagenet_image_size: int = 64):
        """
        Initialize network metadata.
        
        Parameters
        ----------
        network_type : int
            Type of network (0-6)
        layers_grouping_level : int, optional
            Level of layer grouping (default: 0)
        epoch : int, optional
            Epoch number for specific model version (default: None)
        seed : int, optional
            Random seed for model (default: 0)
        imagenet_image_size : int, optional
            ImageNet image size in pixels (default: 64)
        """
        self.network_type = network_type
        self.layers_grouping_level = layers_grouping_level
        self.epoch = epoch
        self.seed = seed
        self.imagenet_image_size = imagenet_image_size
        
        # Metadata will be loaded from MAT file
        self.network_name = self.NETWORK_TYPES.get(network_type)
        if not self.network_name:
            raise ValueError(f"Unknown network type: {network_type}")
        
        # Initialize metadata containers
        self.layer_names = []
        self.layer_types = []
        self.layer_parent_ids = []
        self.layer_dimensions = []
        self.layer_indices = []
        self.layer_sizes = []
        self.layer_type_markers = []
        self.N_LAYERS = 0
        self.ACTIVE_LAYERS = []
        self.parent_ids = []
        self.active_type_markers_names = []
        self.layers = []
        self.all_type_marker_names = []
    
    def load_from_file(self, metadata_file: Path) -> None:
        """
        Load metadata from MATLAB MAT file.
        
        Parameters
        ----------
        metadata_file : Path
            Path to the metadata MAT file
        """
        try:
            metadata = sio.loadmat(str(metadata_file))
            
            self.layer_names = [str(name[0]) if isinstance(name, np.ndarray) else str(name)
                               for name in metadata.get('layer_names', [])]
            self.layer_types = [str(ltype[0]) if isinstance(ltype, np.ndarray) else str(ltype)
                               for ltype in metadata.get('layer_types', [])]
            self.layer_parent_ids = metadata.get('layer_parent_ids', [])
            self.layer_dimensions = metadata.get('layer_sizes', [])
            self.layer_indices = metadata.get('layer_indices', [])
            self.N_LAYERS = int(metadata.get('N_LAYERS', 0)) + 1
            
            logger.info(f"Loaded network metadata from {metadata_file}")
        
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {metadata_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def compute_layer_sizes(self) -> np.ndarray:
        """
        Compute layer sizes from dimensions.
        
        Returns
        -------
        np.ndarray
            Array of layer sizes
        """
        if not self.layer_dimensions:
            return np.array([])
        
        sizes = []
        for dims in self.layer_dimensions:
            if isinstance(dims, (list, np.ndarray)):
                sizes.append(int(np.prod(dims)))
            else:
                sizes.append(int(dims))
        
        return np.array(sizes)
    
    def get_type_markers(self, merge_relu: bool = False) -> str:
        """
        Get type markers for visualization.
        
        Parameters
        ----------
        merge_relu : bool, optional
            Whether to merge ReLU layers with previous (default: False)
        
        Returns
        -------
        str
            String of type markers
        """
        marker_key = 'merged_relu' if (self.layers_grouping_level & 1) == 1 else 'full'
        return ''.join(self.TYPE_MARKERS[marker_key])


class NetworkMetadataLoader:
    """Loader for network metadata with caching support."""
    
    _cache = {}
    
    @classmethod
    def load(cls, network_type: int, layers_grouping_level: int = 0,
             epoch: Optional[int] = None, seed: int = 0,
             imagenet_image_size: int = 64) -> NetworkMetadata:
        """
        Load network metadata with caching.
        
        Parameters
        ----------
        network_type : int
            Type of network
        layers_grouping_level : int, optional
            Level of layer grouping (default: 0)
        epoch : int, optional
            Epoch number (default: None)
        seed : int, optional
            Random seed (default: 0)
        imagenet_image_size : int, optional
            ImageNet image size (default: 64)
        
        Returns
        -------
        NetworkMetadata
            Loaded network metadata
        """
        cache_key = (network_type, layers_grouping_level, epoch, seed, imagenet_image_size)
        
        if cache_key not in cls._cache:
            metadata = NetworkMetadata(
                network_type,
                layers_grouping_level,
                epoch,
                seed,
                imagenet_image_size
            )
            cls._cache[cache_key] = metadata
        
        return cls._cache[cache_key]
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the metadata cache."""
        cls._cache.clear()


def load_network_metadata(network_type: int, layers_grouping_level: int = 0,
                         epoch: Optional[int] = None, seed: int = 0,
                         imagenet_image_size: int = 64) -> Tuple:
    """
    Load network metadata (MATLAB-compatible interface).
    
    Parameters
    ----------
    network_type : int
        Type of network (0-6)
    layers_grouping_level : int, optional
        Level of layer grouping (default: 0)
    epoch : int, optional
        Epoch number (default: None)
    seed : int, optional
        Random seed (default: 0)
    imagenet_image_size : int, optional
        ImageNet image size (default: 64)
    
    Returns
    -------
    tuple
        (network_name, N_LAYERS, ACTIVE_LAYERS, layer_names, layer_sizes,
         type_markers, layer_type_markers, parent_ids, multi_parent_ids,
         ACTIVE_MARKERS, active_type_markers_names, layers, layer_types,
         all_type_marker_names, layer_indices, layer_dimensions)
    """
    metadata = NetworkMetadataLoader.load(
        network_type,
        layers_grouping_level,
        epoch,
        seed,
        imagenet_image_size
    )
    
    # Return in MATLAB-compatible format
    return (
        metadata.network_name,
        metadata.N_LAYERS,
        metadata.ACTIVE_LAYERS,
        metadata.layer_names,
        metadata.layer_sizes if hasattr(metadata, 'layer_sizes') else [],
        metadata.get_type_markers(merge_relu=(layers_grouping_level & 1) == 1),
        metadata.layer_type_markers,
        metadata.parent_ids,
        metadata.layer_parent_ids,
        [],  # ACTIVE_MARKERS placeholder
        metadata.active_type_markers_names,
        metadata.layers,
        metadata.layer_types,
        metadata.all_type_marker_names,
        metadata.layer_indices,
        metadata.layer_dimensions,
    )

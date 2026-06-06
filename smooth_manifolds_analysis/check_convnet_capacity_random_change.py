"""
Convolutional Network Capacity Analysis for Random Changes.

This module provides tools to analyze the capacity and separability properties
of neural network representations across different layers and manifold directions.
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import time

import numpy as np
import scipy.io as sio

from load_network_metadata import load_network_metadata
from init_imagenet import get_imagenet_config, ImageNetConfig
from sample_indices import sample_random_labels
from theory_alpha0 import TheoreticalAlpha0
from square_corrcoeff_full_cost import square_corrcoeff_full_cost

# These modules should be available or created separately:
try:
    from low_dimension_manifold_calculator import calc_low_dimension_manifold
except ImportError:
    def calc_low_dimension_manifold(*args, **kwargs):
        raise NotImplementedError("calc_low_dimension_manifold must be implemented")

try:
    from binary_dichotomies_checker import check_binary_dichotomies_capacity
except ImportError:
    def check_binary_dichotomies_capacity(*args, **kwargs):
        raise NotImplementedError("check_binary_dichotomies_capacity must be implemented")


logger = logging.getLogger(__name__)


class PreprocessingType(IntEnum):
    """Enumeration for preprocessing types."""
    NONE = 0
    ORTHOGONALIZE_CENTERS = 1
    RANDOM_CENTERS = 2
    PERMUTED_MANIFOLD = 3
    MANIFOLD_RANDOM_UNIFORM_CENTERS = 4
    AXES_RANDOM = 5
    PERMUTE_RANDOM = 7


class GlobalPreprocessingType(IntEnum):
    """Enumeration for global preprocessing types."""
    NONE = 0
    ZNORM = 1
    WHITENING = 2
    CENTERS_DECORRELATION = 3


class RandomLabelingType(IntEnum):
    """Enumeration for random labeling types."""
    BINARY_IID = 0
    BALANCED = 1
    SPARSE = 2


class FeaturesType(IntEnum):
    """Enumeration for features types."""
    SUB_SAMPLE = 0
    FIRST_N_FEATURES = 1
    RANDOM_PROJECTIONS = 2


@dataclass
class AnalysisConfig:
    """Configuration for capacity analysis."""
    
    P: int
    range_factor: float
    N_SAMPLES: int
    network_type: int
    degrees_of_freedom: int
    local_preprocessing: int = PreprocessingType.NONE
    input_suffix: str = ""
    run_id: int = 0
    random_labeling_type: int = RandomLabelingType.BALANCED
    layers_grouping_level: Optional[int] = None
    use_half_samples: bool = False
    features_type: int = FeaturesType.RANDOM_PROJECTIONS
    N_OBJECTS: Optional[int] = None
    
    # Constants
    global_preprocessing: int = GlobalPreprocessingType.NONE
    N_NEURON_SAMPLES: int = 41
    N_DICHOTOMIES: int = 1
    max_samples: int = 2000
    precision: int = 1
    N_DIRECTIONS: int = 2
    IMAGENET_IMAGE_SIZE: int = 64
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.layers_grouping_level is None:
            self.layers_grouping_level = 2 if self.network_type == 2 else 0
        
        if self.N_OBJECTS is None:
            self.N_OBJECTS = self.P
        else:
            assert self.N_OBJECTS <= self.P, "N_OBJECTS must be <= P"


@dataclass
class AnalysisResults:
    """Container for analysis results."""
    
    N_DIRECTIONS: int
    N_LAYERS: int
    N_NEURONS: int
    N_NEURON_SAMPLES: int
    N_OBJECTS: int
    
    capacity_results: np.ndarray = field(init=False)
    separability_results: np.ndarray = field(init=False)
    radius_results: np.ndarray = field(init=False)
    mean_half_width_results: np.ndarray = field(init=False)
    mean_argmax_norm_results: np.ndarray = field(init=False)
    mean_half_width2_results: np.ndarray = field(init=False)
    mean_argmax_norm2_results: np.ndarray = field(init=False)
    effective_dimension_results: np.ndarray = field(init=False)
    effective_dimension2_results: np.ndarray = field(init=False)
    alphac_hat_results: np.ndarray = field(init=False)
    features_used_results: np.ndarray = field(init=False)
    labels_used_results: np.ndarray = field(init=False)
    
    def __post_init__(self):
        """Initialize result arrays with NaN."""
        self.capacity_results = np.full((self.N_DIRECTIONS, self.N_LAYERS), np.nan)
        self.separability_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURONS), np.nan)
        self.radius_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURON_SAMPLES, self.N_OBJECTS), np.nan)
        self.mean_half_width_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURON_SAMPLES, self.N_OBJECTS), np.nan)
        self.mean_argmax_norm_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURON_SAMPLES, self.N_OBJECTS), np.nan)
        self.mean_half_width2_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURON_SAMPLES, self.N_OBJECTS), np.nan)
        self.mean_argmax_norm2_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURON_SAMPLES, self.N_OBJECTS), np.nan)
        self.effective_dimension_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURON_SAMPLES, self.N_OBJECTS), np.nan)
        self.effective_dimension2_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURON_SAMPLES, self.N_OBJECTS), np.nan)
        self.alphac_hat_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURON_SAMPLES, self.N_OBJECTS), np.nan)
        self.features_used_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURON_SAMPLES, self.N_NEURONS), np.nan)
        self.labels_used_results = np.full((self.N_DIRECTIONS, self.N_LAYERS, self.N_NEURON_SAMPLES, self.N_OBJECTS), np.nan)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert results to dictionary for saving."""
        return {
            'capacity_results': self.capacity_results,
            'separability_results': self.separability_results,
            'radius_results': self.radius_results,
            'mean_half_width_results': self.mean_half_width_results,
            'mean_argmax_norm_results': self.mean_argmax_norm_results,
            'mean_half_width2_results': self.mean_half_width2_results,
            'mean_argmax_norm2_results': self.mean_argmax_norm2_results,
            'effective_dimension_results': self.effective_dimension_results,
            'effective_dimension2_results': self.effective_dimension2_results,
            'alphac_hat_results': self.alphac_hat_results,
            'features_used_results': self.features_used_results,
            'labels_used_results': self.labels_used_results,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> 'AnalysisResults':
        """Load results from dictionary."""
        N_DIRECTIONS, N_LAYERS = data['capacity_results'].shape
        N_NEURONS = data['separability_results'].shape[2]
        N_NEURON_SAMPLES, N_OBJECTS = data['radius_results'].shape[2:4]
        
        results = cls(
            N_DIRECTIONS=N_DIRECTIONS,
            N_LAYERS=N_LAYERS,
            N_NEURONS=N_NEURONS,
            N_NEURON_SAMPLES=N_NEURON_SAMPLES,
            N_OBJECTS=N_OBJECTS,
        )
        
        for key, value in data.items():
            setattr(results, key, value)
        
        return results


class ResultsFilenameBuilder:
    """Builder for generating results filenames with proper formatting."""
    
    def __init__(self, config: AnalysisConfig, network_name: str):
        """Initialize filename builder."""
        self.config = config
        self.network_name = network_name
    
    def build(self) -> str:
        """Build the output filename."""
        prefix = self._build_prefix()
        suffix = self._build_suffix()
        
        range_str = f"{self.config.range_factor:.1f}" if self.config.range_factor >= 0.1 else f"{self.config.range_factor:.6f}"
        
        filename = (
            f"{prefix}_range{range_str}_P{self.config.N_OBJECTS}_M{self.config.N_SAMPLES}"
            f"{suffix}{self.config.input_suffix}.mat"
        )
        
        return filename
    
    def _build_prefix(self) -> str:
        """Build prefix for filename."""
        prefix = f"check_{self.network_name}_capacity_random_change_dof{self.config.degrees_of_freedom}"
        
        if self.config.IMAGENET_IMAGE_SIZE != 64:
            prefix = f"{prefix}_{self.config.IMAGENET_IMAGE_SIZE}px"
        
        if self.config.features_type == FeaturesType.RANDOM_PROJECTIONS:
            prefix = f"{prefix}_projected"
        elif self.config.features_type != FeaturesType.SUB_SAMPLE:
            raise ValueError(f"Unknown features type: {self.config.features_type}")
        
        return prefix
    
    def _build_suffix(self) -> str:
        """Build suffix for filename."""
        suffix = ""
        
        # Random labeling type
        if self.config.random_labeling_type == RandomLabelingType.BALANCED:
            suffix += "_balanced"
        elif self.config.random_labeling_type == RandomLabelingType.SPARSE:
            suffix += "_sparse"
        
        # Global preprocessing
        if self.config.global_preprocessing == GlobalPreprocessingType.ZNORM:
            suffix += "_znorm"
        elif self.config.global_preprocessing == GlobalPreprocessingType.WHITENING:
            suffix += "_whiten"
        elif self.config.global_preprocessing == GlobalPreprocessingType.CENTERS_DECORRELATION:
            suffix += "_centers_whiten"
        
        # Local preprocessing
        if self.config.local_preprocessing == PreprocessingType.ORTHOGONALIZE_CENTERS:
            suffix += "_orth"
        elif self.config.local_preprocessing == PreprocessingType.RANDOM_CENTERS:
            suffix += "_centers_random"
        elif self.config.local_preprocessing == PreprocessingType.PERMUTED_MANIFOLD:
            suffix += "_manifold_random"
        elif self.config.local_preprocessing == PreprocessingType.MANIFOLD_RANDOM_UNIFORM_CENTERS:
            suffix += "_manifold_random_uniform_centers"
        elif self.config.local_preprocessing == PreprocessingType.AXES_RANDOM:
            suffix += "_axes_random"
        elif self.config.local_preprocessing == PreprocessingType.PERMUTE_RANDOM:
            suffix += "_permute_random"
        elif self.config.local_preprocessing != PreprocessingType.NONE:
            raise ValueError(f"Unknown local preprocessing type: {self.config.local_preprocessing}")
        
        # Use half samples
        if self.config.use_half_samples:
            suffix += "_half"
        
        # Run ID
        if self.config.run_id > 0:
            suffix += f"_{self.config.run_id}"
        
        return suffix


class InputFilenameBuilder:
    """Builder for input data filenames."""
    
    def __init__(self, config: AnalysisConfig, network_name: str):
        """Initialize input filename builder."""
        self.config = config
        self.network_name = network_name
    
    def build(self, layer_name: str) -> str:
        """Build input filename for a specific layer."""
        prefix = f"{self.network_name}/generate_{self.network_name}_random_change_dof{self.config.degrees_of_freedom}"
        
        if self.config.IMAGENET_IMAGE_SIZE != 64:
            prefix = f"{prefix}_{self.config.IMAGENET_IMAGE_SIZE}px"
        
        range_str = f"{self.config.range_factor:.1f}" if self.config.range_factor >= 0.1 else f"{self.config.range_factor:.6f}"
        
        filename = (
            f"{prefix}_range{range_str}_P{self.config.P}_M{self.config.N_SAMPLES}"
            f"_{layer_name}{self.config.input_suffix}.mat"
        )
        
        return filename


class ConvNetCapacityAnalyzer:
    """Main class for analyzing convolutional network capacity."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the analyzer with configuration."""
        self.config = config
        self._setup_logging()
        self._load_network_metadata()
        self._initialize_results()
    
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_network_metadata(self):
        """Load network metadata."""
        (
            self.network_name,
            self.N_LAYERS,
            _,
            self.layer_names,
            *_,
            self.layers,
            *_
        ) = load_network_metadata(self.config.network_type, self.config.layers_grouping_level)
        
        # Create enabled layers mask
        self.ENABLED_LAYERS = np.zeros(self.N_LAYERS, dtype=bool)
        self.ENABLED_LAYERS[self.layers] = True
        
        # Set N_NEURONS from global config
        self.N_NEURONS = self._get_n_hmax_features()
        assert self.N_NEURONS is not None, "Run init_imagenet to set N_HMAX_FEATURES"
    
    def _get_n_hmax_features(self) -> Optional[int]:
        """Get N_HMAX_FEATURES from global configuration or environment."""
        try:
            config = get_imagenet_config()
            return config.N_HMAX_FEATURES
        except Exception:
            return None
    
    def _initialize_results(self):
        """Initialize results data structures."""
        self.results = AnalysisResults(
            N_DIRECTIONS=self.config.N_DIRECTIONS,
            N_LAYERS=self.N_LAYERS,
            N_NEURONS=self.N_NEURONS,
            N_NEURON_SAMPLES=self.config.N_NEURON_SAMPLES,
            N_OBJECTS=self.config.N_OBJECTS,
        )
    
    def run(self, output_dir: Path = Path(".")):
        """Run the capacity analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle multiple run IDs
        if isinstance(self.config.run_id, (list, tuple, np.ndarray)):
            self._handle_multiple_runs(self.config.run_id, output_dir)
            return
        
        # Single run
        start_time = time.time()
        self._run_single_analysis(output_dir)
        elapsed_hours = (time.time() - start_time) / 3600
        logger.info(f"Done. (took {elapsed_hours:.1f} hours)")
    
    def _handle_multiple_runs(self, run_ids: List[int], output_dir: Path):
        """Handle aggregation of multiple run results."""
        # Create aggregated filename
        temp_config = AnalysisConfig(**self.config.__dict__)
        temp_config.run_id = 0
        filename_builder = ResultsFilenameBuilder(temp_config, self.network_name)
        out_name = output_dir / filename_builder.build()
        
        logger.info(f"Results saved to {out_name}")
        
        # Load and aggregate results from each run
        for run_id in run_ids:
            run_config = AnalysisConfig(**self.config.__dict__)
            run_config.run_id = run_id
            filename_builder = ResultsFilenameBuilder(run_config, self.network_name)
            run_name = output_dir / filename_builder.build()
            
            logger.info(f"Results loaded from {run_name}")
            
            # Convert run_id to layer and direction indices
            l, param_id = np.unravel_index(run_id, (self.N_LAYERS, self.config.N_DIRECTIONS))
            
            if not self.ENABLED_LAYERS[l] and not run_name.exists():
                logger.info(f"Skipping missing file: {run_name}")
                continue
            
            if not run_name.exists():
                logger.info(f"Skipping missing file: {run_name}")
                continue
            
            # Load and merge results
            self._merge_results_from_file(str(run_name), l, param_id)
        
        # Save aggregated results
        self._save_results(str(out_name))
    
    def _merge_results_from_file(self, filename: str, layer_idx: int, direction_idx: int):
        """Merge results from a single run file."""
        try:
            run_data = sio.loadmat(filename)
            
            self.results.capacity_results[direction_idx, layer_idx] = run_data['capacity_results'][direction_idx, layer_idx]
            self.results.separability_results[direction_idx, layer_idx, :] = run_data['separability_results'][direction_idx, layer_idx, :]
            self.results.radius_results[direction_idx, layer_idx, :, :] = run_data['radius_results'][direction_idx, layer_idx, :, :]
            self.results.mean_half_width_results[direction_idx, layer_idx, :, :] = run_data['mean_half_width_results'][direction_idx, layer_idx, :, :]
            self.results.mean_argmax_norm_results[direction_idx, layer_idx, :, :] = run_data['mean_argmax_norm_results'][direction_idx, layer_idx, :, :]
            self.results.mean_half_width2_results[direction_idx, layer_idx, :, :] = run_data['mean_half_width2_results'][direction_idx, layer_idx, :, :]
            self.results.mean_argmax_norm2_results[direction_idx, layer_idx, :, :] = run_data['mean_argmax_norm2_results'][direction_idx, layer_idx, :, :]
            self.results.effective_dimension_results[direction_idx, layer_idx, :, :] = run_data['effective_dimension_results'][direction_idx, layer_idx, :, :]
            
            if 'effective_dimension2_results' in run_data:
                self.results.effective_dimension2_results[direction_idx, layer_idx, :, :] = run_data['effective_dimension2_results'][direction_idx, layer_idx, :, :]
            
            self.results.alphac_hat_results[direction_idx, layer_idx, :, :] = run_data['alphac_hat_results'][direction_idx, layer_idx, :, :]
            self.results.features_used_results[direction_idx, layer_idx, :, :] = run_data['features_used_results'][direction_idx, layer_idx, :, :]
            self.results.labels_used_results[direction_idx, layer_idx, :, :] = run_data['labels_used_results'][direction_idx, layer_idx, :, :]
        
        except Exception as e:
            logger.error(f"Error loading results from {filename}: {e}")
    
    def _run_single_analysis(self, output_dir: Path):
        """Run analysis for a single configuration."""
        filename_builder = ResultsFilenameBuilder(self.config, self.network_name)
        out_name = output_dir / filename_builder.build()
        
        logger.info(f"Results saved to {out_name}")
        
        # Load existing results if available
        if out_name.exists():
            logger.info("Loading existing results")
            try:
                data = sio.loadmat(str(out_name))
                self.results = AnalysisResults.from_dict(data)
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
        
        # Determine which layers and directions to process
        if self.config.run_id == 0:
            run_directions = range(self.config.N_DIRECTIONS)
            run_layers = range(self.N_LAYERS)
        else:
            l, param_id = np.unravel_index(
                self.config.run_id,
                (self.N_LAYERS, self.config.N_DIRECTIONS)
            )
            run_directions = [param_id]
            run_layers = [l]
        
        # Process each layer and direction
        input_builder = InputFilenameBuilder(self.config, self.network_name)
        
        for param_id in run_directions:
            for l in run_layers:
                if not self.ENABLED_LAYERS[l]:
                    logger.info(f"Skipping disabled layer {self.layer_names[l]}")
                    continue
                
                # Check if already computed
                if np.isfinite(self.results.capacity_results[param_id, l]):
                    logger.info(f"Skipping existing {self.layer_names[l]}")
                    continue
                
                logger.info(f"Working on {self.layer_names[l]}")
                
                # Process layer
                self._process_layer(param_id, l, input_builder, str(out_name))
        
        # Save final results
        self._save_results(str(out_name))
    
    def _process_layer(self, param_id: int, layer_idx: int, input_builder: InputFilenameBuilder, output_filename: str):
        """Process a single layer."""
        # Load input data
        input_filename = input_builder.build(self.layer_names[layer_idx])
        
        load_start = time.time()
        try:
            input_data = sio.loadmat(input_filename)
            tuning_function = input_data['tuning_function']
        except Exception as e:
            logger.error(f"Error loading {input_filename}: {e}")
            return
        
        # Extract and prepare tuning function
        full_tuning_function = np.squeeze(tuning_function[param_id, :self.config.N_OBJECTS, :, :])
        full_tuning_function = np.transpose(full_tuning_function, (2, 1, 0))
        full_tuning_function = full_tuning_function.astype(np.float64)
        
        assert full_tuning_function.shape == (self.N_NEURONS, self.config.N_SAMPLES, self.config.N_OBJECTS)
        
        # Filter neurons with non-zero firing rate
        mean_square_firing_rate = np.mean(np.mean(full_tuning_function**2, axis=2), axis=1)
        nz_indices = np.where(mean_square_firing_rate > 0)[0]
        N = len(nz_indices)
        
        logger.info(f"Loaded data (took {time.time() - load_start:.1f} sec)")
        
        # Apply local preprocessing
        if self.config.local_preprocessing == PreprocessingType.NONE:
            properties_type = 1
            current_tuning_function = full_tuning_function[nz_indices, :, :]
        else:
            properties_type = 0
            min_n = min(self.config.N_SAMPLES, N)
            current_tuning_function = calc_low_dimension_manifold(
                full_tuning_function[nz_indices, :, :],
                min_n,
                self.config.local_preprocessing
            )
        
        # Use half samples if requested
        if self.config.use_half_samples:
            current_tuning_function = current_tuning_function[:, ::2, :]
        
        # Compute capacity and other metrics
        (
            capacity,
            separability,
            _,
            radius,
            mean_half_width1,
            mean_argmax_norm1,
            mean_half_width2,
            mean_argmax_norm2,
            effective_dimension,
            effective_dimension2,
            alphac_hat,
            features_used,
            labels_used
        ) = check_binary_dichotomies_capacity(
            current_tuning_function,
            self.config.N_NEURON_SAMPLES,
            self.config.N_DICHOTOMIES,
            True,
            self.config.random_labeling_type,
            self.config.precision,
            self.config.max_samples,
            self.config.global_preprocessing,
            self.config.features_type,
            min(512, N),
            properties_type
        )
        
        # Store results
        self.results.capacity_results[param_id, layer_idx] = capacity
        self.results.separability_results[param_id, layer_idx, :N] = separability
        self.results.radius_results[param_id, layer_idx, :, :] = radius
        self.results.mean_half_width_results[param_id, layer_idx, :, :] = mean_half_width1
        self.results.mean_argmax_norm_results[param_id, layer_idx, :, :] = mean_argmax_norm1
        self.results.mean_half_width2_results[param_id, layer_idx, :, :] = mean_half_width2
        self.results.mean_argmax_norm2_results[param_id, layer_idx, :, :] = mean_argmax_norm2
        self.results.effective_dimension_results[param_id, layer_idx, :, :] = effective_dimension
        self.results.effective_dimension2_results[param_id, layer_idx, :, :] = effective_dimension2
        self.results.alphac_hat_results[param_id, layer_idx, :, :] = alphac_hat
        
        if features_used is not None and len(features_used) > 0:
            self.results.features_used_results[param_id, layer_idx, :, :int(capacity)] = features_used
        
        self.results.labels_used_results[param_id, layer_idx, :, :] = labels_used
        
        # Save after each layer
        self._save_results(output_filename)
    
    def _save_results(self, output_filename: str):
        """Save results to MAT file."""
        sio.savemat(
            output_filename,
            self.results.to_dict(),
            long_field_names=True,
            do_compression=True
        )


def check_convnet_capacity_random_change(
    P: int,
    range_factor: float,
    N_SAMPLES: int,
    network_type: int,
    degrees_of_freedom: int,
    local_preprocessing: int = PreprocessingType.NONE,
    input_suffix: str = "",
    run_id: int = 0,
    random_labeling_type: int = RandomLabelingType.BALANCED,
    layers_grouping_level: Optional[int] = None,
    use_half_samples: bool = False,
    features_type: int = FeaturesType.RANDOM_PROJECTIONS,
    N_OBJECTS: Optional[int] = None,
    output_dir: Path = Path("."),
) -> ConvNetCapacityAnalyzer:
    """
    Main function to check convolutional network capacity with random changes.
    
    Parameters
    ----------
    P : int
        Number of stimuli patterns
    range_factor : float
        Range factor for random changes
    N_SAMPLES : int
        Number of samples
    network_type : int
        Type of network (0, 1, 2, etc.)
    degrees_of_freedom : int
        Degrees of freedom for manifold
    local_preprocessing : int, optional
        Type of local preprocessing (default: NONE)
    input_suffix : str, optional
        Suffix for input filenames (default: "")
    run_id : int or list, optional
        Run ID for parallel processing (default: 0)
    random_labeling_type : int, optional
        Type of random labeling (default: BALANCED)
    layers_grouping_level : int, optional
        Layer grouping level (default: auto-determined by network type)
    use_half_samples : bool, optional
        Whether to use half samples (default: False)
    features_type : int, optional
        Type of features to use (default: RANDOM_PROJECTIONS)
    N_OBJECTS : int, optional
        Number of objects (default: P)
    output_dir : Path, optional
        Output directory (default: current directory)
    
    Returns
    -------
    ConvNetCapacityAnalyzer
        Analyzer instance with results
    """
    config = AnalysisConfig(
        P=P,
        range_factor=range_factor,
        N_SAMPLES=N_SAMPLES,
        network_type=network_type,
        degrees_of_freedom=degrees_of_freedom,
        local_preprocessing=local_preprocessing,
        input_suffix=input_suffix,
        run_id=run_id,
        random_labeling_type=random_labeling_type,
        layers_grouping_level=layers_grouping_level,
        use_half_samples=use_half_samples,
        features_type=features_type,
        N_OBJECTS=N_OBJECTS,
    )
    
    analyzer = ConvNetCapacityAnalyzer(config)
    analyzer.run(output_dir)
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Configure parameters
    params = {
        "P": 1000,
        "range_factor": 0.5,
        "N_SAMPLES": 100,
        "network_type": 2,
        "degrees_of_freedom": 2,
        "local_preprocessing": PreprocessingType.ORTHOGONALIZE_CENTERS,
        "random_labeling_type": RandomLabelingType.BALANCED,
        "use_half_samples": False,
        "features_type": FeaturesType.RANDOM_PROJECTIONS,
    }
    
    # Run analysis
    analyzer = check_convnet_capacity_random_change(**params)

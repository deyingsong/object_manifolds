"""
Quick Reference: MATLAB to Python Function Mappings

This module provides a quick lookup for equivalent Python implementations
of the translated MATLAB functions.
"""

# ============================================================================
# HOSTNAME (hostname.m → hostname.py)
# ============================================================================
# MATLAB:  name = hostname()
# Python:  from hostname import get_hostname
#          name = get_hostname()

# ============================================================================
# IMAGENET INITIALIZATION (init_imagenet.m → init_imagenet.py)
# ============================================================================
# MATLAB (Script - globals):
#   global IMAGENET_IMAGE_SIZE
#   global IMAGENET_OBJECT_SIZE
#   global IMAGENET_FRAME
#   ... etc
#
# Python:
#   from init_imagenet import init_imagenet, get_imagenet_config
#
#   # Initialize with default values
#   config = init_imagenet()
#
#   # Or with custom values
#   config = init_imagenet(object_size=48, frame=8, surround_factor=3)
#
#   # Access configuration
#   config.IMAGE_SIZE
#   config.N_HMAX_FEATURES

# ============================================================================
# LOAD NETWORK METADATA (load_network_metadata.m → load_network_metadata.py)
# ============================================================================
# MATLAB:
#   [network_name, N_LAYERS, ACTIVE_LAYERS, layer_names, ...] = ...
#       load_network_metadata(network_type, layers_grouping_level);
#
# Python:
#   from load_network_metadata import load_network_metadata
#   network_name, N_LAYERS, ACTIVE_LAYERS, layer_names, ... = \
#       load_network_metadata(network_type=2, layers_grouping_level=0)

# ============================================================================
# SAMPLING (sample_indices.m, sample_random_labels.m → sample_indices.py)
# ============================================================================
# MATLAB:
#   samples = sample_indices(N, K, R)
#   y = sample_random_labels(N_OBJECTS, random_labeling_type)
#
# Python:
#   from sample_indices import sample_indices, sample_random_labels
#   samples = sample_indices(N=100, K=50, R=10)  # Returns (10, 50) array
#   y = sample_random_labels(N_OBJECTS=1000, random_labeling_type=1)

# ============================================================================
# OPTIMAL LOW RANK (optimal_low_rank_structure2.m → optimal_low_rank_structure2.py)
# ============================================================================
# MATLAB:
#   [Vopt, Xopt, Kopt, residual_norms, ...] = optimal_low_rank_structure2(
#       X, MAX_K, verbose, minSquare, N_REPEATS)
#
# Python:
#   from optimal_low_rank_structure2 import OptimalLowRankStructure
#   optimizer = OptimalLowRankStructure(verbose=1, minimize_square=True, n_repeats=1)
#   Vopt, Xopt, Kopt, residual_norms, sq_corr, abs_corr, sq_corr_raw, abs_corr_raw = \
#       optimizer.compute(X, max_k=None)

# ============================================================================
# SQUARE CORRELATION COEFFICIENT (square_corrcoeff_full_cost.m → square_corrcoeff_full_cost.py)
# ============================================================================
# MATLAB:
#   [cost, gradient] = square_corrcoeff_full_cost(V, X)
#
# Python:
#   from square_corrcoeff_full_cost import square_corrcoeff_full_cost
#   cost, gradient = square_corrcoeff_full_cost(V, X)

# ============================================================================
# THEORY ALPHA0 (theory_alpha0.m, theory_alpha0_cached.m → theory_alpha0.py)
# ============================================================================
# MATLAB (direct):
#   a = theory_alpha0(kappa)
#
# Python (direct):
#   from theory_alpha0 import theory_alpha0
#   a = theory_alpha0(2.5)
#
# MATLAB (cached):
#   alpha = theory_alpha0_cached(kappa)
#
# Python (cached):
#   from theory_alpha0 import TheoreticalAlpha0
#   alpha = TheoreticalAlpha0.compute(2.5)
#   
#   # For array input
#   kappas = np.array([0, 1.5, 50, 150])
#   alphas = TheoreticalAlpha0.compute(kappas)

# ============================================================================
# ENUMERATIONS (from check_convnet_capacity_random_change.py)
# ============================================================================
# Preprocessing Types:
#   PreprocessingType.NONE = 0
#   PreprocessingType.ORTHOGONALIZE_CENTERS = 1
#   PreprocessingType.RANDOM_CENTERS = 2
#   PreprocessingType.PERMUTED_MANIFOLD = 3
#   PreprocessingType.MANIFOLD_RANDOM_UNIFORM_CENTERS = 4
#   PreprocessingType.AXES_RANDOM = 5
#   PreprocessingType.PERMUTE_RANDOM = 7
#
# Global Preprocessing Types:
#   GlobalPreprocessingType.NONE = 0
#   GlobalPreprocessingType.ZNORM = 1
#   GlobalPreprocessingType.WHITENING = 2
#   GlobalPreprocessingType.CENTERS_DECORRELATION = 3
#
# Random Labeling Types:
#   RandomLabelingType.BINARY_IID = 0
#   RandomLabelingType.BALANCED = 1
#   RandomLabelingType.SPARSE = 2
#
# Features Types:
#   FeaturesType.SUB_SAMPLE = 0
#   FeaturesType.FIRST_N_FEATURES = 1
#   FeaturesType.RANDOM_PROJECTIONS = 2

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================
# MATLAB:
#   check_convnet_capacity_random_change(P, range_factor, N_SAMPLES, 
#       network_type, degrees_of_freedom, ...)
#
# Python:
#   from check_convnet_capacity_random_change import check_convnet_capacity_random_change
#   analyzer = check_convnet_capacity_random_change(
#       P=1000,
#       range_factor=0.5,
#       N_SAMPLES=100,
#       network_type=2,
#       degrees_of_freedom=2,
#       local_preprocessing=1,
#       random_labeling_type=1,
#       use_half_samples=False,
#       features_type=2,
#   )

# ============================================================================
# KEY DIFFERENCES FROM MATLAB
# ============================================================================
#
# 1. Global Variables → Configuration Objects
#    Instead of MATLAB's global keyword, use dataclass configuration objects
#    Example: ImageNetConfig instead of global IMAGENET_IMAGE_SIZE
#
# 2. MATLAB's sprintf() → Python's f-strings or .format()
#    Example: f'{prefix}_{value:.1f}' instead of sprintf('%s_%1.1f', prefix, value)
#
# 3. MATLAB's nan → np.nan
#    Example: np.full(shape, np.nan) instead of nan(shape)
#
# 4. MATLAB's cell arrays → Python lists or numpy object arrays
#    Automatic conversion by scipy.io.loadmat
#
# 5. Matrix indexing (1-based in MATLAB → 0-based in Python)
#    Be careful when converting MATLAB indexing!
#
# 6. Random number generation
#    MATLAB: rand, randi, randn
#    Python: np.random.* functions
#
# 7. Array operations
#    MATLAB: A .* B (element-wise multiplication)
#    Python: A * B (with numpy arrays)

# ============================================================================
# PERFORMANCE NOTES
# ============================================================================
#
# 1. Caching: Most modules implement caching for better performance
#    - ImageNetConfig: Global singleton
#    - NetworkMetadataLoader: Dictionary cache
#    - TheoreticalAlpha0: Interpolation cache
#
# 2. NumPy operations are generally faster than MATLAB equivalents
#    when using properly vectorized operations
#
# 3. For intensive numerical computations, consider using:
#    - Numba for JIT compilation: @numba.jit
#    - CuPy for GPU acceleration
#    - PyManopt for Riemannian optimization

"""
Quick reference guide - see individual module docstrings for detailed information
"""

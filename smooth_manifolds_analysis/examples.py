"""
Example usage of translated Python modules.

This file demonstrates how to use the translated Python modules with realistic examples.
"""

import numpy as np
from pathlib import Path

# ============================================================================
# Example 1: Using ImageNet Configuration
# ============================================================================
def example_imagenet_config():
    """Demonstrate ImageNet configuration usage."""
    from init_imagenet import init_imagenet, get_imagenet_config
    
    # Initialize with default 64px images
    config = init_imagenet()
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Object size: {config.OBJECT_SIZE}x{config.OBJECT_SIZE}")
    print(f"HMAX features: {config.N_HMAX_FEATURES}")
    
    # Access the global configuration
    global_config = get_imagenet_config()
    assert global_config is config
    
    # Get formatted info
    print(global_config.get_info_string())


# ============================================================================
# Example 2: Getting System Hostname
# ============================================================================
def example_hostname():
    """Get system hostname."""
    from hostname import get_hostname
    
    hostname = get_hostname()
    print(f"Hostname: {hostname}")


# ============================================================================
# Example 3: Sampling Operations
# ============================================================================
def example_sampling():
    """Demonstrate sampling utilities."""
    from sample_indices import sample_indices, sample_random_labels
    
    # Sample 3 sets of 10 indices from 100 values
    samples = sample_indices(N=100, K=10, R=3)
    print(f"Sampled indices shape: {samples.shape}")  # (3, 10)
    print(f"Sample 1: {samples[0]}")
    
    # Generate random labels (balanced)
    balanced_labels = sample_random_labels(N_OBJECTS=100, random_labeling_type=1)
    print(f"Balanced labels sum: {np.sum(balanced_labels)}")  # Should be ±1
    print(f"Label count: +1: {np.sum(balanced_labels == 1)}, -1: {np.sum(balanced_labels == -1)}")
    
    # Generate sparse labels
    sparse_labels = sample_random_labels(N_OBJECTS=100, random_labeling_type=2)
    print(f"Sparse labels sum: {np.sum(sparse_labels)}")  # Should be 98


# ============================================================================
# Example 4: Theoretical Alpha0 Computation
# ============================================================================
def example_theory_alpha0():
    """Demonstrate theoretical alpha0 computation."""
    from theory_alpha0 import TheoreticalAlpha0, theory_alpha0
    
    # Compute for single value (with caching)
    alpha = TheoreticalAlpha0.compute(2.5)
    print(f"Alpha0(kappa=2.5) = {alpha:.6f}")
    
    # Compute for array of values
    kappas = np.array([0, 1, 2, 5, 10, 50, 150])
    alphas = TheoreticalAlpha0.compute(kappas)
    
    print("\nKappa -> Alpha0 mapping:")
    for k, a in zip(kappas, alphas):
        print(f"  κ={k:3.0f} -> α={a:.6f}")
    
    # Direct computation (no caching)
    alpha_direct = theory_alpha0(2.5)
    assert np.abs(alpha - alpha_direct) < 1e-10


# ============================================================================
# Example 5: Loading Network Metadata
# ============================================================================
def example_network_metadata():
    """Demonstrate network metadata loading."""
    from load_network_metadata import NetworkMetadataLoader, load_network_metadata
    
    # Load metadata using the class
    metadata = NetworkMetadataLoader.load(network_type=1, layers_grouping_level=0)
    print(f"Network: {metadata.network_name}")
    print(f"Number of layers: {metadata.N_LAYERS}")
    
    # Use MATLAB-compatible interface
    network_name, N_LAYERS, ACTIVE_LAYERS, layer_names, *_ = \
        load_network_metadata(network_type=2, layers_grouping_level=2)
    print(f"\nNetwork: {network_name}, Layers: {N_LAYERS}")
    
    # Check cache
    metadata2 = NetworkMetadataLoader.load(network_type=1, layers_grouping_level=0)
    assert metadata is metadata2  # Same object from cache


# ============================================================================
# Example 6: Square Correlation Coefficient
# ============================================================================
def example_square_corrcoeff():
    """Demonstrate square correlation coefficient computation."""
    from square_corrcoeff_full_cost import square_corrcoeff_full_cost
    
    # Generate random data
    np.random.seed(42)
    N = 100  # Dimension
    K = 10   # Rank
    P = 50   # Number of samples
    
    X = np.random.randn(P, N)
    V = np.random.randn(N, K)
    
    # Orthonormalize V (project to Stiefel manifold)
    Q, _ = np.linalg.qr(V)
    V = Q[:, :K]
    
    # Compute cost and gradient
    cost, gradient = square_corrcoeff_full_cost(V, X)
    
    print(f"Data shape (P, N): ({P}, {N})")
    print(f"Basis vectors shape (N, K): ({N}, {K})")
    print(f"Cost: {cost:.6f}")
    print(f"Gradient shape: {gradient.shape}")
    print(f"Gradient norm: {np.linalg.norm(gradient):.6f}")


# ============================================================================
# Example 7: Optimal Low-Rank Structure
# ============================================================================
def example_optimal_low_rank():
    """Demonstrate optimal low-rank structure computation."""
    from optimal_low_rank_structure2 import OptimalLowRankStructure
    
    # Generate random data
    np.random.seed(42)
    N = 100  # Dimension
    P = 50   # Number of samples
    
    X = np.random.randn(N, P)
    
    # Create optimizer
    optimizer = OptimalLowRankStructure(verbose=0, minimize_square=True, n_repeats=1)
    
    # Compute optimal structure
    Vopt, Xopt, Kopt, residual_norms, sq_corr, abs_corr, sq_corr_raw, abs_corr_raw = \
        optimizer.compute(X, max_k=10)
    
    print(f"Input data shape: {X.shape}")
    print(f"Optimal rank: {Kopt}")
    print(f"Residual norms shape: {residual_norms.shape}")
    print(f"Mean square correlations at each rank:")
    for k in range(min(5, len(sq_corr))):
        print(f"  k={k}: {sq_corr[k]:.6f}")


# ============================================================================
# Example 8: Configuration Classes
# ============================================================================
def example_configuration():
    """Demonstrate configuration classes."""
    from check_convnet_capacity_random_change import (
        AnalysisConfig, PreprocessingType, RandomLabelingType, FeaturesType
    )
    
    # Create configuration
    config = AnalysisConfig(
        P=1000,
        range_factor=0.5,
        N_SAMPLES=100,
        network_type=2,
        degrees_of_freedom=2,
        local_preprocessing=PreprocessingType.ORTHOGONALIZE_CENTERS,
        random_labeling_type=RandomLabelingType.BALANCED,
        use_half_samples=False,
        features_type=FeaturesType.RANDOM_PROJECTIONS,
    )
    
    print(f"Configuration:")
    print(f"  P (patterns): {config.P}")
    print(f"  N_OBJECTS: {config.N_OBJECTS}")
    print(f"  Range factor: {config.range_factor}")
    print(f"  Network type: {config.network_type}")
    print(f"  Degrees of freedom: {config.degrees_of_freedom}")
    print(f"  Max samples: {config.max_samples}")


# ============================================================================
# Example 9: Results Container
# ============================================================================
def example_results():
    """Demonstrate results container."""
    from check_convnet_capacity_random_change import AnalysisResults
    
    # Create results container
    results = AnalysisResults(
        N_DIRECTIONS=2,
        N_LAYERS=10,
        N_NEURONS=512,
        N_NEURON_SAMPLES=41,
        N_OBJECTS=100,
    )
    
    print(f"Results container created:")
    print(f"  Shape of capacity results: {results.capacity_results.shape}")
    print(f"  Shape of separability results: {results.separability_results.shape}")
    print(f"  Shape of radius results: {results.radius_results.shape}")
    
    # Fill some results
    results.capacity_results[0, 0] = 0.85
    results.separability_results[0, 0, :50] = np.random.rand(50)
    
    # Convert to dictionary for saving
    results_dict = results.to_dict()
    print(f"\nResults dictionary keys: {list(results_dict.keys())}")


# ============================================================================
# Example 10: Full Analysis Workflow (Conceptual)
# ============================================================================
def example_full_workflow():
    """Demonstrate a complete analysis workflow (conceptual)."""
    from check_convnet_capacity_random_change import (
        check_convnet_capacity_random_change,
        PreprocessingType,
        RandomLabelingType,
    )
    
    print("Full analysis workflow example:")
    print("-" * 60)
    
    # NOTE: This example is conceptual. The actual implementation requires:
    # 1. Data files to be present
    # 2. calc_low_dimension_manifold to be implemented
    # 3. check_binary_dichotomies_capacity to be implemented
    
    # Create analysis configuration
    analysis_config = {
        "P": 1000,
        "range_factor": 0.5,
        "N_SAMPLES": 100,
        "network_type": 2,
        "degrees_of_freedom": 2,
        "local_preprocessing": PreprocessingType.ORTHOGONALIZE_CENTERS,
        "random_labeling_type": RandomLabelingType.BALANCED,
        "use_half_samples": False,
        "features_type": 2,
        "output_dir": Path("./results"),
    }
    
    print(f"Analysis config: {analysis_config}")
    print("(Would run analysis if dependencies were available)")


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TRANSLATED PYTHON MODULES - USAGE EXAMPLES")
    print("=" * 70)
    
    examples = [
        ("ImageNet Configuration", example_imagenet_config),
        ("System Hostname", example_hostname),
        ("Sampling Operations", example_sampling),
        ("Theoretical Alpha0", example_theory_alpha0),
        ("Network Metadata", example_network_metadata),
        ("Square Correlation Coefficient", example_square_corrcoeff),
        ("Optimal Low-Rank Structure", example_optimal_low_rank),
        ("Configuration Classes", example_configuration),
        ("Results Container", example_results),
        ("Full Analysis Workflow", example_full_workflow),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{i}. {name}")
        print("-" * 70)
        try:
            func()
        except ImportError as e:
            print(f"(Skipped - missing dependency: {e})")
        except NotImplementedError as e:
            print(f"(Skipped - not implemented: {e})")
        except Exception as e:
            print(f"(Error: {e})")
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)

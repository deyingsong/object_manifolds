# Separability and Geometry of Object Manifolds in Deep Neural Networks

This repository provides a Python toolkit for studying how neural networks organize variations of the same object—such as translated, rotated, or scaled images—into geometric “object manifolds” in their internal feature spaces. It can generate these manifolds from ImageNet images and pretrained vision models or analyze your own neural and DNN activation data, then quantify how easily object classes can be linearly separated and explain that performance through interpretable geometric properties including manifold radius, effective dimension, and correlations. By combining empirical classification-capacity measurements with predictions from mean-field theory, the project helps researchers and practitioners investigate how representation geometry evolves across network layers and how that geometry affects downstream classification.

The toolkit uses IBM ILOG CPLEX to solve the quadratic and constrained least-squares optimization problems required by several of its manifold-geometry and separability analyses.

Python version of the original [MATLAB](https://github.com/sompolinsky-lab/dnn-object-manifolds) repository by Cohen, Chung, Lee & Sompolinsky (2020).

## Contents

- [Overview](#overview)
- [Package Structure](#package-structure)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [How to Use with Your Data](#how-to-use-with-your-data)
- [Citation](#citation)
- [License](./dnn-object-manifolds-master/LICENSE)


## Overview

This repository provides a **Python implementation** of the algorithms described in the paper, allowing for:

- direct estimation of classification capacity;
- numerical estimation of object manifold geometry (radius, dimension, and inter-manifold correlations) and the classification capacity predicted by the mean-field theory (MFT) analysis used in our work.

The package is a full object-oriented Python rewrite of the original MATLAB codebase. The QP and least-squares solvers use IBM ILOG CPLEX via its Python API, exactly mirroring the original `cplexqp` / `cplexlsqlin` calls. MatConvNet is replaced by PyTorch (`torchvision`).

## Package Structure

```
object_manifolds/
├── FOptM/                          # Stiefel & multi-ball manifold optimisation
│   ├── gram_schmidt.py             #   Modified Gram-Schmidt orthogonalisation
│   ├── opt_stiefel_gbb.py          #   OptStiefelGBB (Wen & Yin 2010)
│   ├── opt_mani_multi_ball_gbb.py  #   OptManiMultiBallGBB
│   └── demos/                      #   Eigenvalue & max-cut demo scripts
│
├── library/                        # Core reusable analysis library
│   ├── utils.py                    #   assert_warn, hostname, sample_indices, …
│   ├── theory.py                   #   TheoryAlpha0, theory_alpha0 (MFT capacity)
│   ├── manifold_properties.py      #   ManifoldPropertiesIterative/LS/LSCorr
│   ├── cplex_interface.py          #   cplexoptimset, cplexqp, cplexlsqlin
│   ├── low_rank.py                 #   ConstrainedLeastSquares, OptimalLowRankStructure
│   ├── separability.py             #   LinearSeparabilitySVM (primal & dual)
│   ├── preprocessing.py            #   TuningFunctionPreprocessor, LowDimensionalManifold
│   └── capacity.py                 #   BinaryDichotomiesCapacity, HierarchicalCapacity
│
├── point_cloud_analysis/           # MFT capacity for point-cloud manifolds
│   └── manifold_analysis.py        #   ManifoldStableAnalysisCorr
│
├── smooth_manifolds_generation/    # DNN image manifold generation
│   ├── imagenet.py                 #   ImageNet dataset utilities
│   ├── network.py                  #   ConvNetExtractor (PyTorch / torchvision)
│   ├── transforms.py               #   7 affine image transformations
│   └── generation.py               #   1-d and random manifold generators
│
└── smooth_manifolds_analysis/      # High-level capacity & geometry analysis
    ├── capacity_analysis.py        #   OneDimensionalCapacityAnalysis, RandomChangeCapacityAnalysis
    └── low_rank_analysis.py        #   CovarianceLowRankAnalysis
```


## System Requirements

- **Python** 3.10 (required for CPLEX; see note below)
- **NumPy**, **SciPy**, **PyTorch**, **torchvision**
- **IBM CPLEX** Studio 22.1.1 or later — Python bindings (see [Install CPLEX](#install-cplex))
- Any OS supported by Python and CPLEX (tested on macOS)

> **Python version note:** IBM CPLEX Studio 2211 ships Python bindings for versions 3.8, 3.9, and 3.10 only. Create a Python 3.10 virtual environment to use the CPLEX solvers. On Python 3.12+ the `library/cplex_interface.py` module loads cleanly but raises an `ImportError` with setup instructions when a solver function is first called.

## Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/deyingsong/object_manifolds.git
cd object_manifolds
```

### 2. Create a Python 3.10 virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install numpy scipy torch torchvision
```

### 3. Install CPLEX

Download CPLEX Studio from IBM's website — it is available for free for academic use
([IBM Academic Initiative](https://developer.ibm.com/learningpaths/ibm-academic-initiative-cplex/)).
After installation, install the Python bindings into your virtual environment:

```bash
pip install /Applications/CPLEX_Studio2211/cplex/python/3.10/x86-64_osx
```

Adjust the path to match your OS and installation directory (`x86-64_linux` on Linux).

### 4. Download the ImageNet thumbnails

Download `imagenet_all_thumbnails_64px.mat`
[from figshare](https://doi.org/10.6084/m9.figshare.11494314) and save it at
`smooth_manifolds_generation/imagenet_all_thumbnails_64px.mat`.

## Demo

All demos below assume the working directory is the `object_manifolds/` package root and the Python 3.10 virtual environment is active.

### Smooth 1-d manifolds

**Generate manifolds**:

```python
from smooth_manifolds_generation.imagenet import init_imagenet
from smooth_manifolds_generation.network import NetworkType
from smooth_manifolds_generation.generation import OneDimensionalManifoldGenerator, GenerationConfig

init_imagenet()

config = GenerationConfig(
    network_type=NetworkType.ALEXNET,   # ALEXNET, RESNET50, or VGG16
    range_factor=0.5,
    n_objects=128,
    n_samples=15,
)
generator = OneDimensionalManifoldGenerator(config)

for batch_id in range(1, 29):
    generator.generate(batch_id=batch_id)

# Collect all batches into a single tuning-function array
tuning_function = generator.collect(batch_ids=range(1, 29))
```

**Direct estimation of classification capacity**:

```python
from smooth_manifolds_analysis.capacity_analysis import (
    OneDimensionalCapacityAnalysis, CapacityAnalysisConfig,
)

cfg = CapacityAnalysisConfig(n_objects=128, range_factor=0.5, n_samples=15)
analysis = OneDimensionalCapacityAnalysis(cfg)
results = analysis.run(tuning_function, layer_number=20)  # 20 = feature layer for AlexNet
print(results)   # LayerCapacityResults with capacity_alpha_c per layer
```

**MFT geometry estimation**:

```python
from smooth_manifolds_analysis.low_rank_analysis import (
    CovarianceLowRankAnalysis, LowRankAnalysisConfig,
)

cfg = LowRankAnalysisConfig(n_objects=128, range_factor=0.5, n_samples=15, max_k=5)
analysis = CovarianceLowRankAnalysis(cfg)
results = analysis.run(tuning_function, layer_number=20)
# results.theory_capacity, results.mean_radius, results.mean_dimension
```

### Smooth 2-d manifolds

**Generate manifolds** (2 affine transformations):

```python
from smooth_manifolds_generation.generation import RandomManifoldGenerator, GenerationConfig

config = GenerationConfig(
    network_type=NetworkType.ALEXNET,
    range_factor=0.5,
    n_objects=64,
    n_samples=201,
    n_transform_dims=2,
    n_batches=4,
)
generator = RandomManifoldGenerator(config)

for batch_id in range(1, config.n_batches * 2 + 1):
    generator.generate(batch_id=batch_id)

tuning_function = generator.collect(batch_ids=range(1, config.n_batches * 2 + 1))
```

**Direct capacity estimation**:

```python
from smooth_manifolds_analysis.capacity_analysis import (
    RandomChangeCapacityAnalysis, CapacityAnalysisConfig,
)

cfg = CapacityAnalysisConfig(n_objects=64, range_factor=0.5, n_samples=201, n_transform_dims=2)
results = RandomChangeCapacityAnalysis(cfg).run(tuning_function, layer_number=20)
```

**MFT geometry estimation**=:

```python
cfg = LowRankAnalysisConfig(n_objects=64, range_factor=0.5, n_samples=201, max_k=5, n_transform_dims=2)
results = CovarianceLowRankAnalysis(cfg).run(tuning_function, layer_number=20)
```

### Point-cloud manifolds

```python
import numpy as np
from point_cloud_analysis.manifold_analysis import ManifoldStableAnalysisCorr, AnalysisOptions

# tuning_function: np.ndarray of shape (N_NEURONS, N_SAMPLES, N_OBJECTS)
tuning_function = np.load("my_data.npy")

opts = AnalysisOptions(kappa=0.0, n_t=200)
result = ManifoldStableAnalysisCorr(opts).analyze(tuning_function)

print(f"Capacity alpha_c = {result.alpha_c:.4f}")
print(f"Mean radius  R_M = {result.mean_radius:.4f}")
print(f"Mean dim     D_M = {result.mean_dimension:.4f}")
```

## How to Use with Your Data

### Prepare your data

Organise your neural or DNN activations as a NumPy array with shape `(N_NEURONS, N_SAMPLES, N_OBJECTS)`:

- `N_NEURONS` — number of units / features in this layer or brain region
- `N_SAMPLES` — number of stimulus samples per object (images, transformations, etc.)
- `N_OBJECTS` — number of object classes

Use `np.nan` for missing neurons or missing samples. For multi-session data, use a list or an additional leading dimension `(N_SESSIONS, N_NEURONS, N_SAMPLES, N_OBJECTS)`.

Save your data in any format readable by NumPy (`.npy`, `.npz`, or via `scipy.io.loadmat` for `.mat` files).

### Direct estimation of capacity

```python
import numpy as np
from library.capacity import BinaryDichotomiesCapacity

tuning_function = np.load("my_data.npy")   # (N_NEURONS, N_SAMPLES, N_OBJECTS)

estimator = BinaryDichotomiesCapacity(n_rep=10)
results = estimator.estimate(tuning_function)

# Critical load (objects per neuron at 50% separability)
alpha_c = results.n_objects / results.capacity_n_neurons
print(f"alpha_c = {alpha_c:.4f}")
```

### Numerical estimation of capacity and manifold geometry

```python
from library.manifold_properties import ManifoldPropertiesLSCorr
from library.low_rank import OptimalLowRankStructure

# Compute manifold geometry (radius, dimension, capacity)
geom = ManifoldPropertiesLSCorr().compute(tuning_function)
print(f"R_M = {geom.mean_half_width2:.4f}")
print(f"D_M = {geom.effective_dimension2:.4f}")
print(f"alpha_c (theory) = {geom.alphac_hat2:.4f}")

# Find optimal low-rank structure (inter-manifold correlation removal)
centers = tuning_function.mean(axis=1)   # (N_NEURONS, N_OBJECTS)
opt_lr = OptimalLowRankStructure(verbose=1)
V_opt, X_opt, K_opt, *metrics = opt_lr.compute(centers, max_k=10)
print(f"Optimal rank K = {K_opt}")
```

## Citation

Please cite the *Nature Communications* paper:

```bibtex
@article{cohen2020separability,
  title={Separability and geometry of object manifolds in deep neural networks},
  author={Cohen*, Uri and Chung*, SueYeon and Lee, Daniel D and Sompolinsky, Haim},
  journal={Nature Communications},
  volume={11},
  number={1},
  pages={1--13},
  year={2020},
  publisher={Nature Publishing Group}
}
```

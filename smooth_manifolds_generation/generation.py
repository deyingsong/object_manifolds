"""
Manifold generation: forward-pass images through a DNN under affine transforms.

Workflow :

    # 1. Init
    state = init_imagenet()
    config = GenerationConfig(network_type=NetworkType.ALEXNET,
                              range_factor=0.5, n_objects=128, n_samples=15)
    gen = OneDimensionalManifoldGenerator(config, state)

    # 2. Generate: one call per (direction × batch) pair
    #    run_id goes 1..N_DIRECTIONS*N_BATCHES  (= 1..28 with 7 dirs × 4 batches)
    for run_id in range(1, 29):
        gen.generate(run_id)            # or: gen.generate(batch_id=run_id)

    # 3. Collect: assemble all partial files
    results = gen.collect(range(1, 29)) # {layer_name → (N_DIR, N_OBJ, N_SMP, N_FEAT)}
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .transforms import AffineTransformFactory, ImageNetConfig, TransformType, AffineTransform
from .network import ConvNetExtractor, load_network_metadata, NetworkType
from .imagenet import ImageNetState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Configuration mirroring the arguments of the MATLAB generation scripts."""

    n_objects: int = 128        # P  – number of object images
    n_samples: int = 15         # M  – number of transform samples per manifold
    range_factor: float = 0.5   # controls transform magnitude
    network_type: int = 1       # NetworkType enum value (1=AlexNet, 3=ResNet50, 5=VGG16)
    n_batches: int = 4          # N_BATCHES in MATLAB (divides N_OBJECTS)
    n_transform_dims: int = 2   # N_TRANSFORMATIONS – random transform realizations (RandomManifoldGenerator)
    n_features: int = 4096      # N_HMAX_FEATURES – max features per layer
    feature_seed: int = 1       # RNG seed for random feature sub-selection
    random_seed: int = 0        # RNG seed for image index selection
    output_dir: str = "."       # where to save per-run files
    device: str = "cpu"         # torch device
    epoch: Optional[int] = None # training epoch (None = fully trained; 0 = random init)


# ---------------------------------------------------------------------------
# One-dimensional (7 affine transform types)
# ---------------------------------------------------------------------------

class OneDimensionalManifoldGenerator:
    """
    Generate DNN response manifolds by sweeping one affine-transform parameter.

    run_id maps to a (direction, batch) pair via MATLAB's ind2sub convention::

        param_id     = (run_id - 1) % N_DIRECTIONS + 1   # 1..7
        batch_number = (run_id - 1) // N_DIRECTIONS + 1  # 1..N_BATCHES
    """

    N_DIRECTIONS = 7
    DIRECTION_NAMES = [
        "x-translation", "y-translation",
        "x-scale",        "y-scale",
        "x-shear",        "y-shear",
        "rotation",
    ]

    def __init__(
        self,
        config: GenerationConfig,
        imagenet_state: Optional[ImageNetState] = None,
    ) -> None:
        self.config = config
        self._imagenet_state: Optional[ImageNetState] = imagenet_state
        self._img_config: Optional[ImageNetConfig] = None
        self._factory: Optional[AffineTransformFactory] = None
        self._image_indices: Optional[np.ndarray] = None
        self._feature_indices: Optional[Dict[str, np.ndarray]] = None

        nt = NetworkType(config.network_type)
        self._network_name = nt.name.lower()   # e.g. "alexnet"
        out = Path(config.output_dir) / self._network_name
        out.mkdir(parents=True, exist_ok=True)
        self._out_dir = out

    @property
    def imagenet(self) -> ImageNetState:
        """Lazy-initialise ImageNet state if not provided at construction."""
        if self._imagenet_state is None:
            from .imagenet import init_imagenet
            self._imagenet_state = init_imagenet()
        return self._imagenet_state

    @property
    def img_config(self) -> ImageNetConfig:
        if self._img_config is None:
            st = self.imagenet
            self._img_config = ImageNetConfig(
                image_size=st.image_size,
                frame_size=st.frame_size,
                object_size=st.object_size,
            )
        return self._img_config

    @property
    def factory(self) -> AffineTransformFactory:
        if self._factory is None:
            self._factory = AffineTransformFactory(self.img_config)
        return self._factory

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        run_id: Optional[int] = None,
        *,
        batch_id: Optional[int] = None,
    ) -> str:
        """
        Compute and save tuning functions for one (direction × batch) pair.

        Parameters
        ----------
        run_id : int
            Linear index 1 .. N_DIRECTIONS × N_BATCHES.
            ``batch_id`` is accepted as an alias.

        Returns
        -------
        str
            Path to the saved ``.npz`` file.
        """
        if run_id is None:
            if batch_id is not None:
                run_id = batch_id
            else:
                raise ValueError("Provide run_id or batch_id.")

        n_dir = self.N_DIRECTIONS
        n_batches = self.config.n_batches
        assert 1 <= run_id <= n_dir * n_batches, (
            f"run_id must be in [1, {n_dir * n_batches}], got {run_id}"
        )

        # MATLAB ind2sub([N_DIRECTIONS, N_BATCHES], run_id) – column-major
        param_id    = (run_id - 1) % n_dir + 1        # 1..7
        batch_number = (run_id - 1) // n_dir + 1      # 1..n_batches

        batch_size = self.config.n_objects // n_batches
        all_indices = self._get_image_indices()
        batch_indices = all_indices[
            (batch_number - 1) * batch_size : batch_number * batch_size
        ]   # 1-based image ids, shape (batch_size,)

        out_path = self._run_path(run_id)
        if out_path.exists():
            print(f"  Skipping existing: {out_path.name}")
            return str(out_path)

        print(
            f"  run_id={run_id:3d}  dir={self.DIRECTION_NAMES[param_id-1]:18s}"
            f"  batch {batch_number}/{n_batches}"
        )

        results = self._run_batch(param_id, batch_indices)
        # results: {layer_name: (batch_size, n_samples, n_features)} float32

        np.savez_compressed(
            out_path,
            image_indices=batch_indices,
            param_id=np.array(param_id),
            batch_number=np.array(batch_number),
            direction_name=np.array(self.DIRECTION_NAMES[param_id - 1]),
            **{f"layer_{k}": v for k, v in results.items()},
        )
        return str(out_path)

    def collect(
        self,
        run_ids: Optional[Iterable[int]] = None,
        *,
        batch_ids: Optional[Iterable[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load all saved per-run files and assemble the full tuning function.

        Parameters
        ----------
        run_ids : iterable of int, optional
            Which run_ids to collect.  ``batch_ids`` is accepted as an alias.
            Defaults to all 1..N_DIRECTIONS × N_BATCHES.

        Returns
        -------
        dict  {layer_name → np.ndarray  (N_DIRECTIONS, N_OBJECTS, N_SAMPLES, N_FEATURES)}
        """
        if run_ids is None:
            run_ids = batch_ids
        if run_ids is None:
            n = self.N_DIRECTIONS * self.config.n_batches
            run_ids = range(1, n + 1)
        run_ids = list(run_ids)

        n_dir = self.N_DIRECTIONS
        n_obj = self.config.n_objects
        n_smp = self.config.n_samples
        n_bat = self.config.n_batches
        batch_size = n_obj // n_bat

        # First pass: discover layer names
        first = np.load(self._run_path(run_ids[0]), allow_pickle=True)
        layer_names = [
            k[len("layer_"):] for k in first.files if k.startswith("layer_")
        ]
        n_feat = {ln: first[f"layer_{ln}"].shape[-1] for ln in layer_names}

        # Allocate output
        full: Dict[str, np.ndarray] = {
            ln: np.full((n_dir, n_obj, n_smp, n_feat[ln]), np.nan, dtype=np.float32)
            for ln in layer_names
        }

        for run_id in run_ids:
            param_id    = (run_id - 1) % n_dir + 1
            batch_number = (run_id - 1) // n_dir + 1
            obj_lo = (batch_number - 1) * batch_size
            obj_hi = batch_number * batch_size

            data = np.load(self._run_path(run_id), allow_pickle=True)
            for ln in layer_names:
                # shape: (batch_size, n_samples, n_features)
                full[ln][param_id - 1, obj_lo:obj_hi, :, :] = data[f"layer_{ln}"]

        return full

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_path(self, run_id: int) -> Path:
        cfg = self.config
        rf = cfg.range_factor
        tag = f"{rf:.1f}" if rf >= 0.1 else f"{rf:f}"
        return self._out_dir / (
            f"gen1d_{self._network_name}_range{tag}"
            f"_P{cfg.n_objects}_M{cfg.n_samples}_run{run_id:03d}.npz"
        )

    def _get_image_indices(self) -> np.ndarray:
        """Return fixed 1-based image indices (one per call, cached)."""
        if self._image_indices is None:
            from .imagenet import choose_imagenet_template_images
            self._image_indices = choose_imagenet_template_images(
                self.config.n_objects,
                random_seed=self.config.random_seed if self.config.random_seed != 0 else None,
            )
        return self._image_indices

    def _run_batch(
        self,
        param_id: int,
        batch_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run the network on all (object, sample) pairs for one direction.

        Returns
        -------
        {layer_name: (batch_size, n_samples, n_features)} float32
        """
        from .imagenet import read_imagenet_thumbnails

        cfg = self.config
        n_smp = cfg.n_samples
        batch_size = len(batch_indices)

        # --- Lazy-load the network and get layer names ---
        meta = load_network_metadata(cfg.network_type, epoch=cfg.epoch)
        layer_names = meta.layer_names   # e.g. ['conv1', 'relu1', ...]

        extractor = ConvNetExtractor(
            cfg.network_type,
            layer_names=layer_names,
            device=cfg.device,
            n_features=cfg.n_features,
            feature_seed=cfg.feature_seed,
            epoch=cfg.epoch,
        )

        # Allocate result buffers (filled incrementally)
        results: Optional[Dict[str, np.ndarray]] = None

        T0 = time.time()
        for ii, image_id in enumerate(batch_indices):
            # Load base thumbnail: (C=3, H=144, W=144) uint8
            img_chw_uint8 = read_imagenet_thumbnails(int(image_id))
            img_chw = img_chw_uint8.astype(np.float32)  # still (3, H, W)

            # Build N_SAMPLES warped + preprocessed images
            batch_hwc = []
            for j in range(n_smp):
                transform = self.factory.create_1d(
                    cfg.range_factor, param_id, j, n_smp
                )
                warped_hwc = self._calc_imagenet_warp(img_chw, transform)   # (H_out, W_out, 3) float32
                batch_hwc.append(warped_hwc)

            batch = np.stack(batch_hwc, axis=0)   # (n_smp, H_out, W_out, 3)

            # Forward pass: {layer_name → (n_smp, n_feat)}
            feats = extractor.extract(batch)

            # Allocate on first object
            if results is None:
                results = {
                    ln: np.zeros((batch_size, n_smp, arr.shape[-1]), dtype=np.float32)
                    for ln, arr in feats.items()
                }

            for ln, arr in feats.items():
                if ln in results:
                    results[ln][ii] = arr   # (n_smp, n_feat)

            elapsed = time.time() - T0
            eta = elapsed / (ii + 1) * (batch_size - ii - 1)
            if eta > 60:
                print(f"    object {ii+1}/{batch_size}  ETA {eta/60:.1f} min")
            elif elapsed > 5:
                print(f"    object {ii+1}/{batch_size}  ETA {eta:.0f} sec")

        extractor.close()
        return results

    def _calc_imagenet_warp(
        self,
        img_chw: np.ndarray,
        transform: AffineTransform,
    ) -> np.ndarray:
        """
        input  → (FRAME_SIZE × FRAME_SIZE) world range [-s_in,  s_in]
        output → (IMAGE_SIZE × IMAGE_SIZE) world range [-s_out, s_out]

        Parameters
        ----------
        img_chw : (3, H_in, W_in) float32, values in [0, 255]
            H_in = W_in = frame_size (144 for 64-px setting).
        transform : AffineTransform
            3×3 matrix in MATLAB affine2d convention:
            ``[x', y', 1] = [x, y, 1] @ T``.

        Returns
        -------
        np.ndarray  (H_out, W_out, 3) float32, values in [0, 255]
        """
        from scipy.ndimage import affine_transform as _ndi_warp

        state = self.imagenet
        N_in  = state.frame_size   # 144
        N_out = state.image_size   # 64
        s_in  = float(state.frame_size)  / state.object_size   # 3.0
        s_out = float(state.image_size)  / state.object_size   # 4/3 ≈ 1.3333

        T = transform.matrix          # 3×3, MATLAB convention
        T_inv = np.linalg.inv(T)      # maps output world → input world

        # Pixel-spacing in world units
        # output: pixel col/row → world x/y
        scale_out = 2.0 * s_out / (N_out - 1)   # world units per output pixel
        # input: world x/y → pixel col/row
        scale_in  = (N_in - 1) / (2.0 * s_in)   # input pixels per world unit

        sc = scale_in * scale_out   # combined scale (≈ 1.009 for identity)

        # Build scipy matrix and offset:
        #   in_coord = matrix @ out_coord + offset
        #   coord = [row, col] = [y, x]
        #
        # From derivation (x=col, y=row in image):
        #   row_in = sc*T_inv[1,1]*row_out + sc*T_inv[0,1]*col_out + b_row
        #   col_in = sc*T_inv[1,0]*row_out + sc*T_inv[0,0]*col_out + b_col
        matrix = np.array([
            [sc * T_inv[1, 1],  sc * T_inv[0, 1]],   # row_in from (row_out, col_out)
            [sc * T_inv[1, 0],  sc * T_inv[0, 0]],   # col_in from (row_out, col_out)
        ])
        offset = np.array([
            scale_in * (-s_out * (T_inv[0, 1] + T_inv[1, 1]) + T_inv[2, 1] + s_in),
            scale_in * (-s_out * (T_inv[0, 0] + T_inv[1, 0]) + T_inv[2, 0] + s_in),
        ])

        # Apply to each channel
        out_hwc = np.zeros((N_out, N_out, 3), dtype=np.float32)
        for c in range(3):
            out_hwc[:, :, c] = _ndi_warp(
                img_chw[c].astype(np.float64),
                matrix=matrix,
                offset=offset,
                output_shape=(N_out, N_out),
                order=1,         # bilinear – matches MATLAB imwarp default
                mode="nearest",  # border handling
                prefilter=False,
            )
        return out_hwc


# ---------------------------------------------------------------------------
# Two-dimensional (random transform pairs)
# ---------------------------------------------------------------------------

class RandomManifoldGenerator:
    """
    Generate DNN response manifolds with random affine transforms.

    run_id mapping:

    n_total   = n_batches × n_transform_dims
    batch_num = (run_id - 1) % n_batches + 1          # 1..n_batches
    xform_id  = (run_id - 1) // n_batches + 1         # 1..n_transform_dims

    collect() returns:
        {layer_name → (N_TRANSFORM_DIMS, N_OBJECTS, N_SAMPLES, N_FEATURES)}
    """

    def __init__(
        self,
        config: GenerationConfig,
        imagenet_state: Optional[ImageNetState] = None,
    ) -> None:
        self.config = config
        self._imagenet_state: Optional[ImageNetState] = imagenet_state
        self._img_config: Optional[ImageNetConfig] = None
        self._factory: Optional[AffineTransformFactory] = None
        self._image_indices: Optional[np.ndarray] = None

        nt = NetworkType(config.network_type)
        self._network_name = nt.name.lower()
        out = Path(config.output_dir) / self._network_name
        out.mkdir(parents=True, exist_ok=True)
        self._out_dir = out

    @property
    def imagenet(self) -> ImageNetState:
        if self._imagenet_state is None:
            from .imagenet import init_imagenet
            self._imagenet_state = init_imagenet()
        return self._imagenet_state

    @property
    def img_config(self) -> ImageNetConfig:
        if self._img_config is None:
            st = self.imagenet
            self._img_config = ImageNetConfig(
                image_size=st.image_size,
                frame_size=st.frame_size,
                object_size=st.object_size,
            )
        return self._img_config

    @property
    def factory(self) -> AffineTransformFactory:
        if self._factory is None:
            self._factory = AffineTransformFactory(self.img_config)
        return self._factory

    def generate(
        self,
        run_id: Optional[int] = None,
        *,
        batch_id: Optional[int] = None,
    ) -> str:
        """
        Generate and save one (batch × transform-realization) run.

        Parameters
        ----------
        run_id / batch_id : int
            Linear index 1 .. n_batches × n_transform_dims.
        """
        if run_id is None:
            if batch_id is not None:
                run_id = batch_id
            else:
                raise ValueError("Provide run_id or batch_id.")

        cfg = self.config
        n_batches  = cfg.n_batches
        n_xdims    = cfg.n_transform_dims
        n_total    = n_batches * n_xdims
        assert 1 <= run_id <= n_total, (
            f"run_id must be in [1, {n_total}], got {run_id}"
        )

        # Column-major decomposition (batch iterates fastest)
        batch_number = (run_id - 1) % n_batches + 1   # 1..n_batches
        xform_id     = (run_id - 1) // n_batches + 1  # 1..n_transform_dims

        out_path = self._run_path(run_id)
        if out_path.exists():
            print(f"  Skipping existing: {out_path.name}")
            return str(out_path)

        print(
            f"  run_id={run_id:3d}  batch={batch_number}/{n_batches}"
            f"  xform_realization={xform_id}/{n_xdims}"
        )

        batch_size  = cfg.n_objects // n_batches
        all_indices = self._get_image_indices()
        batch_indices = all_indices[
            (batch_number - 1) * batch_size : batch_number * batch_size
        ]

        meta = load_network_metadata(cfg.network_type, epoch=cfg.epoch)
        extractor = ConvNetExtractor(
            cfg.network_type,
            layer_names=meta.layer_names,
            device=cfg.device,
            n_features=cfg.n_features,
            feature_seed=cfg.feature_seed,
            epoch=cfg.epoch,
        )

        # Each (run_id) gets a unique RNG seed so transforms are independent
        seed = (cfg.random_seed, run_id) if cfg.random_seed else run_id
        rng  = np.random.default_rng(seed)
        n_smp = cfg.n_samples

        results: Optional[Dict[str, np.ndarray]] = None
        T0 = time.time()
        for ii, image_id in enumerate(batch_indices):
            from .imagenet import read_imagenet_thumbnails
            img_chw = read_imagenet_thumbnails(int(image_id)).astype(np.float32)

            batch_hwc = []
            for _ in range(n_smp):
                transform = self.factory.create_random(cfg.range_factor, n_smp, rng)
                warped = self._calc_imagenet_warp(img_chw, transform)
                batch_hwc.append(warped)

            feats = extractor.extract(np.stack(batch_hwc))
            if results is None:
                results = {
                    ln: np.zeros((batch_size, n_smp, v.shape[-1]), dtype=np.float32)
                    for ln, v in feats.items()
                }
            for ln, arr in feats.items():
                if ln in results:
                    results[ln][ii] = arr

            elapsed = time.time() - T0
            eta = elapsed / (ii + 1) * (batch_size - ii - 1)
            if eta > 60:
                print(f"    object {ii+1}/{batch_size}  ETA {eta/60:.1f} min")
            elif elapsed > 5:
                print(f"    object {ii+1}/{batch_size}  ETA {eta:.0f} sec")

        extractor.close()
        np.savez_compressed(
            out_path,
            image_indices=batch_indices,
            batch_number=np.array(batch_number),
            xform_id=np.array(xform_id),
            **{f"layer_{k}": v for k, v in results.items()},
        )
        return str(out_path)

    def collect(
        self,
        run_ids: Optional[Iterable[int]] = None,
        *,
        batch_ids: Optional[Iterable[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Assemble the full tuning function from per-run files.

        Returns
        -------
        dict  {layer_name → np.ndarray (N_TRANSFORM_DIMS, N_OBJECTS, N_SAMPLES, N_FEATURES)}
        """
        if run_ids is None:
            run_ids = batch_ids
        if run_ids is None:
            n_total = self.config.n_batches * self.config.n_transform_dims
            run_ids = range(1, n_total + 1)
        run_ids = list(run_ids)

        cfg       = self.config
        n_obj     = cfg.n_objects
        n_smp     = cfg.n_samples
        n_batches = cfg.n_batches
        n_xdims   = cfg.n_transform_dims
        batch_size = n_obj // n_batches

        first = np.load(self._run_path(run_ids[0]), allow_pickle=True)
        layer_names = [k[len("layer_"):] for k in first.files if k.startswith("layer_")]
        n_feat = {ln: first[f"layer_{ln}"].shape[-1] for ln in layer_names}

        # Output: (N_TRANSFORM_DIMS, N_OBJECTS, N_SAMPLES, N_FEATURES)
        full: Dict[str, np.ndarray] = {
            ln: np.full((n_xdims, n_obj, n_smp, n_feat[ln]), np.nan, dtype=np.float32)
            for ln in layer_names
        }

        for run_id in run_ids:
            batch_number = (run_id - 1) % n_batches + 1
            xform_id     = (run_id - 1) // n_batches + 1
            obj_lo = (batch_number - 1) * batch_size
            obj_hi =  batch_number      * batch_size

            data = np.load(self._run_path(run_id), allow_pickle=True)
            for ln in layer_names:
                full[ln][xform_id - 1, obj_lo:obj_hi, :, :] = data[f"layer_{ln}"]

        return full

    def _run_path(self, run_id: int) -> Path:
        cfg = self.config
        tag = f"{cfg.range_factor:.1f}" if cfg.range_factor >= 0.1 else f"{cfg.range_factor:f}"
        return self._out_dir / (
            f"genrand_{self._network_name}_range{tag}"
            f"_P{cfg.n_objects}_M{cfg.n_samples}_run{run_id:03d}.npz"
        )

    def _get_image_indices(self) -> np.ndarray:
        if self._image_indices is None:
            from .imagenet import choose_imagenet_template_images
            self._image_indices = choose_imagenet_template_images(
                self.config.n_objects,
                random_seed=self.config.random_seed or None,
            )
        return self._image_indices

    def _calc_imagenet_warp(
        self, img_chw: np.ndarray, transform: AffineTransform
    ) -> np.ndarray:
        """Same world-coordinate warp as in OneDimensionalManifoldGenerator."""
        from scipy.ndimage import affine_transform as _ndi_warp

        state = self.imagenet
        N_in  = state.frame_size
        N_out = state.image_size
        s_in  = float(state.frame_size) / state.object_size
        s_out = float(state.image_size) / state.object_size

        T = transform.matrix
        T_inv = np.linalg.inv(T)
        scale_out = 2.0 * s_out / (N_out - 1)
        scale_in  = (N_in - 1) / (2.0 * s_in)
        sc = scale_in * scale_out

        matrix = np.array([
            [sc * T_inv[1, 1],  sc * T_inv[0, 1]],
            [sc * T_inv[1, 0],  sc * T_inv[0, 0]],
        ])
        offset = np.array([
            scale_in * (-s_out * (T_inv[0, 1] + T_inv[1, 1]) + T_inv[2, 1] + s_in),
            scale_in * (-s_out * (T_inv[0, 0] + T_inv[1, 0]) + T_inv[2, 0] + s_in),
        ])
        out_hwc = np.zeros((N_out, N_out, 3), dtype=np.float32)
        for c in range(3):
            out_hwc[:, :, c] = _ndi_warp(
                img_chw[c].astype(np.float64),
                matrix=matrix,
                offset=offset,
                output_shape=(N_out, N_out),
                order=1,
                mode="nearest",
                prefilter=False,
            )
        return out_hwc

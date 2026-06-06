"""
ImageNet initialization and configuration module.

Provides global configuration for ImageNet image sizes, object sizes, and features.
"""

from dataclasses import dataclass


@dataclass
class ImageNetConfig:
    """Configuration for ImageNet processing."""
    
    # Image processing parameters
    OBJECT_SIZE: int = 48
    FRAME: int = 8
    SURROUND_FACTOR: int = 3
    USE_NEW_IMAGENET_WARP: bool = True
    
    # Derived parameters (computed automatically)
    FRAME_SIZE: int = 0
    IMAGE_SIZE: int = 0
    FRAME_LIMITS: tuple = None
    
    # Feature parameters
    N_HMAX_FEATURES: int = 4096
    
    def __post_init__(self):
        """Compute derived parameters."""
        self.FRAME_SIZE = self.OBJECT_SIZE * self.SURROUND_FACTOR
        self.IMAGE_SIZE = self.OBJECT_SIZE + 2 * self.FRAME
        frame_start = 1 + (self.FRAME_SIZE - self.IMAGE_SIZE) / 2
        frame_end = self.FRAME_SIZE / 2 + self.IMAGE_SIZE / 2
        self.FRAME_LIMITS = (frame_start, frame_end)
    
    def get_info_string(self) -> str:
        """Get formatted information string."""
        return (
            f"Objects are {self.OBJECT_SIZE} x {self.OBJECT_SIZE} in the middle of "
            f"{self.IMAGE_SIZE} x {self.IMAGE_SIZE} image"
        )


# Global singleton configuration instance
_global_config = None


def get_imagenet_config() -> ImageNetConfig:
    """Get the global ImageNet configuration."""
    global _global_config
    if _global_config is None:
        _global_config = ImageNetConfig()
    return _global_config


def set_imagenet_config(config: ImageNetConfig) -> None:
    """Set the global ImageNet configuration."""
    global _global_config
    _global_config = config


def init_imagenet(object_size: int = 48, frame: int = 8, surround_factor: int = 3,
                  n_hmax_features: int = 4096) -> ImageNetConfig:
    """
    Initialize ImageNet configuration with specified parameters.
    
    Parameters
    ----------
    object_size : int, optional
        Size of the square object (default: 48)
    frame : int, optional
        Size of the frame around object (default: 8)
    surround_factor : int, optional
        Surround factor (default: 3)
    n_hmax_features : int, optional
        Number of HMAX features (default: 4096)
    
    Returns
    -------
    ImageNetConfig
        The initialized configuration
    
    Notes
    -----
    Image width and height = frame + object_size + frame
    """
    config = ImageNetConfig(
        OBJECT_SIZE=object_size,
        FRAME=frame,
        SURROUND_FACTOR=surround_factor,
        N_HMAX_FEATURES=n_hmax_features
    )
    set_imagenet_config(config)
    print(config.get_info_string())
    return config

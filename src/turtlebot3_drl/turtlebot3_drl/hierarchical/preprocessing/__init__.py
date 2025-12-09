"""
Preprocessing modules for hierarchical navigation.

Contains:
- LiDAR processing (downsampling, normalization, sector conversion)
- Attention mechanism for LiDAR feature extraction
- Path feature extraction module
"""

from .lidar_processor import LidarProcessor, LidarProcessorTorch
from .attention import (
    LidarAttention,
    LidarAttentionEfficient,
    PathModule,
    CombinedFeatureExtractor,
    make_mlp
)

__all__ = [
    # LiDAR processing
    'LidarProcessor',
    'LidarProcessorTorch',
    # Attention
    'LidarAttention',
    'LidarAttentionEfficient',
    'PathModule',
    'CombinedFeatureExtractor',
    'make_mlp'
]

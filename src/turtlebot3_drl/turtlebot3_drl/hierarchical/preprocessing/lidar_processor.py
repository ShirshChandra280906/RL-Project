"""
LiDAR Preprocessor for Hierarchical Navigation

Downsamples and preprocesses LiDAR scans for the Subgoal Agent.
Based on the paper: "Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Key features:
- Downsamples 360 rays to 80 rays via min-pooling
- Clips values at max range
- Prepares data for attention mechanism (10 sectors × 8 rays)
"""

import numpy as np
from typing import Tuple, Optional


class LidarProcessor:
    """
    LiDAR scan preprocessor for hierarchical navigation.
    
    Downsamples raw LiDAR scans using min-pooling to preserve
    obstacle information while reducing dimensionality.
    """
    
    def __init__(
        self,
        input_rays: int = 360,
        output_rays: int = 80,
        max_range: float = 3.5,
        clip_range: float = 4.0,
        num_sectors: int = 10
    ):
        """
        Initialize LiDAR processor.
        
        Args:
            input_rays: Number of rays in raw scan (TurtleBot3: 360)
            output_rays: Number of rays after downsampling (paper: 80)
            max_range: LiDAR max range in meters (TurtleBot3 LDS-01: 3.5m)
            clip_range: Clip range for normalization (paper: 4.0m)
            num_sectors: Number of angular sectors for attention (paper: 10)
        """
        self.input_rays = input_rays
        self.output_rays = output_rays
        self.max_range = max_range
        self.clip_range = clip_range
        self.num_sectors = num_sectors
        
        # Calculate pooling parameters
        # For 360->80: pool_size = 360/80 = 4.5, use 4 with some overlap handling
        self.pool_size = input_rays // output_rays
        self.rays_per_sector = output_rays // num_sectors  # 80/10 = 8
        
        # Validate configuration
        assert output_rays % num_sectors == 0, \
            f"output_rays ({output_rays}) must be divisible by num_sectors ({num_sectors})"
        
        # Pre-compute pooling indices for efficiency
        self._compute_pooling_indices()
    
    def _compute_pooling_indices(self) -> None:
        """Pre-compute indices for min-pooling operation."""
        self.pool_indices = []
        
        # Handle case where input_rays is not perfectly divisible
        step = self.input_rays / self.output_rays
        
        for i in range(self.output_rays):
            start_idx = int(i * step)
            end_idx = int((i + 1) * step)
            # Ensure at least one ray per output
            if end_idx == start_idx:
                end_idx = start_idx + 1
            self.pool_indices.append((start_idx, min(end_idx, self.input_rays)))
    
    def process(self, raw_scan: np.ndarray) -> np.ndarray:
        """
        Process raw LiDAR scan.
        
        Args:
            raw_scan: Raw LiDAR scan of shape (input_rays,) or (batch, input_rays)
            
        Returns:
            Processed scan of shape (output_rays,) or (batch, output_rays)
        """
        # Handle single scan vs batch
        single_scan = raw_scan.ndim == 1
        if single_scan:
            raw_scan = raw_scan[np.newaxis, :]
        
        batch_size = raw_scan.shape[0]
        processed = np.zeros((batch_size, self.output_rays), dtype=np.float32)
        
        # Replace inf/nan with max_range
        raw_scan = np.nan_to_num(raw_scan, nan=self.max_range, posinf=self.max_range)
        
        # Min-pooling: take minimum value in each pool (closest obstacle)
        for i, (start, end) in enumerate(self.pool_indices):
            processed[:, i] = np.min(raw_scan[:, start:end], axis=1)
        
        # Clip to max range
        processed = np.clip(processed, 0, self.clip_range)
        
        if single_scan:
            processed = processed[0]
        
        return processed
    
    def process_normalized(self, raw_scan: np.ndarray) -> np.ndarray:
        """
        Process and normalize LiDAR scan to [0, 1] range.
        
        Args:
            raw_scan: Raw LiDAR scan
            
        Returns:
            Normalized scan with values in [0, 1]
        """
        processed = self.process(raw_scan)
        return processed / self.clip_range
    
    def to_sectors(self, processed_scan: np.ndarray) -> np.ndarray:
        """
        Reshape processed scan into sectors for attention mechanism.
        
        Args:
            processed_scan: Processed scan of shape (output_rays,) or (batch, output_rays)
            
        Returns:
            Sectored scan of shape (num_sectors, rays_per_sector) or 
            (batch, num_sectors, rays_per_sector)
        """
        single_scan = processed_scan.ndim == 1
        if single_scan:
            processed_scan = processed_scan[np.newaxis, :]
        
        batch_size = processed_scan.shape[0]
        sectors = processed_scan.reshape(batch_size, self.num_sectors, self.rays_per_sector)
        
        if single_scan:
            sectors = sectors[0]
        
        return sectors
    
    def get_closest_obstacle(self, processed_scan: np.ndarray) -> Tuple[float, int]:
        """
        Get the distance and index of the closest obstacle.
        
        Args:
            processed_scan: Processed LiDAR scan (1D)
            
        Returns:
            Tuple of (distance, ray_index)
        """
        min_idx = np.argmin(processed_scan)
        min_dist = processed_scan[min_idx]
        return float(min_dist), int(min_idx)
    
    def get_sector_minimums(self, processed_scan: np.ndarray) -> np.ndarray:
        """
        Get the minimum distance in each sector.
        
        Args:
            processed_scan: Processed LiDAR scan
            
        Returns:
            Array of shape (num_sectors,) with min distance per sector
        """
        sectors = self.to_sectors(processed_scan)
        if sectors.ndim == 2:
            return np.min(sectors, axis=1)
        else:
            return np.min(sectors, axis=2)
    
    def get_angular_resolution(self) -> float:
        """Get angular resolution of processed scan in degrees."""
        return 360.0 / self.output_rays
    
    def ray_to_angle(self, ray_index: int) -> float:
        """
        Convert ray index to angle in radians.
        
        Args:
            ray_index: Index of the ray (0 to output_rays-1)
            
        Returns:
            Angle in radians (0 = front, positive = counterclockwise)
        """
        angle_deg = ray_index * self.get_angular_resolution()
        return np.radians(angle_deg)
    
    def angle_to_ray(self, angle_rad: float) -> int:
        """
        Convert angle to ray index.
        
        Args:
            angle_rad: Angle in radians
            
        Returns:
            Closest ray index
        """
        angle_deg = np.degrees(angle_rad) % 360
        ray_idx = int(round(angle_deg / self.get_angular_resolution()))
        return ray_idx % self.output_rays


class LidarProcessorTorch:
    """
    PyTorch-compatible LiDAR processor for use in neural networks.
    
    Provides the same functionality as LidarProcessor but operates
    on PyTorch tensors for GPU acceleration.
    """
    
    def __init__(
        self,
        input_rays: int = 360,
        output_rays: int = 80,
        max_range: float = 3.5,
        clip_range: float = 4.0,
        num_sectors: int = 10
    ):
        """Initialize PyTorch LiDAR processor."""
        import torch
        
        self.input_rays = input_rays
        self.output_rays = output_rays
        self.max_range = max_range
        self.clip_range = clip_range
        self.num_sectors = num_sectors
        self.rays_per_sector = output_rays // num_sectors
        
        # Pre-compute pooling indices
        step = input_rays / output_rays
        self.pool_indices = []
        for i in range(output_rays):
            start_idx = int(i * step)
            end_idx = int((i + 1) * step)
            if end_idx == start_idx:
                end_idx = start_idx + 1
            self.pool_indices.append((start_idx, min(end_idx, input_rays)))
    
    def process(self, raw_scan):
        """
        Process raw LiDAR scan (PyTorch tensor).
        
        Args:
            raw_scan: Tensor of shape (batch, input_rays)
            
        Returns:
            Processed tensor of shape (batch, output_rays)
        """
        import torch
        
        device = raw_scan.device
        batch_size = raw_scan.shape[0]
        
        # Replace inf/nan with max_range
        raw_scan = torch.where(
            torch.isnan(raw_scan) | torch.isinf(raw_scan),
            torch.tensor(self.max_range, device=device),
            raw_scan
        )
        
        # Min-pooling
        processed = torch.zeros(batch_size, self.output_rays, device=device)
        for i, (start, end) in enumerate(self.pool_indices):
            processed[:, i] = torch.min(raw_scan[:, start:end], dim=1)[0]
        
        # Clip
        processed = torch.clamp(processed, 0, self.clip_range)
        
        return processed
    
    def to_sectors(self, processed_scan):
        """
        Reshape to sectors for attention.
        
        Args:
            processed_scan: Tensor of shape (batch, output_rays)
            
        Returns:
            Tensor of shape (batch, num_sectors, rays_per_sector)
        """
        batch_size = processed_scan.shape[0]
        return processed_scan.view(batch_size, self.num_sectors, self.rays_per_sector)


if __name__ == "__main__":
    # Test LiDAR processor
    print("=" * 60)
    print("LiDAR Processor Test")
    print("=" * 60)
    
    processor = LidarProcessor(
        input_rays=360,
        output_rays=80,
        max_range=3.5,
        clip_range=4.0,
        num_sectors=10
    )
    
    # Test 1: Basic processing
    print("\n--- Test 1: Basic Processing ---")
    raw_scan = np.full(360, 3.5)  # All at max range
    processed = processor.process(raw_scan)
    print(f"Input shape: {raw_scan.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"All values at max range: {np.allclose(processed, 3.5)}")
    
    # Test 2: Obstacle detection
    print("\n--- Test 2: Obstacle Detection ---")
    raw_scan = np.full(360, 3.5)
    raw_scan[40:50] = 0.5  # Obstacle at ~40-50° range
    processed = processor.process(raw_scan)
    
    min_dist, min_idx = processor.get_closest_obstacle(processed)
    print(f"Closest obstacle: {min_dist:.2f}m at ray {min_idx}")
    print(f"Expected: ~0.5m at ray ~11 (40°/4.5°)")
    
    # Test 3: Sector conversion
    print("\n--- Test 3: Sector Conversion ---")
    sectors = processor.to_sectors(processed)
    print(f"Sectors shape: {sectors.shape}")
    print(f"Expected: (10, 8)")
    
    sector_mins = processor.get_sector_minimums(processed)
    print(f"Sector minimums: {sector_mins}")
    
    # Test 4: Batch processing
    print("\n--- Test 4: Batch Processing ---")
    batch_scan = np.random.uniform(0.5, 3.5, (4, 360))
    batch_processed = processor.process(batch_scan)
    print(f"Batch input shape: {batch_scan.shape}")
    print(f"Batch output shape: {batch_processed.shape}")
    
    # Test 5: Normalization
    print("\n--- Test 5: Normalization ---")
    normalized = processor.process_normalized(raw_scan)
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"Expected: [0, 1]")
    
    # Test 6: Angular conversion
    print("\n--- Test 6: Angular Conversion ---")
    print(f"Angular resolution: {processor.get_angular_resolution():.2f}°")
    print(f"Ray 0 angle: {np.degrees(processor.ray_to_angle(0)):.1f}°")
    print(f"Ray 20 angle: {np.degrees(processor.ray_to_angle(20)):.1f}°")
    print(f"90° -> ray: {processor.angle_to_ray(np.radians(90))}")
    
    print("\n✓ LiDAR Processor test complete!")

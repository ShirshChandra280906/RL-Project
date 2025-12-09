"""
Attention Module for Hierarchical Navigation

Implements sector-based attention mechanism for LiDAR processing.
Based on the paper: "Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Key features:
- Processes 80 LiDAR rays as 10 sectors × 8 rays
- Shared embedding network across sectors
- Attention-weighted feature aggregation
- Outputs 64-dimensional feature vector
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


def make_mlp(layer_sizes: List[int], activation: nn.Module = nn.ReLU) -> nn.Sequential:
    """
    Create a multi-layer perceptron.
    
    Args:
        layer_sizes: List of layer sizes [input, hidden1, ..., output]
        activation: Activation function to use between layers
        
    Returns:
        Sequential MLP module
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:  # No activation after last layer
            layers.append(activation())
    return nn.Sequential(*layers)


class LidarAttention(nn.Module):
    """
    Attention-based LiDAR feature extractor.
    
    Architecture from paper:
    - Embedding module: [512, 256, 128] - shared across sectors
    - Feature module: [256, 128, 64] - extracts features from embeddings
    - Score module: [128, 64, 1] - computes attention scores
    - Output: Weighted sum of features (64-dim)
    """
    
    def __init__(
        self,
        num_sectors: int = 10,
        rays_per_sector: int = 8,
        feature_dim: int = 64,
        embedding_layers: List[int] = None,
        feature_layers: List[int] = None,
        score_layers: List[int] = None
    ):
        """
        Initialize attention module.
        
        Args:
            num_sectors: Number of angular sectors (default: 10)
            rays_per_sector: Rays per sector (default: 8)
            feature_dim: Output feature dimension (default: 64)
            embedding_layers: Layer sizes for embedding MLP
            feature_layers: Layer sizes for feature MLP
            score_layers: Layer sizes for score MLP
        """
        super().__init__()
        
        self.num_sectors = num_sectors
        self.rays_per_sector = rays_per_sector
        self.feature_dim = feature_dim
        
        # Default layer sizes from paper
        if embedding_layers is None:
            embedding_layers = [rays_per_sector, 512, 256, 128]
        else:
            embedding_layers = [rays_per_sector] + embedding_layers
            
        if feature_layers is None:
            feature_layers = [128, 256, 128, feature_dim]
        else:
            feature_layers = [embedding_layers[-1]] + feature_layers
            
        if score_layers is None:
            score_layers = [128, 128, 64, 1]
        else:
            score_layers = [embedding_layers[-1]] + score_layers
        
        # Shared embedding network (processes each sector)
        self.embedding_net = make_mlp(embedding_layers)
        
        # Feature extraction network
        self.feature_net = make_mlp(feature_layers)
        
        # Attention score network
        self.score_net = make_mlp(score_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        lidar: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through attention module.
        
        Args:
            lidar: LiDAR tensor of shape (batch, 80) or (batch, 10, 8)
            return_attention: Whether to return attention weights
            
        Returns:
            features: Output features of shape (batch, feature_dim)
            attention_weights: Attention weights of shape (batch, num_sectors)
                              or None if return_attention=False
        """
        batch_size = lidar.shape[0]
        
        # Reshape to sectors if needed
        if lidar.dim() == 2 and lidar.shape[1] == self.num_sectors * self.rays_per_sector:
            lidar = lidar.view(batch_size, self.num_sectors, self.rays_per_sector)
        
        assert lidar.shape == (batch_size, self.num_sectors, self.rays_per_sector), \
            f"Expected shape ({batch_size}, {self.num_sectors}, {self.rays_per_sector}), got {lidar.shape}"
        
        # Process each sector through shared embedding network
        # Shape: (batch, num_sectors, embedding_dim)
        embeddings = []
        for i in range(self.num_sectors):
            sector_data = lidar[:, i, :]  # (batch, rays_per_sector)
            embedding = self.embedding_net(sector_data)  # (batch, 128)
            embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings, dim=1)  # (batch, num_sectors, 128)
        
        # Extract features from embeddings
        # Shape: (batch, num_sectors, feature_dim)
        features = []
        for i in range(self.num_sectors):
            feature = self.feature_net(embeddings[:, i, :])  # (batch, 64)
            features.append(feature)
        
        features = torch.stack(features, dim=1)  # (batch, num_sectors, 64)
        
        # Compute attention scores
        # Shape: (batch, num_sectors, 1) -> (batch, num_sectors)
        scores = []
        for i in range(self.num_sectors):
            score = self.score_net(embeddings[:, i, :])  # (batch, 1)
            scores.append(score)
        
        scores = torch.cat(scores, dim=1)  # (batch, num_sectors)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, num_sectors)
        
        # Weighted sum of features
        # attention_weights: (batch, num_sectors, 1)
        # features: (batch, num_sectors, feature_dim)
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (batch, num_sectors, 1)
        weighted_features = features * attention_weights_expanded  # (batch, num_sectors, 64)
        output = weighted_features.sum(dim=1)  # (batch, 64)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None
    
    def get_attention_visualization(
        self,
        lidar: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Get attention weights for visualization.
        
        Args:
            lidar: Single LiDAR scan (80,) or (1, 80)
            
        Returns:
            sector_values: Mean LiDAR value per sector
            attention_weights: Attention weight per sector
        """
        if lidar.dim() == 1:
            lidar = lidar.unsqueeze(0)
        
        with torch.no_grad():
            _, attention_weights = self.forward(lidar)
        
        # Reshape to sectors and get mean
        sectors = lidar.view(self.num_sectors, self.rays_per_sector)
        sector_values = sectors.mean(dim=1)
        
        return sector_values.cpu().numpy(), attention_weights[0].cpu().numpy()


class LidarAttentionEfficient(nn.Module):
    """
    Efficient implementation of LiDAR attention using batch operations.
    
    More efficient for GPU by avoiding loops over sectors.
    """
    
    def __init__(
        self,
        num_sectors: int = 10,
        rays_per_sector: int = 8,
        feature_dim: int = 64,
        embedding_dim: int = 128
    ):
        """
        Initialize efficient attention module.
        
        Args:
            num_sectors: Number of angular sectors
            rays_per_sector: Rays per sector
            feature_dim: Output feature dimension
            embedding_dim: Intermediate embedding dimension
        """
        super().__init__()
        
        self.num_sectors = num_sectors
        self.rays_per_sector = rays_per_sector
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Embedding network
        self.embed1 = nn.Linear(rays_per_sector, 512)
        self.embed2 = nn.Linear(512, 256)
        self.embed3 = nn.Linear(256, embedding_dim)
        
        # Feature network
        self.feat1 = nn.Linear(embedding_dim, 256)
        self.feat2 = nn.Linear(256, 128)
        self.feat3 = nn.Linear(128, feature_dim)
        
        # Score network
        self.score1 = nn.Linear(embedding_dim, 128)
        self.score2 = nn.Linear(128, 64)
        self.score3 = nn.Linear(64, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        lidar: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Efficient forward pass.
        
        Args:
            lidar: LiDAR tensor of shape (batch, 80)
            return_attention: Whether to return attention weights
            
        Returns:
            features: Output features of shape (batch, feature_dim)
            attention_weights: Attention weights of shape (batch, num_sectors)
        """
        batch_size = lidar.shape[0]
        
        # Reshape: (batch, 80) -> (batch * num_sectors, rays_per_sector)
        lidar_sectors = lidar.view(batch_size * self.num_sectors, self.rays_per_sector)
        
        # Embedding: shared weights applied to all sectors at once
        embed = F.relu(self.embed1(lidar_sectors))
        embed = F.relu(self.embed2(embed))
        embed = F.relu(self.embed3(embed))  # (batch * num_sectors, embedding_dim)
        
        # Features
        feat = F.relu(self.feat1(embed))
        feat = F.relu(self.feat2(feat))
        feat = self.feat3(feat)  # (batch * num_sectors, feature_dim)
        
        # Scores
        score = F.relu(self.score1(embed))
        score = F.relu(self.score2(score))
        score = self.score3(score)  # (batch * num_sectors, 1)
        
        # Reshape back
        feat = feat.view(batch_size, self.num_sectors, self.feature_dim)
        score = score.view(batch_size, self.num_sectors)
        
        # Attention
        attention_weights = F.softmax(score, dim=1)
        
        # Weighted sum
        output = (feat * attention_weights.unsqueeze(-1)).sum(dim=1)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None


class PathModule(nn.Module):
    """
    Path processing module for waypoint features.
    
    Processes 5 waypoints (10 values: x1,y1,x2,y2,...) into
    a 32-dimensional feature vector.
    """
    
    def __init__(
        self,
        input_dim: int = 10,  # 5 waypoints × 2 coords
        output_dim: int = 32,
        hidden_layers: List[int] = None
    ):
        """
        Initialize path module.
        
        Args:
            input_dim: Input dimension (num_waypoints * 2)
            output_dim: Output feature dimension
            hidden_layers: Hidden layer sizes
        """
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64]
        
        layer_sizes = [input_dim] + hidden_layers + [output_dim]
        self.network = make_mlp(layer_sizes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        """
        Process waypoints.
        
        Args:
            waypoints: Waypoint tensor of shape (batch, 10)
            
        Returns:
            Features of shape (batch, output_dim)
        """
        return self.network(waypoints)


class CombinedFeatureExtractor(nn.Module):
    """
    Combined feature extractor for Subgoal Agent.
    
    Combines:
    - LiDAR attention features (64-dim)
    - Path features (32-dim)
    
    Output: 96-dim combined feature vector
    """
    
    def __init__(
        self,
        lidar_rays: int = 80,
        num_waypoints: int = 5,
        lidar_feature_dim: int = 64,
        path_feature_dim: int = 32
    ):
        """
        Initialize combined feature extractor.
        
        Args:
            lidar_rays: Number of LiDAR rays
            num_waypoints: Number of waypoints
            lidar_feature_dim: LiDAR feature dimension
            path_feature_dim: Path feature dimension
        """
        super().__init__()
        
        self.lidar_attention = LidarAttentionEfficient(
            num_sectors=10,
            rays_per_sector=lidar_rays // 10,
            feature_dim=lidar_feature_dim
        )
        
        self.path_module = PathModule(
            input_dim=num_waypoints * 2,
            output_dim=path_feature_dim
        )
        
        self.lidar_feature_dim = lidar_feature_dim
        self.path_feature_dim = path_feature_dim
        self.output_dim = lidar_feature_dim + path_feature_dim
    
    def forward(
        self,
        lidar: torch.Tensor,
        waypoints: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract combined features.
        
        Args:
            lidar: LiDAR tensor (batch, 80)
            waypoints: Waypoint tensor (batch, 10)
            return_attention: Whether to return attention weights
            
        Returns:
            combined: Combined features (batch, 96)
            attention: Attention weights if requested
        """
        lidar_features, attention = self.lidar_attention(lidar, return_attention)
        path_features = self.path_module(waypoints)
        
        combined = torch.cat([lidar_features, path_features], dim=1)
        
        return combined, attention


# Update preprocessing __init__.py exports
__all__ = [
    'LidarAttention',
    'LidarAttentionEfficient', 
    'PathModule',
    'CombinedFeatureExtractor',
    'make_mlp'
]


if __name__ == "__main__":
    # Test attention module
    print("=" * 60)
    print("Attention Module Test")
    print("=" * 60)
    
    # Test 1: Basic LidarAttention
    print("\n--- Test 1: LidarAttention ---")
    attention = LidarAttention(
        num_sectors=10,
        rays_per_sector=8,
        feature_dim=64
    )
    
    lidar_batch = torch.rand(4, 80)
    output, weights = attention(lidar_batch)
    
    print(f"Input shape: {lidar_batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (4, 64)")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Expected: (4, 10)")
    print(f"Attention sums to 1: {torch.allclose(weights.sum(dim=1), torch.ones(4))}")
    
    # Test 2: Efficient version
    print("\n--- Test 2: LidarAttentionEfficient ---")
    efficient_attention = LidarAttentionEfficient(
        num_sectors=10,
        rays_per_sector=8,
        feature_dim=64
    )
    
    output_eff, weights_eff = efficient_attention(lidar_batch)
    
    print(f"Output shape: {output_eff.shape}")
    print(f"Attention weights shape: {weights_eff.shape}")
    print(f"Attention sums to 1: {torch.allclose(weights_eff.sum(dim=1), torch.ones(4))}")
    
    # Test 3: PathModule
    print("\n--- Test 3: PathModule ---")
    path_module = PathModule(input_dim=10, output_dim=32)
    
    waypoints = torch.rand(4, 10)  # 5 waypoints × 2 coords
    path_features = path_module(waypoints)
    
    print(f"Input shape: {waypoints.shape}")
    print(f"Output shape: {path_features.shape}")
    print(f"Expected: (4, 32)")
    
    # Test 4: CombinedFeatureExtractor
    print("\n--- Test 4: CombinedFeatureExtractor ---")
    extractor = CombinedFeatureExtractor(
        lidar_rays=80,
        num_waypoints=5,
        lidar_feature_dim=64,
        path_feature_dim=32
    )
    
    combined, attention = extractor(lidar_batch, waypoints, return_attention=True)
    
    print(f"Combined output shape: {combined.shape}")
    print(f"Expected: (4, 96)")
    print(f"Attention shape: {attention.shape}")
    
    # Test 5: No NaN/Inf
    print("\n--- Test 5: Numerical Stability ---")
    has_nan = torch.isnan(output).any() or torch.isnan(weights).any()
    has_inf = torch.isinf(output).any() or torch.isinf(weights).any()
    print(f"Has NaN: {has_nan}")
    print(f"Has Inf: {has_inf}")
    
    # Test 6: Gradient flow
    print("\n--- Test 6: Gradient Flow ---")
    lidar_grad = torch.rand(2, 80, requires_grad=True)
    waypoints_grad = torch.rand(2, 10, requires_grad=True)
    
    output, _ = extractor(lidar_grad, waypoints_grad)
    loss = output.sum()
    loss.backward()
    
    print(f"LiDAR gradient exists: {lidar_grad.grad is not None}")
    print(f"Waypoints gradient exists: {waypoints_grad.grad is not None}")
    print(f"LiDAR gradient norm: {lidar_grad.grad.norm():.4f}")
    print(f"Waypoints gradient norm: {waypoints_grad.grad.norm():.4f}")
    
    # Count parameters
    print("\n--- Parameter Count ---")
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"LidarAttention parameters: {total_params:,}")
    
    total_params_eff = sum(p.numel() for p in efficient_attention.parameters())
    print(f"LidarAttentionEfficient parameters: {total_params_eff:,}")
    
    total_params_combined = sum(p.numel() for p in extractor.parameters())
    print(f"CombinedFeatureExtractor parameters: {total_params_combined:,}")
    
    print("\n✓ Attention Module test complete!")

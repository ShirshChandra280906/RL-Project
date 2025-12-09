"""
Neural Network Architectures for Hierarchical Navigation Agents

Contains network architectures for:
- Subgoal Agent (SA): Actor and Critic networks with attention
- Motion Agent (MA): Actor and Critic networks (simple MLP)

Based on the paper: "Lightweight Motion Planning via Hierarchical Reinforcement Learning"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


def make_mlp(
    layer_sizes: List[int],
    activation: nn.Module = nn.ReLU,
    output_activation: bool = False
) -> nn.Sequential:
    """
    Create a multi-layer perceptron.
    
    Args:
        layer_sizes: List of layer sizes [input, hidden1, ..., output]
        activation: Activation function to use between layers
        output_activation: Whether to add activation after last layer
        
    Returns:
        Sequential MLP module
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2 or output_activation:
            layers.append(activation())
    return nn.Sequential(*layers)


def init_weights(module: nn.Module, gain: float = 1.0):
    """Initialize weights with xavier uniform."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


# =============================================================================
# SUBGOAL AGENT NETWORKS (DDPG)
# =============================================================================

class SALidarModule(nn.Module):
    """
    LiDAR attention module for Subgoal Agent.
    
    Architecture from paper:
    - Embedding: [8 -> 512 -> 256 -> 128]
    - Feature: [128 -> 256 -> 128 -> 64]
    - Score: [128 -> 128 -> 64 -> 1]
    - Output: 64-dim weighted sum
    """
    
    def __init__(
        self,
        num_sectors: int = 10,
        rays_per_sector: int = 8,
        embedding_layers: List[int] = None,
        feature_layers: List[int] = None,
        score_layers: List[int] = None
    ):
        super().__init__()
        
        self.num_sectors = num_sectors
        self.rays_per_sector = rays_per_sector
        
        # Default architecture from paper
        if embedding_layers is None:
            embedding_layers = [512, 256, 128]
        if feature_layers is None:
            feature_layers = [256, 128, 64]
        if score_layers is None:
            score_layers = [128, 64, 1]
        
        self.embedding_dim = embedding_layers[-1]
        self.feature_dim = feature_layers[-1]
        
        # Build networks
        self.embedding_net = make_mlp(
            [rays_per_sector] + embedding_layers,
            output_activation=True
        )
        self.feature_net = make_mlp(
            [self.embedding_dim] + feature_layers,
            output_activation=False  # No activation on feature output
        )
        self.score_net = make_mlp(
            [self.embedding_dim] + score_layers,
            output_activation=False  # No activation before softmax
        )
        
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))
    
    def forward(self, lidar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process LiDAR through attention mechanism.
        
        Args:
            lidar: LiDAR tensor (batch, 80)
            
        Returns:
            features: Attention-weighted features (batch, 64)
            attention: Attention weights (batch, 10)
        """
        batch_size = lidar.shape[0]
        
        # Reshape to sectors: (batch, 10, 8)
        sectors = lidar.view(batch_size, self.num_sectors, self.rays_per_sector)
        
        # Process all sectors at once: (batch * 10, 8) -> (batch * 10, 128)
        sectors_flat = sectors.view(-1, self.rays_per_sector)
        embeddings = self.embedding_net(sectors_flat)
        
        # Extract features and scores
        features = self.feature_net(embeddings)  # (batch * 10, 64)
        scores = self.score_net(embeddings)      # (batch * 10, 1)
        
        # Reshape back
        features = features.view(batch_size, self.num_sectors, self.feature_dim)
        scores = scores.view(batch_size, self.num_sectors)
        
        # Softmax attention
        attention = F.softmax(scores, dim=1)
        
        # Weighted sum: (batch, 10, 1) * (batch, 10, 64) -> sum -> (batch, 64)
        weighted = features * attention.unsqueeze(-1)
        output = weighted.sum(dim=1)
        
        return output, attention


class SAPathModule(nn.Module):
    """
    Path processing module for Subgoal Agent.
    
    Architecture from paper: [128, 64, 32]
    Input: 5 waypoints × 2 coords = 10
    Output: 32-dim features
    """
    
    def __init__(
        self,
        num_waypoints: int = 5,
        hidden_layers: List[int] = None
    ):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        input_dim = num_waypoints * 2
        self.output_dim = hidden_layers[-1]
        
        self.network = make_mlp(
            [input_dim] + hidden_layers,
            output_activation=True
        )
        
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))
    
    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        """
        Process waypoints.
        
        Args:
            waypoints: Waypoint tensor (batch, 10)
            
        Returns:
            Path features (batch, 32)
        """
        return self.network(waypoints)


class SAOutputModule(nn.Module):
    """
    Output module for Subgoal Agent.
    
    Architecture from paper: [128, 64, 64] + final layer
    - Actor: final layer outputs 2 (l, θ)
    - Critic: final layer outputs 1 (Q-value)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int] = None
    ):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 64]
        
        self.hidden = make_mlp(
            [input_dim] + hidden_layers,
            output_activation=True
        )
        
        # Final layer without activation (paper: "non-activated layer")
        self.output = nn.Linear(hidden_layers[-1], output_dim)
        
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))
        # Smaller init for output layer
        nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
        nn.init.constant_(self.output.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through output module."""
        return self.output(self.hidden(x))


class SubgoalActorNetwork(nn.Module):
    """
    Actor network for Subgoal Agent (DDPG).
    
    Input: LiDAR (80) + Waypoints (10)
    Output: Subgoal action (l, θ) = 2
    
    Architecture:
    - LiDAR module with attention -> 64-dim
    - Path module -> 32-dim
    - Concatenate -> 96-dim
    - Output module [128, 64, 64] -> 2
    """
    
    def __init__(
        self,
        lidar_rays: int = 80,
        num_waypoints: int = 5,
        max_distance: float = 0.6,
        min_distance: float = 0.0,
        max_angle: float = 2 * math.pi,
        min_angle: float = 0.0
    ):
        super().__init__()
        
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.max_angle = max_angle
        self.min_angle = min_angle
        
        # Submodules
        self.lidar_module = SALidarModule(
            num_sectors=10,
            rays_per_sector=lidar_rays // 10
        )
        self.path_module = SAPathModule(num_waypoints=num_waypoints)
        
        # Combined feature dim: 64 (lidar) + 32 (path) = 96
        combined_dim = 64 + self.path_module.output_dim
        
        self.output_module = SAOutputModule(
            input_dim=combined_dim,
            output_dim=2  # (l, θ)
        )
    
    def forward(
        self,
        lidar: torch.Tensor,
        waypoints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate subgoal action.
        
        Args:
            lidar: LiDAR tensor (batch, 80)
            waypoints: Waypoint tensor (batch, 10)
            
        Returns:
            action: Subgoal (l, θ) in (batch, 2)
            attention: Attention weights (batch, 10)
        """
        # Process inputs
        lidar_features, attention = self.lidar_module(lidar)
        path_features = self.path_module(waypoints)
        
        # Concatenate and output
        combined = torch.cat([lidar_features, path_features], dim=1)
        raw_output = self.output_module(combined)
        
        # Scale outputs to valid ranges
        # l: sigmoid -> [0, 1] -> [min_distance, max_distance]
        # θ: sigmoid -> [0, 1] -> [min_angle, max_angle]
        l = torch.sigmoid(raw_output[:, 0]) * (self.max_distance - self.min_distance) + self.min_distance
        theta = torch.sigmoid(raw_output[:, 1]) * (self.max_angle - self.min_angle) + self.min_angle
        
        action = torch.stack([l, theta], dim=1)
        
        return action, attention


class SubgoalCriticNetwork(nn.Module):
    """
    Critic network for Subgoal Agent (DDPG).
    
    Input: LiDAR (80) + Waypoints (10) + Action (2)
    Output: Q-value (1)
    
    Note from paper: "the critic shares the same architecture...
    However, one difference lies in the path module, which additionally 
    takes the predicted subgoal position (action) as input for the critic."
    """
    
    def __init__(
        self,
        lidar_rays: int = 80,
        num_waypoints: int = 5,
        action_dim: int = 2
    ):
        super().__init__()
        
        # LiDAR module (same as actor)
        self.lidar_module = SALidarModule(
            num_sectors=10,
            rays_per_sector=lidar_rays // 10
        )
        
        # Path module takes waypoints + action
        # Input: 10 (waypoints) + 2 (action) = 12
        self.path_module = nn.Sequential(
            nn.Linear(num_waypoints * 2 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.path_output_dim = 32
        
        # Combined: 64 (lidar) + 32 (path+action) = 96
        combined_dim = 64 + self.path_output_dim
        
        self.output_module = SAOutputModule(
            input_dim=combined_dim,
            output_dim=1  # Q-value
        )
        
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))
    
    def forward(
        self,
        lidar: torch.Tensor,
        waypoints: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-value.
        
        Args:
            lidar: LiDAR tensor (batch, 80)
            waypoints: Waypoint tensor (batch, 10)
            action: Action tensor (batch, 2)
            
        Returns:
            Q-value (batch, 1)
        """
        # Process LiDAR
        lidar_features, _ = self.lidar_module(lidar)
        
        # Process path + action
        path_action = torch.cat([waypoints, action], dim=1)
        path_features = self.path_module(path_action)
        
        # Concatenate and output
        combined = torch.cat([lidar_features, path_features], dim=1)
        q_value = self.output_module(combined)
        
        return q_value


# =============================================================================
# MOTION AGENT NETWORKS (TD3)
# =============================================================================

class MotionActorNetwork(nn.Module):
    """
    Actor network for Motion Agent (TD3).
    
    Input: State (v*, ω*, px, py, θdiff) = 5
    Output: Action (v, ω) = 2
    
    Architecture from paper: [256, 128, 64, 64] + output layer
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden_layers: List[int] = None,
        max_linear_vel: float = 0.5,
        min_linear_vel: float = 0.0,
        max_angular_vel: float = math.pi / 2,
        min_angular_vel: float = -math.pi / 2
    ):
        super().__init__()
        
        self.max_linear_vel = max_linear_vel
        self.min_linear_vel = min_linear_vel
        self.max_angular_vel = max_angular_vel
        self.min_angular_vel = min_angular_vel
        
        if hidden_layers is None:
            hidden_layers = [256, 128, 64, 64]
        
        # Build network
        self.hidden = make_mlp(
            [state_dim] + hidden_layers,
            output_activation=True
        )
        
        # Output layer (non-activated)
        self.output = nn.Linear(hidden_layers[-1], action_dim)
        
        # Initialize
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))
        nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
        nn.init.constant_(self.output.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate velocity command.
        
        Args:
            state: State (v*, ω*, px, py, θdiff) in (batch, 5)
            
        Returns:
            action: (v, ω) in (batch, 2)
        """
        x = self.hidden(state)
        raw_output = self.output(x)
        
        # Scale to action ranges using tanh
        # v: tanh -> [-1, 1] -> scale to [min, max]
        # ω: tanh -> [-1, 1] -> scale to [min, max]
        v = torch.tanh(raw_output[:, 0])
        v = (v + 1) / 2 * (self.max_linear_vel - self.min_linear_vel) + self.min_linear_vel
        
        omega = torch.tanh(raw_output[:, 1])
        omega = omega * self.max_angular_vel  # Symmetric around 0
        
        action = torch.stack([v, omega], dim=1)
        
        return action


class MotionCriticNetwork(nn.Module):
    """
    Critic network for Motion Agent (TD3).
    
    Input: State (5) + Action (2) = 7
    Output: Q-value (1)
    
    Architecture from paper: [256, 128, 64, 64] + output layer
    
    TD3 uses twin critics (Q1, Q2).
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden_layers: List[int] = None
    ):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [256, 128, 64, 64]
        
        input_dim = state_dim + action_dim
        
        # Q1
        self.q1_hidden = make_mlp(
            [input_dim] + hidden_layers,
            output_activation=True
        )
        self.q1_output = nn.Linear(hidden_layers[-1], 1)
        
        # Q2
        self.q2_hidden = make_mlp(
            [input_dim] + hidden_layers,
            output_activation=True
        )
        self.q2_output = nn.Linear(hidden_layers[-1], 1)
        
        # Initialize
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))
        nn.init.uniform_(self.q1_output.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q2_output.weight, -3e-3, 3e-3)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values from both critics.
        
        Args:
            state: State tensor (batch, 5)
            action: Action tensor (batch, 2)
            
        Returns:
            q1: Q-value from critic 1 (batch, 1)
            q2: Q-value from critic 2 (batch, 1)
        """
        x = torch.cat([state, action], dim=1)
        
        q1 = self.q1_output(self.q1_hidden(x))
        q2 = self.q2_output(self.q2_hidden(x))
        
        return q1, q2
    
    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q1 only (for actor update)."""
        x = torch.cat([state, action], dim=1)
        return self.q1_output(self.q1_hidden(x))


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Network Architecture Tests")
    print("=" * 60)
    
    # Test SA Actor
    print("\n--- Subgoal Agent Actor ---")
    sa_actor = SubgoalActorNetwork()
    lidar = torch.rand(4, 80)
    waypoints = torch.rand(4, 10)
    action, attention = sa_actor(lidar, waypoints)
    print(f"Input: lidar {lidar.shape}, waypoints {waypoints.shape}")
    print(f"Output: action {action.shape}, attention {attention.shape}")
    print(f"Action range: l=[{action[:, 0].min():.3f}, {action[:, 0].max():.3f}], "
          f"θ=[{action[:, 1].min():.3f}, {action[:, 1].max():.3f}]")
    
    # Test SA Critic
    print("\n--- Subgoal Agent Critic ---")
    sa_critic = SubgoalCriticNetwork()
    q_value = sa_critic(lidar, waypoints, action)
    print(f"Input: lidar {lidar.shape}, waypoints {waypoints.shape}, action {action.shape}")
    print(f"Output: Q-value {q_value.shape}")
    
    # Test MA Actor
    print("\n--- Motion Agent Actor ---")
    ma_actor = MotionActorNetwork()
    state = torch.rand(4, 5)
    ma_action = ma_actor(state)
    print(f"Input: state {state.shape}")
    print(f"Output: action {ma_action.shape}")
    print(f"Action range: v=[{ma_action[:, 0].min():.3f}, {ma_action[:, 0].max():.3f}], "
          f"ω=[{ma_action[:, 1].min():.3f}, {ma_action[:, 1].max():.3f}]")
    
    # Test MA Critic
    print("\n--- Motion Agent Critic ---")
    ma_critic = MotionCriticNetwork()
    q1, q2 = ma_critic(state, ma_action)
    print(f"Input: state {state.shape}, action {ma_action.shape}")
    print(f"Output: Q1 {q1.shape}, Q2 {q2.shape}")
    
    # Parameter counts
    print("\n--- Parameter Counts ---")
    sa_actor_params = sum(p.numel() for p in sa_actor.parameters())
    sa_critic_params = sum(p.numel() for p in sa_critic.parameters())
    ma_actor_params = sum(p.numel() for p in ma_actor.parameters())
    ma_critic_params = sum(p.numel() for p in ma_critic.parameters())
    print(f"SA Actor: {sa_actor_params:,}")
    print(f"SA Critic: {sa_critic_params:,}")
    print(f"MA Actor: {ma_actor_params:,}")
    print(f"MA Critic: {ma_critic_params:,}")
    
    print("\n✓ All network tests passed!")

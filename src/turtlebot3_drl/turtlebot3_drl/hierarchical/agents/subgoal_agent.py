"""
Subgoal Agent (SA) using DDPG Algorithm

The Subgoal Agent predicts local subgoal positions (l, θ) for collision avoidance.
It operates at 5 Hz (∆tSA = 0.2s) and uses attention-based LiDAR processing.

Based on the paper: "Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Key features:
- DDPG algorithm (found to converge fastest for SA)
- Attention mechanism for LiDAR processing
- Path processing module for waypoint features
- Replans A* path every 3 subgoal predictions
"""

import os
import math
import copy
import numpy as np
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .networks import SubgoalActorNetwork, SubgoalCriticNetwork

# Import config - handle both standalone and installed package
try:
    from ..config import HierarchicalConfig
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hierarchical.config import HierarchicalConfig


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""
    
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2
    ):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
    
    def reset(self):
        """Reset noise to mean."""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """Generate noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class ReplayBuffer:
    """Experience replay buffer for DDPG."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate buffers
        # State: (lidar, waypoints) = (80, 10)
        self.lidar = np.zeros((capacity, 80), dtype=np.float32)
        self.waypoints = np.zeros((capacity, 10), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_lidar = np.zeros((capacity, 80), dtype=np.float32)
        self.next_waypoints = np.zeros((capacity, 10), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(
        self,
        lidar: np.ndarray,
        waypoints: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_lidar: np.ndarray,
        next_waypoints: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.lidar[self.ptr] = lidar
        self.waypoints[self.ptr] = waypoints
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_lidar[self.ptr] = next_lidar
        self.next_waypoints[self.ptr] = next_waypoints
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'lidar': torch.FloatTensor(self.lidar[indices]).to(device),
            'waypoints': torch.FloatTensor(self.waypoints[indices]).to(device),
            'actions': torch.FloatTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_lidar': torch.FloatTensor(self.next_lidar[indices]).to(device),
            'next_waypoints': torch.FloatTensor(self.next_waypoints[indices]).to(device),
            'dones': torch.FloatTensor(self.dones[indices]).to(device)
        }
    
    def __len__(self) -> int:
        return self.size


class SubgoalAgent:
    """
    Subgoal Agent using DDPG.
    
    Predicts subgoal positions (l, θ) based on LiDAR and waypoints.
    Handles collision avoidance through learned subgoal selection.
    """
    
    def __init__(
        self,
        config: HierarchicalConfig = None,
        device: str = 'auto'
    ):
        """
        Initialize Subgoal Agent.
        
        Args:
            config: Configuration object
            device: 'cuda', 'cpu', or 'auto'
        """
        if config is None:
            config = HierarchicalConfig()
        self.config = config
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Networks
        self.actor = SubgoalActorNetwork(
            lidar_rays=config.LIDAR_RAYS,
            num_waypoints=config.NUM_WAYPOINTS,
            max_distance=config.SUBGOAL_MAX_DISTANCE,
            min_distance=config.SUBGOAL_MIN_DISTANCE,
            max_angle=config.SUBGOAL_ANGLE_MAX,
            min_angle=config.SUBGOAL_ANGLE_MIN
        ).to(self.device)
        
        self.critic = SubgoalCriticNetwork(
            lidar_rays=config.LIDAR_RAYS,
            num_waypoints=config.NUM_WAYPOINTS,
            action_dim=2
        ).to(self.device)
        
        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.SA_LEARNING_RATE_ACTOR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.SA_LEARNING_RATE_CRITIC
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.SA_BUFFER_SIZE,
            state_dim=config.LIDAR_RAYS + config.NUM_WAYPOINTS * 2,
            action_dim=2
        )
        
        # Exploration noise
        self.noise = OUNoise(action_dim=2)
        
        # Training parameters
        self.gamma = config.SA_GAMMA
        self.tau = config.SA_TAU
        self.batch_size = config.SA_BATCH_SIZE
        
        # Counters
        self.prediction_count = 0
        self.replan_interval = config.ASTAR_REPLAN_INTERVAL
        self.total_steps = 0
        
        # Training flag
        self.training = True
        
        # Last attention for visualization
        self.last_attention = None
    
    def select_action(
        self,
        lidar: np.ndarray,
        waypoints: np.ndarray,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, bool]:
        """
        Select subgoal action.
        
        Args:
            lidar: Normalized LiDAR scan (80,)
            waypoints: Waypoints in robot frame (10,) = 5 × (x, y)
            add_noise: Whether to add exploration noise
            
        Returns:
            action: Subgoal (l, θ) as numpy array
            should_replan: Whether to trigger A* replanning
        """
        # Increment counter
        self.prediction_count += 1
        should_replan = (self.prediction_count % self.replan_interval == 0)
        
        # Convert to tensors
        lidar_t = torch.FloatTensor(lidar).unsqueeze(0).to(self.device)
        waypoints_t = torch.FloatTensor(waypoints).unsqueeze(0).to(self.device)
        
        # Get action from actor
        with torch.no_grad():
            action, attention = self.actor(lidar_t, waypoints_t)
            action = action.cpu().numpy()[0]
            self.last_attention = attention.cpu().numpy()[0]
        
        # Add exploration noise during training
        if add_noise and self.training:
            noise = self.noise.sample()
            # Scale noise appropriately
            action[0] += noise[0] * 0.1  # Distance noise (smaller)
            action[1] += noise[1] * 0.3  # Angle noise (larger)
            
            # Clip to valid ranges
            action[0] = np.clip(
                action[0],
                self.config.SUBGOAL_MIN_DISTANCE,
                self.config.SUBGOAL_MAX_DISTANCE
            )
            action[1] = np.clip(
                action[1],
                self.config.SUBGOAL_ANGLE_MIN,
                self.config.SUBGOAL_ANGLE_MAX
            )
        
        return action, should_replan
    
    def store_transition(
        self,
        lidar: np.ndarray,
        waypoints: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_lidar: np.ndarray,
        next_waypoints: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.add(
            lidar, waypoints, action, reward,
            next_lidar, next_waypoints, done
        )
    
    def update(self) -> Dict[str, float]:
        """
        Perform one DDPG update step.
        
        Returns:
            Dictionary with loss values
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        
        # ============ Update Critic ============
        with torch.no_grad():
            # Get next actions from target actor
            next_actions, _ = self.actor_target(
                batch['next_lidar'],
                batch['next_waypoints']
            )
            
            # Compute target Q-value
            target_q = self.critic_target(
                batch['next_lidar'],
                batch['next_waypoints'],
                next_actions
            )
            target_q = batch['rewards'] + (1 - batch['dones']) * self.gamma * target_q
        
        # Current Q-value
        current_q = self.critic(
            batch['lidar'],
            batch['waypoints'],
            batch['actions']
        )
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ============ Update Actor ============
        # Actor loss: maximize Q-value
        actions, _ = self.actor(batch['lidar'], batch['waypoints'])
        actor_loss = -self.critic(
            batch['lidar'],
            batch['waypoints'],
            actions
        ).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ============ Soft Update Targets ============
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return {
            'sa_actor_loss': actor_loss.item(),
            'sa_critic_loss': critic_loss.item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def reset_noise(self):
        """Reset exploration noise."""
        self.noise.reset()
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()
    
    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'prediction_count': self.prediction_count
        }, path)
    
    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.prediction_count = checkpoint.get('prediction_count', 0)
    
    def train_step(self) -> Dict[str, float]:
        """Alias for update() for compatibility with trainer."""
        return self.update()
    
    def subgoal_to_cartesian(
        self,
        l: float,
        theta: float
    ) -> Tuple[float, float]:
        """
        Convert polar subgoal to Cartesian coordinates (robot frame).
        
        Args:
            l: Distance to subgoal
            theta: Angle to subgoal (0 = forward, increases CCW)
            
        Returns:
            (px, py): Cartesian position in robot frame
        """
        px = l * math.cos(theta)
        py = l * math.sin(theta)
        return px, py
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get last attention weights for visualization."""
        return self.last_attention


# =============================================================================
# REWARD FUNCTION
# =============================================================================

def compute_sa_reward(
    d_astar: float,
    d_astar_prev: float,
    min_lidar: float,
    collision: bool,
    goal_reached: bool,
    config: HierarchicalConfig
) -> float:
    """
    Compute Subgoal Agent reward.
    
    r_SA = r_path + r_safety + r_collision + r_goal
    
    Args:
        d_astar: Current A* distance to goal
        d_astar_prev: Previous A* distance to goal
        min_lidar: Minimum LiDAR reading (closest obstacle)
        collision: Whether collision occurred
        goal_reached: Whether goal was reached
        config: Configuration object
        
    Returns:
        Total reward
    """
    reward = 0.0
    
    # Path progress reward (negative of distance change)
    # Encourages moving toward goal along A* path
    r_path = config.SA_REWARD_PATH_COEFF * (d_astar - d_astar_prev)
    reward += r_path
    
    # Safety penalty (when too close to obstacles)
    if min_lidar < config.SA_SAFETY_DISTANCE:
        r_safety = config.SA_REWARD_SAFETY_COEFF * (
            1 - min_lidar / config.SA_SAFETY_DISTANCE
        )
        reward += r_safety
    
    # Collision penalty
    if collision:
        reward += config.SA_REWARD_COLLISION
    
    # Goal reward
    if goal_reached:
        reward += config.SA_REWARD_GOAL
    
    return reward


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Subgoal Agent Test")
    print("=" * 60)
    
    # Create agent
    config = HierarchicalConfig()
    agent = SubgoalAgent(config, device='cpu')
    
    print(f"Device: {agent.device}")
    print(f"Replan interval: {agent.replan_interval}")
    
    # Test action selection
    print("\n--- Action Selection ---")
    lidar = np.random.rand(80).astype(np.float32)
    waypoints = np.random.rand(10).astype(np.float32)
    
    for i in range(5):
        action, should_replan = agent.select_action(lidar, waypoints)
        print(f"Step {i+1}: action=(l={action[0]:.3f}, θ={action[1]:.3f}), "
              f"replan={should_replan}")
    
    # Test storing transitions
    print("\n--- Storing Transitions ---")
    for i in range(100):
        next_lidar = np.random.rand(80).astype(np.float32)
        next_waypoints = np.random.rand(10).astype(np.float32)
        agent.store_transition(
            lidar, waypoints, action, -0.5,
            next_lidar, next_waypoints, False
        )
        lidar = next_lidar
        waypoints = next_waypoints
    
    print(f"Buffer size: {len(agent.replay_buffer)}")
    
    # Test update
    print("\n--- Training Update ---")
    losses = agent.update()
    if losses:
        print(f"Actor loss: {losses['sa_actor_loss']:.4f}")
        print(f"Critic loss: {losses['sa_critic_loss']:.4f}")
    
    # Test attention
    print("\n--- Attention Weights ---")
    attention = agent.get_attention_weights()
    if attention is not None:
        print(f"Attention shape: {attention.shape}")
        print(f"Attention sum: {attention.sum():.4f}")
        print(f"Max attention sector: {attention.argmax()}")
    
    # Test subgoal conversion
    print("\n--- Subgoal Conversion ---")
    px, py = agent.subgoal_to_cartesian(0.5, math.pi/4)
    print(f"Polar (l=0.5, θ=π/4) -> Cartesian ({px:.3f}, {py:.3f})")
    
    print("\n✓ Subgoal Agent test complete!")

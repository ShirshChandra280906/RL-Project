"""
Motion Agent (MA) using TD3 Algorithm

The Motion Agent controls robot velocity to reach subgoal positions.
It operates at 20 Hz (∆tMA = 0.05s) and does NOT handle collision avoidance.

Based on the paper: "Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Key features:
- TD3 algorithm (more stable velocities than DDPG)
- Simple state: (v*, ω*, px, py, θdiff)
- Does not use LiDAR - relies on SA for collision avoidance
- Pre-trained separately before SA training
"""

import os
import math
import copy
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .networks import MotionActorNetwork, MotionCriticNetwork

# Import config
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hierarchical.config import HierarchicalConfig


class GaussianNoise:
    """Gaussian noise for exploration."""
    
    def __init__(self, action_dim: int, sigma: float = 0.1):
        self.action_dim = action_dim
        self.sigma = sigma
    
    def sample(self) -> np.ndarray:
        return np.random.randn(self.action_dim) * self.sigma


class MAReplayBuffer:
    """Experience replay buffer for Motion Agent."""
    
    def __init__(self, capacity: int, state_dim: int = 5, action_dim: int = 2):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # State: (v*, ω*, px, py, θdiff)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.FloatTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.FloatTensor(self.dones[indices]).to(device)
        }
    
    def __len__(self) -> int:
        return self.size


class MotionAgent:
    """
    Motion Agent using TD3.
    
    Controls robot velocity to reach nearby subgoal positions.
    Pre-trained separately, does not handle collision avoidance.
    """
    
    def __init__(
        self,
        config: HierarchicalConfig = None,
        device: str = 'auto'
    ):
        """
        Initialize Motion Agent.
        
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
        self.actor = MotionActorNetwork(
            state_dim=config.MA_STATE_DIM,
            action_dim=config.MA_ACTION_DIM,
            hidden_layers=config.MA_LAYERS,
            max_linear_vel=config.MA_MAX_LINEAR_VEL,
            min_linear_vel=config.MA_MIN_LINEAR_VEL,
            max_angular_vel=config.MA_MAX_ANGULAR_VEL,
            min_angular_vel=config.MA_MIN_ANGULAR_VEL
        ).to(self.device)
        
        self.critic = MotionCriticNetwork(
            state_dim=config.MA_STATE_DIM,
            action_dim=config.MA_ACTION_DIM,
            hidden_layers=config.MA_LAYERS
        ).to(self.device)
        
        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.MA_LEARNING_RATE_ACTOR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.MA_LEARNING_RATE_CRITIC
        )
        
        # Replay buffer
        self.replay_buffer = MAReplayBuffer(
            capacity=config.MA_BUFFER_SIZE,
            state_dim=config.MA_STATE_DIM,
            action_dim=config.MA_ACTION_DIM
        )
        
        # Exploration noise
        self.noise = GaussianNoise(action_dim=2, sigma=0.1)
        
        # Training parameters
        self.gamma = config.MA_GAMMA
        self.tau = config.MA_TAU
        self.batch_size = config.MA_BATCH_SIZE
        self.policy_noise = config.MA_POLICY_NOISE
        self.noise_clip = config.MA_NOISE_CLIP
        self.policy_delay = config.MA_POLICY_DELAY
        
        # Update counter for delayed policy updates
        self.update_count = 0
        
        # Action limits for target smoothing
        self.max_action = np.array([config.MA_MAX_LINEAR_VEL, config.MA_MAX_ANGULAR_VEL])
        self.min_action = np.array([config.MA_MIN_LINEAR_VEL, config.MA_MIN_ANGULAR_VEL])
        
        # Training flag
        self.training = True
        
        # Pre-training tracking
        self.consecutive_successes = 0
        self.convergence_threshold = config.MA_CONVERGENCE_EPISODES
        self.converged = False
    
    def build_state(
        self,
        prev_v: float,
        prev_omega: float,
        subgoal_x: float,
        subgoal_y: float,
        subgoal_theta: float = None
    ) -> np.ndarray:
        """
        Build Motion Agent state vector.
        
        State: (v*, ω*, px, py, θdiff)
        
        Args:
            prev_v: Previous linear velocity command
            prev_omega: Previous angular velocity command
            subgoal_x: Subgoal x position (robot frame)
            subgoal_y: Subgoal y position (robot frame)
            subgoal_theta: Angle to subgoal (optional, computed if None)
            
        Returns:
            State vector (5,)
        """
        # Compute θdiff if not provided
        if subgoal_theta is None:
            subgoal_theta = math.atan2(subgoal_y, subgoal_x)
        
        # Normalize θdiff to [-π, π]
        theta_diff = subgoal_theta
        while theta_diff > math.pi:
            theta_diff -= 2 * math.pi
        while theta_diff < -math.pi:
            theta_diff += 2 * math.pi
        
        state = np.array([
            prev_v,
            prev_omega,
            subgoal_x,
            subgoal_y,
            theta_diff
        ], dtype=np.float32)
        
        return state
    
    def select_action(
        self,
        state: np.ndarray,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Select velocity action.
        
        Args:
            state: State vector (v*, ω*, px, py, θdiff)
            add_noise: Whether to add exploration noise
            
        Returns:
            action: (v, ω) as numpy array
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        
        # Add exploration noise during training
        if add_noise and self.training:
            noise = self.noise.sample()
            action[0] += noise[0] * 0.1  # Smaller noise for linear vel
            action[1] += noise[1] * 0.3  # Larger noise for angular vel
            
            # Clip to valid ranges
            action[0] = np.clip(
                action[0],
                self.config.MA_MIN_LINEAR_VEL,
                self.config.MA_MAX_LINEAR_VEL
            )
            action[1] = np.clip(
                action[1],
                self.config.MA_MIN_ANGULAR_VEL,
                self.config.MA_MAX_ANGULAR_VEL
            )
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """
        Perform one TD3 update step.
        
        Returns:
            Dictionary with loss values
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        self.update_count += 1
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        
        # ============ Update Critics ============
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.actor_target(batch['next_states'])
            
            # Add clipped noise for target smoothing (TD3 feature)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = next_actions + noise
            # Clip actions to valid range
            next_actions[:, 0] = next_actions[:, 0].clamp(
                self.config.MA_MIN_LINEAR_VEL,
                self.config.MA_MAX_LINEAR_VEL
            )
            next_actions[:, 1] = next_actions[:, 1].clamp(
                self.config.MA_MIN_ANGULAR_VEL,
                self.config.MA_MAX_ANGULAR_VEL
            )
            
            # Compute target Q-values (use minimum of two critics)
            target_q1, target_q2 = self.critic_target(
                batch['next_states'],
                next_actions
            )
            target_q = torch.min(target_q1, target_q2)
            target_q = batch['rewards'] + (1 - batch['dones']) * self.gamma * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(batch['states'], batch['actions'])
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        losses = {'ma_critic_loss': critic_loss.item()}
        
        # ============ Delayed Policy Update (TD3 feature) ============
        if self.update_count % self.policy_delay == 0:
            # Actor loss: maximize Q1
            actions = self.actor(batch['states'])
            actor_loss = -self.critic.q1_forward(batch['states'], actions).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            losses['ma_actor_loss'] = actor_loss.item()
            
            # Soft update targets
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        return losses
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()
    
    def record_episode_result(self, success: bool):
        """
        Record episode result for convergence tracking.
        
        Args:
            success: Whether episode was successful
        """
        if success:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0
        
        if self.consecutive_successes >= self.convergence_threshold:
            self.converged = True
    
    def is_converged(self) -> bool:
        """Check if pre-training has converged."""
        return self.converged
    
    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'update_count': self.update_count,
            'consecutive_successes': self.consecutive_successes,
            'converged': self.converged
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
        self.update_count = checkpoint.get('update_count', 0)
        self.consecutive_successes = checkpoint.get('consecutive_successes', 0)
        self.converged = checkpoint.get('converged', False)


# =============================================================================
# REWARD FUNCTION
# =============================================================================

def compute_ma_reward(
    distance_to_subgoal: float,
    subgoal_reached: bool,
    config: HierarchicalConfig
) -> float:
    """
    Compute Motion Agent reward.
    
    r_MA = r_reach + r_dist
    
    Args:
        distance_to_subgoal: Euclidean distance to subgoal
        subgoal_reached: Whether subgoal was reached
        config: Configuration object
        
    Returns:
        Total reward
    """
    reward = 0.0
    
    # Reach reward
    if subgoal_reached:
        reward += config.MA_REWARD_REACH  # +2
    
    # Distance penalty
    reward += config.MA_REWARD_DIST_COEFF * distance_to_subgoal  # -1 * distance
    
    return reward


# =============================================================================
# SUBGOAL SAMPLING FOR PRE-TRAINING
# =============================================================================

class SubgoalSampler:
    """
    Samples subgoal positions for Motion Agent pre-training.
    
    From paper:
    - Straight line (p=0.2)
    - Curvy line with ±π/2 direction changes (p=0.3)
    - Fully random direction (p=0.5)
    - Distance range: (0, 0.7] m
    """
    
    def __init__(self, config: HierarchicalConfig = None):
        if config is None:
            config = HierarchicalConfig()
        self.config = config
        
        self.max_distance = config.MA_SUBGOAL_SAMPLE_DISTANCE_MAX
        self.straight_prob = config.MA_SUBGOAL_STRAIGHT_PROB
        self.curvy_prob = config.MA_SUBGOAL_CURVY_PROB
        # random_prob = 1 - straight - curvy
    
    def sample(self) -> Tuple[float, float]:
        """
        Sample a random subgoal position.
        
        Returns:
            (x, y): Subgoal position in robot frame
        """
        # Sample distance (0, max]
        distance = np.random.uniform(0.1, self.max_distance)
        
        # Sample direction based on probabilities
        p = np.random.random()
        
        if p < self.straight_prob:
            # Straight line (forward)
            theta = 0.0
        elif p < self.straight_prob + self.curvy_prob:
            # Curvy: ±π/2 range
            theta = np.random.uniform(-math.pi/2, math.pi/2)
        else:
            # Fully random
            theta = np.random.uniform(-math.pi, math.pi)
        
        x = distance * math.cos(theta)
        y = distance * math.sin(theta)
        
        return x, y
    
    def sample_batch(self, batch_size: int) -> List[Tuple[float, float]]:
        """Sample a batch of subgoal positions."""
        return [self.sample() for _ in range(batch_size)]


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Motion Agent Test")
    print("=" * 60)
    
    # Create agent
    config = HierarchicalConfig()
    agent = MotionAgent(config, device='cpu')
    
    print(f"Device: {agent.device}")
    print(f"Convergence threshold: {agent.convergence_threshold} episodes")
    
    # Test state building
    print("\n--- State Building ---")
    state = agent.build_state(
        prev_v=0.2,
        prev_omega=0.1,
        subgoal_x=0.3,
        subgoal_y=0.2
    )
    print(f"State: {state}")
    print(f"State shape: {state.shape}")
    
    # Test action selection
    print("\n--- Action Selection ---")
    for i in range(5):
        action = agent.select_action(state)
        print(f"Step {i+1}: action=(v={action[0]:.3f}, ω={action[1]:.3f})")
    
    # Test storing transitions
    print("\n--- Storing Transitions ---")
    for i in range(100):
        next_state = np.random.randn(5).astype(np.float32)
        agent.store_transition(state, action, -0.5, next_state, False)
        state = next_state
    
    print(f"Buffer size: {len(agent.replay_buffer)}")
    
    # Test update
    print("\n--- Training Update ---")
    for i in range(5):
        losses = agent.update()
        if losses:
            print(f"Update {i+1}: critic={losses.get('ma_critic_loss', 0):.4f}, "
                  f"actor={losses.get('ma_actor_loss', 'N/A')}")
    
    # Test subgoal sampler
    print("\n--- Subgoal Sampling ---")
    sampler = SubgoalSampler(config)
    for i in range(5):
        x, y = sampler.sample()
        dist = math.sqrt(x**2 + y**2)
        theta = math.atan2(y, x)
        print(f"Sample {i+1}: ({x:.3f}, {y:.3f}), dist={dist:.3f}, θ={math.degrees(theta):.1f}°")
    
    # Test reward computation
    print("\n--- Reward Computation ---")
    print(f"Subgoal reached: {compute_ma_reward(0.05, True, config):.2f}")
    print(f"Far from subgoal: {compute_ma_reward(0.5, False, config):.2f}")
    print(f"Close to subgoal: {compute_ma_reward(0.1, False, config):.2f}")
    
    # Test convergence tracking
    print("\n--- Convergence Tracking ---")
    for i in range(55):
        agent.record_episode_result(True)
    print(f"Consecutive successes: {agent.consecutive_successes}")
    print(f"Converged: {agent.is_converged()}")
    
    print("\n✓ Motion Agent test complete!")

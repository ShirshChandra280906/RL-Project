"""
Training Pipeline for Hierarchical DRL Navigation

Implements two-stage training based on the paper:
"Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Training Stages:
1. MA Pre-training: Train Motion Agent with sampled subgoals until convergence
   - Convergence: 50 consecutive successful episodes
   - Uses random subgoals (20% straight, 30% curvy, 50% random)
   
2. SA Training: Train Subgoal Agent with frozen MA weights
   - MA follows SA's predicted subgoals
   - SA learns collision avoidance and path following
"""

import os
import math
import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

import torch

# Import hierarchical components
import sys
# Add parent of hierarchical to path (turtlebot3_drl directory)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_hierarchical_dir = os.path.dirname(_current_dir)
_turtlebot3_drl_dir = os.path.dirname(_hierarchical_dir)
if _turtlebot3_drl_dir not in sys.path:
    sys.path.insert(0, _turtlebot3_drl_dir)

from hierarchical.config import HierarchicalConfig
from hierarchical.environments.scenes import SceneType, SceneFactory
from hierarchical.environments.hierarchical_env import (
    HierarchicalEnvironment,
    MAPretrainingEnvironment,
    TerminationReason
)
from hierarchical.agents.subgoal_agent import SubgoalAgent
from hierarchical.agents.motion_agent import MotionAgent


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingStage(Enum):
    """Training stages."""
    MA_PRETRAINING = "ma_pretraining"
    SA_TRAINING = "sa_training"
    JOINT_FINETUNING = "joint_finetuning"


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    episode: int = 0
    total_steps: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    success: bool = False
    collision: bool = False
    timeout: bool = False
    
    # MA-specific metrics
    ma_consecutive_successes: int = 0
    ma_avg_reward: float = 0.0
    
    # SA-specific metrics
    sa_avg_reward: float = 0.0
    dist_to_goal: float = 0.0
    
    # Loss values
    actor_loss: float = 0.0
    critic_loss: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    # General
    seed: int = 42
    device: str = 'auto'
    save_dir: str = './models'
    log_interval: int = 10
    save_interval: int = 100
    
    # MA Pre-training
    ma_pretrain_episodes: int = 10000  # Max episodes for MA pre-training
    ma_convergence_threshold: int = 50  # Consecutive successes
    
    # SA Training
    sa_train_episodes: int = 50000
    sa_eval_interval: int = 100
    
    # Scene switching
    scene_types: List[SceneType] = field(default_factory=lambda: list(SceneType))
    scene_switch_interval: int = 500  # Episodes before switching scene type


class MAPretrainer:
    """
    Motion Agent Pre-trainer.
    
    Pre-trains MA with sampled subgoals until convergence
    (50 consecutive successful episodes).
    """
    
    def __init__(
        self,
        config: HierarchicalConfig = None,
        training_config: TrainingConfig = None
    ):
        if config is None:
            config = HierarchicalConfig()
        if training_config is None:
            training_config = TrainingConfig()
        
        self.config = config
        self.train_config = training_config
        
        # Create environment
        self.env = MAPretrainingEnvironment(config)
        
        # Create agent
        self.agent = MotionAgent(config, device=training_config.device)
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.consecutive_successes = 0
        self.converged = False
        
        # Metrics history
        self.metrics_history: List[TrainingMetrics] = []
    
    def train_episode(self) -> TrainingMetrics:
        """
        Train for one episode.
        
        Returns:
            Episode metrics
        """
        state = self.env.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        
        done = False
        while not done:
            # Select action
            action = self.agent.select_action(state, add_noise=True)
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            losses = self.agent.update()
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
        
        # Record result
        success = info.get('success', False)
        self.agent.record_episode_result(success)
        self.consecutive_successes = self.agent.consecutive_successes
        self.converged = self.agent.is_converged()
        
        self.episode_count += 1
        
        metrics = TrainingMetrics(
            episode=self.episode_count,
            total_steps=self.total_steps,
            episode_reward=episode_reward,
            episode_length=episode_steps,
            success=success,
            ma_consecutive_successes=self.consecutive_successes,
            ma_avg_reward=episode_reward / max(1, episode_steps),
            actor_loss=losses.get('ma_actor_loss', 0.0),
            critic_loss=losses.get('ma_critic_loss', 0.0)
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def train(self, max_episodes: int = None) -> bool:
        """
        Train until convergence or max episodes.
        
        Args:
            max_episodes: Maximum episodes (None = use config)
            
        Returns:
            True if converged, False if max episodes reached
        """
        if max_episodes is None:
            max_episodes = self.train_config.ma_pretrain_episodes
        
        logger.info(f"Starting MA pre-training (max {max_episodes} episodes)")
        logger.info(f"Convergence: {self.train_config.ma_convergence_threshold} consecutive successes")
        
        while self.episode_count < max_episodes and not self.converged:
            metrics = self.train_episode()
            
            if self.episode_count % self.train_config.log_interval == 0:
                logger.info(
                    f"MA Episode {self.episode_count}: "
                    f"reward={metrics.episode_reward:.2f}, "
                    f"success={metrics.success}, "
                    f"consecutive={self.consecutive_successes}/{self.train_config.ma_convergence_threshold}"
                )
            
            if self.episode_count % self.train_config.save_interval == 0:
                self.save_checkpoint()
        
        if self.converged:
            logger.info(f"MA converged after {self.episode_count} episodes!")
        else:
            logger.warning(f"MA did not converge after {max_episodes} episodes")
        
        return self.converged
    
    def save_checkpoint(self, path: str = None):
        """Save agent checkpoint."""
        if path is None:
            path = os.path.join(
                self.train_config.save_dir,
                f"ma_pretrain_ep{self.episode_count}.pt"
            )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)
        logger.info(f"Saved MA checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load agent from checkpoint."""
        self.agent.load(path)
        logger.info(f"Loaded MA from {path}")


class SATrainer:
    """
    Subgoal Agent Trainer.
    
    Trains SA with frozen MA weights.
    MA follows SA's predicted subgoals.
    """
    
    def __init__(
        self,
        config: HierarchicalConfig = None,
        training_config: TrainingConfig = None,
        pretrained_ma: MotionAgent = None
    ):
        if config is None:
            config = HierarchicalConfig()
        if training_config is None:
            training_config = TrainingConfig()
        
        self.config = config
        self.train_config = training_config
        
        # Create environment
        self.env = HierarchicalEnvironment(config)
        
        # Create SA
        self.sa_agent = SubgoalAgent(config, device=training_config.device)
        
        # Use pre-trained MA or create new
        if pretrained_ma is not None:
            self.ma_agent = pretrained_ma
        else:
            self.ma_agent = MotionAgent(config, device=training_config.device)
        
        # Freeze MA
        self.ma_agent.set_training(False)
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.current_scene_idx = 0
        
        # Metrics
        self.metrics_history: List[TrainingMetrics] = []
        self.success_rate = 0.0
    
    def train_episode(self, scene_type: SceneType = None) -> TrainingMetrics:
        """
        Train for one episode.
        
        Args:
            scene_type: Scene type (None = random from config)
            
        Returns:
            Episode metrics
        """
        # Select scene
        if scene_type is None:
            scene_type = random.choice(self.train_config.scene_types)
        
        # Reset environment
        obs = self.env.reset(scene_type)
        
        episode_reward = 0.0
        episode_steps = 0
        sa_losses = {}
        
        done = False
        while not done:
            # Get SA state
            lidar = obs['lidar']
            waypoints = obs['waypoints']
            
            # SA selects subgoal
            sa_action, should_replan = self.sa_agent.select_action(
                lidar, waypoints, add_noise=True
            )
            
            # Convert to subgoal position
            l, theta = sa_action
            subgoal_x = l * math.cos(theta)
            subgoal_y = l * math.sin(theta)
            
            # Execute MA steps
            ma_total_reward = 0.0
            for _ in range(self.config.MA_STEPS_PER_SA):
                # Build MA state
                ma_state = self.ma_agent.build_state(
                    prev_v=self.env.robot.linear_vel,
                    prev_omega=self.env.robot.angular_vel,
                    subgoal_x=subgoal_x,
                    subgoal_y=subgoal_y
                )
                
                # MA selects action (no noise - frozen)
                ma_action = self.ma_agent.select_action(ma_state, add_noise=False)
                
                # Execute in environment
                _, ma_reward, done, info = self.env.step_ma(ma_action)
                ma_total_reward += ma_reward
                
                # Update subgoal in robot frame (robot moved)
                # Simplified: reduce distance by velocity * dt
                v = ma_action[0]
                omega = ma_action[1]
                dt = self.config.MA_TIME_STEP
                
                subgoal_x -= v * dt
                c, s = math.cos(-omega * dt), math.sin(-omega * dt)
                new_x = c * subgoal_x - s * subgoal_y
                new_y = s * subgoal_x + c * subgoal_y
                subgoal_x, subgoal_y = new_x, new_y
                
                episode_steps += 1
                self.total_steps += 1
                
                if done:
                    break
            
            # Get next observation
            next_obs = self.env._get_observation()
            
            # Compute SA reward
            termination = TerminationReason[info['termination'].upper()] if done else TerminationReason.NONE
            sa_reward = self.env._compute_sa_reward(done, termination)
            episode_reward += sa_reward
            
            # Store SA transition
            self.sa_agent.store_transition(
                lidar=lidar,
                waypoints=waypoints,
                action=sa_action,
                reward=sa_reward,
                next_lidar=next_obs['lidar'],
                next_waypoints=next_obs['waypoints'],
                done=done
            )
            
            # Update SA
            sa_losses = self.sa_agent.update()
            
            obs = next_obs
        
        # Determine outcome
        termination = TerminationReason[info['termination'].upper()]
        success = termination == TerminationReason.GOAL_REACHED
        collision = termination == TerminationReason.COLLISION
        timeout = termination == TerminationReason.TIMEOUT
        
        self.episode_count += 1
        
        metrics = TrainingMetrics(
            episode=self.episode_count,
            total_steps=self.total_steps,
            episode_reward=episode_reward,
            episode_length=episode_steps,
            success=success,
            collision=collision,
            timeout=timeout,
            sa_avg_reward=episode_reward / max(1, episode_steps // self.config.MA_STEPS_PER_SA),
            dist_to_goal=obs['dist_to_goal'],
            actor_loss=sa_losses.get('sa_actor_loss', 0.0),
            critic_loss=sa_losses.get('sa_critic_loss', 0.0)
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def train(self, num_episodes: int = None) -> None:
        """
        Train SA for specified episodes.
        
        Args:
            num_episodes: Number of episodes (None = use config)
        """
        if num_episodes is None:
            num_episodes = self.train_config.sa_train_episodes
        
        logger.info(f"Starting SA training for {num_episodes} episodes")
        
        successes = []
        
        for ep in range(num_episodes):
            # Switch scene periodically
            if ep % self.train_config.scene_switch_interval == 0:
                self.current_scene_idx = (self.current_scene_idx + 1) % len(self.train_config.scene_types)
            
            scene_type = self.train_config.scene_types[self.current_scene_idx]
            
            metrics = self.train_episode(scene_type)
            successes.append(metrics.success)
            
            # Compute success rate over last 100 episodes
            recent_successes = successes[-100:]
            self.success_rate = sum(recent_successes) / len(recent_successes)
            
            if (ep + 1) % self.train_config.log_interval == 0:
                logger.info(
                    f"SA Episode {self.episode_count}: "
                    f"reward={metrics.episode_reward:.2f}, "
                    f"success={metrics.success}, "
                    f"success_rate={self.success_rate:.2%}, "
                    f"scene={scene_type.value}"
                )
            
            if (ep + 1) % self.train_config.save_interval == 0:
                self.save_checkpoint()
    
    def save_checkpoint(self, path: str = None):
        """Save agent checkpoint."""
        if path is None:
            path = os.path.join(
                self.train_config.save_dir,
                f"sa_ep{self.episode_count}.pt"
            )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.sa_agent.save(path)
        logger.info(f"Saved SA checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load SA from checkpoint."""
        self.sa_agent.load(path)
        logger.info(f"Loaded SA from {path}")


class HierarchicalTrainer:
    """
    Complete hierarchical training pipeline.
    
    Stage 1: Pre-train MA
    Stage 2: Train SA with frozen MA
    """
    
    def __init__(
        self,
        config: HierarchicalConfig = None,
        training_config: TrainingConfig = None
    ):
        if config is None:
            config = HierarchicalConfig()
        if training_config is None:
            training_config = TrainingConfig()
        
        self.config = config
        self.train_config = training_config
        
        # Set random seeds
        self._set_seeds(training_config.seed)
        
        # Training components (created lazily)
        self.ma_pretrainer: Optional[MAPretrainer] = None
        self.sa_trainer: Optional[SATrainer] = None
        
        # Current stage
        self.current_stage = TrainingStage.MA_PRETRAINING
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def pretrain_ma(self, max_episodes: int = None) -> bool:
        """
        Stage 1: Pre-train Motion Agent.
        
        Args:
            max_episodes: Max episodes (None = use config)
            
        Returns:
            True if converged
        """
        logger.info("="*60)
        logger.info("STAGE 1: Motion Agent Pre-training")
        logger.info("="*60)
        
        self.current_stage = TrainingStage.MA_PRETRAINING
        
        self.ma_pretrainer = MAPretrainer(self.config, self.train_config)
        converged = self.ma_pretrainer.train(max_episodes)
        
        # Save final model
        save_path = os.path.join(self.train_config.save_dir, "ma_final.pt")
        self.ma_pretrainer.save_checkpoint(save_path)
        
        return converged
    
    def train_sa(
        self,
        num_episodes: int = None,
        ma_checkpoint: str = None
    ) -> None:
        """
        Stage 2: Train Subgoal Agent.
        
        Args:
            num_episodes: Number of episodes (None = use config)
            ma_checkpoint: Path to pre-trained MA (None = use from Stage 1)
        """
        logger.info("="*60)
        logger.info("STAGE 2: Subgoal Agent Training")
        logger.info("="*60)
        
        self.current_stage = TrainingStage.SA_TRAINING
        
        # Get pre-trained MA
        if ma_checkpoint is not None:
            ma_agent = MotionAgent(self.config, device=self.train_config.device)
            ma_agent.load(ma_checkpoint)
        elif self.ma_pretrainer is not None:
            ma_agent = self.ma_pretrainer.agent
        else:
            logger.warning("No pre-trained MA available, creating new MA")
            ma_agent = MotionAgent(self.config, device=self.train_config.device)
        
        self.sa_trainer = SATrainer(
            self.config,
            self.train_config,
            pretrained_ma=ma_agent
        )
        
        self.sa_trainer.train(num_episodes)
        
        # Save final model
        save_path = os.path.join(self.train_config.save_dir, "sa_final.pt")
        self.sa_trainer.save_checkpoint(save_path)
    
    def train_full(self) -> Dict[str, Any]:
        """
        Run complete two-stage training.
        
        Returns:
            Training results
        """
        start_time = time.time()
        
        # Stage 1
        ma_converged = self.pretrain_ma()
        
        # Stage 2
        self.train_sa()
        
        total_time = time.time() - start_time
        
        results = {
            'ma_converged': ma_converged,
            'ma_episodes': self.ma_pretrainer.episode_count if self.ma_pretrainer else 0,
            'sa_episodes': self.sa_trainer.episode_count if self.sa_trainer else 0,
            'sa_success_rate': self.sa_trainer.success_rate if self.sa_trainer else 0.0,
            'total_time_seconds': total_time
        }
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"MA converged: {results['ma_converged']}")
        logger.info(f"MA episodes: {results['ma_episodes']}")
        logger.info(f"SA episodes: {results['sa_episodes']}")
        logger.info(f"SA success rate: {results['sa_success_rate']:.2%}")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        
        return results
    
    def evaluate(
        self,
        num_episodes: int = 100,
        scene_type: SceneType = None
    ) -> Dict[str, float]:
        """
        Evaluate trained agents.
        
        Args:
            num_episodes: Number of evaluation episodes
            scene_type: Scene type (None = all types)
            
        Returns:
            Evaluation metrics
        """
        if self.sa_trainer is None:
            raise RuntimeError("No trained SA available")
        
        # Set agents to eval mode
        self.sa_trainer.sa_agent.set_training(False)
        self.sa_trainer.ma_agent.set_training(False)
        
        successes = []
        collisions = []
        timeouts = []
        rewards = []
        
        scene_types = [scene_type] if scene_type else list(SceneType)
        
        for _ in range(num_episodes):
            st = random.choice(scene_types)
            metrics = self.sa_trainer.train_episode(st)  # No training updates in eval
            
            successes.append(metrics.success)
            collisions.append(metrics.collision)
            timeouts.append(metrics.timeout)
            rewards.append(metrics.episode_reward)
        
        return {
            'success_rate': np.mean(successes),
            'collision_rate': np.mean(collisions),
            'timeout_rate': np.mean(timeouts),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards)
        }


if __name__ == "__main__":
    # Test training pipeline
    print("Testing Training Pipeline...")
    
    config = HierarchicalConfig()
    train_config = TrainingConfig(
        save_dir='./test_models',
        log_interval=5,
        save_interval=50
    )
    
    # Test MA pre-trainer
    print("\n" + "="*50)
    print("Testing MA Pre-trainer (5 episodes)")
    print("="*50)
    
    ma_trainer = MAPretrainer(config, train_config)
    for i in range(5):
        metrics = ma_trainer.train_episode()
        print(f"Episode {i+1}: reward={metrics.episode_reward:.2f}, success={metrics.success}")
    
    # Test SA trainer
    print("\n" + "="*50)
    print("Testing SA Trainer (3 episodes)")
    print("="*50)
    
    sa_trainer = SATrainer(config, train_config, pretrained_ma=ma_trainer.agent)
    for i in range(3):
        metrics = sa_trainer.train_episode()
        print(f"Episode {i+1}: reward={metrics.episode_reward:.2f}, success={metrics.success}")
    
    print("\n✓ Training pipeline tests complete!")

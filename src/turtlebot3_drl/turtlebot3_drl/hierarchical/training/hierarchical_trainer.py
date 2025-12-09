"""
Hierarchical DRL Training Pipeline

Two-stage training process based on the paper:
"Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Stage 1: Pre-train Motion Agent (MA) with sampled subgoals until convergence
         (50 consecutive successes)
Stage 2: Train Subgoal Agent (SA) with frozen MA

Usage:
    python hierarchical_trainer.py --stage 1  # Pre-train MA
    python hierarchical_trainer.py --stage 2  # Train SA (after MA converged)
    python hierarchical_trainer.py --stage full  # Both stages
"""

import os
import sys
import time
import argparse
import math
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

# Handle imports for both standalone and installed package
try:
    from ..config import HierarchicalConfig
    from ..agents.subgoal_agent import SubgoalAgent
    from ..agents.motion_agent import MotionAgent
    from ..environments.hierarchical_env import (
        HierarchicalEnvironment,
        MAPretrainingEnvironment,
        TerminationReason
    )
    from ..environments.scenes import SceneType
except ImportError:
    # Add parent path for standalone execution
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _hierarchical_dir = os.path.dirname(_current_dir)
    _turtlebot3_drl_dir = os.path.dirname(_hierarchical_dir)
    if _turtlebot3_drl_dir not in sys.path:
        sys.path.insert(0, _turtlebot3_drl_dir)
    
    from hierarchical.config import HierarchicalConfig
    from hierarchical.agents.subgoal_agent import SubgoalAgent
    from hierarchical.agents.motion_agent import MotionAgent
    from hierarchical.environments.hierarchical_env import (
        HierarchicalEnvironment,
        MAPretrainingEnvironment,
        TerminationReason
    )
    from hierarchical.environments.scenes import SceneType

import torch


class TrainingLogger:
    """Logs training metrics."""
    
    def __init__(self, log_dir: str, stage: str):
        self.log_dir = log_dir
        self.stage = stage
        
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{stage}_{timestamp}.json")
        
        self.metrics: Dict[str, List] = {
            'episode': [],
            'reward': [],
            'steps': [],
            'success': [],
            'time': []
        }
        
        self.start_time = time.time()
    
    def log(self, episode: int, reward: float, steps: int, success: bool, **kwargs):
        """Log episode metrics."""
        self.metrics['episode'].append(episode)
        self.metrics['reward'].append(reward)
        self.metrics['steps'].append(steps)
        self.metrics['success'].append(success)
        self.metrics['time'].append(time.time() - self.start_time)
        
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def save(self):
        """Save metrics to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def print_summary(self, window: int = 100):
        """Print recent performance summary."""
        if len(self.metrics['episode']) < window:
            window = len(self.metrics['episode'])
        
        if window == 0:
            return
        
        recent_rewards = self.metrics['reward'][-window:]
        recent_success = self.metrics['success'][-window:]
        
        avg_reward = np.mean(recent_rewards)
        success_rate = np.mean(recent_success) * 100
        
        print(f"  Last {window} episodes: avg_reward={avg_reward:.2f}, "
              f"success_rate={success_rate:.1f}%")


class MAPretrainer:
    """
    Pre-trains the Motion Agent with sampled subgoals.
    
    Training continues until MA achieves 50 consecutive successes.
    """
    
    def __init__(
        self,
        config: HierarchicalConfig,
        model_dir: str,
        device: str = None
    ):
        self.config = config
        self.model_dir = model_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Create MA and environment
        self.ma = MotionAgent(config, device=self.device)
        self.env = MAPretrainingEnvironment(config)
        
        # Logging
        self.logger = TrainingLogger(
            os.path.join(model_dir, 'logs'),
            'ma_pretrain'
        )
        
        # Convergence tracking
        self.consecutive_successes = 0
        self.convergence_threshold = config.MA_CONVERGENCE_EPISODES
    
    def train(self, max_episodes: int = 10000) -> bool:
        """
        Train MA until convergence.
        
        Args:
            max_episodes: Maximum training episodes
            
        Returns:
            True if converged, False if max_episodes reached
        """
        print(f"\n{'='*60}")
        print("STAGE 1: Motion Agent Pre-training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Convergence threshold: {self.convergence_threshold} consecutive successes")
        print(f"Max episodes: {max_episodes}")
        print()
        
        for episode in range(1, max_episodes + 1):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            done = False
            while not done:
                # Select action
                if self.ma.total_steps < self.config.SA_BATCH_SIZE * 10:
                    # Random exploration initially
                    action = np.random.uniform(
                        [self.config.MA_MIN_LINEAR_VEL, self.config.MA_MIN_ANGULAR_VEL],
                        [self.config.MA_MAX_LINEAR_VEL, self.config.MA_MAX_ANGULAR_VEL]
                    )
                else:
                    action = self.ma.select_action(state, add_noise=True)
                
                # Step environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.ma.store_transition(state, action, reward, next_state, done)
                
                # Train
                if self.ma.total_steps >= self.config.MA_BATCH_SIZE:
                    self.ma.train_step()
                
                state = next_state
                episode_reward += reward
                steps += 1
                self.ma.total_steps += 1
            
            # Track success
            success = info.get('success', False)
            
            if success:
                self.consecutive_successes += 1
            else:
                self.consecutive_successes = 0
            
            # Log
            self.logger.log(
                episode, episode_reward, steps, success,
                consecutive_successes=self.consecutive_successes
            )
            
            # Print progress
            if episode % 100 == 0:
                print(f"Episode {episode}/{max_episodes}")
                self.logger.print_summary()
                print(f"  Consecutive successes: {self.consecutive_successes}/{self.convergence_threshold}")
            
            # Check convergence
            if self.consecutive_successes >= self.convergence_threshold:
                print(f"\n✓ MA CONVERGED at episode {episode}!")
                self._save_model('converged')
                self.logger.save()
                return True
            
            # Periodic save
            if episode % 500 == 0:
                self._save_model(f'ep{episode}')
                self.logger.save()
        
        print(f"\n✗ MA did not converge within {max_episodes} episodes")
        self._save_model('final')
        self.logger.save()
        return False
    
    def _save_model(self, suffix: str):
        """Save MA model."""
        path = os.path.join(self.model_dir, f'ma_{suffix}.pth')
        self.ma.save(path)
        print(f"Saved MA model to {path}")


class SATrainer:
    """
    Trains the Subgoal Agent with a frozen Motion Agent.
    """
    
    def __init__(
        self,
        config: HierarchicalConfig,
        model_dir: str,
        ma_model_path: str,
        device: str = None
    ):
        self.config = config
        self.model_dir = model_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Create SA
        self.sa = SubgoalAgent(config, device=self.device)
        
        # Load pre-trained MA
        self.ma = MotionAgent(config, device=self.device)
        if os.path.exists(ma_model_path):
            self.ma.load(ma_model_path)
            print(f"Loaded MA from {ma_model_path}")
        else:
            print(f"WARNING: MA model not found at {ma_model_path}")
        
        # Freeze MA
        self.ma.freeze()
        
        # Create environment
        self.env = HierarchicalEnvironment(config)
        
        # Logging
        self.logger = TrainingLogger(
            os.path.join(model_dir, 'logs'),
            'sa_train'
        )
    
    def train(self, num_episodes: int = 5000):
        """
        Train SA for specified episodes.
        
        Args:
            num_episodes: Number of training episodes
        """
        print(f"\n{'='*60}")
        print("STAGE 2: Subgoal Agent Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Episodes: {num_episodes}")
        print(f"SA time step: {self.config.SA_TIME_STEP}s")
        print(f"MA time step: {self.config.MA_TIME_STEP}s")
        print()
        
        best_success_rate = 0
        recent_successes = deque(maxlen=100)
        
        for episode in range(1, num_episodes + 1):
            # Reset environment (random scene)
            obs = self.env.reset()
            
            episode_reward = 0
            sa_steps = 0
            
            done = False
            while not done:
                # SA predicts subgoal
                lidar = obs['lidar']
                waypoints = obs['waypoints']
                
                if self.sa.total_steps < self.config.SA_BATCH_SIZE * 5:
                    # Random exploration
                    sa_action = np.array([
                        np.random.uniform(0, self.config.SUBGOAL_MAX_DISTANCE),
                        np.random.uniform(0, 2 * np.pi)
                    ])
                    should_replan = False
                else:
                    sa_action, should_replan = self.sa.select_action(lidar, waypoints, add_noise=True)
                
                # Execute SA step (internally runs MA steps)
                # For now, we'll manually run MA steps
                l, theta = sa_action[0], sa_action[1]
                subgoal_x = l * np.cos(theta)
                subgoal_y = l * np.sin(theta)
                self.env.current_subgoal = (subgoal_x, subgoal_y)
                
                # Run MA steps
                ma_rewards = []
                for _ in range(self.config.MA_STEPS_PER_SA):
                    ma_state = self.env._get_ma_state()
                    ma_action = self.ma.select_action(ma_state, add_noise=False)
                    _, ma_reward, done, info = self.env.step_ma(ma_action)
                    ma_rewards.append(ma_reward)
                    
                    if done:
                        break
                
                # Get new observation
                new_obs = self.env._get_observation()
                
                # Compute SA reward
                termination = TerminationReason(info.get('termination', 'none'))
                sa_reward = self.env._compute_sa_reward(done, termination)
                
                # Store transition
                new_lidar = new_obs['lidar']
                new_waypoints = new_obs['waypoints']
                self.sa.store_transition(
                    lidar, waypoints, sa_action, sa_reward,
                    new_lidar, new_waypoints, done
                )
                
                # Train SA
                if self.sa.total_steps >= self.config.SA_BATCH_SIZE:
                    self.sa.train_step()
                
                obs = new_obs
                episode_reward += sa_reward
                sa_steps += 1
                self.sa.total_steps += 1
            
            # Track success
            success = termination == TerminationReason.GOAL_REACHED
            recent_successes.append(success)
            
            # Log
            self.logger.log(
                episode, episode_reward, sa_steps, success,
                termination=termination.value
            )
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}")
                self.logger.print_summary()
                
                # Check for best model
                if len(recent_successes) >= 100:
                    current_rate = np.mean(list(recent_successes))
                    if current_rate > best_success_rate:
                        best_success_rate = current_rate
                        self._save_model('best')
                        print(f"  New best! Success rate: {best_success_rate*100:.1f}%")
            
            # Periodic save
            if episode % 500 == 0:
                self._save_model(f'ep{episode}')
                self.logger.save()
        
        print(f"\nTraining complete. Best success rate: {best_success_rate*100:.1f}%")
        self._save_model('final')
        self.logger.save()
    
    def _save_model(self, suffix: str):
        """Save SA model."""
        path = os.path.join(self.model_dir, f'sa_{suffix}.pth')
        self.sa.save(path)
        print(f"Saved SA model to {path}")


class HierarchicalTrainer:
    """
    Complete hierarchical training pipeline.
    
    Manages both MA pre-training and SA training stages.
    """
    
    def __init__(
        self,
        config: HierarchicalConfig = None,
        output_dir: str = None,
        device: str = None
    ):
        if config is None:
            config = HierarchicalConfig()
        
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up directories
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Get the turtlebot3_drl package directory
            _current_file = os.path.abspath(__file__)
            _training_dir = os.path.dirname(_current_file)
            _hierarchical_dir = os.path.dirname(_training_dir)
            _turtlebot3_drl_dir = os.path.dirname(_hierarchical_dir)
            
            output_dir = os.path.join(
                _turtlebot3_drl_dir,
                'model',
                'hierarchical',
                f'session_{timestamp}'
            )
        
        self.output_dir = output_dir
        self.ma_dir = os.path.join(output_dir, 'ma')
        self.sa_dir = os.path.join(output_dir, 'sa')
        
        os.makedirs(self.ma_dir, exist_ok=True)
        os.makedirs(self.sa_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self._config_to_dict(), f, indent=2)
    
    def _config_to_dict(self) -> Dict:
        """Convert config to dictionary for saving."""
        return {
            key: getattr(self.config, key)
            for key in dir(self.config)
            if not key.startswith('_') and not callable(getattr(self.config, key))
        }
    
    def train_ma(self, max_episodes: int = 10000) -> bool:
        """
        Stage 1: Pre-train Motion Agent.
        
        Returns:
            True if converged
        """
        trainer = MAPretrainer(self.config, self.ma_dir, self.device)
        return trainer.train(max_episodes)
    
    def train_sa(
        self,
        num_episodes: int = 5000,
        ma_model: str = 'converged'
    ):
        """
        Stage 2: Train Subgoal Agent.
        
        Args:
            num_episodes: Training episodes
            ma_model: Which MA model to use ('converged', 'final', or path)
        """
        # Find MA model
        if os.path.isfile(ma_model):
            ma_path = ma_model
        else:
            ma_path = os.path.join(self.ma_dir, f'ma_{ma_model}.pth')
        
        if not os.path.exists(ma_path):
            raise FileNotFoundError(f"MA model not found: {ma_path}")
        
        trainer = SATrainer(self.config, self.sa_dir, ma_path, self.device)
        trainer.train(num_episodes)
    
    def train_full(
        self,
        ma_max_episodes: int = 10000,
        sa_episodes: int = 5000
    ):
        """
        Full two-stage training.
        
        Args:
            ma_max_episodes: Max episodes for MA pre-training
            sa_episodes: Episodes for SA training
        """
        print("\n" + "="*60)
        print("HIERARCHICAL DRL TRAINING PIPELINE")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print()
        
        # Stage 1
        ma_converged = self.train_ma(ma_max_episodes)
        
        if not ma_converged:
            print("\nWARNING: MA did not converge. Continuing with best model...")
        
        # Stage 2
        self.train_sa(sa_episodes)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Models saved to: {self.output_dir}")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description='Hierarchical DRL Navigation Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training (both stages)
  python hierarchical_trainer.py --stage full

  # Pre-train Motion Agent only
  python hierarchical_trainer.py --stage 1

  # Train Subgoal Agent (requires pre-trained MA)
  python hierarchical_trainer.py --stage 2 --ma-model path/to/ma.pth

  # Continue from existing session
  python hierarchical_trainer.py --stage 2 --output-dir path/to/session
        """
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        choices=['1', '2', 'full', 'ma', 'sa'],
        default='full',
        help='Training stage: 1/ma (MA pre-training), 2/sa (SA training), full (both)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for models and logs'
    )
    
    parser.add_argument(
        '--ma-model',
        type=str,
        default='converged',
        help='MA model to use for SA training (path or "converged"/"final")'
    )
    
    parser.add_argument(
        '--ma-episodes',
        type=int,
        default=10000,
        help='Max episodes for MA pre-training'
    )
    
    parser.add_argument(
        '--sa-episodes',
        type=int,
        default=5000,
        help='Episodes for SA training'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = HierarchicalTrainer(
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run appropriate stage
    if args.stage in ['full']:
        trainer.train_full(args.ma_episodes, args.sa_episodes)
    elif args.stage in ['1', 'ma']:
        trainer.train_ma(args.ma_episodes)
    elif args.stage in ['2', 'sa']:
        trainer.train_sa(args.sa_episodes, args.ma_model)


if __name__ == '__main__':
    main()

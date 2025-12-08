#!/usr/bin/env python3
"""
Remote DRL Agent - Offloads NN inference to remote GPU server

This module provides a drop-in replacement for the local DRL agent that:
1. Uses the same ROS2 interface (services, topics)
2. Sends state observations to a remote GPU server for inference
3. Receives action predictions from the server
4. Maintains compatibility with existing training/testing workflows

The remote server is accessed via SSH tunnel:
    ssh -L 8000:localhost:8000 rl-group13@192.168.3.214

Key differences from local agent:
- No local PyTorch/neural network computation
- All forward passes happen on the remote GPU server
- Training still happens on the server (if implemented there)
"""

import copy
import os
import sys
import time
import numpy as np
from typing import List, Optional, Tuple

from ..common.settings import ENABLE_VISUAL, ENABLE_STACKING, OBSERVE_STEPS, MODEL_STORE_INTERVAL, GRAPH_DRAW_INTERVAL

from ..common.storagemanager import StorageManager
from ..common.graph import Graph
from ..common.logger import Logger
from ..common import utilities as util

# Import the GPU client for remote inference
from .gpu_client import GPUInferenceClient, GPU_SERVER_URL

from turtlebot3_msgs.srv import DrlStep, Goal
from std_srvs.srv import Empty

import rclpy
from rclpy.node import Node
from ..common.replaybuffer import ReplayBuffer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Remote server configuration
REMOTE_SERVER_URL = GPU_SERVER_URL
REQUEST_TIMEOUT = 5.0

# Action space configuration (must match server model)
ACTION_SIZE = 2  # [linear_velocity, angular_velocity]


class RemotePolicy:
    """
    Wrapper for remote GPU inference that mimics the local model interface.
    
    This class provides the same interface as DQN/DDPG/TD3 for get_action(),
    but sends requests to the remote GPU server instead of running locally.
    """
    
    def __init__(self, server_url: str = REMOTE_SERVER_URL, timeout: float = REQUEST_TIMEOUT):
        """
        Initialize remote policy.
        
        Args:
            server_url: URL of the GPU inference server
            timeout: Request timeout in seconds
        """
        self.gpu_client = GPUInferenceClient(server_url, timeout)
        self.action_size = ACTION_SIZE
        
        # These parameters are used by the agent for compatibility
        # They should match what the remote model expects
        self.state_size = 26  # Default: 24 LiDAR + 2 goal (adjust if needed)
        self.hidden_size = 512
        self.batch_size = 128
        self.buffer_size = 1000000
        self.discount_factor = 0.99
        self.learning_rate = 0.003
        self.tau = 0.003
        self.step_time = 0.0  # No delay needed for remote
        self.reward_function = "A"
        self.backward_enabled = False
        self.stacking_enabled = ENABLE_STACKING
        self.stack_depth = 1
        self.frame_skip = 1
        self.input_size = self.state_size
        
        # Placeholder for visual (not used in remote mode)
        self.visual = None
        self.networks = []
        
        # For DQN compatibility
        self.possible_actions = [[0.3, -1.0], [0.3, -0.5], [1.0, 0.0], [0.3, 0.5], [0.3, 1.0]]
        
        print(f"Remote policy initialized, server: {server_url}")
        
        # Test connection
        if self.gpu_client.check_connection():
            print("✓ Connected to GPU server")
        else:
            print(f"✗ Warning: Could not connect to GPU server: {self.gpu_client.last_error}")
            print("  Make sure SSH tunnel is running:")
            print("  ssh -L 8000:localhost:8000 rl-group13@192.168.3.214")
    
    def get_action(self, state, is_training: bool = False, step: int = 0, visualize: bool = False) -> List[float]:
        """
        Get action from remote GPU server.
        
        This is the main inference method that replaces local NN forward pass.
        
        Args:
            state: State observation (list or numpy array)
            is_training: Whether in training mode (affects exploration)
            step: Current step (used for exploration scheduling)
            visualize: Whether to visualize (not supported in remote mode)
            
        Returns:
            Action as list of floats [linear_vel, angular_vel]
        """
        # Convert state to list if needed
        if isinstance(state, np.ndarray):
            state_list = state.tolist()
        else:
            state_list = list(state)
        
        # ===========================================================
        # REMOTE INFERENCE - Send state to GPU server
        # ===========================================================
        print(f"[REMOTE INFERENCE] Sending {len(state_list)} values to GPU server...")
        action = self.gpu_client.infer(state_list)
        
        if action is None:
            print(f"[REMOTE INFERENCE] FAILED! Error: {self.gpu_client.last_error}")
            print(f"[REMOTE INFERENCE] Using random action as fallback")
            return self.get_action_random()
        
        print(f"[REMOTE INFERENCE] SUCCESS! Received action: {action}")
        
        # Clip actions to valid range [-1, 1]
        action = [max(-1.0, min(1.0, a)) for a in action]
        
        return action
    
    def get_action_random(self) -> List[float]:
        """
        Get random action (fallback when server unavailable).
        
        Returns:
            Random action within [-1, 1] for each dimension
        """
        return [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)] * self.action_size
    
    def _train(self, replay_buffer) -> Tuple[float, float]:
        """
        Training step - sends batch to remote server for training.
        
        Note: This requires the server to support training endpoints.
        For inference-only mode, this is a no-op.
        
        Args:
            replay_buffer: Replay buffer with experiences
            
        Returns:
            Tuple of (critic_loss, actor_loss) - zeros if not supported
        """
        # TODO: Implement remote training if server supports it
        # For now, return zeros (inference-only mode)
        print("Note: Remote training not implemented, inference-only mode")
        return 0.0, 0.0
    
    def get_model_parameters(self) -> str:
        """Get model parameters as string."""
        parameters = [
            self.batch_size, self.buffer_size, self.state_size, self.action_size, self.hidden_size,
            self.discount_factor, self.learning_rate, self.tau, self.step_time, self.reward_function,
            self.backward_enabled, self.stacking_enabled, self.stack_depth, self.frame_skip
        ]
        return ', '.join(map(str, parameters))
    
    def get_model_configuration(self) -> str:
        """Get model configuration as string."""
        return f"RemotePolicy(server={REMOTE_SERVER_URL})"
    
    def attach_visual(self, visual):
        """Attach visualization (not supported in remote mode)."""
        self.visual = visual
        print("Note: Visualization not supported in remote inference mode")


class DrlAgentRemote(Node):
    """
    Remote DRL Agent Node.
    
    This is a modified version of DrlAgent that uses remote GPU inference
    instead of local PyTorch models. It maintains the same ROS2 interface
    and can be used as a drop-in replacement.
    
    Usage:
        1. Start SSH tunnel: ssh -L 8000:localhost:8000 rl-group13@192.168.3.214
        2. Run: ros2 run turtlebot3_drl remote_agent <algorithm>
    """
    
    def __init__(self, training: int = 0, algorithm: str = "td3", 
                 load_session: str = "", load_episode: int = 0, real_robot: int = 0):
        super().__init__(algorithm + '_remote_agent')
        
        self.algorithm = algorithm
        self.training = int(training)
        self.load_session = load_session
        self.episode = int(load_episode)
        self.real_robot = real_robot
        
        # Check if we need to test but have no server
        self.device = util.check_gpu()  # Still useful for logging
        self.sim_speed = util.get_simulation_speed(util.stage) if not self.real_robot else 1
        
        print(f"Remote Agent Mode: {'training' if self.training else 'testing'}")
        print(f"Algorithm: {algorithm}, Stage: {util.stage}")
        
        self.total_steps = 0
        self.observe_steps = OBSERVE_STEPS
        
        # ===========================================================
        # USE REMOTE POLICY INSTEAD OF LOCAL MODEL
        # ===========================================================
        self.model = RemotePolicy(REMOTE_SERVER_URL, REQUEST_TIMEOUT)
        
        # These are still used for compatibility
        self.replay_buffer = ReplayBuffer(self.model.buffer_size)
        self.graph = Graph()
        
        # Storage manager (for logging, not model storage in remote mode)
        self.sm = StorageManager(self.algorithm, self.load_session, self.episode, self.device, util.stage)
        
        if not self.load_session:
            self.sm.new_session_dir(util.stage)
        
        self.graph.session_dir = self.sm.session_dir
        self.logger = Logger(
            self.training, self.sm.machine_dir, self.sm.session_dir, 
            self.sm.session, self.model.get_model_parameters(), 
            self.model.get_model_configuration(), str(util.stage), 
            self.algorithm, self.episode
        )
        
        # ROS2 service clients
        self.step_comm_client = self.create_client(DrlStep, 'step_comm')
        self.goal_comm_client = self.create_client(Goal, 'goal_comm')
        
        if not self.real_robot:
            self.gazebo_pause = self.create_client(Empty, '/pause_physics')
            self.gazebo_unpause = self.create_client(Empty, '/unpause_physics')
        
        # Start processing
        self.process()
    
    def process(self):
        """
        Main processing loop.
        
        Same structure as local agent, but uses remote inference.
        """
        util.pause_simulation(self, self.real_robot)
        
        while True:
            util.wait_new_goal(self)
            episode_done = False
            step, reward_sum, loss_critic, loss_actor = 0, 0, 0, 0
            action_past = [0.0, 0.0]
            state = util.init_episode(self)
            
            if ENABLE_STACKING:
                frame_buffer = [0.0] * (self.model.state_size * self.model.stack_depth * self.model.frame_skip)
                state = [0.0] * (self.model.state_size * (self.model.stack_depth - 1)) + list(state)
                next_state = [0.0] * (self.model.state_size * self.model.stack_depth)
            
            util.unpause_simulation(self, self.real_robot)
            time.sleep(0.5)
            episode_start = time.perf_counter()
            
            while not episode_done:
                # ===========================================================
                # GET ACTION FROM REMOTE GPU SERVER
                # ===========================================================
                if self.training and self.total_steps < self.observe_steps:
                    action = self.model.get_action_random()
                else:
                    # This calls the remote server!
                    action = self.model.get_action(state, self.training, step)
                
                action_current = action
                if self.algorithm == 'dqn':
                    action_current = self.model.possible_actions[action]
                
                # Take a step in the environment
                next_state, reward, episode_done, outcome, distance_traveled = util.step(
                    self, action_current, action_past
                )
                action_past = copy.deepcopy(action_current)
                reward_sum += reward
                
                if ENABLE_STACKING:
                    frame_buffer = frame_buffer[self.model.state_size:] + list(next_state)
                    next_state = []
                    for depth in range(self.model.stack_depth):
                        start = self.model.state_size * (self.model.frame_skip - 1) + (self.model.state_size * self.model.frame_skip * depth)
                        next_state += frame_buffer[start : start + self.model.state_size]
                
                # Training (if supported by server)
                if self.training:
                    self.replay_buffer.add_sample(state, action, [reward], next_state, [episode_done])
                    if self.replay_buffer.get_length() >= self.model.batch_size:
                        loss_c, loss_a = self.model._train(self.replay_buffer)
                        loss_critic += loss_c
                        loss_actor += loss_a
                
                state = copy.deepcopy(next_state)
                step += 1
                time.sleep(self.model.step_time)
            
            # Episode done
            util.pause_simulation(self, self.real_robot)
            self.total_steps += step
            duration = time.perf_counter() - episode_start
            
            self.finish_episode(step, duration, outcome, distance_traveled, reward_sum, loss_critic, loss_actor)
    
    def finish_episode(self, step, eps_duration, outcome, dist_traveled, reward_sum, loss_critic, loss_actor):
        """Handle episode completion."""
        if self.total_steps < self.observe_steps:
            print(f"Observe phase: {self.total_steps}/{self.observe_steps} steps")
            return
        
        self.episode += 1
        print(f"Epi: {self.episode:<5}R: {reward_sum:<8.0f}outcome: {util.translate_outcome(outcome):<13}", end='')
        print(f"steps: {step:<6}steps_total: {self.total_steps:<7}time: {eps_duration:<6.2f}")
        
        if not self.training:
            self.logger.update_test_results(step, outcome, dist_traveled, eps_duration, 0)
            return
        
        self.graph.update_data(step, self.total_steps, outcome, reward_sum, loss_critic, loss_actor)
        self.logger.file_log.write(
            f"{self.episode}, {reward_sum}, {outcome}, {eps_duration}, {step}, {self.total_steps}, "
            f"{self.replay_buffer.get_length()}, {loss_critic / step if step > 0 else 0}, {loss_actor / step if step > 0 else 0}\n"
        )
        
        if (self.episode % MODEL_STORE_INTERVAL == 0) or (self.episode == 1):
            self.logger.update_comparison_file(self.episode, self.graph.get_success_count(), self.graph.get_reward_average())
        
        if (self.episode % GRAPH_DRAW_INTERVAL == 0) or (self.episode == 1):
            self.graph.draw_plots(self.episode)


def main_train(args=sys.argv[1:]):
    """Training mode entry point - uses remote GPU server."""
    rclpy.init()
    
    algorithm = args[0] if len(args) > 0 else "ddpg"
    load_session = args[1] if len(args) > 1 else ""
    load_episode = int(args[2]) if len(args) > 2 else 0
    
    print("\n" + "="*60)
    print("REMOTE TRAINING MODE")
    print(f"Algorithm: {algorithm}")
    print("GPU Server: http://localhost:8000/infer")
    print("="*60 + "\n")
    
    try:
        agent = DrlAgentRemote(
            training=1,
            algorithm=algorithm,
            load_session=load_session,
            load_episode=load_episode,
            real_robot=0
        )
        rclpy.spin(agent)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        rclpy.shutdown()


def main_test(args=sys.argv[1:]):
    """Test mode entry point - uses remote GPU server."""
    rclpy.init()
    
    algorithm = args[0] if len(args) > 0 else "ddpg"
    load_session = args[1] if len(args) > 1 else ""
    load_episode = int(args[2]) if len(args) > 2 else 0
    
    print("\n" + "="*60)
    print("REMOTE TESTING MODE")
    print(f"Algorithm: {algorithm}")
    print("GPU Server: http://localhost:8000/infer")
    print("="*60 + "\n")
    
    try:
        agent = DrlAgentRemote(
            training=0,
            algorithm=algorithm,
            load_session=load_session,
            load_episode=load_episode,
            real_robot=0
        )
        rclpy.spin(agent)
    except KeyboardInterrupt:
        print("\nTesting interrupted.")
    finally:
        rclpy.shutdown()


def main_real(args=sys.argv[1:]):
    """Real robot mode entry point."""
    rclpy.init()
    
    algorithm = args[0] if len(args) > 0 else "ddpg"
    
    try:
        agent = DrlAgentRemote(
            training=0,
            algorithm=algorithm,
            load_session="",
            load_episode=0,
            real_robot=1
        )
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


def main(args=sys.argv[1:]):
    """Default entry point - training mode."""
    main_train(args)


if __name__ == '__main__':
    main()

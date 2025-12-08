"""
Hierarchical Environment for DRL Navigation

Coordinates the Subgoal Agent (SA) and Motion Agent (MA) in a hierarchical
control structure for robot navigation.

Based on the paper: "Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Timing:
- SA operates at 5 Hz (∆tSA = 0.2s)
- MA operates at 20 Hz (∆tMA = 0.05s)
- 4 MA steps per SA step

Episode termination:
- Goal reached (success)
- Collision (failure)
- Timeout (failure)
"""

import math
import time
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Import hierarchical components
import sys
import os
# Add parent of hierarchical to path (turtlebot3_drl directory)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_hierarchical_dir = os.path.dirname(_current_dir)
_turtlebot3_drl_dir = os.path.dirname(_hierarchical_dir)
if _turtlebot3_drl_dir not in sys.path:
    sys.path.insert(0, _turtlebot3_drl_dir)

from hierarchical.config import HierarchicalConfig
from hierarchical.environments.scenes import BaseScene, SceneFactory, SceneType
from hierarchical.environments.obstacles import ObstacleManager, Obstacle
from hierarchical.planners.astar import AStarPlanner
from hierarchical.planners.waypoint_manager import WaypointManager
from hierarchical.preprocessing.lidar_processor import LidarProcessor


class TerminationReason(Enum):
    """Episode termination reasons."""
    NONE = "none"
    GOAL_REACHED = "goal_reached"
    COLLISION = "collision"
    TIMEOUT = "timeout"


@dataclass
class RobotState:
    """Robot state container."""
    x: float
    y: float
    theta: float
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.linear_vel, self.angular_vel])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'RobotState':
        return cls(x=arr[0], y=arr[1], theta=arr[2], 
                   linear_vel=arr[3] if len(arr) > 3 else 0.0,
                   angular_vel=arr[4] if len(arr) > 4 else 0.0)


@dataclass
class HierarchicalStep:
    """Result of a hierarchical step."""
    # SA outputs
    subgoal: Tuple[float, float]  # (l, theta) or (px, py) in robot frame
    should_replan: bool
    sa_reward: float
    
    # MA accumulated outputs
    ma_rewards: List[float]
    ma_actions: List[np.ndarray]
    
    # Terminal state
    done: bool
    termination: TerminationReason
    info: Dict[str, Any]


class HierarchicalEnvironment:
    """
    Hierarchical Navigation Environment.
    
    Coordinates SA and MA for navigation:
    1. SA predicts subgoal every 0.2s (5 Hz)
    2. MA controls velocity to reach subgoal at 20 Hz (4 steps per SA step)
    3. A* replanning every 3 SA steps
    
    This environment can operate in different modes:
    - Full: Both SA and MA active
    - MA-only: For MA pre-training with sampled subgoals
    - SA-only: For SA training with frozen MA
    """
    
    def __init__(
        self,
        config: HierarchicalConfig = None,
        scene_type: SceneType = None,
        use_ros: bool = False
    ):
        """
        Initialize hierarchical environment.
        
        Args:
            config: Configuration object
            scene_type: Type of scene (None = random)
            use_ros: Whether to use ROS2/Gazebo (False = simulation)
        """
        if config is None:
            config = HierarchicalConfig()
        self.config = config
        
        # Timing
        self.sa_timestep = config.SA_TIME_STEP
        self.ma_timestep = config.MA_TIME_STEP
        self.ma_steps_per_sa = config.MA_STEPS_PER_SA
        
        # Scene setup
        self.scene_type = scene_type
        self.scene: Optional[BaseScene] = None
        
        # Components
        self.astar_planner = AStarPlanner(
            grid_resolution=config.ASTAR_RESOLUTION,
            robot_radius=config.ASTAR_ROBOT_RADIUS,
            inflation_radius=config.ASTAR_INFLATION_RADIUS
        )
        
        self.waypoint_manager = WaypointManager(
            num_waypoints=config.NUM_WAYPOINTS,
            waypoint_spacing=config.WAYPOINT_SPACING
        )
        
        self.lidar_processor = LidarProcessor(
            input_rays=config.LIDAR_RAW_RAYS,
            output_rays=config.LIDAR_RAYS,
            max_range=config.LIDAR_MAX_RANGE,
            clip_range=config.LIDAR_CLIP_RANGE,
            num_sectors=config.LIDAR_SECTORS
        )
        
        self.obstacle_manager = ObstacleManager(
            num_dynamic=2,
            num_static=1
        )
        
        # Robot state
        self.robot = RobotState(0, 0, 0)
        self.goal_pos: Tuple[float, float] = (0, 0)
        
        # Episode tracking
        self.step_count = 0
        self.sa_step_count = 0
        self.episode_count = 0
        
        # Current subgoal
        self.current_subgoal: Optional[Tuple[float, float]] = None  # In robot frame
        
        # Path tracking
        self.global_path: List[Tuple[float, float]] = []
        
        # ROS flag
        self.use_ros = use_ros
        
        # Simulated LiDAR (for non-ROS mode)
        self._simulated_lidar: Optional[np.ndarray] = None
    
    def reset(
        self,
        scene_type: SceneType = None
    ) -> Dict[str, Any]:
        """
        Reset environment for a new episode.
        
        Args:
            scene_type: Override scene type for this episode
            
        Returns:
            Initial observation dictionary
        """
        # Create or reset scene
        if scene_type is None:
            scene_type = self.scene_type
        
        if scene_type is None:
            self.scene = SceneFactory.create_random(self.config.ASTAR_RESOLUTION)
        else:
            self.scene = SceneFactory.create(scene_type, self.config.ASTAR_RESOLUTION)
        
        # Generate scene and get start/goal
        scene_info = self.scene.reset()
        
        # Initialize robot
        self.robot = RobotState(
            x=scene_info['start'][0],
            y=scene_info['start'][1],
            theta=scene_info['start_theta']
        )
        self.goal_pos = scene_info['goal']
        
        # Set up A* planner
        self.astar_planner.set_occupancy_grid(
            scene_info['grid'],
            scene_info['origin'][0],
            scene_info['origin'][1]
        )
        
        # Plan initial path
        self.global_path = self.astar_planner.plan(
            (self.robot.x, self.robot.y),
            self.goal_pos
        ) or []
        
        # Initialize waypoint manager
        self.waypoint_manager.set_path(self.global_path)
        
        # Initialize obstacles
        self.obstacle_manager.set_scene(self.scene, self.astar_planner)
        self.obstacle_manager.reset(
            robot_start=(self.robot.x, self.robot.y),
            robot_goal=self.goal_pos,
            robot_path=self.global_path
        )
        
        # Reset counters
        self.step_count = 0
        self.sa_step_count = 0
        self.episode_count += 1
        
        # Initial subgoal (straight ahead)
        self.current_subgoal = (self.config.SUBGOAL_MAX_DISTANCE / 2, 0.0)
        
        # Generate initial observation
        obs = self._get_observation()
        
        return obs
    
    def step_sa(
        self,
        sa_action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one Subgoal Agent step (0.2s = 4 MA steps).
        
        Args:
            sa_action: Subgoal prediction (l, theta)
            
        Returns:
            (observation, reward, done, info)
        """
        self.sa_step_count += 1
        
        # Convert SA action to subgoal position (robot frame)
        l, theta = sa_action[0], sa_action[1]
        subgoal_x = l * math.cos(theta)
        subgoal_y = l * math.sin(theta)
        self.current_subgoal = (subgoal_x, subgoal_y)
        
        # Check if we should replan A*
        should_replan = (self.sa_step_count % self.config.ASTAR_REPLAN_INTERVAL == 0)
        
        if should_replan and len(self.global_path) > 0:
            new_path = self.astar_planner.plan(
                (self.robot.x, self.robot.y),
                self.goal_pos
            )
            if new_path:
                self.global_path = new_path
                self.waypoint_manager.set_path(new_path)
        
        # Collect reward components
        total_reward = 0.0
        done = False
        info = {'ma_steps': 0, 'subgoal': (l, theta)}
        
        # Execute MA steps
        for _ in range(self.ma_steps_per_sa):
            # Get MA state
            ma_state = self._get_ma_state()
            
            # For now, return without MA execution (MA will be handled externally)
            # This allows flexible training modes
            
            self.step_count += 1
            
            # Update obstacles
            self.obstacle_manager.update(self.ma_timestep)
            
            # Check termination conditions
            done, termination = self._check_termination()
            if done:
                break
            
            info['ma_steps'] += 1
        
        # Compute SA reward
        reward = self._compute_sa_reward(done, termination if done else TerminationReason.NONE)
        
        # Get new observation
        obs = self._get_observation()
        
        info['termination'] = termination.value if done else 'none'
        info['should_replan'] = should_replan
        
        return obs, reward, done, info
    
    def step_ma(
        self,
        ma_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one Motion Agent step (0.05s).
        
        Args:
            ma_action: Velocity command (v, omega)
            
        Returns:
            (ma_state, reward, done, info)
        """
        self.step_count += 1
        
        # Apply velocity command
        v, omega = ma_action[0], ma_action[1]
        self._apply_velocity(v, omega, self.ma_timestep)
        
        # Update obstacles
        self.obstacle_manager.update(self.ma_timestep)
        
        # Check termination
        done, termination = self._check_termination()
        
        # Compute MA reward
        reward = self._compute_ma_reward(done, termination)
        
        # Get new MA state
        ma_state = self._get_ma_state()
        
        info = {
            'termination': termination.value if done else 'none',
            'robot_pos': (self.robot.x, self.robot.y),
            'subgoal': self.current_subgoal
        }
        
        return ma_state, reward, done, info
    
    def _apply_velocity(self, v: float, omega: float, dt: float) -> None:
        """
        Apply velocity command to robot (differential drive kinematics).
        
        Args:
            v: Linear velocity (m/s)
            omega: Angular velocity (rad/s)
            dt: Time step (s)
        """
        # Clip velocities
        v = np.clip(v, self.config.MA_MIN_LINEAR_VEL, self.config.MA_MAX_LINEAR_VEL)
        omega = np.clip(omega, self.config.MA_MIN_ANGULAR_VEL, self.config.MA_MAX_ANGULAR_VEL)
        
        # Update orientation
        new_theta = self.robot.theta + omega * dt
        # Normalize to [-pi, pi]
        while new_theta > math.pi:
            new_theta -= 2 * math.pi
        while new_theta < -math.pi:
            new_theta += 2 * math.pi
        
        # Update position
        avg_theta = (self.robot.theta + new_theta) / 2
        new_x = self.robot.x + v * math.cos(avg_theta) * dt
        new_y = self.robot.y + v * math.sin(avg_theta) * dt
        
        # Update robot state
        self.robot.x = new_x
        self.robot.y = new_y
        self.robot.theta = new_theta
        self.robot.linear_vel = v
        self.robot.angular_vel = omega
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get full observation for SA.
        
        Returns:
            Dictionary with lidar, waypoints, robot_state, etc.
        """
        # Get LiDAR scan
        raw_lidar = self._get_lidar_scan()
        processed_lidar = self.lidar_processor.process_normalized(raw_lidar)
        
        # Get waypoints in robot frame
        waypoints = self.waypoint_manager.get_waypoints_robot_frame(
            self.robot.x,
            self.robot.y,
            self.robot.theta
        )
        waypoints_flat = waypoints.flatten() if waypoints is not None else np.zeros(10)
        
        # Distance to goal
        dist_to_goal = math.sqrt(
            (self.goal_pos[0] - self.robot.x)**2 +
            (self.goal_pos[1] - self.robot.y)**2
        )
        
        return {
            'lidar': processed_lidar,
            'waypoints': waypoints_flat,
            'robot_state': self.robot.to_array(),
            'goal': self.goal_pos,
            'dist_to_goal': dist_to_goal,
            'step_count': self.step_count,
            'sa_step_count': self.sa_step_count
        }
    
    def _get_ma_state(self) -> np.ndarray:
        """
        Get Motion Agent state.
        
        State: (v*, omega*, px, py, theta_diff)
        """
        if self.current_subgoal is None:
            return np.zeros(5, dtype=np.float32)
        
        px, py = self.current_subgoal
        theta_diff = math.atan2(py, px)
        
        # Normalize theta_diff to [-pi, pi]
        while theta_diff > math.pi:
            theta_diff -= 2 * math.pi
        while theta_diff < -math.pi:
            theta_diff += 2 * math.pi
        
        state = np.array([
            self.robot.linear_vel,
            self.robot.angular_vel,
            px,
            py,
            theta_diff
        ], dtype=np.float32)
        
        return state
    
    def _get_lidar_scan(self) -> np.ndarray:
        """
        Get LiDAR scan (simulated or from ROS).
        
        Returns:
            Raw LiDAR scan (360 rays)
        """
        if self.use_ros:
            # TODO: Get from ROS topic
            return np.ones(self.config.LIDAR_RAW_RAYS) * self.config.LIDAR_MAX_RANGE
        else:
            # Simulated LiDAR
            return self._simulate_lidar()
    
    def _simulate_lidar(self) -> np.ndarray:
        """
        Simulate LiDAR scan based on scene and obstacles.
        
        Returns:
            Simulated LiDAR scan (360 rays)
        """
        num_rays = self.config.LIDAR_RAW_RAYS
        max_range = self.config.LIDAR_MAX_RANGE
        
        scan = np.ones(num_rays) * max_range
        
        if self.scene is None or self.scene.grid is None:
            return scan
        
        # Ray casting for each angle
        for i in range(num_rays):
            angle = self.robot.theta + (2 * math.pi * i / num_rays)
            
            # Cast ray
            for r in np.arange(0.05, max_range, 0.05):
                rx = self.robot.x + r * math.cos(angle)
                ry = self.robot.y + r * math.sin(angle)
                
                # Check scene grid
                if not self.scene.is_free(rx, ry, margin=0.0):
                    scan[i] = r
                    break
                
                # Check obstacles
                for obs in self.obstacle_manager.obstacles:
                    dx = rx - obs.position[0]
                    dy = ry - obs.position[1]
                    dist = math.sqrt(dx**2 + dy**2)
                    obs_radius = max(obs.size[0], obs.size[1]) / 2
                    
                    if dist < obs_radius:
                        scan[i] = r
                        break
                else:
                    continue
                break
        
        return scan
    
    def _check_termination(self) -> Tuple[bool, TerminationReason]:
        """
        Check episode termination conditions.
        
        Returns:
            (done, reason)
        """
        # Goal reached
        dist_to_goal = math.sqrt(
            (self.goal_pos[0] - self.robot.x)**2 +
            (self.goal_pos[1] - self.robot.y)**2
        )
        if dist_to_goal < self.config.GOAL_THRESHOLD:
            return True, TerminationReason.GOAL_REACHED
        
        # Collision with walls
        if self.scene and not self.scene.is_free(self.robot.x, self.robot.y, 
                                                  margin=self.config.COLLISION_DISTANCE):
            return True, TerminationReason.COLLISION
        
        # Collision with obstacles
        collision, _ = self.obstacle_manager.check_collision(
            (self.robot.x, self.robot.y),
            self.config.COLLISION_DISTANCE
        )
        if collision:
            return True, TerminationReason.COLLISION
        
        # Timeout
        if self.step_count >= self.config.EPISODE_TIMEOUT:
            return True, TerminationReason.TIMEOUT
        
        return False, TerminationReason.NONE
    
    def _compute_sa_reward(
        self,
        done: bool,
        termination: TerminationReason
    ) -> float:
        """
        Compute Subgoal Agent reward.
        
        Reward components (from paper):
        - Collision: -10
        - Path distance penalty: -0.5 * dist_to_path
        - Safety penalty: -2 * safety_violation
        - Goal reached: +100
        """
        reward = 0.0
        
        if termination == TerminationReason.GOAL_REACHED:
            reward += self.config.SA_REWARD_GOAL
        
        elif termination == TerminationReason.COLLISION:
            reward += self.config.SA_REWARD_COLLISION
        
        else:
            # Path distance penalty
            if self.global_path:
                min_dist = float('inf')
                for pt in self.global_path:
                    dist = math.sqrt((self.robot.x - pt[0])**2 + (self.robot.y - pt[1])**2)
                    min_dist = min(min_dist, dist)
                reward += self.config.SA_REWARD_PATH_COEFF * min_dist
            
            # Safety penalty (distance to closest obstacle)
            closest_dist = float('inf')
            for obs in self.obstacle_manager.obstacles:
                dist = math.sqrt(
                    (self.robot.x - obs.position[0])**2 +
                    (self.robot.y - obs.position[1])**2
                )
                closest_dist = min(closest_dist, dist)
            
            if closest_dist < self.config.SA_SAFETY_DISTANCE:
                safety_violation = self.config.SA_SAFETY_DISTANCE - closest_dist
                reward += self.config.SA_REWARD_SAFETY_COEFF * safety_violation
        
        return reward
    
    def _compute_ma_reward(
        self,
        done: bool,
        termination: TerminationReason
    ) -> float:
        """
        Compute Motion Agent reward.
        
        Reward components (from paper):
        - Reach subgoal: +2
        - Distance penalty: -1 * dist_to_subgoal
        """
        if self.current_subgoal is None:
            return 0.0
        
        # Distance to subgoal (in robot frame = just magnitude)
        px, py = self.current_subgoal
        dist_to_subgoal = math.sqrt(px**2 + py**2)
        
        if dist_to_subgoal < self.config.MA_SUBGOAL_THRESHOLD:
            return self.config.MA_REWARD_REACH
        
        return self.config.MA_REWARD_DIST_COEFF * dist_to_subgoal
    
    def update_subgoal_robot_frame(self) -> None:
        """
        Update subgoal position in robot frame after robot moved.
        
        This should be called after each MA step to keep subgoal
        in robot-centric coordinates.
        """
        if self.current_subgoal is None:
            return
        
        # The subgoal is already in robot frame and gets updated
        # as the robot moves toward it
        px, py = self.current_subgoal
        
        # After robot moves, subgoal appears closer/rotated
        # This is handled by the MA state construction
    
    def get_goal_in_robot_frame(self) -> Tuple[float, float]:
        """Get goal position in robot frame."""
        dx = self.goal_pos[0] - self.robot.x
        dy = self.goal_pos[1] - self.robot.y
        
        # Rotate to robot frame
        c = math.cos(-self.robot.theta)
        s = math.sin(-self.robot.theta)
        
        px = c * dx - s * dy
        py = s * dx + c * dy
        
        return px, py
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the current environment state.
        
        Returns:
            RGB image array or None
        """
        # TODO: Implement visualization
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass


class MAPretrainingEnvironment:
    """
    Simplified environment for Motion Agent pre-training.
    
    Uses sampled subgoals instead of SA predictions.
    Based on paper's pre-training procedure.
    """
    
    def __init__(self, config: HierarchicalConfig = None):
        if config is None:
            config = HierarchicalConfig()
        self.config = config
        
        # Subgoal sampling parameters
        self.max_distance = config.MA_SUBGOAL_SAMPLE_DISTANCE_MAX
        self.straight_prob = config.MA_SUBGOAL_STRAIGHT_PROB
        self.curvy_prob = config.MA_SUBGOAL_CURVY_PROB
        
        # State
        self.subgoal: Tuple[float, float] = (0, 0)
        self.prev_v = 0.0
        self.prev_omega = 0.0
        self.step_count = 0
        self.max_steps = 50  # Steps to reach subgoal
        
        # Position tracking (relative to start)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.theta = 0.0
    
    def reset(self) -> np.ndarray:
        """Reset and sample new subgoal."""
        self.step_count = 0
        self.prev_v = 0.0
        self.prev_omega = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.theta = 0.0
        
        # Sample subgoal
        self.subgoal = self._sample_subgoal()
        
        return self._get_state()
    
    def _sample_subgoal(self) -> Tuple[float, float]:
        """
        Sample a subgoal for pre-training.
        
        From paper:
        - 20% straight-line
        - 30% curvy (±π/2)
        - 50% random
        """
        import random
        
        r = random.random()
        distance = random.uniform(0.1, self.max_distance)
        
        if r < self.straight_prob:
            # Straight ahead
            return (distance, 0.0)
        elif r < self.straight_prob + self.curvy_prob:
            # Curvy (±π/2)
            angle = random.choice([-math.pi/2, math.pi/2])
            angle += random.uniform(-0.3, 0.3)  # Add some variation
            px = distance * math.cos(angle)
            py = distance * math.sin(angle)
            return (px, py)
        else:
            # Random
            angle = random.uniform(-math.pi, math.pi)
            px = distance * math.cos(angle)
            py = distance * math.sin(angle)
            return (px, py)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step.
        
        Args:
            action: (v, omega)
            
        Returns:
            (state, reward, done, info)
        """
        self.step_count += 1
        
        v, omega = action[0], action[1]
        dt = self.config.MA_TIME_STEP
        
        # Apply velocity (update robot position)
        self.theta += omega * dt
        self.pos_x += v * math.cos(self.theta) * dt
        self.pos_y += v * math.sin(self.theta) * dt
        
        self.prev_v = v
        self.prev_omega = omega
        
        # Update subgoal in robot frame
        self._update_subgoal_robot_frame(v, omega, dt)
        
        # Check termination
        px, py = self.subgoal
        dist = math.sqrt(px**2 + py**2)
        
        done = dist < self.config.MA_SUBGOAL_THRESHOLD or self.step_count >= self.max_steps
        success = dist < self.config.MA_SUBGOAL_THRESHOLD
        
        # Compute reward
        if success:
            reward = self.config.MA_REWARD_REACH
        else:
            reward = self.config.MA_REWARD_DIST_COEFF * dist
        
        info = {'success': success, 'dist_to_subgoal': dist}
        
        return self._get_state(), reward, done, info
    
    def _update_subgoal_robot_frame(self, v: float, omega: float, dt: float):
        """Update subgoal position after robot moved."""
        px, py = self.subgoal
        
        # Robot moved forward by v*dt and rotated by omega*dt
        # Subgoal appears to move backward and rotate
        
        # Translate (robot moved forward)
        px -= v * dt
        
        # Rotate (robot rotated)
        c = math.cos(-omega * dt)
        s = math.sin(-omega * dt)
        new_px = c * px - s * py
        new_py = s * px + c * py
        
        self.subgoal = (new_px, new_py)
    
    def _get_state(self) -> np.ndarray:
        """Get MA state."""
        px, py = self.subgoal
        theta_diff = math.atan2(py, px)
        
        return np.array([
            self.prev_v,
            self.prev_omega,
            px,
            py,
            theta_diff
        ], dtype=np.float32)


if __name__ == "__main__":
    # Test hierarchical environment
    print("Testing Hierarchical Environment...")
    
    config = HierarchicalConfig()
    env = HierarchicalEnvironment(config)
    
    # Test reset
    obs = env.reset(SceneType.CORRIDOR)
    print(f"\nScene: CORRIDOR")
    print(f"Robot start: ({env.robot.x:.2f}, {env.robot.y:.2f})")
    print(f"Goal: {env.goal_pos}")
    print(f"LiDAR shape: {obs['lidar'].shape}")
    print(f"Waypoints shape: {obs['waypoints'].shape}")
    
    # Test SA step
    sa_action = np.array([0.3, 0.0])  # 0.3m forward
    obs, reward, done, info = env.step_sa(sa_action)
    print(f"\nSA step: action={sa_action}, reward={reward:.2f}, done={done}")
    
    # Test MA step
    ma_action = np.array([0.2, 0.0])  # Forward
    ma_state, ma_reward, ma_done, ma_info = env.step_ma(ma_action)
    print(f"MA step: action={ma_action}, reward={ma_reward:.2f}")
    print(f"MA state: {ma_state}")
    
    # Test MA pre-training environment
    print("\n" + "="*50)
    print("Testing MA Pre-training Environment...")
    
    ma_env = MAPretrainingEnvironment(config)
    state = ma_env.reset()
    print(f"Initial state: {state}")
    print(f"Subgoal: {ma_env.subgoal}")
    
    # Run a few steps
    total_reward = 0
    for i in range(20):
        action = np.array([0.3, 0.1])  # Forward + slight turn
        state, reward, done, info = ma_env.step(action)
        total_reward += reward
        if done:
            print(f"Episode done at step {i+1}: success={info['success']}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    
    print("\n✓ Hierarchical environment tests complete!")

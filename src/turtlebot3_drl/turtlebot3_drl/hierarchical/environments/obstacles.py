"""
Obstacle Manager for Hierarchical DRL Navigation

Manages dynamic and static obstacles during training episodes.
Based on the paper: "Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Obstacle Configuration (from paper):
- 2 dynamic obstacles (cuboids representing pedestrians)
- 1 static unknown obstacle (not in given map)
- Dynamic obstacles move on A* paths at 0.1-0.5 m/s
- One dynamic obstacle on robot's global path (middle/end to start)
- One dynamic obstacle crosses robot's path
- Static obstacle placed randomly on robot's planned path
"""

import math
import random
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ObstacleType(Enum):
    """Types of obstacles."""
    DYNAMIC = "dynamic"
    STATIC = "static"


@dataclass
class Obstacle:
    """
    Represents an obstacle in the environment.
    
    Attributes:
        id: Unique identifier
        type: DYNAMIC or STATIC
        position: Current (x, y) position
        size: (width, depth, height) in meters
        velocity: Speed for dynamic obstacles (m/s)
        path: A* path for dynamic obstacles
        path_index: Current index on path
        direction: +1 (forward) or -1 (backward) on path
    """
    id: int
    type: ObstacleType
    position: Tuple[float, float]
    size: Tuple[float, float, float] = (0.4, 0.4, 1.0)  # Pedestrian-like
    velocity: float = 0.0
    path: List[Tuple[float, float]] = None
    path_index: int = 0
    direction: int = 1
    
    def __post_init__(self):
        if self.path is None:
            self.path = []


class DynamicObstacleMode(Enum):
    """Mode for dynamic obstacle path placement."""
    ON_ROBOT_PATH = "on_robot_path"  # Middle/end to start of robot's path
    CROSSING = "crossing"  # Crosses the robot's path


class ObstacleManager:
    """
    Manages obstacles during training episodes.
    
    Features:
    - Spawns dynamic obstacles that move on A* paths
    - Spawns static obstacles on robot's planned path
    - Updates obstacle positions during simulation
    - Provides obstacle states for collision checking
    """
    
    # Speed range from paper
    SPEED_MIN = 0.1  # m/s
    SPEED_MAX = 0.5  # m/s
    
    # Default obstacle size (pedestrian approximation)
    DEFAULT_SIZE = (0.4, 0.4, 1.0)  # width, depth, height
    
    def __init__(
        self,
        num_dynamic: int = 2,
        num_static: int = 1,
        obstacle_size: Tuple[float, float, float] = None
    ):
        """
        Initialize obstacle manager.
        
        Args:
            num_dynamic: Number of dynamic obstacles (default: 2)
            num_static: Number of static obstacles (default: 1)
            obstacle_size: (width, depth, height) in meters
        """
        self.num_dynamic = num_dynamic
        self.num_static = num_static
        self.obstacle_size = obstacle_size or self.DEFAULT_SIZE
        
        # Current obstacles
        self.obstacles: List[Obstacle] = []
        self.dynamic_obstacles: List[Obstacle] = []
        self.static_obstacles: List[Obstacle] = []
        
        # Scene reference (for A* planning)
        self.scene = None
        self.astar_planner = None
        
        # Robot's global path (for obstacle placement)
        self.robot_path: List[Tuple[float, float]] = []
        
        # Counter for unique IDs
        self._next_id = 0
    
    def set_scene(self, scene, astar_planner) -> None:
        """
        Set the scene and planner for obstacle path generation.
        
        Args:
            scene: Current scene instance
            astar_planner: A* planner for dynamic obstacle paths
        """
        self.scene = scene
        self.astar_planner = astar_planner
    
    def reset(
        self,
        robot_start: Tuple[float, float],
        robot_goal: Tuple[float, float],
        robot_path: List[Tuple[float, float]]
    ) -> List[Obstacle]:
        """
        Reset obstacles for a new episode.
        
        Args:
            robot_start: Robot's start position
            robot_goal: Robot's goal position
            robot_path: Robot's planned A* path
            
        Returns:
            List of all obstacles
        """
        self.obstacles = []
        self.dynamic_obstacles = []
        self.static_obstacles = []
        self.robot_path = robot_path
        self._next_id = 0
        
        if not robot_path or len(robot_path) < 2:
            return self.obstacles
        
        # Spawn dynamic obstacles
        self._spawn_dynamic_obstacles(robot_start, robot_goal)
        
        # Spawn static obstacles
        self._spawn_static_obstacles()
        
        return self.obstacles
    
    def _spawn_dynamic_obstacles(
        self,
        robot_start: Tuple[float, float],
        robot_goal: Tuple[float, float]
    ) -> None:
        """
        Spawn dynamic obstacles according to paper specifications.
        
        - One on robot's path (from middle/end to start)
        - One crossing the robot's path
        """
        if self.num_dynamic < 1:
            return
        
        # First dynamic obstacle: on robot's path
        if self.num_dynamic >= 1:
            obs = self._create_on_path_obstacle(robot_start, robot_goal)
            if obs:
                self.obstacles.append(obs)
                self.dynamic_obstacles.append(obs)
        
        # Second dynamic obstacle: crossing robot's path
        if self.num_dynamic >= 2:
            obs = self._create_crossing_obstacle()
            if obs:
                self.obstacles.append(obs)
                self.dynamic_obstacles.append(obs)
        
        # Additional dynamic obstacles (random positions)
        for i in range(2, self.num_dynamic):
            obs = self._create_random_dynamic_obstacle()
            if obs:
                self.obstacles.append(obs)
                self.dynamic_obstacles.append(obs)
    
    def _create_on_path_obstacle(
        self,
        robot_start: Tuple[float, float],
        robot_goal: Tuple[float, float]
    ) -> Optional[Obstacle]:
        """
        Create an obstacle that moves on robot's global path.
        
        Paper: "moves from middle or end to the start position of the robot's global A* path"
        """
        if len(self.robot_path) < 3:
            return None
        
        # Random speed
        speed = random.uniform(self.SPEED_MIN, self.SPEED_MAX)
        
        # Start from middle or end of robot's path
        if random.random() < 0.5:
            # Start from middle
            start_idx = len(self.robot_path) // 2
        else:
            # Start from end (near goal)
            start_idx = len(self.robot_path) - 1
        
        # Path: from start_idx to robot start (reversed direction)
        obstacle_path = list(reversed(self.robot_path[:start_idx + 1]))
        
        if len(obstacle_path) < 2:
            obstacle_path = list(reversed(self.robot_path))
        
        start_pos = obstacle_path[0]
        
        obs = Obstacle(
            id=self._next_id,
            type=ObstacleType.DYNAMIC,
            position=start_pos,
            size=self.obstacle_size,
            velocity=speed,
            path=obstacle_path,
            path_index=0,
            direction=1
        )
        self._next_id += 1
        
        return obs
    
    def _create_crossing_obstacle(self) -> Optional[Obstacle]:
        """
        Create an obstacle that crosses robot's path.
        
        Paper: "The other one is sampled so that it crosses the robot's global path"
        """
        if not self.robot_path or len(self.robot_path) < 3:
            return None
        
        if self.scene is None or self.astar_planner is None:
            return None
        
        # Pick a crossing point on robot's path
        cross_idx = random.randint(len(self.robot_path) // 4, 3 * len(self.robot_path) // 4)
        cross_point = self.robot_path[cross_idx]
        
        # Find perpendicular direction to path at crossing point
        if cross_idx > 0 and cross_idx < len(self.robot_path) - 1:
            prev_pt = self.robot_path[cross_idx - 1]
            next_pt = self.robot_path[cross_idx + 1]
            dx = next_pt[0] - prev_pt[0]
            dy = next_pt[1] - prev_pt[1]
        else:
            dx, dy = 1.0, 0.0
        
        # Perpendicular direction
        length = math.sqrt(dx**2 + dy**2)
        if length > 0.001:
            perp_x, perp_y = -dy / length, dx / length
        else:
            perp_x, perp_y = 0.0, 1.0
        
        # Sample start and end positions perpendicular to path
        offset = random.uniform(1.5, 3.0)
        start_pos = (cross_point[0] - perp_x * offset, cross_point[1] - perp_y * offset)
        end_pos = (cross_point[0] + perp_x * offset, cross_point[1] + perp_y * offset)
        
        # Check if positions are valid
        if not self._is_valid_position(start_pos) or not self._is_valid_position(end_pos):
            # Fall back to random obstacle
            return self._create_random_dynamic_obstacle()
        
        # Create path (simple straight line through crossing point)
        obstacle_path = self._create_line_path(start_pos, end_pos)
        
        if len(obstacle_path) < 2:
            return self._create_random_dynamic_obstacle()
        
        speed = random.uniform(self.SPEED_MIN, self.SPEED_MAX)
        
        obs = Obstacle(
            id=self._next_id,
            type=ObstacleType.DYNAMIC,
            position=start_pos,
            size=self.obstacle_size,
            velocity=speed,
            path=obstacle_path,
            path_index=0,
            direction=1
        )
        self._next_id += 1
        
        return obs
    
    def _create_random_dynamic_obstacle(self) -> Optional[Obstacle]:
        """Create a random dynamic obstacle with its own A* path."""
        if self.scene is None:
            return None
        
        # Random start position
        for _ in range(20):
            start_x = random.uniform(0.5, self.scene.length - 0.5)
            start_y = random.uniform(0.5, self.scene.width - 0.5)
            
            if self._is_valid_position((start_x, start_y)):
                break
        else:
            return None
        
        # Random end position
        for _ in range(20):
            end_x = random.uniform(0.5, self.scene.length - 0.5)
            end_y = random.uniform(0.5, self.scene.width - 0.5)
            
            dist = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            if dist > 1.0 and self._is_valid_position((end_x, end_y)):
                break
        else:
            return None
        
        # Create simple line path
        obstacle_path = self._create_line_path((start_x, start_y), (end_x, end_y))
        
        if len(obstacle_path) < 2:
            return None
        
        speed = random.uniform(self.SPEED_MIN, self.SPEED_MAX)
        
        obs = Obstacle(
            id=self._next_id,
            type=ObstacleType.DYNAMIC,
            position=(start_x, start_y),
            size=self.obstacle_size,
            velocity=speed,
            path=obstacle_path,
            path_index=0,
            direction=1
        )
        self._next_id += 1
        
        return obs
    
    def _create_line_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        resolution: float = 0.1
    ) -> List[Tuple[float, float]]:
        """Create a straight line path from start to end."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < resolution:
            return [start, end]
        
        num_points = int(distance / resolution) + 1
        path = []
        
        for i in range(num_points + 1):
            t = i / num_points
            x = start[0] + t * dx
            y = start[1] + t * dy
            path.append((x, y))
        
        return path
    
    def _spawn_static_obstacles(self) -> None:
        """
        Spawn static obstacles on robot's planned path.
        
        Paper: "The unknown static obstacle is placed randomly on the robot's planned path"
        """
        if self.num_static < 1 or len(self.robot_path) < 3:
            return
        
        for _ in range(self.num_static):
            # Pick a random point on robot's path (not too close to start/end)
            margin = max(1, len(self.robot_path) // 5)
            path_idx = random.randint(margin, len(self.robot_path) - margin - 1)
            
            pos = self.robot_path[path_idx]
            
            # Add small random offset
            offset_x = random.uniform(-0.3, 0.3)
            offset_y = random.uniform(-0.3, 0.3)
            pos = (pos[0] + offset_x, pos[1] + offset_y)
            
            obs = Obstacle(
                id=self._next_id,
                type=ObstacleType.STATIC,
                position=pos,
                size=self.obstacle_size,
                velocity=0.0
            )
            self._next_id += 1
            
            self.obstacles.append(obs)
            self.static_obstacles.append(obs)
    
    def _is_valid_position(self, pos: Tuple[float, float]) -> bool:
        """Check if position is valid (in free space)."""
        if self.scene is None:
            return True
        return self.scene.is_free(pos[0], pos[1], margin=0.3)
    
    def update(self, dt: float) -> None:
        """
        Update obstacle positions.
        
        Args:
            dt: Time step in seconds
        """
        for obs in self.dynamic_obstacles:
            self._update_dynamic_obstacle(obs, dt)
    
    def _update_dynamic_obstacle(self, obs: Obstacle, dt: float) -> None:
        """
        Update a single dynamic obstacle.
        
        Obstacle moves along its path, reversing at endpoints.
        """
        if not obs.path or len(obs.path) < 2:
            return
        
        # Calculate movement distance
        move_dist = obs.velocity * dt
        
        # Current position on path
        current_pos = obs.position
        target_idx = obs.path_index + obs.direction
        
        # Check bounds
        if target_idx >= len(obs.path):
            obs.direction = -1
            target_idx = len(obs.path) - 2
        elif target_idx < 0:
            obs.direction = 1
            target_idx = 1
        
        if target_idx < 0 or target_idx >= len(obs.path):
            return
        
        target_pos = obs.path[target_idx]
        
        # Move toward target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 0.001:
            # Already at target, move to next waypoint
            obs.path_index = target_idx
            return
        
        if move_dist >= dist:
            # Reach target this step
            obs.position = target_pos
            obs.path_index = target_idx
        else:
            # Move toward target
            ratio = move_dist / dist
            new_x = current_pos[0] + dx * ratio
            new_y = current_pos[1] + dy * ratio
            obs.position = (new_x, new_y)
    
    def get_obstacle_positions(self) -> List[Tuple[float, float]]:
        """Get all obstacle positions."""
        return [obs.position for obs in self.obstacles]
    
    def get_dynamic_positions(self) -> List[Tuple[float, float]]:
        """Get dynamic obstacle positions."""
        return [obs.position for obs in self.dynamic_obstacles]
    
    def get_static_positions(self) -> List[Tuple[float, float]]:
        """Get static obstacle positions."""
        return [obs.position for obs in self.static_obstacles]
    
    def check_collision(
        self,
        robot_pos: Tuple[float, float],
        robot_radius: float = 0.18
    ) -> Tuple[bool, Optional[Obstacle]]:
        """
        Check if robot collides with any obstacle.
        
        Args:
            robot_pos: Robot (x, y) position
            robot_radius: Robot collision radius
            
        Returns:
            (collision, obstacle) - True if collision, with colliding obstacle
        """
        for obs in self.obstacles:
            # Simple circular collision check
            dx = robot_pos[0] - obs.position[0]
            dy = robot_pos[1] - obs.position[1]
            dist = math.sqrt(dx**2 + dy**2)
            
            # Use obstacle width for radius
            obs_radius = max(obs.size[0], obs.size[1]) / 2
            
            if dist < (robot_radius + obs_radius):
                return True, obs
        
        return False, None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current obstacle state for rendering/debugging."""
        return {
            'num_dynamic': len(self.dynamic_obstacles),
            'num_static': len(self.static_obstacles),
            'dynamic_positions': self.get_dynamic_positions(),
            'static_positions': self.get_static_positions(),
            'obstacles': [
                {
                    'id': obs.id,
                    'type': obs.type.value,
                    'position': obs.position,
                    'velocity': obs.velocity
                }
                for obs in self.obstacles
            ]
        }


if __name__ == "__main__":
    # Test obstacle manager
    print("Testing Obstacle Manager...")
    
    # Create a simple mock scene
    class MockScene:
        def __init__(self):
            self.width = 10.0
            self.length = 10.0
        
        def is_free(self, x, y, margin=0.2):
            return 0 < x < self.width and 0 < y < self.length
    
    scene = MockScene()
    
    # Create manager
    manager = ObstacleManager(num_dynamic=2, num_static=1)
    manager.set_scene(scene, None)
    
    # Create a mock robot path
    robot_path = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
    
    # Reset with robot path
    obstacles = manager.reset(robot_start=(1, 1), robot_goal=(8, 8), robot_path=robot_path)
    
    print(f"\nSpawned {len(obstacles)} obstacles:")
    for obs in obstacles:
        print(f"  {obs.type.value}: id={obs.id}, pos={obs.position}, vel={obs.velocity:.2f} m/s")
    
    # Test update
    print("\nSimulating 1 second of movement (20 steps at 0.05s)...")
    for _ in range(20):
        manager.update(0.05)
    
    print("\nAfter update:")
    for obs in manager.dynamic_obstacles:
        print(f"  Dynamic {obs.id}: pos={obs.position}")
    
    # Test collision
    collision, obs = manager.check_collision((5, 5), robot_radius=0.5)
    print(f"\nCollision at (5,5) with r=0.5: {collision}")
    
    print("\n✓ Obstacle manager test complete!")

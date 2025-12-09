"""
Obstacle Management for Hierarchical Navigation

Handles both static and dynamic obstacles in the environment.
Dynamic obstacles move according to simple patterns.
"""

import math
import random
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ObstacleType(Enum):
    """Obstacle types."""
    STATIC = "static"
    DYNAMIC_LINEAR = "dynamic_linear"
    DYNAMIC_CIRCULAR = "dynamic_circular"
    DYNAMIC_RANDOM = "dynamic_random"


@dataclass
class Obstacle:
    """Obstacle container."""
    id: int
    type: ObstacleType
    position: Tuple[float, float]
    size: Tuple[float, float]  # (width, height) or (radius, radius)
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    # For circular motion
    center: Optional[Tuple[float, float]] = None
    radius: float = 0.0
    angular_vel: float = 0.0
    angle: float = 0.0
    
    # Bounds for random motion
    bounds: Optional[Tuple[float, float, float, float]] = None
    
    def get_radius(self) -> float:
        """Get collision radius."""
        return max(self.size[0], self.size[1]) / 2


class ObstacleManager:
    """
    Manages obstacles in the environment.
    
    Handles:
    - Static obstacles (from scene)
    - Dynamic obstacles (moving)
    - Collision checking
    """
    
    def __init__(
        self,
        num_dynamic: int = 2,
        num_static: int = 0,
        dynamic_speed: float = 0.3
    ):
        """
        Initialize obstacle manager.
        
        Args:
            num_dynamic: Number of dynamic obstacles
            num_static: Number of additional static obstacles (beyond scene obstacles)
            dynamic_speed: Speed of dynamic obstacles (m/s)
        """
        self.num_dynamic = num_dynamic
        self.num_static = num_static
        self.dynamic_speed = dynamic_speed
        
        self.obstacles: List[Obstacle] = []
        self.scene = None
        self.planner = None
        
        self._obstacle_id_counter = 0
    
    def set_scene(self, scene: Any, planner: Any = None) -> None:
        """
        Set the current scene.
        
        Args:
            scene: Scene object with bounds and is_free method
            planner: Optional A* planner for path-aware obstacle placement
        """
        self.scene = scene
        self.planner = planner
    
    def reset(
        self,
        robot_start: Tuple[float, float],
        robot_goal: Tuple[float, float],
        robot_path: List[Tuple[float, float]] = None
    ) -> None:
        """
        Reset obstacles for a new episode.
        
        Args:
            robot_start: Robot starting position
            robot_goal: Robot goal position
            robot_path: Optional robot planned path (to avoid blocking)
        """
        self.obstacles = []
        self._obstacle_id_counter = 0
        
        if self.scene is None:
            return
        
        bounds = self.scene.bounds
        
        # Create dynamic obstacles
        for _ in range(self.num_dynamic):
            obs = self._create_dynamic_obstacle(
                bounds, robot_start, robot_goal, robot_path
            )
            if obs:
                self.obstacles.append(obs)
        
        # Create static obstacles
        for _ in range(self.num_static):
            obs = self._create_static_obstacle(
                bounds, robot_start, robot_goal, robot_path
            )
            if obs:
                self.obstacles.append(obs)
    
    def _create_dynamic_obstacle(
        self,
        bounds: Tuple[float, float, float, float],
        robot_start: Tuple[float, float],
        robot_goal: Tuple[float, float],
        robot_path: List[Tuple[float, float]] = None
    ) -> Optional[Obstacle]:
        """Create a dynamic obstacle."""
        min_dist = 1.0  # Minimum distance from start/goal
        
        for _ in range(20):  # Max attempts
            # Random position within bounds
            x = random.uniform(bounds[0] + 0.5, bounds[2] - 0.5)
            y = random.uniform(bounds[1] + 0.5, bounds[3] - 0.5)
            
            # Check distance from start and goal
            dist_start = math.sqrt((x - robot_start[0])**2 + (y - robot_start[1])**2)
            dist_goal = math.sqrt((x - robot_goal[0])**2 + (y - robot_goal[1])**2)
            
            if dist_start < min_dist or dist_goal < min_dist:
                continue
            
            # Check if position is free
            if self.scene and not self.scene.is_free(x, y, margin=0.3):
                continue
            
            # Choose motion type
            motion_type = random.choice([
                ObstacleType.DYNAMIC_LINEAR,
                ObstacleType.DYNAMIC_CIRCULAR,
                ObstacleType.DYNAMIC_RANDOM
            ])
            
            self._obstacle_id_counter += 1
            size = (random.uniform(0.2, 0.4), random.uniform(0.2, 0.4))
            
            if motion_type == ObstacleType.DYNAMIC_LINEAR:
                # Linear back-and-forth motion
                angle = random.uniform(0, 2 * math.pi)
                vx = self.dynamic_speed * math.cos(angle)
                vy = self.dynamic_speed * math.sin(angle)
                
                return Obstacle(
                    id=self._obstacle_id_counter,
                    type=motion_type,
                    position=(x, y),
                    size=size,
                    velocity=(vx, vy),
                    bounds=bounds
                )
            
            elif motion_type == ObstacleType.DYNAMIC_CIRCULAR:
                # Circular motion
                radius = random.uniform(0.5, 1.0)
                angular_vel = self.dynamic_speed / radius
                
                return Obstacle(
                    id=self._obstacle_id_counter,
                    type=motion_type,
                    position=(x, y),
                    size=size,
                    center=(x, y),
                    radius=radius,
                    angular_vel=angular_vel,
                    angle=random.uniform(0, 2 * math.pi)
                )
            
            else:  # DYNAMIC_RANDOM
                return Obstacle(
                    id=self._obstacle_id_counter,
                    type=motion_type,
                    position=(x, y),
                    size=size,
                    velocity=(0.0, 0.0),
                    bounds=bounds
                )
        
        return None
    
    def _create_static_obstacle(
        self,
        bounds: Tuple[float, float, float, float],
        robot_start: Tuple[float, float],
        robot_goal: Tuple[float, float],
        robot_path: List[Tuple[float, float]] = None
    ) -> Optional[Obstacle]:
        """Create a static obstacle."""
        min_dist = 0.8
        
        for _ in range(20):
            x = random.uniform(bounds[0] + 0.5, bounds[2] - 0.5)
            y = random.uniform(bounds[1] + 0.5, bounds[3] - 0.5)
            
            dist_start = math.sqrt((x - robot_start[0])**2 + (y - robot_start[1])**2)
            dist_goal = math.sqrt((x - robot_goal[0])**2 + (y - robot_goal[1])**2)
            
            if dist_start < min_dist or dist_goal < min_dist:
                continue
            
            if self.scene and not self.scene.is_free(x, y, margin=0.3):
                continue
            
            self._obstacle_id_counter += 1
            
            return Obstacle(
                id=self._obstacle_id_counter,
                type=ObstacleType.STATIC,
                position=(x, y),
                size=(random.uniform(0.3, 0.5), random.uniform(0.3, 0.5))
            )
        
        return None
    
    def update(self, dt: float) -> None:
        """
        Update obstacle positions.
        
        Args:
            dt: Time step in seconds
        """
        for obs in self.obstacles:
            if obs.type == ObstacleType.STATIC:
                continue
            
            elif obs.type == ObstacleType.DYNAMIC_LINEAR:
                self._update_linear(obs, dt)
            
            elif obs.type == ObstacleType.DYNAMIC_CIRCULAR:
                self._update_circular(obs, dt)
            
            elif obs.type == ObstacleType.DYNAMIC_RANDOM:
                self._update_random(obs, dt)
    
    def _update_linear(self, obs: Obstacle, dt: float) -> None:
        """Update obstacle with linear motion."""
        new_x = obs.position[0] + obs.velocity[0] * dt
        new_y = obs.position[1] + obs.velocity[1] * dt
        
        # Bounce off walls
        if obs.bounds:
            margin = max(obs.size) / 2 + 0.1
            if new_x < obs.bounds[0] + margin or new_x > obs.bounds[2] - margin:
                obs.velocity = (-obs.velocity[0], obs.velocity[1])
                new_x = obs.position[0] + obs.velocity[0] * dt
            if new_y < obs.bounds[1] + margin or new_y > obs.bounds[3] - margin:
                obs.velocity = (obs.velocity[0], -obs.velocity[1])
                new_y = obs.position[1] + obs.velocity[1] * dt
        
        obs.position = (new_x, new_y)
    
    def _update_circular(self, obs: Obstacle, dt: float) -> None:
        """Update obstacle with circular motion."""
        obs.angle += obs.angular_vel * dt
        
        if obs.center:
            new_x = obs.center[0] + obs.radius * math.cos(obs.angle)
            new_y = obs.center[1] + obs.radius * math.sin(obs.angle)
            obs.position = (new_x, new_y)
    
    def _update_random(self, obs: Obstacle, dt: float) -> None:
        """Update obstacle with random motion."""
        # Occasionally change direction
        if random.random() < 0.05:
            angle = random.uniform(0, 2 * math.pi)
            speed = self.dynamic_speed * random.uniform(0.5, 1.0)
            obs.velocity = (speed * math.cos(angle), speed * math.sin(angle))
        
        # Move
        new_x = obs.position[0] + obs.velocity[0] * dt
        new_y = obs.position[1] + obs.velocity[1] * dt
        
        # Bounce off walls
        if obs.bounds:
            margin = max(obs.size) / 2 + 0.1
            if new_x < obs.bounds[0] + margin or new_x > obs.bounds[2] - margin:
                obs.velocity = (-obs.velocity[0], obs.velocity[1])
                new_x = obs.position[0]
            if new_y < obs.bounds[1] + margin or new_y > obs.bounds[3] - margin:
                obs.velocity = (obs.velocity[0], -obs.velocity[1])
                new_y = obs.position[1]
        
        obs.position = (new_x, new_y)
    
    def check_collision(
        self,
        position: Tuple[float, float],
        robot_radius: float = 0.18
    ) -> Tuple[bool, Optional[Obstacle]]:
        """
        Check collision with any obstacle.
        
        Args:
            position: Position to check (x, y)
            robot_radius: Robot collision radius
            
        Returns:
            (collision, obstacle) - obstacle is None if no collision
        """
        for obs in self.obstacles:
            dist = math.sqrt(
                (position[0] - obs.position[0])**2 +
                (position[1] - obs.position[1])**2
            )
            min_dist = robot_radius + obs.get_radius()
            
            if dist < min_dist:
                return True, obs
        
        return False, None
    
    def get_closest_obstacle(
        self,
        position: Tuple[float, float]
    ) -> Tuple[float, Optional[Obstacle]]:
        """
        Get closest obstacle to position.
        
        Args:
            position: Query position (x, y)
            
        Returns:
            (distance, obstacle) - distance is inf if no obstacles
        """
        min_dist = float('inf')
        closest = None
        
        for obs in self.obstacles:
            dist = math.sqrt(
                (position[0] - obs.position[0])**2 +
                (position[1] - obs.position[1])**2
            ) - obs.get_radius()
            
            if dist < min_dist:
                min_dist = dist
                closest = obs
        
        return min_dist, closest
    
    def get_obstacles_in_range(
        self,
        position: Tuple[float, float],
        range_m: float
    ) -> List[Obstacle]:
        """
        Get all obstacles within range.
        
        Args:
            position: Query position
            range_m: Range in meters
            
        Returns:
            List of obstacles within range
        """
        result = []
        for obs in self.obstacles:
            dist = math.sqrt(
                (position[0] - obs.position[0])**2 +
                (position[1] - obs.position[1])**2
            )
            if dist < range_m + obs.get_radius():
                result.append(obs)
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state of all obstacles."""
        return {
            'obstacles': [
                {
                    'id': obs.id,
                    'type': obs.type.value,
                    'position': obs.position,
                    'size': obs.size,
                    'velocity': obs.velocity
                }
                for obs in self.obstacles
            ]
        }


if __name__ == "__main__":
    # Test obstacle manager
    print("Testing Obstacle Manager...")
    
    from scenes import EmptyScene
    
    scene = EmptyScene()
    scene.reset()
    
    manager = ObstacleManager(num_dynamic=3, num_static=2)
    manager.set_scene(scene)
    manager.reset(
        robot_start=(-1.5, 0),
        robot_goal=(1.5, 0)
    )
    
    print(f"\nCreated {len(manager.obstacles)} obstacles:")
    for obs in manager.obstacles:
        print(f"  {obs.type.value}: pos={obs.position}, size={obs.size}")
    
    # Test update
    for _ in range(10):
        manager.update(0.1)
    
    print(f"\nAfter 1s of updates:")
    for obs in manager.obstacles:
        print(f"  {obs.type.value}: pos=({obs.position[0]:.2f}, {obs.position[1]:.2f})")
    
    # Test collision
    collision, obs = manager.check_collision((0, 0), 0.2)
    print(f"\nCollision at (0,0): {collision}")
    
    dist, closest = manager.get_closest_obstacle((0, 0))
    print(f"Closest obstacle: dist={dist:.2f}")
    
    print("\n✓ Obstacle manager tests passed!")

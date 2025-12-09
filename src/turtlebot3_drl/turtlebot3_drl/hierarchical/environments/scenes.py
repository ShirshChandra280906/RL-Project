"""
Scene Definitions for Hierarchical Navigation

Provides various scene types for training and testing the hierarchical
navigation system.

Scenes define:
- Environment boundaries
- Static obstacles (walls)
- Start/goal positions
- Occupancy grid for A* planning
"""

import math
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import random


class SceneType(Enum):
    """Available scene types."""
    EMPTY = "empty"
    CORRIDOR = "corridor"
    ROOM = "room"
    MAZE_SIMPLE = "maze_simple"
    MAZE_COMPLEX = "maze_complex"
    RANDOM_OBSTACLES = "random_obstacles"


@dataclass
class SceneInfo:
    """Scene information container."""
    start: Tuple[float, float]
    start_theta: float
    goal: Tuple[float, float]
    grid: np.ndarray
    origin: Tuple[float, float]
    resolution: float
    bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


class BaseScene:
    """Base class for all scenes."""
    
    def __init__(self, resolution: float = 0.1):
        """
        Initialize scene.
        
        Args:
            resolution: Grid resolution in meters
        """
        self.resolution = resolution
        self.grid: Optional[np.ndarray] = None
        self.origin: Tuple[float, float] = (0, 0)
        self.bounds: Tuple[float, float, float, float] = (-2, -2, 2, 2)
        
        self.start: Tuple[float, float] = (0, 0)
        self.start_theta: float = 0.0
        self.goal: Tuple[float, float] = (0, 0)
        
        self.static_obstacles: List[Dict[str, Any]] = []
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset scene (regenerate if needed).
        
        Returns:
            Scene information dictionary
        """
        self._generate()
        
        return {
            'start': self.start,
            'start_theta': self.start_theta,
            'goal': self.goal,
            'grid': self.grid,
            'origin': self.origin,
            'resolution': self.resolution,
            'bounds': self.bounds
        }
    
    def _generate(self) -> None:
        """Generate the scene. Override in subclasses."""
        raise NotImplementedError
    
    def _create_grid(self, width: float, height: float) -> np.ndarray:
        """
        Create empty occupancy grid.
        
        Args:
            width: Grid width in meters
            height: Grid height in meters
            
        Returns:
            Empty occupancy grid (0 = free, 1 = occupied)
        """
        cols = int(width / self.resolution)
        rows = int(height / self.resolution)
        return np.zeros((rows, cols), dtype=np.uint8)
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return row, col
    
    def _grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        x = col * self.resolution + self.origin[0]
        y = row * self.resolution + self.origin[1]
        return x, y
    
    def _add_wall(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        thickness: float = 0.1
    ) -> None:
        """Add a wall to the grid."""
        if self.grid is None:
            return
        
        # Discretize wall
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        steps = max(2, int(length / (self.resolution / 2)))
        
        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Fill cells around the wall point
            for dx in np.arange(-thickness, thickness + self.resolution, self.resolution):
                for dy in np.arange(-thickness, thickness + self.resolution, self.resolution):
                    row, col = self._world_to_grid(x + dx, y + dy)
                    if 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]:
                        self.grid[row, col] = 1
    
    def _add_rectangle(
        self,
        center_x: float, center_y: float,
        width: float, height: float,
        filled: bool = True
    ) -> None:
        """Add a rectangular obstacle."""
        if self.grid is None:
            return
        
        half_w = width / 2
        half_h = height / 2
        
        if filled:
            for x in np.arange(center_x - half_w, center_x + half_w, self.resolution):
                for y in np.arange(center_y - half_h, center_y + half_h, self.resolution):
                    row, col = self._world_to_grid(x, y)
                    if 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]:
                        self.grid[row, col] = 1
        else:
            # Just walls
            self._add_wall(center_x - half_w, center_y - half_h,
                          center_x + half_w, center_y - half_h)
            self._add_wall(center_x + half_w, center_y - half_h,
                          center_x + half_w, center_y + half_h)
            self._add_wall(center_x + half_w, center_y + half_h,
                          center_x - half_w, center_y + half_h)
            self._add_wall(center_x - half_w, center_y + half_h,
                          center_x - half_w, center_y - half_h)
        
        self.static_obstacles.append({
            'type': 'rectangle',
            'center': (center_x, center_y),
            'size': (width, height)
        })
    
    def is_free(self, x: float, y: float, margin: float = 0.0) -> bool:
        """
        Check if a position is free (not occupied).
        
        Args:
            x, y: World coordinates
            margin: Additional safety margin
            
        Returns:
            True if free, False if occupied
        """
        if self.grid is None:
            return True
        
        # Check bounds
        if x < self.bounds[0] + margin or x > self.bounds[2] - margin:
            return False
        if y < self.bounds[1] + margin or y > self.bounds[3] - margin:
            return False
        
        # Check grid with margin
        for dx in np.arange(-margin, margin + self.resolution, self.resolution):
            for dy in np.arange(-margin, margin + self.resolution, self.resolution):
                row, col = self._world_to_grid(x + dx, y + dy)
                if 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]:
                    if self.grid[row, col] == 1:
                        return False
        
        return True


class EmptyScene(BaseScene):
    """Empty scene with just boundary walls."""
    
    def __init__(self, resolution: float = 0.1, size: float = 4.0):
        super().__init__(resolution)
        self.size = size
    
    def _generate(self) -> None:
        half = self.size / 2
        self.bounds = (-half, -half, half, half)
        self.origin = (-half, -half)
        
        self.grid = self._create_grid(self.size, self.size)
        
        # Add boundary walls
        wall_thick = 0.05
        self._add_wall(-half, -half, half, -half, wall_thick)  # Bottom
        self._add_wall(half, -half, half, half, wall_thick)    # Right
        self._add_wall(half, half, -half, half, wall_thick)    # Top
        self._add_wall(-half, half, -half, -half, wall_thick)  # Left
        
        # Random start and goal
        self.start = (
            random.uniform(-half + 0.5, -half + 1.0),
            random.uniform(-half + 0.5, half - 0.5)
        )
        self.start_theta = random.uniform(-math.pi, math.pi)
        
        self.goal = (
            random.uniform(half - 1.0, half - 0.5),
            random.uniform(-half + 0.5, half - 0.5)
        )


class CorridorScene(BaseScene):
    """Corridor-like scene."""
    
    def __init__(self, resolution: float = 0.1):
        super().__init__(resolution)
    
    def _generate(self) -> None:
        width = 6.0
        height = 3.0
        
        self.bounds = (-width/2, -height/2, width/2, height/2)
        self.origin = (-width/2, -height/2)
        
        self.grid = self._create_grid(width, height)
        
        # Boundary walls
        wall_thick = 0.05
        self._add_wall(-width/2, -height/2, width/2, -height/2, wall_thick)
        self._add_wall(width/2, -height/2, width/2, height/2, wall_thick)
        self._add_wall(width/2, height/2, -width/2, height/2, wall_thick)
        self._add_wall(-width/2, height/2, -width/2, -height/2, wall_thick)
        
        # Add some obstacles in corridor
        self._add_rectangle(-1.0, 0.0, 0.5, 0.5)
        self._add_rectangle(1.5, 0.3, 0.4, 0.6)
        
        # Start on left, goal on right
        self.start = (-width/2 + 0.5, 0.0)
        self.start_theta = 0.0
        self.goal = (width/2 - 0.5, 0.0)


class RoomScene(BaseScene):
    """Room with doorway."""
    
    def __init__(self, resolution: float = 0.1):
        super().__init__(resolution)
    
    def _generate(self) -> None:
        size = 5.0
        half = size / 2
        
        self.bounds = (-half, -half, half, half)
        self.origin = (-half, -half)
        
        self.grid = self._create_grid(size, size)
        
        # Outer walls
        wall_thick = 0.05
        self._add_wall(-half, -half, half, -half, wall_thick)
        self._add_wall(half, -half, half, half, wall_thick)
        self._add_wall(half, half, -half, half, wall_thick)
        self._add_wall(-half, half, -half, -half, wall_thick)
        
        # Center dividing wall with doorway
        door_y = random.uniform(-0.5, 0.5)
        door_width = 0.8
        
        self._add_wall(0, -half, 0, door_y - door_width/2, wall_thick)
        self._add_wall(0, door_y + door_width/2, 0, half, wall_thick)
        
        # Start in left room, goal in right room
        self.start = (-half + 0.5, random.uniform(-half + 0.5, half - 0.5))
        self.start_theta = 0.0
        self.goal = (half - 0.5, random.uniform(-half + 0.5, half - 0.5))


class SimpleMazeScene(BaseScene):
    """Simple maze with a few walls."""
    
    def __init__(self, resolution: float = 0.1):
        super().__init__(resolution)
    
    def _generate(self) -> None:
        size = 5.0
        half = size / 2
        
        self.bounds = (-half, -half, half, half)
        self.origin = (-half, -half)
        
        self.grid = self._create_grid(size, size)
        
        # Outer walls
        wall_thick = 0.05
        self._add_wall(-half, -half, half, -half, wall_thick)
        self._add_wall(half, -half, half, half, wall_thick)
        self._add_wall(half, half, -half, half, wall_thick)
        self._add_wall(-half, half, -half, -half, wall_thick)
        
        # Internal maze walls
        self._add_wall(-1.5, -half, -1.5, 0.5, wall_thick)
        self._add_wall(0.0, -0.5, 0.0, half, wall_thick)
        self._add_wall(1.5, -half, 1.5, 1.0, wall_thick)
        
        # Start bottom-left, goal top-right
        self.start = (-half + 0.5, -half + 0.5)
        self.start_theta = math.pi / 4
        self.goal = (half - 0.5, half - 0.5)


class RandomObstacleScene(BaseScene):
    """Scene with randomly placed obstacles."""
    
    def __init__(
        self,
        resolution: float = 0.1,
        num_obstacles: int = 5,
        size: float = 5.0
    ):
        super().__init__(resolution)
        self.num_obstacles = num_obstacles
        self.size = size
    
    def _generate(self) -> None:
        half = self.size / 2
        
        self.bounds = (-half, -half, half, half)
        self.origin = (-half, -half)
        
        self.grid = self._create_grid(self.size, self.size)
        self.static_obstacles = []
        
        # Outer walls
        wall_thick = 0.05
        self._add_wall(-half, -half, half, -half, wall_thick)
        self._add_wall(half, -half, half, half, wall_thick)
        self._add_wall(half, half, -half, half, wall_thick)
        self._add_wall(-half, half, -half, -half, wall_thick)
        
        # Random start and goal
        self.start = (-half + 0.5, random.uniform(-half + 0.5, half - 0.5))
        self.start_theta = 0.0
        self.goal = (half - 0.5, random.uniform(-half + 0.5, half - 0.5))
        
        # Add random obstacles (avoiding start and goal)
        min_dist_from_endpoints = 1.0
        
        for _ in range(self.num_obstacles):
            for attempt in range(20):  # Max attempts
                cx = random.uniform(-half + 0.5, half - 0.5)
                cy = random.uniform(-half + 0.5, half - 0.5)
                
                # Check distance from start and goal
                dist_start = math.sqrt((cx - self.start[0])**2 + (cy - self.start[1])**2)
                dist_goal = math.sqrt((cx - self.goal[0])**2 + (cy - self.goal[1])**2)
                
                if dist_start > min_dist_from_endpoints and dist_goal > min_dist_from_endpoints:
                    # Add obstacle
                    w = random.uniform(0.3, 0.6)
                    h = random.uniform(0.3, 0.6)
                    self._add_rectangle(cx, cy, w, h)
                    break


class SceneFactory:
    """Factory for creating scenes."""
    
    _scene_classes = {
        SceneType.EMPTY: EmptyScene,
        SceneType.CORRIDOR: CorridorScene,
        SceneType.ROOM: RoomScene,
        SceneType.MAZE_SIMPLE: SimpleMazeScene,
        SceneType.RANDOM_OBSTACLES: RandomObstacleScene,
    }
    
    @classmethod
    def create(cls, scene_type: SceneType, resolution: float = 0.1) -> BaseScene:
        """Create a scene of the given type."""
        if scene_type not in cls._scene_classes:
            raise ValueError(f"Unknown scene type: {scene_type}")
        return cls._scene_classes[scene_type](resolution)
    
    @classmethod
    def create_random(cls, resolution: float = 0.1) -> BaseScene:
        """Create a random scene type."""
        scene_type = random.choice(list(cls._scene_classes.keys()))
        return cls.create(scene_type, resolution)
    
    @classmethod
    def available_types(cls) -> List[SceneType]:
        """Get list of available scene types."""
        return list(cls._scene_classes.keys())


if __name__ == "__main__":
    # Test scenes
    print("Testing Scene Generation...")
    
    for scene_type in SceneFactory.available_types():
        scene = SceneFactory.create(scene_type)
        info = scene.reset()
        
        print(f"\n{scene_type.value}:")
        print(f"  Start: {info['start']}")
        print(f"  Goal: {info['goal']}")
        print(f"  Grid shape: {info['grid'].shape}")
        print(f"  Bounds: {info['bounds']}")
        
        # Test is_free
        assert scene.is_free(info['start'][0], info['start'][1], margin=0.1), "Start should be free"
        assert scene.is_free(info['goal'][0], info['goal'][1], margin=0.1), "Goal should be free"
    
    print("\n✓ All scene tests passed!")

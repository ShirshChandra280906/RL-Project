"""
Scene Classes for Hierarchical DRL Navigation

Implements three scene types based on the paper:
"Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Scene Types:
1. Corridor: Width [1.8m, 3m], Length [10m, 14m]
2. Intersection: Hallway width [1.8m, 2.5m], Length [4m, 6m]
3. Office: Fixed 7x7m outer walls, randomized inner walls

Key Features:
- Wall placement randomization for generalization
- Start/goal position sampling per scene type
- Occupancy grid generation for A* planning
"""

import math
import random
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum


class SceneType(Enum):
    """Enum for scene types."""
    CORRIDOR = "corridor"
    INTERSECTION = "intersection"
    OFFICE = "office"


class BaseScene(ABC):
    """
    Abstract base class for training scenes.
    
    Defines the interface for scene generation, occupancy grid creation,
    and start/goal position sampling.
    """
    
    def __init__(self, grid_resolution: float = 0.1):
        """
        Initialize base scene.
        
        Args:
            grid_resolution: Size of each grid cell in meters
        """
        self.resolution = grid_resolution
        
        # Scene bounds (set by subclasses)
        self.width: float = 0.0
        self.length: float = 0.0
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        
        # Occupancy grid (0=free, 1=occupied)
        self.grid: Optional[np.ndarray] = None
        self.grid_width: int = 0
        self.grid_height: int = 0
        
        # Wall segments for rendering/collision (list of (x1, y1, x2, y2))
        self.walls: List[Tuple[float, float, float, float]] = []
        
        # Start and goal positions
        self.start_pos: Tuple[float, float] = (0.0, 0.0)
        self.goal_pos: Tuple[float, float] = (0.0, 0.0)
        
        # Robot start orientation
        self.start_theta: float = 0.0
    
    @abstractmethod
    def generate(self) -> None:
        """Generate a new random scene layout."""
        pass
    
    @abstractmethod
    def sample_start_goal(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Sample start and goal positions.
        
        Returns:
            (start, goal) tuples of (x, y) positions
        """
        pass
    
    def get_occupancy_grid(self) -> Tuple[np.ndarray, float, float]:
        """
        Get the occupancy grid for A* planning.
        
        Returns:
            (grid, origin_x, origin_y)
        """
        if self.grid is None:
            self.generate()
        return self.grid.copy(), self.origin_x, self.origin_y
    
    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int(round((wx - self.origin_x) / self.resolution))
        gy = int(round((wy - self.origin_y) / self.resolution))
        return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        wx = gx * self.resolution + self.origin_x
        wy = gy * self.resolution + self.origin_y
        return wx, wy
    
    def is_free(self, x: float, y: float, margin: float = 0.2) -> bool:
        """
        Check if a position is free (not in obstacle).
        
        Args:
            x, y: Position in world coordinates
            margin: Safety margin from obstacles
        """
        if self.grid is None:
            return True
        
        gx, gy = self.world_to_grid(x, y)
        
        # Check surrounding cells for margin
        margin_cells = int(margin / self.resolution)
        
        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    if self.grid[ny, nx] == 1:
                        return False
        return True
    
    def _add_wall(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        thickness: float = 0.1
    ) -> None:
        """
        Add a wall segment and mark it in the occupancy grid.
        
        Args:
            x1, y1: Start point
            x2, y2: End point
            thickness: Wall thickness in meters
        """
        self.walls.append((x1, y1, x2, y2))
        
        if self.grid is None:
            return
        
        # Mark wall cells in grid
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        
        if length < 0.001:
            return
        
        # Unit direction vector
        ux, uy = dx / length, dy / length
        # Perpendicular vector for thickness
        px, py = -uy, ux
        
        # Sample points along the wall
        num_samples = int(length / (self.resolution / 2)) + 1
        half_thick = thickness / 2
        
        for i in range(num_samples + 1):
            t = i / num_samples
            wx = x1 + t * dx
            wy = y1 + t * dy
            
            # Mark cells across wall thickness
            for j in range(-int(half_thick / self.resolution) - 1,
                          int(half_thick / self.resolution) + 2):
                mark_x = wx + j * self.resolution * px
                mark_y = wy + j * self.resolution * py
                
                gx, gy = self.world_to_grid(mark_x, mark_y)
                if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                    self.grid[gy, gx] = 1
    
    def _add_rectangle_wall(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        thickness: float = 0.1
    ) -> None:
        """Add a rectangular wall outline."""
        # Four walls
        self._add_wall(x, y, x + width, y, thickness)  # Bottom
        self._add_wall(x + width, y, x + width, y + height, thickness)  # Right
        self._add_wall(x + width, y + height, x, y + height, thickness)  # Top
        self._add_wall(x, y + height, x, y, thickness)  # Left
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset scene with new random layout.
        
        Returns:
            Dictionary with scene info
        """
        self.generate()
        start, goal = self.sample_start_goal()
        self.start_pos = start
        self.goal_pos = goal
        
        return {
            'start': start,
            'goal': goal,
            'start_theta': self.start_theta,
            'grid': self.grid.copy(),
            'origin': (self.origin_x, self.origin_y),
            'bounds': (self.width, self.length)
        }


class CorridorScene(BaseScene):
    """
    Corridor scene with randomized width and length.
    
    Paper specs:
    - Width: [1.8m, 3m]
    - Length: [10m, 14m]
    - Start and goal in different dead-ends
    """
    
    # Corridor dimension ranges from paper
    WIDTH_MIN = 1.8
    WIDTH_MAX = 3.0
    LENGTH_MIN = 10.0
    LENGTH_MAX = 14.0
    
    def __init__(self, grid_resolution: float = 0.1):
        super().__init__(grid_resolution)
        self.scene_type = SceneType.CORRIDOR
    
    def generate(self) -> None:
        """Generate a random corridor layout."""
        # Sample random dimensions
        self.width = random.uniform(self.WIDTH_MIN, self.WIDTH_MAX)
        self.length = random.uniform(self.LENGTH_MIN, self.LENGTH_MAX)
        
        # Set origin at bottom-left corner
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        # Create occupancy grid
        self.grid_width = int(self.length / self.resolution) + 1
        self.grid_height = int(self.width / self.resolution) + 1
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Clear walls list
        self.walls = []
        
        # Add corridor walls (long walls on sides)
        wall_thickness = 0.15
        
        # Bottom wall
        self._add_wall(0, 0, self.length, 0, wall_thickness)
        # Top wall
        self._add_wall(0, self.width, self.length, self.width, wall_thickness)
        # Left end wall
        self._add_wall(0, 0, 0, self.width, wall_thickness)
        # Right end wall
        self._add_wall(self.length, 0, self.length, self.width, wall_thickness)
    
    def sample_start_goal(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Sample start and goal at opposite ends of corridor.
        
        Returns:
            (start, goal) positions
        """
        margin = 0.5
        corridor_center_y = self.width / 2
        
        # Random slight offset from center
        y_offset = random.uniform(-0.3, 0.3)
        
        if random.random() < 0.5:
            # Start at left, goal at right
            start = (margin, corridor_center_y + y_offset)
            goal = (self.length - margin, corridor_center_y + random.uniform(-0.3, 0.3))
            self.start_theta = 0.0  # Facing right
        else:
            # Start at right, goal at left
            start = (self.length - margin, corridor_center_y + y_offset)
            goal = (margin, corridor_center_y + random.uniform(-0.3, 0.3))
            self.start_theta = math.pi  # Facing left
        
        return start, goal


class IntersectionScene(BaseScene):
    """
    Intersection (cross) scene with randomized hallway dimensions.
    
    Paper specs:
    - Hallway width: [1.8m, 2.5m]
    - Hallway length: [4m, 6m]
    - Start and goal in different dead-ends
    """
    
    # Intersection dimension ranges from paper
    HALLWAY_WIDTH_MIN = 1.8
    HALLWAY_WIDTH_MAX = 2.5
    HALLWAY_LENGTH_MIN = 4.0
    HALLWAY_LENGTH_MAX = 6.0
    
    def __init__(self, grid_resolution: float = 0.1):
        super().__init__(grid_resolution)
        self.scene_type = SceneType.INTERSECTION
        self.hallway_width: float = 2.0
        self.hallway_length: float = 5.0
    
    def generate(self) -> None:
        """Generate a random intersection layout."""
        # Sample random dimensions
        self.hallway_width = random.uniform(self.HALLWAY_WIDTH_MIN, self.HALLWAY_WIDTH_MAX)
        self.hallway_length = random.uniform(self.HALLWAY_LENGTH_MIN, self.HALLWAY_LENGTH_MAX)
        
        # Total scene size
        self.width = 2 * self.hallway_length + self.hallway_width
        self.length = self.width  # Square scene
        
        # Set origin at bottom-left corner
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        # Create occupancy grid
        self.grid_width = int(self.length / self.resolution) + 1
        self.grid_height = int(self.width / self.resolution) + 1
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Clear walls list
        self.walls = []
        
        # Calculate intersection geometry
        hw = self.hallway_width
        hl = self.hallway_length
        center = self.width / 2
        
        wall_thickness = 0.15
        
        # Create the cross-shaped intersection
        # Four corner blocks (obstacles outside the cross)
        
        # Bottom-left corner block
        self._fill_rectangle(0, 0, hl, hl)
        # Bottom-right corner block
        self._fill_rectangle(hl + hw, 0, hl, hl)
        # Top-left corner block
        self._fill_rectangle(0, hl + hw, hl, hl)
        # Top-right corner block
        self._fill_rectangle(hl + hw, hl + hw, hl, hl)
        
        # Add outer boundary walls
        self._add_rectangle_wall(0, 0, self.length, self.width, wall_thickness)
    
    def _fill_rectangle(self, x: float, y: float, width: float, height: float) -> None:
        """Fill a rectangular area as obstacle in the grid."""
        gx1, gy1 = self.world_to_grid(x, y)
        gx2, gy2 = self.world_to_grid(x + width, y + height)
        
        for gx in range(max(0, gx1), min(self.grid_width, gx2 + 1)):
            for gy in range(max(0, gy1), min(self.grid_height, gy2 + 1)):
                self.grid[gy, gx] = 1
    
    def sample_start_goal(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Sample start and goal in different dead-ends.
        
        Dead-ends: bottom, top, left, right of the cross
        """
        hw = self.hallway_width
        hl = self.hallway_length
        center = self.width / 2
        
        margin = 0.5
        
        # Define dead-end positions (center of each arm)
        dead_ends = [
            # (position, orientation when starting here)
            ((center, margin), math.pi / 2),  # Bottom arm, facing up
            ((center, self.width - margin), -math.pi / 2),  # Top arm, facing down
            ((margin, center), 0.0),  # Left arm, facing right
            ((self.length - margin, center), math.pi),  # Right arm, facing left
        ]
        
        # Sample two different dead-ends
        start_idx = random.randint(0, 3)
        goal_idx = random.randint(0, 3)
        while goal_idx == start_idx:
            goal_idx = random.randint(0, 3)
        
        start_pos, self.start_theta = dead_ends[start_idx]
        goal_pos, _ = dead_ends[goal_idx]
        
        # Add small random offset
        start = (
            start_pos[0] + random.uniform(-0.2, 0.2),
            start_pos[1] + random.uniform(-0.2, 0.2)
        )
        goal = (
            goal_pos[0] + random.uniform(-0.2, 0.2),
            goal_pos[1] + random.uniform(-0.2, 0.2)
        )
        
        return start, goal


class OfficeScene(BaseScene):
    """
    Office scene with fixed outer walls and randomized inner walls.
    
    Paper specs:
    - Fixed outer: 7m × 7m
    - Inner walls create different room configurations
    - Start and goal in opposing corners
    """
    
    # Office dimensions from paper
    OUTER_SIZE = 7.0
    
    def __init__(self, grid_resolution: float = 0.1):
        super().__init__(grid_resolution)
        self.scene_type = SceneType.OFFICE
        self.inner_walls: List[Tuple[float, float, float, float]] = []
    
    def generate(self) -> None:
        """Generate a random office layout with inner walls."""
        self.width = self.OUTER_SIZE
        self.length = self.OUTER_SIZE
        
        # Set origin at bottom-left corner
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        # Create occupancy grid
        self.grid_width = int(self.length / self.resolution) + 1
        self.grid_height = int(self.width / self.resolution) + 1
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Clear walls list
        self.walls = []
        self.inner_walls = []
        
        wall_thickness = 0.15
        
        # Add outer boundary walls
        self._add_rectangle_wall(0, 0, self.length, self.width, wall_thickness)
        
        # Generate random inner walls
        self._generate_inner_walls(wall_thickness)
    
    def _generate_inner_walls(self, wall_thickness: float) -> None:
        """Generate random inner wall configurations."""
        # Different room configurations
        config = random.randint(0, 4)
        
        margin = 0.8  # Keep walls away from corners
        door_width = 1.0  # Width of doorways
        
        if config == 0:
            # Simple horizontal divider with door
            y = self.width / 2
            door_x = random.uniform(1.5, self.length - 1.5)
            self._add_wall(margin, y, door_x - door_width/2, y, wall_thickness)
            self._add_wall(door_x + door_width/2, y, self.length - margin, y, wall_thickness)
            
        elif config == 1:
            # Simple vertical divider with door
            x = self.length / 2
            door_y = random.uniform(1.5, self.width - 1.5)
            self._add_wall(x, margin, x, door_y - door_width/2, wall_thickness)
            self._add_wall(x, door_y + door_width/2, x, self.width - margin, wall_thickness)
            
        elif config == 2:
            # Four rooms (cross pattern with doors)
            cx, cy = self.length / 2, self.width / 2
            
            # Horizontal walls
            self._add_wall(margin, cy, cx - door_width/2, cy, wall_thickness)
            self._add_wall(cx + door_width/2, cy, self.length - margin, cy, wall_thickness)
            
            # Vertical walls
            self._add_wall(cx, margin, cx, cy - door_width/2, wall_thickness)
            self._add_wall(cx, cy + door_width/2, cx, self.width - margin, wall_thickness)
            
        elif config == 3:
            # L-shaped room
            wall_x = random.uniform(2.5, 4.5)
            wall_y = random.uniform(2.5, 4.5)
            
            # Horizontal part of L
            self._add_wall(wall_x, wall_y, self.length - margin, wall_y, wall_thickness)
            # Vertical part of L
            self._add_wall(wall_x, margin, wall_x, wall_y, wall_thickness)
            
        elif config == 4:
            # Multiple small obstacles/partitions
            num_partitions = random.randint(2, 4)
            
            for _ in range(num_partitions):
                # Random partition
                x1 = random.uniform(1.5, self.length - 1.5)
                y1 = random.uniform(1.5, self.width - 1.5)
                
                if random.random() < 0.5:
                    # Horizontal partition
                    length = random.uniform(1.0, 2.0)
                    x2 = min(x1 + length, self.length - margin)
                    self._add_wall(x1, y1, x2, y1, wall_thickness)
                else:
                    # Vertical partition
                    length = random.uniform(1.0, 2.0)
                    y2 = min(y1 + length, self.width - margin)
                    self._add_wall(x1, y1, x1, y2, wall_thickness)
    
    def sample_start_goal(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Sample start and goal in opposing corners.
        
        Paper: "start and goal pose are sampled in opposing corners of the outer walls"
        """
        margin = 0.6
        
        # Corner positions
        corners = [
            (margin, margin),  # Bottom-left
            (margin, self.width - margin),  # Top-left
            (self.length - margin, margin),  # Bottom-right
            (self.length - margin, self.width - margin),  # Top-right
        ]
        
        # Opposing corner pairs
        opposing_pairs = [(0, 3), (1, 2), (3, 0), (2, 1)]
        
        pair_idx = random.randint(0, 3)
        start_corner, goal_corner = opposing_pairs[pair_idx]
        
        start = corners[start_corner]
        goal = corners[goal_corner]
        
        # Calculate start orientation (facing toward goal)
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        self.start_theta = math.atan2(dy, dx)
        
        # Add small random offset (ensure still in free space)
        for _ in range(10):  # Max attempts
            start_offset = (
                start[0] + random.uniform(-0.3, 0.3),
                start[1] + random.uniform(-0.3, 0.3)
            )
            if self.is_free(start_offset[0], start_offset[1]):
                start = start_offset
                break
        
        for _ in range(10):
            goal_offset = (
                goal[0] + random.uniform(-0.3, 0.3),
                goal[1] + random.uniform(-0.3, 0.3)
            )
            if self.is_free(goal_offset[0], goal_offset[1]):
                goal = goal_offset
                break
        
        return start, goal


class SceneFactory:
    """Factory for creating scenes."""
    
    @staticmethod
    def create(scene_type: SceneType, grid_resolution: float = 0.1) -> BaseScene:
        """
        Create a scene of the specified type.
        
        Args:
            scene_type: Type of scene to create
            grid_resolution: Grid resolution in meters
            
        Returns:
            Scene instance
        """
        if scene_type == SceneType.CORRIDOR:
            return CorridorScene(grid_resolution)
        elif scene_type == SceneType.INTERSECTION:
            return IntersectionScene(grid_resolution)
        elif scene_type == SceneType.OFFICE:
            return OfficeScene(grid_resolution)
        else:
            raise ValueError(f"Unknown scene type: {scene_type}")
    
    @staticmethod
    def create_random(grid_resolution: float = 0.1) -> BaseScene:
        """Create a random scene type."""
        scene_type = random.choice(list(SceneType))
        return SceneFactory.create(scene_type, grid_resolution)


if __name__ == "__main__":
    # Test scene generation
    print("Testing Scene Generation...")
    
    for scene_type in SceneType:
        print(f"\n{'='*50}")
        print(f"Testing {scene_type.value} scene")
        print('='*50)
        
        scene = SceneFactory.create(scene_type)
        info = scene.reset()
        
        print(f"Scene dimensions: {scene.width:.2f} x {scene.length:.2f} m")
        print(f"Grid size: {scene.grid_width} x {scene.grid_height}")
        print(f"Start: {info['start']}")
        print(f"Goal: {info['goal']}")
        print(f"Start theta: {math.degrees(info['start_theta']):.1f}°")
        
        # Check occupancy
        free_cells = np.sum(scene.grid == 0)
        occupied_cells = np.sum(scene.grid == 1)
        print(f"Free cells: {free_cells}, Occupied: {occupied_cells}")
    
    print("\n✓ All scene types generated successfully!")

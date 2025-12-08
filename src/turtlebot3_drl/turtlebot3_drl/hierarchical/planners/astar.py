"""
A* Path Planner for Hierarchical Navigation

Implements the A* algorithm for global path planning using known static obstacles.
Based on the paper: "Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Key features:
- Uses only known static obstacles (ignores dynamic obstacles)
- Replans every 3 subgoal predictions
- Provides waypoints for the Subgoal Agent
"""

import heapq
import math
import numpy as np
from typing import List, Tuple, Optional, Set


class Node:
    """A* search node."""
    
    def __init__(self, x: int, y: int, g: float = 0, h: float = 0, parent: 'Node' = None):
        self.x = x
        self.y = y
        self.g = g  # Cost from start
        self.h = h  # Heuristic (estimated cost to goal)
        self.f = g + h  # Total cost
        self.parent = parent
    
    def __lt__(self, other: 'Node') -> bool:
        return self.f < other.f
    
    def __eq__(self, other: 'Node') -> bool:
        return self.x == other.x and self.y == other.y
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))


class AStarPlanner:
    """
    A* Global Path Planner.
    
    Plans paths on a 2D occupancy grid using only known static obstacles.
    The grid uses binary values: 0 = free, 1 = occupied.
    """
    
    def __init__(
        self,
        grid_resolution: float = 0.1,
        robot_radius: float = 0.2,
        inflation_radius: float = 0.1,
        diagonal_movement: bool = True
    ):
        """
        Initialize A* planner.
        
        Args:
            grid_resolution: Size of each grid cell in meters
            robot_radius: Robot radius for collision checking
            inflation_radius: Additional safety margin
            diagonal_movement: Allow 8-connected movement (vs 4-connected)
        """
        self.resolution = grid_resolution
        self.robot_radius = robot_radius
        self.inflation_radius = inflation_radius
        self.diagonal = diagonal_movement
        
        # Occupancy grid (set via set_occupancy_grid)
        self.grid: Optional[np.ndarray] = None
        self.grid_width: int = 0
        self.grid_height: int = 0
        
        # World bounds (for coordinate conversion)
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        
        # Movement directions
        if diagonal_movement:
            # 8-connected: cardinal + diagonal
            self.dx = [1, 0, -1, 0, 1, 1, -1, -1]
            self.dy = [0, 1, 0, -1, 1, -1, 1, -1]
            self.costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        else:
            # 4-connected: cardinal only
            self.dx = [1, 0, -1, 0]
            self.dy = [0, 1, 0, -1]
            self.costs = [1.0, 1.0, 1.0, 1.0]
    
    def set_occupancy_grid(
        self,
        grid: np.ndarray,
        origin_x: float = 0.0,
        origin_y: float = 0.0
    ) -> None:
        """
        Set the occupancy grid for planning.
        
        Args:
            grid: 2D numpy array (0=free, 1=occupied)
            origin_x: X coordinate of grid origin in world frame
            origin_y: Y coordinate of grid origin in world frame
        """
        self.grid = grid.copy()
        self.grid_height, self.grid_width = grid.shape
        self.origin_x = origin_x
        self.origin_y = origin_y
        
        # Inflate obstacles by robot radius + safety margin
        self._inflate_obstacles()
    
    def _inflate_obstacles(self) -> None:
        """Inflate obstacles by robot radius for collision-free planning."""
        if self.grid is None:
            return
        
        inflate_cells = int(math.ceil((self.robot_radius + self.inflation_radius) / self.resolution))
        
        if inflate_cells <= 0:
            return
        
        # Create inflated grid
        inflated = self.grid.copy()
        
        # Find all obstacle cells
        obstacle_coords = np.argwhere(self.grid == 1)
        
        # Inflate each obstacle
        for (y, x) in obstacle_coords:
            for dy in range(-inflate_cells, inflate_cells + 1):
                for dx in range(-inflate_cells, inflate_cells + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        # Check if within inflation radius (circular)
                        dist = math.sqrt(dx**2 + dy**2) * self.resolution
                        if dist <= self.robot_radius + self.inflation_radius:
                            inflated[ny, nx] = 1
        
        self.grid = inflated
    
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
    
    def is_valid(self, gx: int, gy: int) -> bool:
        """Check if grid cell is valid and free."""
        if gx < 0 or gx >= self.grid_width:
            return False
        if gy < 0 or gy >= self.grid_height:
            return False
        if self.grid is None:
            return False
        return self.grid[gy, gx] == 0
    
    def heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate heuristic (Euclidean distance)."""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal.
        
        Args:
            start: Start position (x, y) in world coordinates
            goal: Goal position (x, y) in world coordinates
            
        Returns:
            List of waypoints [(x, y), ...] in world coordinates, or None if no path found
        """
        if self.grid is None:
            raise ValueError("Occupancy grid not set. Call set_occupancy_grid first.")
        
        # Convert to grid coordinates
        start_gx, start_gy = self.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.world_to_grid(goal[0], goal[1])
        
        # Check validity
        if not self.is_valid(start_gx, start_gy):
            print(f"Warning: Start position {start} is invalid or in obstacle")
            # Try to find nearest valid cell
            start_gx, start_gy = self._find_nearest_free(start_gx, start_gy)
            if start_gx is None:
                return None
        
        if not self.is_valid(goal_gx, goal_gy):
            print(f"Warning: Goal position {goal} is invalid or in obstacle")
            goal_gx, goal_gy = self._find_nearest_free(goal_gx, goal_gy)
            if goal_gx is None:
                return None
        
        # Initialize A*
        start_node = Node(start_gx, start_gy, 0, self.heuristic(start_gx, start_gy, goal_gx, goal_gy))
        
        open_set: List[Node] = []
        heapq.heappush(open_set, start_node)
        
        closed_set: Set[Tuple[int, int]] = set()
        g_scores = {(start_gx, start_gy): 0}
        
        while open_set:
            current = heapq.heappop(open_set)
            
            # Goal reached
            if current.x == goal_gx and current.y == goal_gy:
                return self._reconstruct_path(current)
            
            if (current.x, current.y) in closed_set:
                continue
            
            closed_set.add((current.x, current.y))
            
            # Explore neighbors
            for i in range(len(self.dx)):
                nx = current.x + self.dx[i]
                ny = current.y + self.dy[i]
                
                if not self.is_valid(nx, ny):
                    continue
                
                if (nx, ny) in closed_set:
                    continue
                
                tentative_g = current.g + self.costs[i]
                
                if (nx, ny) not in g_scores or tentative_g < g_scores[(nx, ny)]:
                    g_scores[(nx, ny)] = tentative_g
                    h = self.heuristic(nx, ny, goal_gx, goal_gy)
                    neighbor = Node(nx, ny, tentative_g, h, current)
                    heapq.heappush(open_set, neighbor)
        
        # No path found
        return None
    
    def _find_nearest_free(self, gx: int, gy: int, max_radius: int = 10) -> Tuple[Optional[int], Optional[int]]:
        """Find nearest free cell to given position."""
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:  # Only check boundary
                        nx, ny = gx + dx, gy + dy
                        if self.is_valid(nx, ny):
                            return nx, ny
        return None, None
    
    def _reconstruct_path(self, goal_node: Node) -> List[Tuple[float, float]]:
        """Reconstruct path from goal node back to start."""
        path = []
        current = goal_node
        
        while current is not None:
            wx, wy = self.grid_to_world(current.x, current.y)
            path.append((wx, wy))
            current = current.parent
        
        # Reverse to get start-to-goal order
        path.reverse()
        return path
    
    def smooth_path(
        self,
        path: List[Tuple[float, float]],
        weight_data: float = 0.5,
        weight_smooth: float = 0.1,
        tolerance: float = 0.001
    ) -> List[Tuple[float, float]]:
        """
        Apply path smoothing using gradient descent.
        
        Args:
            path: Original path
            weight_data: Weight for staying close to original path
            weight_smooth: Weight for smoothness
            tolerance: Convergence tolerance
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        # Convert to numpy for easier manipulation
        new_path = [list(p) for p in path]
        
        change = tolerance + 1
        while change > tolerance:
            change = 0
            for i in range(1, len(path) - 1):
                for j in range(2):
                    old_val = new_path[i][j]
                    
                    # Gradient descent update
                    new_path[i][j] += weight_data * (path[i][j] - new_path[i][j])
                    new_path[i][j] += weight_smooth * (
                        new_path[i-1][j] + new_path[i+1][j] - 2 * new_path[i][j]
                    )
                    
                    change += abs(old_val - new_path[i][j])
        
        return [(p[0], p[1]) for p in new_path]
    
    def get_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total path length in meters."""
        if not path or len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            length += math.sqrt(dx**2 + dy**2)
        
        return length


def create_test_grid(width: int = 50, height: int = 50) -> np.ndarray:
    """Create a test occupancy grid with obstacles."""
    grid = np.zeros((height, width), dtype=np.int8)
    
    # Add some obstacles
    # Vertical wall with gap
    grid[15:35, 20] = 1
    grid[20:25, 20] = 0  # Gap in wall
    
    # Horizontal wall
    grid[10, 30:45] = 1
    
    # L-shaped obstacle
    grid[35:45, 10:13] = 1
    grid[42:45, 10:20] = 1
    
    # Small blocks
    grid[5:8, 8:11] = 1
    grid[25:28, 40:43] = 1
    
    return grid


def visualize_path_ascii(
    grid: np.ndarray,
    path: List[Tuple[float, float]],
    start: Tuple[float, float],
    goal: Tuple[float, float],
    resolution: float = 0.1
) -> str:
    """Create ASCII visualization of path on grid."""
    height, width = grid.shape
    
    # Create character grid
    vis = []
    for y in range(height):
        row = []
        for x in range(width):
            if grid[y, x] == 1:
                row.append('█')  # Obstacle
            else:
                row.append('·')  # Free space
        vis.append(row)
    
    # Mark path
    for (px, py) in path:
        gx = int(round(px / resolution))
        gy = int(round(py / resolution))
        if 0 <= gx < width and 0 <= gy < height:
            vis[gy][gx] = '*'
    
    # Mark start and goal
    sx, sy = int(round(start[0] / resolution)), int(round(start[1] / resolution))
    gx, gy = int(round(goal[0] / resolution)), int(round(goal[1] / resolution))
    
    if 0 <= sx < width and 0 <= sy < height:
        vis[sy][sx] = 'S'
    if 0 <= gx < width and 0 <= gy < height:
        vis[gy][gx] = 'G'
    
    # Convert to string (flip Y for normal orientation)
    lines = [''.join(row) for row in reversed(vis)]
    return '\n'.join(lines)


if __name__ == "__main__":
    # Test A* planner
    print("=" * 60)
    print("A* Path Planner Test")
    print("=" * 60)
    
    # Create test grid (50x50 at 0.1m resolution = 5m x 5m)
    grid = create_test_grid(50, 50)
    
    # Initialize planner
    planner = AStarPlanner(
        grid_resolution=0.1,
        robot_radius=0.15,
        inflation_radius=0.05
    )
    planner.set_occupancy_grid(grid)
    
    # Test cases
    test_cases = [
        ((0.5, 0.5), (4.5, 4.5), "Diagonal across map"),
        ((0.5, 2.5), (4.0, 2.5), "Through wall gap"),
        ((0.5, 0.5), (0.5, 4.5), "Along left edge"),
    ]
    
    for start, goal, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Start: {start}, Goal: {goal}")
        
        path = planner.plan(start, goal)
        
        if path:
            print(f"✓ Path found with {len(path)} waypoints")
            print(f"  Path length: {planner.get_path_length(path):.2f}m")
            print(f"  First 5 points: {path[:5]}")
            
            # Smooth the path
            smooth = planner.smooth_path(path)
            print(f"  Smoothed length: {planner.get_path_length(smooth):.2f}m")
        else:
            print("✗ No path found")
    
    # Visual test
    print("\n" + "=" * 60)
    print("Visual Test (ASCII Map)")
    print("=" * 60)
    
    start, goal = (0.5, 0.5), (4.5, 4.5)
    path = planner.plan(start, goal)
    
    if path:
        # Show smaller section for readability
        small_grid = grid[:30, :30]
        small_path = [(x, y) for (x, y) in path if x < 3.0 and y < 3.0]
        print(visualize_path_ascii(small_grid, small_path, start, goal, 0.1))
        print("\nLegend: S=Start, G=Goal, *=Path, █=Obstacle, ·=Free")
    
    print("\n✓ A* Planner test complete!")

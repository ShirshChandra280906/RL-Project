"""
Waypoint Manager for Hierarchical Navigation

Manages waypoints extracted from the A* global path for the Subgoal Agent.
Based on the paper: "Lightweight Motion Planning via Hierarchical Reinforcement Learning"

Key features:
- Extracts 5 waypoints at 0.3m spacing (1.2m total coverage)
- Converts waypoints to robot-centric coordinates
- Handles path updates and edge cases
"""

import math
import numpy as np
from typing import List, Tuple, Optional


class WaypointManager:
    """
    Manages waypoints along the A* planned path.
    
    Extracts a fixed number of waypoints at regular intervals from the
    global path and transforms them to robot-centric coordinates.
    """
    
    def __init__(
        self,
        num_waypoints: int = 5,
        waypoint_spacing: float = 0.3,
        lookahead_distance: float = 0.0
    ):
        """
        Initialize waypoint manager.
        
        Args:
            num_waypoints: Number of waypoints to extract (default: 5)
            waypoint_spacing: Distance between consecutive waypoints (default: 0.3m)
            lookahead_distance: How far ahead to start waypoints (default: 0.0m)
        """
        self.num_waypoints = num_waypoints
        self.spacing = waypoint_spacing
        self.lookahead = lookahead_distance
        
        # Current path (list of (x, y) tuples in world frame)
        self.path: List[Tuple[float, float]] = []
        
        # Interpolated path for smoother waypoint extraction
        self.interpolated_path: List[Tuple[float, float]] = []
        self.interpolation_resolution = 0.05  # 5cm resolution
        
        # Path progress tracking
        self.current_path_index: int = 0
        
    def set_path(self, path: List[Tuple[float, float]]) -> None:
        """
        Set a new global path.
        
        Args:
            path: List of (x, y) waypoints in world frame
        """
        self.path = path
        self.current_path_index = 0
        
        # Interpolate path for smoother waypoint extraction
        self.interpolated_path = self._interpolate_path(path)
    
    def _interpolate_path(
        self,
        path: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Interpolate path to have points at regular intervals.
        
        Args:
            path: Original path points
            
        Returns:
            Interpolated path with points at interpolation_resolution spacing
        """
        if not path or len(path) < 2:
            return path.copy() if path else []
        
        interpolated = [path[0]]
        
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            segment_length = math.sqrt(dx**2 + dy**2)
            
            if segment_length < self.interpolation_resolution:
                continue
            
            num_points = int(segment_length / self.interpolation_resolution)
            
            for j in range(1, num_points + 1):
                t = j / num_points
                x = p1[0] + t * dx
                y = p1[1] + t * dy
                interpolated.append((x, y))
        
        # Ensure goal is included
        if interpolated[-1] != path[-1]:
            interpolated.append(path[-1])
        
        return interpolated
    
    def _find_closest_point_index(
        self,
        robot_x: float,
        robot_y: float
    ) -> int:
        """
        Find the index of the closest point on the interpolated path.
        
        Args:
            robot_x: Robot x position in world frame
            robot_y: Robot y position in world frame
            
        Returns:
            Index of closest point
        """
        if not self.interpolated_path:
            return 0
        
        min_dist = float('inf')
        closest_idx = self.current_path_index
        
        # Search from current index forward (with some backward tolerance)
        search_start = max(0, self.current_path_index - 5)
        
        for i in range(search_start, len(self.interpolated_path)):
            px, py = self.interpolated_path[i]
            dist = math.sqrt((px - robot_x)**2 + (py - robot_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx
    
    def _transform_to_robot_frame(
        self,
        world_x: float,
        world_y: float,
        robot_x: float,
        robot_y: float,
        robot_theta: float
    ) -> Tuple[float, float]:
        """
        Transform a point from world frame to robot-centric frame.
        
        Args:
            world_x, world_y: Point in world frame
            robot_x, robot_y: Robot position in world frame
            robot_theta: Robot heading in world frame (radians)
            
        Returns:
            (x, y) in robot frame where +x is forward, +y is left
        """
        # Translate to robot position
        dx = world_x - robot_x
        dy = world_y - robot_y
        
        # Rotate to robot frame
        cos_theta = math.cos(-robot_theta)
        sin_theta = math.sin(-robot_theta)
        
        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta
        
        return local_x, local_y
    
    def get_waypoints(
        self,
        robot_pose: Tuple[float, float, float]
    ) -> List[Tuple[float, float]]:
        """
        Get waypoints in robot-centric coordinates.
        
        Args:
            robot_pose: (x, y, theta) robot pose in world frame
            
        Returns:
            List of (x, y) waypoints in robot frame.
            Returns num_waypoints * 2 values flattened if needed.
        """
        robot_x, robot_y, robot_theta = robot_pose
        
        if not self.interpolated_path:
            # Return zeros if no path
            return [(0.0, 0.0)] * self.num_waypoints
        
        # Find closest point on path
        closest_idx = self._find_closest_point_index(robot_x, robot_y)
        self.current_path_index = closest_idx
        
        # Extract waypoints at regular spacing
        waypoints_world = []
        current_distance = self.lookahead
        path_idx = closest_idx
        
        for _ in range(self.num_waypoints):
            # Move along path until we reach target distance
            target_distance = current_distance + self.spacing
            
            while path_idx < len(self.interpolated_path) - 1:
                p1 = self.interpolated_path[path_idx]
                p2 = self.interpolated_path[path_idx + 1]
                
                segment_length = math.sqrt(
                    (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
                )
                
                if current_distance + segment_length >= target_distance:
                    # Interpolate within this segment
                    t = (target_distance - current_distance) / segment_length
                    wx = p1[0] + t * (p2[0] - p1[0])
                    wy = p1[1] + t * (p2[1] - p1[1])
                    waypoints_world.append((wx, wy))
                    current_distance = target_distance
                    break
                else:
                    current_distance += segment_length
                    path_idx += 1
            else:
                # Reached end of path, use goal
                waypoints_world.append(self.interpolated_path[-1])
                current_distance = target_distance  # Continue with spacing
        
        # Pad with goal if not enough waypoints
        while len(waypoints_world) < self.num_waypoints:
            waypoints_world.append(self.interpolated_path[-1])
        
        # Transform to robot frame
        waypoints_robot = []
        for wx, wy in waypoints_world:
            rx, ry = self._transform_to_robot_frame(
                wx, wy, robot_x, robot_y, robot_theta
            )
            waypoints_robot.append((rx, ry))
        
        return waypoints_robot
    
    def get_waypoints_flat(
        self,
        robot_pose: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Get waypoints as a flat numpy array for neural network input.
        
        Args:
            robot_pose: (x, y, theta) robot pose in world frame
            
        Returns:
            Flat array of shape (num_waypoints * 2,)
            Format: [x1, y1, x2, y2, ..., xn, yn]
        """
        waypoints = self.get_waypoints(robot_pose)
        flat = []
        for x, y in waypoints:
            flat.extend([x, y])
        return np.array(flat, dtype=np.float32)
    
    def get_distance_to_goal(self, robot_x: float, robot_y: float) -> float:
        """
        Get Euclidean distance from robot to the goal (end of path).
        
        Args:
            robot_x, robot_y: Robot position in world frame
            
        Returns:
            Distance to goal in meters
        """
        if not self.path:
            return float('inf')
        
        goal = self.path[-1]
        return math.sqrt((goal[0] - robot_x)**2 + (goal[1] - robot_y)**2)
    
    def get_second_waypoint_distance(
        self,
        robot_pose: Tuple[float, float, float]
    ) -> float:
        """
        Get distance to the second-next waypoint (used in SA reward).
        
        This is the dA* term in the paper's reward function.
        
        Args:
            robot_pose: (x, y, theta) robot pose in world frame
            
        Returns:
            Distance to second waypoint in robot frame
        """
        waypoints = self.get_waypoints(robot_pose)
        
        if len(waypoints) >= 2:
            x, y = waypoints[1]  # Second waypoint (index 1)
            return math.sqrt(x**2 + y**2)
        else:
            return 0.0
    
    def has_reached_goal(
        self,
        robot_x: float,
        robot_y: float,
        threshold: float = 0.3
    ) -> bool:
        """
        Check if robot has reached the goal.
        
        Args:
            robot_x, robot_y: Robot position in world frame
            threshold: Distance threshold to consider goal reached
            
        Returns:
            True if goal is reached
        """
        return self.get_distance_to_goal(robot_x, robot_y) < threshold
    
    def get_progress(self) -> float:
        """
        Get path completion progress as a percentage.
        
        Returns:
            Progress from 0.0 to 1.0
        """
        if not self.interpolated_path:
            return 0.0
        
        return self.current_path_index / max(1, len(self.interpolated_path) - 1)


if __name__ == "__main__":
    # Test waypoint manager
    print("=" * 60)
    print("Waypoint Manager Test")
    print("=" * 60)
    
    # Create a simple diagonal path
    path = [(i * 0.2, i * 0.2) for i in range(20)]  # 4m diagonal path
    
    manager = WaypointManager(
        num_waypoints=5,
        waypoint_spacing=0.3
    )
    manager.set_path(path)
    
    print(f"\nPath: {len(path)} points from {path[0]} to {path[-1]}")
    print(f"Interpolated: {len(manager.interpolated_path)} points")
    
    # Test 1: Robot at origin, facing +x direction
    print("\n--- Test 1: Robot at origin, heading +X ---")
    robot_pose = (0.0, 0.0, 0.0)
    waypoints = manager.get_waypoints(robot_pose)
    
    print(f"Number of waypoints: {len(waypoints)}")
    print("Waypoints (robot frame):")
    for i, (x, y) in enumerate(waypoints):
        dist = math.sqrt(x**2 + y**2)
        print(f"  WP{i}: ({x:.3f}, {y:.3f}) - distance: {dist:.3f}m")
    
    # Check spacing
    print("\nSpacing check:")
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        spacing = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        print(f"  WP{i} to WP{i+1}: {spacing:.3f}m (expected: 0.30m)")
    
    # Test 2: Robot at (1, 1), facing +x direction
    print("\n--- Test 2: Robot at (1,1), heading +X ---")
    robot_pose = (1.0, 1.0, 0.0)
    waypoints = manager.get_waypoints(robot_pose)
    
    print("Waypoints (robot frame):")
    for i, (x, y) in enumerate(waypoints):
        dist = math.sqrt(x**2 + y**2)
        print(f"  WP{i}: ({x:.3f}, {y:.3f}) - distance: {dist:.3f}m")
    
    # Test 3: Robot at origin, facing +45 degrees (along path)
    print("\n--- Test 3: Robot at origin, heading +45° (along path) ---")
    robot_pose = (0.0, 0.0, math.pi / 4)  # 45 degrees
    waypoints = manager.get_waypoints(robot_pose)
    
    print("Waypoints (robot frame):")
    for i, (x, y) in enumerate(waypoints):
        dist = math.sqrt(x**2 + y**2)
        angle = math.atan2(y, x) * 180 / math.pi
        print(f"  WP{i}: ({x:.3f}, {y:.3f}) - dist: {dist:.3f}m, angle: {angle:.1f}°")
    
    # Test 4: Flat output for neural network
    print("\n--- Test 4: Flat output for NN ---")
    robot_pose = (0.0, 0.0, 0.0)
    flat = manager.get_waypoints_flat(robot_pose)
    print(f"Shape: {flat.shape}")
    print(f"Values: {flat}")
    
    # Test 5: Distance to second waypoint (reward term)
    print("\n--- Test 5: Distance to second waypoint (dA*) ---")
    robot_pose = (0.0, 0.0, 0.0)
    d_astar = manager.get_second_waypoint_distance(robot_pose)
    print(f"Distance to second waypoint: {d_astar:.3f}m")
    
    # Test 6: Goal checking
    print("\n--- Test 6: Goal checking ---")
    print(f"Distance to goal from (0,0): {manager.get_distance_to_goal(0, 0):.3f}m")
    print(f"Distance to goal from (3,3): {manager.get_distance_to_goal(3, 3):.3f}m")
    print(f"Reached goal at (3.7, 3.7)? {manager.has_reached_goal(3.7, 3.7)}")
    print(f"Reached goal at (3.9, 3.9)? {manager.has_reached_goal(3.9, 3.9)}")
    
    print("\n✓ Waypoint Manager test complete!")

#!/usr/bin/env python3
"""
Verification Script for Hierarchical Navigation - Steps 1-4

This script tests:
1. Directory structure
2. Configuration file
3. A* Path Planner
4. Waypoint Manager

Run this script to verify the initial implementation is working correctly.

Usage:
    cd ~/turtlebot3_drlnav
    python3 src/turtlebot3_drl/turtlebot3_drl/hierarchical/tests/test_steps_1_to_4.py
"""

import sys
import os
import math

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_result(test_name: str, passed: bool, details: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")

def test_step1_directory_structure():
    """Test Step 1: Directory Structure"""
    print_header("STEP 1: Directory Structure")
    
    base_path = os.path.dirname(os.path.dirname(__file__))  # hierarchical/
    
    required_dirs = [
        'planners',
        'preprocessing',
        'agents',
        'environments',
        'training'
    ]
    
    required_files = [
        '__init__.py',
        'config.py',
        'planners/__init__.py',
        'planners/astar.py',
        'planners/waypoint_manager.py',
        'preprocessing/__init__.py',
        'agents/__init__.py',
        'environments/__init__.py',
        'training/__init__.py'
    ]
    
    all_passed = True
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = os.path.join(base_path, dir_name)
        exists = os.path.isdir(dir_path)
        print_result(f"Directory: {dir_name}/", exists)
        all_passed = all_passed and exists
    
    # Check files
    for file_name in required_files:
        file_path = os.path.join(base_path, file_name)
        exists = os.path.isfile(file_path)
        print_result(f"File: {file_name}", exists)
        all_passed = all_passed and exists
    
    return all_passed


def test_step2_config():
    """Test Step 2: Configuration File"""
    print_header("STEP 2: Configuration File")
    
    all_passed = True
    
    try:
        from hierarchical.config import HierarchicalConfig
        print_result("Import HierarchicalConfig", True)
    except ImportError as e:
        print_result("Import HierarchicalConfig", False, str(e))
        return False
    
    config = HierarchicalConfig()
    
    # Test expected values from paper
    tests = [
        ("SA_TIME_STEP", config.SA_TIME_STEP, 0.2),
        ("MA_TIME_STEP", config.MA_TIME_STEP, 0.05),
        ("LIDAR_RAYS", config.LIDAR_RAYS, 80),
        ("NUM_WAYPOINTS", config.NUM_WAYPOINTS, 5),
        ("WAYPOINT_SPACING", config.WAYPOINT_SPACING, 0.3),
        ("SUBGOAL_MAX_DISTANCE", config.SUBGOAL_MAX_DISTANCE, 0.6),
        ("MA_MAX_LINEAR_VEL", config.MA_MAX_LINEAR_VEL, 0.5),
        ("ASTAR_REPLAN_INTERVAL", config.ASTAR_REPLAN_INTERVAL, 3),
    ]
    
    for name, actual, expected in tests:
        passed = actual == expected
        print_result(f"{name} = {actual}", passed, f"expected: {expected}")
        all_passed = all_passed and passed
    
    # Test dimension calculations
    sa_state = config.get_sa_state_dim()
    expected_sa_state = 80 + 10  # 80 LiDAR + 5*2 waypoints
    print_result(f"SA State Dim = {sa_state}", sa_state == expected_sa_state, f"expected: {expected_sa_state}")
    all_passed = all_passed and (sa_state == expected_sa_state)
    
    return all_passed


def test_step3_astar():
    """Test Step 3: A* Path Planner"""
    print_header("STEP 3: A* Path Planner")
    
    import numpy as np
    
    try:
        from hierarchical.planners.astar import AStarPlanner, create_test_grid
        print_result("Import AStarPlanner", True)
    except ImportError as e:
        print_result("Import AStarPlanner", False, str(e))
        return False
    
    all_passed = True
    
    # Create test grid
    grid = create_test_grid(50, 50)
    
    planner = AStarPlanner(
        grid_resolution=0.1,
        robot_radius=0.15,
        inflation_radius=0.05
    )
    planner.set_occupancy_grid(grid)
    
    # Test 1: Basic path finding
    start, goal = (0.5, 0.5), (4.5, 4.5)
    path = planner.plan(start, goal)
    
    passed = path is not None and len(path) > 0
    print_result("Find path from (0.5,0.5) to (4.5,4.5)", passed, f"{len(path) if path else 0} waypoints")
    all_passed = all_passed and passed
    
    # Test 2: Path avoids obstacles
    if path:
        # Check that no point is in original obstacle area
        obstacle_collision = False
        for (px, py) in path:
            gx, gy = int(px / 0.1), int(py / 0.1)
            if 0 <= gx < 50 and 0 <= gy < 50:
                # Check original grid (before inflation)
                test_grid = create_test_grid(50, 50)  # Fresh grid
                if test_grid[gy, gx] == 1:
                    obstacle_collision = True
                    break
        
        print_result("Path avoids obstacles", not obstacle_collision)
        all_passed = all_passed and (not obstacle_collision)
    
    # Test 3: Path continuity (no teleportation)
    if path and len(path) > 1:
        max_gap = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            gap = math.sqrt(dx**2 + dy**2)
            max_gap = max(max_gap, gap)
        
        # Max gap should be < 0.2m (diagonal of 0.1m grid)
        passed = max_gap < 0.2
        print_result("Path is continuous (no teleportation)", passed, f"max gap: {max_gap:.3f}m")
        all_passed = all_passed and passed
    
    # Test 4: Impossible path returns None
    blocked_grid = np.ones((10, 10), dtype=np.int8)  # All obstacles
    blocked_planner = AStarPlanner(grid_resolution=0.1)
    blocked_planner.set_occupancy_grid(blocked_grid)
    
    impossible_path = blocked_planner.plan((0.5, 0.5), (0.8, 0.8))
    passed = impossible_path is None
    print_result("Returns None for impossible path", passed)
    all_passed = all_passed and passed
    
    # Test 5: Path length calculation
    if path:
        length = planner.get_path_length(path)
        # Diagonal from (0.5,0.5) to (4.5,4.5) is ~5.66m minimum
        # With obstacles, should be longer
        passed = length > 5.0
        print_result(f"Path length reasonable", passed, f"{length:.2f}m (min expected: 5.0m)")
        all_passed = all_passed and passed
    
    return all_passed


def test_step4_waypoint_manager():
    """Test Step 4: Waypoint Manager"""
    print_header("STEP 4: Waypoint Manager")
    
    import numpy as np
    
    try:
        from hierarchical.planners.waypoint_manager import WaypointManager
        print_result("Import WaypointManager", True)
    except ImportError as e:
        print_result("Import WaypointManager", False, str(e))
        return False
    
    all_passed = True
    
    # Create a simple diagonal path
    path = [(i * 0.1, i * 0.1) for i in range(50)]  # 5m diagonal
    
    manager = WaypointManager(num_waypoints=5, waypoint_spacing=0.3)
    manager.set_path(path)
    
    # Test 1: Returns correct number of waypoints
    robot_pose = (0.0, 0.0, 0.0)
    waypoints = manager.get_waypoints(robot_pose)
    
    passed = len(waypoints) == 5
    print_result(f"Returns {len(waypoints)} waypoints", passed, "expected: 5")
    all_passed = all_passed and passed
    
    # Test 2: Waypoints are approximately 0.3m apart
    if len(waypoints) >= 2:
        spacings = []
        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]
            spacing = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            spacings.append(spacing)
        
        avg_spacing = sum(spacings) / len(spacings)
        passed = abs(avg_spacing - 0.3) < 0.05  # Within 5cm tolerance
        print_result(f"Waypoint spacing ~0.3m", passed, f"avg: {avg_spacing:.3f}m")
        all_passed = all_passed and passed
    
    # Test 3: Waypoints are in robot frame (should be ahead when facing path)
    robot_pose = (0.0, 0.0, math.pi / 4)  # Facing 45° (along path)
    waypoints = manager.get_waypoints(robot_pose)
    
    # All waypoints should be roughly in front (+x direction)
    all_forward = all(wp[0] > -0.1 for wp in waypoints)  # Allow small tolerance
    print_result("Waypoints in robot frame (ahead)", all_forward)
    all_passed = all_passed and all_forward
    
    # Test 4: Flat output shape
    flat = manager.get_waypoints_flat(robot_pose)
    passed = flat.shape == (10,)  # 5 waypoints * 2 coords
    print_result(f"Flat output shape {flat.shape}", passed, "expected: (10,)")
    all_passed = all_passed and passed
    
    # Test 5: Distance to goal
    dist = manager.get_distance_to_goal(0, 0)
    expected_dist = math.sqrt(4.9**2 + 4.9**2)  # Path ends at (4.9, 4.9)
    passed = abs(dist - expected_dist) < 0.1
    print_result(f"Distance to goal calculation", passed, f"{dist:.2f}m (expected: {expected_dist:.2f}m)")
    all_passed = all_passed and passed
    
    # Test 6: Goal reached detection
    reached_far = manager.has_reached_goal(0, 0)
    reached_near = manager.has_reached_goal(4.8, 4.8)
    
    print_result("Goal NOT reached at (0,0)", not reached_far)
    print_result("Goal reached at (4.8,4.8)", reached_near)
    all_passed = all_passed and (not reached_far) and reached_near
    
    # Test 7: Second waypoint distance (dA* reward term)
    robot_pose = (0.0, 0.0, 0.0)
    d_astar = manager.get_second_waypoint_distance(robot_pose)
    # Second waypoint should be at ~0.6m
    passed = 0.5 < d_astar < 0.8
    print_result(f"Second waypoint distance (dA*)", passed, f"{d_astar:.3f}m (expected: ~0.6m)")
    all_passed = all_passed and passed
    
    return all_passed


def test_integration():
    """Test integration of A* and Waypoint Manager"""
    print_header("INTEGRATION TEST: A* + Waypoint Manager")
    
    import numpy as np
    
    try:
        from hierarchical.planners.astar import AStarPlanner, create_test_grid
        from hierarchical.planners.waypoint_manager import WaypointManager
        from hierarchical.config import HierarchicalConfig
    except ImportError as e:
        print_result("Import modules", False, str(e))
        return False
    
    print_result("Import all modules", True)
    
    all_passed = True
    config = HierarchicalConfig()
    
    # Create environment
    grid = create_test_grid(50, 50)
    
    planner = AStarPlanner(
        grid_resolution=config.ASTAR_RESOLUTION,
        robot_radius=config.ASTAR_ROBOT_RADIUS,
        inflation_radius=config.ASTAR_INFLATION_RADIUS
    )
    planner.set_occupancy_grid(grid)
    
    manager = WaypointManager(
        num_waypoints=config.NUM_WAYPOINTS,
        waypoint_spacing=config.WAYPOINT_SPACING
    )
    
    # Plan path
    start = (0.5, 0.5)
    goal = (4.5, 4.5)
    path = planner.plan(start, goal)
    
    passed = path is not None
    print_result("A* finds path", passed)
    all_passed = all_passed and passed
    
    if path:
        # Set path in waypoint manager
        manager.set_path(path)
        
        # Simulate robot following path
        robot_poses = [
            (0.5, 0.5, 0.0),
            (1.0, 1.0, math.pi/4),
            (2.0, 2.0, math.pi/4),
            (3.0, 3.0, math.pi/4),
        ]
        
        print("\nSimulated path following:")
        for rx, ry, rtheta in robot_poses:
            waypoints = manager.get_waypoints((rx, ry, rtheta))
            dist_to_goal = manager.get_distance_to_goal(rx, ry)
            progress = manager.get_progress()
            
            # First waypoint distance
            wp0_dist = math.sqrt(waypoints[0][0]**2 + waypoints[0][1]**2)
            
            print(f"  Robot at ({rx:.1f},{ry:.1f}): "
                  f"WP0 dist={wp0_dist:.2f}m, "
                  f"Goal dist={dist_to_goal:.2f}m, "
                  f"Progress={progress*100:.0f}%")
        
        print_result("Integration test complete", True)
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" HIERARCHICAL NAVIGATION - VERIFICATION TESTS (Steps 1-4)")
    print("=" * 70)
    
    results = {
        "Step 1: Directory Structure": test_step1_directory_structure(),
        "Step 2: Configuration": test_step2_config(),
        "Step 3: A* Planner": test_step3_astar(),
        "Step 4: Waypoint Manager": test_step4_waypoint_manager(),
        "Integration Test": test_integration(),
    }
    
    # Summary
    print_header("SUMMARY")
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed
    
    print("\n" + "-" * 70)
    if all_passed:
        print("  ✓ ALL TESTS PASSED - Steps 1-4 are working correctly!")
        print("  You can proceed to Step 5 (LiDAR Preprocessor)")
    else:
        print("  ✗ SOME TESTS FAILED - Please fix issues before proceeding")
    print("-" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

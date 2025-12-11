"""Minimal ROS2 environment wrapper for hierarchical DRL.

This is a light, synchronous wrapper that:
- Subscribes to `/scan` and `/odom`
- Publishes `/cmd_vel`
- Optionally calls Gazebo reset services
- Provides processed observations for the Subgoal Agent (LiDAR + waypoints)
- Provides rewards/termination for training loops

It is intentionally conservative: straight-line waypoint generation is used if no
map is available. Replace `build_path_to_goal` with A* if you have a map server.
"""

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from hierarchical.config import HierarchicalConfig
from hierarchical.planners.waypoint_manager import WaypointManager
from hierarchical.preprocessing.lidar_processor import LidarProcessor


def _yaw_from_quaternion(q) -> float:
    """Convert quaternion to yaw (rad)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class HierarchicalEnv(Node):
    """Synchronous environment used by the hierarchical trainer."""

    def __init__(
        self,
        config: Optional[HierarchicalConfig] = None,
        goal: Tuple[float, float] = (1.5, 0.0),
        use_sim_time: bool = True,
    ) -> None:
        if config is None:
            config = HierarchicalConfig()
        self.config = config
        super().__init__("hierarchical_env", allow_undeclared_parameters=True)
        if use_sim_time:
            self.set_parameters([self.create_parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])

        self.goal = goal
        self._scan = None
        self._odom = None
        self._last_cmd = (0.0, 0.0)
        self._step_count = 0

        self.lidar_processor = LidarProcessor(
            input_rays=config.LIDAR_RAW_RAYS,
            output_rays=config.LIDAR_RAYS,
            max_range=config.LIDAR_MAX_RANGE,
            clip_range=config.LIDAR_CLIP_RANGE,
        )
        self.waypoint_manager = WaypointManager(
            num_waypoints=config.NUM_WAYPOINTS,
            waypoint_spacing=config.WAYPOINT_SPACING,
        )

        self._scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self._odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self._reset_clients = [
            self.create_client(Empty, "/reset_world"),
            self.create_client(Empty, "/gazebo/reset_simulation"),
        ]

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _scan_cb(self, msg: LaserScan) -> None:
        self._scan = np.array(msg.ranges, dtype=np.float32)

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom = msg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """Wait until both scan and odom are received."""
        start = time.time()
        while rclpy.ok() and (self._scan is None or self._odom is None):
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - start > timeout:
                return False
        return True

    def _call_reset_service(self) -> None:
        for client in self._reset_clients:
            if client.service_is_ready():
                req = Empty.Request()
                try:
                    client.call_async(req)
                except Exception:
                    pass

    def _robot_pose(self) -> Tuple[float, float, float]:
        assert self._odom is not None
        p = self._odom.pose.pose.position
        q = self._odom.pose.pose.orientation
        theta = _yaw_from_quaternion(q)
        return p.x, p.y, theta

    def build_path_to_goal(self) -> None:
        """Generate a straight-line path to the goal in map frame."""
        rx, ry, _ = self._robot_pose()
        gx, gy = self.goal
        steps = max(2, int(max(abs(gx - rx), abs(gy - ry)) / self.config.WAYPOINT_SPACING) + 1)
        path = []
        for i in range(steps):
            t = i / (steps - 1)
            path.append((rx + t * (gx - rx), ry + t * (gy - ry)))
        self.waypoint_manager.set_path(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, goal: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
        """Reset world (if service available) and return first observation."""
        if goal is not None:
            self.goal = goal
        self._call_reset_service()
        self._step_count = 0
        self._last_cmd = (0.0, 0.0)
        self._cmd_pub.publish(Twist())  # stop
        ok = self.wait_for_data(timeout=5.0)
        if not ok:
            raise RuntimeError("Timed out waiting for /scan and /odom")
        self.build_path_to_goal()
        return self.get_observation()

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Return processed LiDAR and waypoint features for SA."""
        if self._scan is None or self._odom is None:
            raise RuntimeError("Sensors not ready")
        lidar = self.lidar_processor.process(self._scan)
        rx, ry, rtheta = self._robot_pose()
        waypoints = self.waypoint_manager.get_waypoints((rx, ry, rtheta))
        flat_wp = self.waypoint_manager.get_waypoints_flat((rx, ry, rtheta))
        return {
            "lidar": lidar.astype(np.float32),
            "waypoints": flat_wp.astype(np.float32),
            "waypoints_xy": waypoints,
        }

    def step(self, ma_action: Tuple[float, float]) -> Dict[str, float]:
        """Apply Motion Agent action and return transition info."""
        v, w = ma_action
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self._cmd_pub.publish(twist)
        self._last_cmd = (float(v), float(w))

        # Let the robot move for one MA time step
        rclpy.spin_once(self, timeout_sec=self.config.MA_TIME_STEP)
        time.sleep(self.config.MA_TIME_STEP)
        rclpy.spin_once(self, timeout_sec=0.0)

        self._step_count += 1
        obs = self.get_observation()
        reward, done, info = self._compute_reward_done(obs)
        return {
            "reward": reward,
            "done": done,
            "info": info,
            "observation": obs,
        }

    def _compute_reward_done(self, obs: Dict[str, np.ndarray]) -> Tuple[float, bool, Dict[str, float]]:
        rx, ry, _ = self._robot_pose()
        gx, gy = self.goal
        dist_goal = math.hypot(gx - rx, gy - ry)
        min_lidar = float(np.min(obs["lidar"]))

        reward = -dist_goal
        info = {
            "dist_goal": dist_goal,
            "min_lidar": min_lidar,
        }

        collision = min_lidar < self.config.COLLISION_DISTANCE
        success = dist_goal < self.config.GOAL_THRESHOLD
        timeout = self._step_count >= self.config.EPISODE_TIMEOUT

        if collision:
            reward += self.config.SA_REWARD_COLLISION
        if success:
            reward += self.config.SA_REWARD_GOAL

        done = collision or success or timeout
        info.update({"collision": collision, "success": success, "timeout": timeout})
        return reward, done, info

    @property
    def last_cmd(self) -> Tuple[float, float]:
        return self._last_cmd

    def shutdown(self) -> None:
        self._cmd_pub.publish(Twist())
        self.destroy_node()


def launch_env(goal: Tuple[float, float] = (1.5, 0.0)) -> HierarchicalEnv:
    """Helper to create env when used outside rclpy entrypoints."""
    if not rclpy.ok():
        rclpy.init()
    env = HierarchicalEnv(goal=goal)
    env.wait_for_data(timeout=5.0)
    return env

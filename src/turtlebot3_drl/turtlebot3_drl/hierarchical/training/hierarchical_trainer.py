"""Synchronous trainer/runner for the hierarchical TurtleBot3 DRL stack.

This is a pragmatic, minimal implementation to let you:
- Train: `ros2 run turtlebot3_drl train_hierarchical --episodes 10`
- Run/infer: `ros2 run turtlebot3_drl run_hierarchical --episodes 1`

It wires SubgoalAgent (SA) + MotionAgent (MA) with the HierarchicalEnv.
Uses hierarchical reward shaping from the paper:
- SA Reward: collision penalty + path distance penalty + safety penalty + goal bonus
- MA Reward: subgoal reach bonus + distance penalty
"""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rclpy

from hierarchical.agents.motion_agent import MotionAgent
from hierarchical.agents.subgoal_agent import SubgoalAgent
from hierarchical.config import HierarchicalConfig
from hierarchical.environments.hierarchical_env import HierarchicalEnv


class HierarchicalRewardComputer:
    """Computes hierarchical rewards for SA and MA based on the paper."""

    def __init__(self, config: HierarchicalConfig):
        self.config = config
        self._prev_dist_goal = None
        self._prev_dist_subgoal = None

    def reset(self):
        """Reset reward state for new episode."""
        self._prev_dist_goal = None
        self._prev_dist_subgoal = None

    def compute_sa_reward(
        self,
        dist_goal: float,
        min_lidar: float,
        collision: bool,
        success: bool,
        lidar_scan: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Subgoal Agent reward based on hierarchical formulation.
        
        SA Reward components:
        - Collision penalty: SA_REWARD_COLLISION (-10.0)
        - Goal bonus: SA_REWARD_GOAL (100.0)
        - Path distance penalty: SA_REWARD_PATH_COEFF * dist_goal
        - Safety penalty: SA_REWARD_SAFETY_COEFF * (num rays below safety threshold)
        """
        reward = 0.0
        components = {}

        # Collision penalty
        if collision:
            reward += self.config.SA_REWARD_COLLISION
            components["collision_penalty"] = self.config.SA_REWARD_COLLISION

        # Goal bonus
        if success:
            reward += self.config.SA_REWARD_GOAL
            components["goal_bonus"] = self.config.SA_REWARD_GOAL

        # Path distance penalty (negative reward proportional to goal distance)
        path_penalty = self.config.SA_REWARD_PATH_COEFF * dist_goal
        reward += path_penalty
        components["path_penalty"] = path_penalty

        # Safety penalty (penalize getting close to obstacles)
        # Count rays below safety threshold (normalized lidar values)
        safety_threshold = self.config.SA_SAFETY_DISTANCE / self.config.LIDAR_CLIP_RANGE
        unsafe_rays = np.sum(lidar_scan < safety_threshold)
        safety_penalty = self.config.SA_REWARD_SAFETY_COEFF * (unsafe_rays / len(lidar_scan))
        reward += safety_penalty
        components["safety_penalty"] = safety_penalty

        # Progress reward (bonus for getting closer to goal)
        if self._prev_dist_goal is not None:
            progress = self._prev_dist_goal - dist_goal
            progress_reward = progress * 5.0  # Scale factor
            reward += progress_reward
            components["progress_reward"] = progress_reward
        self._prev_dist_goal = dist_goal

        components["total"] = reward
        return reward, components

    def compute_ma_reward(
        self,
        subgoal_x: float,
        subgoal_y: float,
        robot_x: float,
        robot_y: float,
        collision: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Motion Agent reward based on hierarchical formulation.
        
        MA Reward components:
        - Subgoal reach bonus: MA_REWARD_REACH (2.0)
        - Distance penalty: MA_REWARD_DIST_COEFF * distance_to_subgoal
        """
        reward = 0.0
        components = {}

        # Distance to subgoal in robot frame (subgoal is relative to robot)
        dist_subgoal = math.hypot(subgoal_x, subgoal_y)

        # Subgoal reached bonus
        if dist_subgoal < self.config.MA_SUBGOAL_THRESHOLD:
            reward += self.config.MA_REWARD_REACH
            components["subgoal_reached"] = self.config.MA_REWARD_REACH

        # Distance penalty
        dist_penalty = self.config.MA_REWARD_DIST_COEFF * dist_subgoal
        reward += dist_penalty
        components["dist_penalty"] = dist_penalty

        # Progress toward subgoal
        if self._prev_dist_subgoal is not None:
            progress = self._prev_dist_subgoal - dist_subgoal
            progress_reward = progress * 2.0  # Scale factor
            reward += progress_reward
            components["subgoal_progress"] = progress_reward
        self._prev_dist_subgoal = dist_subgoal

        # Collision penalty for MA as well
        if collision:
            reward += self.config.SA_REWARD_COLLISION * 0.5  # Half penalty
            components["collision_penalty"] = self.config.SA_REWARD_COLLISION * 0.5

        components["total"] = reward
        return reward, components


class HierarchicalTrainer:
    """Simple on-robot trainer for hierarchical navigation."""

    def __init__(
        self,
        config: HierarchicalConfig,
        goal: Tuple[float, float] = (1.5, 0.0),
        sa_checkpoint: str | None = None,
        ma_checkpoint: str | None = None,
        log_file: str | None = None,
    ) -> None:
        self.config = config
        self.env = HierarchicalEnv(config=config, goal=goal)
        self.sa = SubgoalAgent(config=config)
        self.ma = MotionAgent(config=config)
        self.reward_computer = HierarchicalRewardComputer(config=config)

        # Optional file logging for detailed run traces
        self.logger = logging.getLogger("hierarchical_trainer")
        self.logger.setLevel(logging.INFO)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
        # Mirror to stdout via ROS logger by default
        self.logger.addHandler(logging.StreamHandler())

        # Load pretrained checkpoints when provided
        if sa_checkpoint and os.path.exists(sa_checkpoint):
            self.sa.load(sa_checkpoint)
            self.env.get_logger().info(f"Loaded SA checkpoint: {sa_checkpoint}")
            self.logger.info(f"Loaded SA checkpoint: {sa_checkpoint}")
        if ma_checkpoint and os.path.exists(ma_checkpoint):
            self.ma.load(ma_checkpoint)
            self.env.get_logger().info(f"Loaded MA checkpoint: {ma_checkpoint}")
            self.logger.info(f"Loaded MA checkpoint: {ma_checkpoint}")

    def train(self, episodes: int = 1) -> None:
        for ep in range(1, episodes + 1):
            obs = self.env.reset()
            self.sa.reset_noise()
            self.reward_computer.reset()
            ep_sa_reward = 0.0
            ep_ma_reward = 0.0

            lidar = obs["lidar"]
            wps = obs["waypoints"]

            for step in range(self.config.EPISODE_TIMEOUT):
                sa_action, _ = self.sa.select_action(lidar, wps, add_noise=True)
                px, py = self.sa.subgoal_to_cartesian(sa_action[0], sa_action[1])

                ma_state = self.ma.build_state(
                    prev_v=self.env.last_cmd[0],
                    prev_omega=self.env.last_cmd[1],
                    subgoal_x=px,
                    subgoal_y=py,
                )
                ma_action = self.ma.select_action(ma_state, add_noise=True)

                step_result = self.env.step(ma_action)
                next_obs = step_result["observation"]
                done = step_result["done"]
                info = step_result["info"]

                # Compute hierarchical rewards
                sa_reward, sa_components = self.reward_computer.compute_sa_reward(
                    dist_goal=info["dist_goal"],
                    min_lidar=info["min_lidar"],
                    collision=info["collision"],
                    success=info["success"],
                    lidar_scan=next_obs["lidar"],
                )
                ma_reward, ma_components = self.reward_computer.compute_ma_reward(
                    subgoal_x=px,
                    subgoal_y=py,
                    robot_x=0.0,  # Subgoal is in robot frame
                    robot_y=0.0,
                    collision=info["collision"],
                )
                ep_sa_reward += sa_reward
                ep_ma_reward += ma_reward

                next_lidar = next_obs["lidar"]
                next_wps = next_obs["waypoints"]
                next_ma_state = self.ma.build_state(
                    prev_v=self.env.last_cmd[0],
                    prev_omega=self.env.last_cmd[1],
                    subgoal_x=px,
                    subgoal_y=py,
                )

                # Store transitions with hierarchical rewards
                self.sa.store_transition(lidar, wps, sa_action, sa_reward, next_lidar, next_wps, done)
                self.ma.store_transition(ma_state, ma_action, ma_reward, next_ma_state, done)

                # Updates
                self.sa.update()
                self.ma.update()

                # Prepare next step
                lidar, wps = next_lidar, next_wps
                self.logger.info(
                    "EP %d step %d | sa_r=%.3f ma_r=%.3f dist_goal=%.3f min_lidar=%.3f collision=%s success=%s",
                    ep,
                    step,
                    sa_reward,
                    ma_reward,
                    info.get("dist_goal", float("nan")),
                    info.get("min_lidar", float("nan")),
                    info.get("collision"),
                    info.get("success"),
                )
                if done:
                    break

            self.env.get_logger().info(f"Episode {ep}/{episodes} | SA_Reward={ep_sa_reward:.2f} MA_Reward={ep_ma_reward:.2f}")
            self.logger.info("Episode %d/%d total_sa_reward=%.3f total_ma_reward=%.3f", ep, episodes, ep_sa_reward, ep_ma_reward)

    def run(self, episodes: int = 1, render: bool = False) -> None:
        """Run inference with hierarchical reward computation."""
        self.sa.set_training(False)
        self.ma.set_training(False)
        
        all_results = []
        
        for ep in range(1, episodes + 1):
            obs = self.env.reset()
            self.reward_computer.reset()
            lidar = obs["lidar"]
            wps = obs["waypoints"]
            
            ep_sa_reward = 0.0
            ep_ma_reward = 0.0
            step_count = 0
            outcome = "TIMEOUT"

            for step in range(self.config.EPISODE_TIMEOUT):
                step_count = step + 1
                sa_action, _ = self.sa.select_action(lidar, wps, add_noise=False)
                px, py = self.sa.subgoal_to_cartesian(sa_action[0], sa_action[1])
                ma_state = self.ma.build_state(
                    prev_v=self.env.last_cmd[0],
                    prev_omega=self.env.last_cmd[1],
                    subgoal_x=px,
                    subgoal_y=py,
                )
                ma_action = self.ma.select_action(ma_state, add_noise=False)
                step_result = self.env.step(ma_action)
                obs = step_result["observation"]
                info = step_result["info"]
                lidar = obs["lidar"]
                wps = obs["waypoints"]
                
                # Compute hierarchical rewards
                sa_reward, sa_components = self.reward_computer.compute_sa_reward(
                    dist_goal=info["dist_goal"],
                    min_lidar=info["min_lidar"],
                    collision=info["collision"],
                    success=info["success"],
                    lidar_scan=obs["lidar"],
                )
                ma_reward, ma_components = self.reward_computer.compute_ma_reward(
                    subgoal_x=px,
                    subgoal_y=py,
                    robot_x=0.0,
                    robot_y=0.0,
                    collision=info["collision"],
                )
                ep_sa_reward += sa_reward
                ep_ma_reward += ma_reward
                
                # Log every 50 steps for detailed progress
                if step % 50 == 0 or step_result["done"]:
                    self.logger.info(
                        "[RUN] EP %d step %d | sa_r=%.2f ma_r=%.2f dist_goal=%.3f min_lidar=%.3f collision=%s success=%s",
                        ep,
                        step,
                        sa_reward,
                        ma_reward,
                        info.get("dist_goal", float("nan")),
                        info.get("min_lidar", float("nan")),
                        info.get("collision"),
                        info.get("success"),
                    )
                
                if step_result["done"]:
                    if info["success"]:
                        outcome = "SUCCESS"
                    elif info["collision"]:
                        outcome = "COLLISION"
                    break
            
            # Episode summary
            ep_result = {
                "episode": ep,
                "outcome": outcome,
                "steps": step_count,
                "sa_reward": ep_sa_reward,
                "ma_reward": ep_ma_reward,
                "total_reward": ep_sa_reward + ep_ma_reward,
            }
            all_results.append(ep_result)
            
            self.env.get_logger().info(
                f"[RUN] Episode {ep}/{episodes} | Outcome={outcome} Steps={step_count} "
                f"SA_R={ep_sa_reward:.2f} MA_R={ep_ma_reward:.2f} Total={ep_sa_reward + ep_ma_reward:.2f}"
            )
            self.logger.info(
                "[RUN] Episode %d/%d outcome=%s steps=%d sa_reward=%.3f ma_reward=%.3f total=%.3f",
                ep, episodes, outcome, step_count, ep_sa_reward, ep_ma_reward, ep_sa_reward + ep_ma_reward
            )
        
        # Final summary
        self.logger.info("=" * 60)
        self.logger.info("FINAL SUMMARY:")
        self.logger.info("=" * 60)
        
        successes = sum(1 for r in all_results if r["outcome"] == "SUCCESS")
        collisions = sum(1 for r in all_results if r["outcome"] == "COLLISION")
        timeouts = sum(1 for r in all_results if r["outcome"] == "TIMEOUT")
        avg_sa = np.mean([r["sa_reward"] for r in all_results])
        avg_ma = np.mean([r["ma_reward"] for r in all_results])
        avg_total = np.mean([r["total_reward"] for r in all_results])
        avg_steps = np.mean([r["steps"] for r in all_results])
        
        for r in all_results:
            self.logger.info(
                "  Episode %d: %s, steps=%d, SA_R=%.2f, MA_R=%.2f, Total=%.2f",
                r["episode"], r["outcome"], r["steps"], r["sa_reward"], r["ma_reward"], r["total_reward"]
            )
        
        self.logger.info("-" * 60)
        self.logger.info("STATISTICS:")
        self.logger.info("  Success Rate: %d/%d (%.1f%%)", successes, episodes, 100 * successes / episodes)
        self.logger.info("  Collisions: %d, Timeouts: %d", collisions, timeouts)
        self.logger.info("  Avg Steps: %.1f", avg_steps)
        self.logger.info("  Avg SA Reward: %.2f", avg_sa)
        self.logger.info("  Avg MA Reward: %.2f", avg_ma)
        self.logger.info("  Avg Total Reward: %.2f", avg_total)
        self.logger.info("=" * 60)
        
        self.env.get_logger().info(
            f"FINAL: Success={successes}/{episodes} ({100*successes/episodes:.1f}%) "
            f"AvgSA={avg_sa:.2f} AvgMA={avg_ma:.2f} AvgTotal={avg_total:.2f}"
        )

    def shutdown(self) -> None:
        self.env.shutdown()


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical DRL trainer")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--goal", type=float, nargs=2, default=[1.5, 0.0], metavar=("X", "Y"))
    parser.add_argument("--mode", type=str, choices=["train", "run"], default="train")
    parser.add_argument("--sa-checkpoint", type=str, default=None, help="Path to SA checkpoint (.pth)")
    parser.add_argument("--ma-checkpoint", type=str, default=None, help="Path to MA checkpoint (.pth)")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file to store detailed run logs")
    return parser.parse_args()


def main_train() -> None:
    args = _parse_args()
    config = HierarchicalConfig()
    rclpy.init()
    trainer = HierarchicalTrainer(
        config=config,
        goal=(args.goal[0], args.goal[1]),
        sa_checkpoint=args.sa_checkpoint,
        ma_checkpoint=args.ma_checkpoint,
        log_file=args.log_file,
    )
    try:
        trainer.train(episodes=args.episodes)
    finally:
        trainer.shutdown()
        rclpy.shutdown()


def main_run() -> None:
    args = _parse_args()
    config = HierarchicalConfig()
    rclpy.init()
    trainer = HierarchicalTrainer(
        config=config,
        goal=(args.goal[0], args.goal[1]),
        sa_checkpoint=args.sa_checkpoint,
        ma_checkpoint=args.ma_checkpoint,
        log_file=args.log_file,
    )
    try:
        if args.mode == "train":
            trainer.train(episodes=args.episodes)
        else:
            trainer.run(episodes=args.episodes)
    finally:
        trainer.shutdown()
        rclpy.shutdown()

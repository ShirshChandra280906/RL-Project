"""
Hierarchical DRL Navigation Configuration

Configuration parameters based on the paper:
"Lightweight Motion Planning via Hierarchical Reinforcement Learning"

All parameters can be modified here without changing other code.
"""

import math


class HierarchicalConfig:
    """Configuration class for hierarchical navigation system."""
    
    def __init__(self):
        # ==================== SUBGOAL AGENT (SA) ====================
        # Timing
        self.SA_TIME_STEP = 0.2          # seconds (5 Hz)
        self.SA_ALGORITHM = 'ddpg'
        
        # LiDAR Configuration
        self.LIDAR_RAW_RAYS = 360        # TurtleBot3 LDS-01 output
        self.LIDAR_RAYS = 80             # Down-sampled rays (paper spec)
        self.LIDAR_SECTORS = 10          # Number of attention sectors
        self.LIDAR_RAYS_PER_SECTOR = 8   # 80 / 10 = 8 rays per sector
        self.LIDAR_MAX_RANGE = 3.5       # TurtleBot3 LDS-01 max range (m)
        self.LIDAR_CLIP_RANGE = 4.0      # Clip range for normalization (m)
        
        # Waypoint Configuration
        self.NUM_WAYPOINTS = 5           # Number of waypoints from A*
        self.WAYPOINT_SPACING = 0.3      # Distance between waypoints (m)
        self.WAYPOINT_COVERAGE = 1.2     # Total coverage (5-1) * 0.3 = 1.2m
        
        # Subgoal Action Space
        self.SUBGOAL_MAX_DISTANCE = 0.6  # Max distance l (m)
        self.SUBGOAL_MIN_DISTANCE = 0.0  # Min distance l (m)
        self.SUBGOAL_ANGLE_MIN = 0.0     # Min angle θ (rad)
        self.SUBGOAL_ANGLE_MAX = 2 * math.pi  # Max angle θ (rad)
        
        # SA Network Architecture
        self.SA_EMBEDDING_LAYERS = [512, 256, 128]
        self.SA_FEATURE_LAYERS = [256, 128, 64]
        self.SA_SCORE_LAYERS = [128, 64, 1]
        self.SA_PATH_LAYERS = [128, 64, 32]
        self.SA_OUTPUT_LAYERS = [128, 64, 64]
        self.SA_ATTENTION_OUTPUT_DIM = 64
        self.SA_PATH_OUTPUT_DIM = 32
        
        # SA Reward Parameters
        self.SA_REWARD_COLLISION = -10.0
        self.SA_REWARD_PATH_COEFF = -0.5     # Coefficient for A* path distance
        self.SA_REWARD_SAFETY_COEFF = -2.0   # Coefficient for safety penalty
        self.SA_SAFETY_DISTANCE = 0.5        # Safety threshold (m)
        self.SA_REWARD_GOAL = 100.0          # Reward for reaching goal
        
        # SA Training Hyperparameters
        self.SA_LEARNING_RATE_ACTOR = 1e-4
        self.SA_LEARNING_RATE_CRITIC = 1e-3
        self.SA_GAMMA = 0.99
        self.SA_TAU = 0.005
        self.SA_BATCH_SIZE = 64
        self.SA_BUFFER_SIZE = 100000
        
        # ==================== MOTION AGENT (MA) ====================
        # Timing
        self.MA_TIME_STEP = 0.05         # seconds (20 Hz)
        self.MA_ALGORITHM = 'td3'
        self.MA_STEPS_PER_SA = 4         # 0.2 / 0.05 = 4 MA steps per SA step
        
        # MA State Space
        self.MA_STATE_DIM = 5            # (v*, ω*, px, py, θdiff)
        
        # MA Action Space
        self.MA_ACTION_DIM = 2           # (v, ω)
        self.MA_MAX_LINEAR_VEL = 0.5     # m/s (TurtleBot3 safe limit)
        self.MA_MIN_LINEAR_VEL = 0.0     # m/s
        self.MA_MAX_ANGULAR_VEL = math.pi / 2  # rad/s (π/2 ≈ 1.57)
        self.MA_MIN_ANGULAR_VEL = -math.pi / 2
        
        # MA Network Architecture
        self.MA_LAYERS = [256, 128, 64, 64]
        
        # MA Reward Parameters
        self.MA_REWARD_REACH = 2.0       # Reward for reaching subgoal
        self.MA_REWARD_DIST_COEFF = -1.0 # Coefficient for distance penalty
        self.MA_SUBGOAL_THRESHOLD = 0.1  # Distance to consider subgoal reached (m)
        
        # MA Training Hyperparameters
        self.MA_LEARNING_RATE_ACTOR = 1e-4
        self.MA_LEARNING_RATE_CRITIC = 1e-3
        self.MA_GAMMA = 0.99
        self.MA_TAU = 0.005
        self.MA_BATCH_SIZE = 64
        self.MA_BUFFER_SIZE = 100000
        self.MA_POLICY_NOISE = 0.2
        self.MA_NOISE_CLIP = 0.5
        self.MA_POLICY_DELAY = 2
        
        # MA Pre-training Configuration
        self.MA_CONVERGENCE_EPISODES = 50  # Consecutive successes to converge
        self.MA_SUBGOAL_SAMPLE_DISTANCE_MAX = 0.7  # Max distance for random subgoals
        self.MA_SUBGOAL_STRAIGHT_PROB = 0.2   # Probability of straight-line subgoal
        self.MA_SUBGOAL_CURVY_PROB = 0.3      # Probability of curvy (±π/2) subgoal
        self.MA_SUBGOAL_RANDOM_PROB = 0.5     # Probability of random subgoal
        
        # ==================== GLOBAL PLANNER (A*) ====================
        # Paper: "replan the robot's global path every three subgoal predictions"
        # NOTE: A* replanning does NOT consider unknown obstacles (static map only)
        self.ASTAR_RESOLUTION = 0.1      # Grid resolution (m)
        self.ASTAR_REPLAN_INTERVAL = 3   # Replan every N SA predictions (paper: 3)
        self.ASTAR_ROBOT_RADIUS = 0.2    # Robot radius for collision check (m)
        self.ASTAR_INFLATION_RADIUS = 0.1  # Additional inflation for safety (m)
        
        # Paper: "fixed time frame of ∆tSA = 0.2s between subsequent states"
        # NOTE: Timing is independent of whether subgoal is reached within ∆tSA
        self.SA_FIXED_TIMESTEP = True    # Always use fixed ∆tSA regardless of subgoal reach
        
        # ==================== EPISODE CONFIGURATION ====================
        self.EPISODE_TIMEOUT = 500       # Max steps per episode
        self.COLLISION_DISTANCE = 0.18   # Distance to consider collision (m)
        self.GOAL_THRESHOLD = 0.3        # Distance to consider goal reached (m)
        
        # ==================== TRAINING CONFIGURATION ====================
        self.STORE_MODEL_INTERVAL = 100  # Save model every N episodes
        self.LOG_INTERVAL = 10           # Log metrics every N episodes
        self.GRAPH_DRAW_INTERVAL = 50    # Draw graphs every N episodes
        
    def get_sa_state_dim(self) -> int:
        """Calculate Subgoal Agent state dimension."""
        return self.LIDAR_RAYS + (self.NUM_WAYPOINTS * 2)  # 80 + 10 = 90
    
    def get_sa_action_dim(self) -> int:
        """Subgoal Agent action dimension (l, θ)."""
        return 2
    
    def get_ma_state_dim(self) -> int:
        """Motion Agent state dimension (v*, ω*, px, py, θdiff)."""
        return self.MA_STATE_DIM
    
    def get_ma_action_dim(self) -> int:
        """Motion Agent action dimension (v, ω)."""
        return self.MA_ACTION_DIM
    
    def __repr__(self) -> str:
        return (
            f"HierarchicalConfig(\n"
            f"  SA: {self.SA_TIME_STEP}s @ {1/self.SA_TIME_STEP:.0f}Hz, "
            f"LiDAR={self.LIDAR_RAYS} rays, Waypoints={self.NUM_WAYPOINTS}\n"
            f"  MA: {self.MA_TIME_STEP}s @ {1/self.MA_TIME_STEP:.0f}Hz, "
            f"v=[{self.MA_MIN_LINEAR_VEL}, {self.MA_MAX_LINEAR_VEL}] m/s\n"
            f"  A*: resolution={self.ASTAR_RESOLUTION}m, "
            f"replan every {self.ASTAR_REPLAN_INTERVAL} SA steps\n"
            f")"
        )


# Global config instance
config = HierarchicalConfig()


if __name__ == "__main__":
    # Test configuration
    cfg = HierarchicalConfig()
    print(cfg)
    print(f"\nSA State Dim: {cfg.get_sa_state_dim()}")
    print(f"SA Action Dim: {cfg.get_sa_action_dim()}")
    print(f"MA State Dim: {cfg.get_ma_state_dim()}")
    print(f"MA Action Dim: {cfg.get_ma_action_dim()}")
    print("\n✓ Config loaded successfully!")

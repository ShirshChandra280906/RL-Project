"""
Agents module for Hierarchical Navigation.

Contains:
- SubgoalAgent: DDPG-based agent for collision avoidance via subgoal selection
- MotionAgent: TD3-based agent for velocity control to reach subgoals
- Network architectures: Actor/Critic networks for both agents
"""

from .networks import (
    SubgoalActorNetwork,
    SubgoalCriticNetwork,
    MotionActorNetwork,
    MotionCriticNetwork,
    SALidarModule,
    SAPathModule,
    SAOutputModule
)

from .subgoal_agent import (
    SubgoalAgent,
    OUNoise,
    ReplayBuffer,
    compute_sa_reward
)

from .motion_agent import (
    MotionAgent,
    GaussianNoise,
    MAReplayBuffer,
    SubgoalSampler,
    compute_ma_reward
)

__all__ = [
    # Networks
    'SubgoalActorNetwork',
    'SubgoalCriticNetwork',
    'MotionActorNetwork',
    'MotionCriticNetwork',
    'SALidarModule',
    'SAPathModule',
    'SAOutputModule',
    # Subgoal Agent
    'SubgoalAgent',
    'OUNoise',
    'ReplayBuffer',
    'compute_sa_reward',
    # Motion Agent
    'MotionAgent',
    'GaussianNoise',
    'MAReplayBuffer',
    'SubgoalSampler',
    'compute_ma_reward'
]

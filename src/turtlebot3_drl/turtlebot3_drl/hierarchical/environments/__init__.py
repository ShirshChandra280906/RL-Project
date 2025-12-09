"""
Environments package for Hierarchical Navigation.

Provides simulation environments for training and testing.
"""

from .scenes import (
    SceneType,
    SceneInfo,
    BaseScene,
    EmptyScene,
    CorridorScene,
    RoomScene,
    SimpleMazeScene,
    RandomObstacleScene,
    SceneFactory
)

from .obstacles import (
    ObstacleType,
    Obstacle,
    ObstacleManager
)

from .hierarchical_env import (
    TerminationReason,
    RobotState,
    HierarchicalStep,
    HierarchicalEnvironment,
    MAPretrainingEnvironment
)

__all__ = [
    # Scenes
    'SceneType',
    'SceneInfo',
    'BaseScene',
    'EmptyScene',
    'CorridorScene',
    'RoomScene',
    'SimpleMazeScene',
    'RandomObstacleScene',
    'SceneFactory',
    # Obstacles
    'ObstacleType',
    'Obstacle',
    'ObstacleManager',
    # Environments
    'TerminationReason',
    'RobotState',
    'HierarchicalStep',
    'HierarchicalEnvironment',
    'MAPretrainingEnvironment',
]

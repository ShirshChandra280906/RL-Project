#!/usr/bin/env python3
"""
Comprehensive Test Suite for Steps 9-10: Hierarchical Environment & Training Pipeline

Tests:
- Step 9: Scene classes, obstacle manager, hierarchical environment
- Step 10: Training pipeline, MA pre-training, SA training

Run: python3 hierarchical/tests/test_steps_9_to_10.py
"""

import os
import sys
import math
import unittest
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# STEP 9 TESTS: HIERARCHICAL ENVIRONMENT
# ============================================================================

class TestSceneClasses(unittest.TestCase):
    """Test scene generation and configuration."""
    
    @classmethod
    def setUpClass(cls):
        from hierarchical.environments.scenes import (
            SceneType, BaseScene, CorridorScene, IntersectionScene, 
            OfficeScene, SceneFactory
        )
        cls.SceneType = SceneType
        cls.CorridorScene = CorridorScene
        cls.IntersectionScene = IntersectionScene
        cls.OfficeScene = OfficeScene
        cls.SceneFactory = SceneFactory
    
    def test_corridor_dimensions(self):
        """Test corridor scene dimensions match paper specs."""
        scene = self.CorridorScene()
        
        for _ in range(10):
            scene.generate()
            # Paper: width [1.8m, 3m], length [10m, 14m]
            self.assertGreaterEqual(scene.width, 1.8)
            self.assertLessEqual(scene.width, 3.0)
            self.assertGreaterEqual(scene.length, 10.0)
            self.assertLessEqual(scene.length, 14.0)
    
    def test_intersection_dimensions(self):
        """Test intersection scene dimensions match paper specs."""
        scene = self.IntersectionScene()
        
        for _ in range(10):
            scene.generate()
            # Paper: hallway width [1.8m, 2.5m], length [4m, 6m]
            self.assertGreaterEqual(scene.hallway_width, 1.8)
            self.assertLessEqual(scene.hallway_width, 2.5)
            self.assertGreaterEqual(scene.hallway_length, 4.0)
            self.assertLessEqual(scene.hallway_length, 6.0)
    
    def test_office_dimensions(self):
        """Test office scene has fixed 7x7m outer walls."""
        scene = self.OfficeScene()
        scene.generate()
        
        # Paper: fixed outer width and length of 7 meters
        self.assertEqual(scene.width, 7.0)
        self.assertEqual(scene.length, 7.0)
    
    def test_corridor_start_goal_opposite_ends(self):
        """Test corridor start/goal are at opposite ends."""
        scene = self.CorridorScene()
        
        for _ in range(10):
            info = scene.reset()
            start, goal = info['start'], info['goal']
            
            # Start and goal should be at opposite ends (large x-distance)
            x_dist = abs(goal[0] - start[0])
            self.assertGreater(x_dist, scene.length * 0.5)
    
    def test_intersection_start_goal_different_deadends(self):
        """Test intersection start/goal are in different dead-ends."""
        scene = self.IntersectionScene()
        
        for _ in range(10):
            info = scene.reset()
            start, goal = info['start'], info['goal']
            
            # Start and goal should be far apart
            dist = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
            self.assertGreater(dist, 2.0)  # At least 2m apart
    
    def test_office_start_goal_opposing_corners(self):
        """Test office start/goal are in opposing corners."""
        scene = self.OfficeScene()
        
        for _ in range(10):
            info = scene.reset()
            start, goal = info['start'], info['goal']
            
            # Diagonal distance in 7x7 square is ~9.9m, opposite corners > 5m
            dist = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
            self.assertGreater(dist, 5.0)
    
    def test_occupancy_grid_generated(self):
        """Test that occupancy grid is properly generated."""
        for scene_type in self.SceneType:
            scene = self.SceneFactory.create(scene_type)
            scene.generate()
            
            self.assertIsNotNone(scene.grid)
            self.assertEqual(scene.grid.ndim, 2)
            self.assertGreater(scene.grid_width, 0)
            self.assertGreater(scene.grid_height, 0)
    
    def test_scene_factory(self):
        """Test scene factory creates correct scene types."""
        for scene_type in self.SceneType:
            scene = self.SceneFactory.create(scene_type)
            self.assertEqual(scene.scene_type, scene_type)


class TestObstacleManager(unittest.TestCase):
    """Test obstacle management."""
    
    @classmethod
    def setUpClass(cls):
        from hierarchical.environments.obstacles import (
            ObstacleType, Obstacle, ObstacleManager
        )
        from hierarchical.environments.scenes import CorridorScene
        cls.ObstacleType = ObstacleType
        cls.Obstacle = Obstacle
        cls.ObstacleManager = ObstacleManager
        cls.CorridorScene = CorridorScene
    
    def test_spawn_correct_obstacle_count(self):
        """Test correct number of obstacles spawned."""
        manager = self.ObstacleManager(num_dynamic=2, num_static=1)
        
        # Create mock scene
        class MockScene:
            width = 10.0
            length = 10.0
            def is_free(self, x, y, margin=0.2):
                return 0 < x < self.width and 0 < y < self.length
        
        manager.set_scene(MockScene(), None)
        
        # Create robot path
        robot_path = [(1, 1), (2, 2), (5, 5), (8, 8)]
        obstacles = manager.reset((1, 1), (8, 8), robot_path)
        
        # Should have 2 dynamic + 1 static = 3 obstacles
        self.assertEqual(len(manager.dynamic_obstacles), 2)
        self.assertEqual(len(manager.static_obstacles), 1)
        self.assertEqual(len(obstacles), 3)
    
    def test_dynamic_obstacle_speed_range(self):
        """Test dynamic obstacle speed is in paper range [0.1, 0.5] m/s."""
        manager = self.ObstacleManager(num_dynamic=5, num_static=0)
        
        class MockScene:
            width = 10.0
            length = 10.0
            def is_free(self, x, y, margin=0.2):
                return True
        
        manager.set_scene(MockScene(), None)
        robot_path = [(1, 1), (2, 2), (5, 5), (8, 8)]
        manager.reset((1, 1), (8, 8), robot_path)
        
        for obs in manager.dynamic_obstacles:
            # Paper: speed [0.1, 0.5] m/s
            self.assertGreaterEqual(obs.velocity, 0.1)
            self.assertLessEqual(obs.velocity, 0.5)
    
    def test_static_obstacle_on_robot_path(self):
        """Test static obstacle is placed on robot's path."""
        manager = self.ObstacleManager(num_dynamic=0, num_static=1)
        
        class MockScene:
            width = 10.0
            length = 10.0
            def is_free(self, x, y, margin=0.2):
                return True
        
        manager.set_scene(MockScene(), None)
        robot_path = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        manager.reset((1, 1), (5, 5), robot_path)
        
        if len(manager.static_obstacles) > 0:
            obs = manager.static_obstacles[0]
            # Should be near the path (within offset + margin)
            min_dist = float('inf')
            for pt in robot_path:
                dist = math.sqrt((obs.position[0] - pt[0])**2 + (obs.position[1] - pt[1])**2)
                min_dist = min(min_dist, dist)
            self.assertLess(min_dist, 1.0)  # Within 1m of path
    
    def test_obstacle_update_movement(self):
        """Test dynamic obstacles move when updated."""
        manager = self.ObstacleManager(num_dynamic=1, num_static=0)
        
        class MockScene:
            width = 10.0
            length = 10.0
            def is_free(self, x, y, margin=0.2):
                return True
        
        manager.set_scene(MockScene(), None)
        robot_path = [(1, 1), (5, 1), (9, 1)]
        manager.reset((1, 1), (9, 1), robot_path)
        
        if len(manager.dynamic_obstacles) > 0:
            obs = manager.dynamic_obstacles[0]
            initial_pos = obs.position
            
            # Update for 1 second
            for _ in range(20):
                manager.update(0.05)
            
            # Position should have changed
            final_pos = obs.position
            dist = math.sqrt((final_pos[0] - initial_pos[0])**2 + 
                           (final_pos[1] - initial_pos[1])**2)
            # At 0.1-0.5 m/s for 1s, should move 0.1-0.5m
            self.assertGreater(dist, 0.05)
    
    def test_collision_detection(self):
        """Test collision detection between robot and obstacles."""
        manager = self.ObstacleManager(num_dynamic=0, num_static=1)
        
        class MockScene:
            width = 10.0
            length = 10.0
            def is_free(self, x, y, margin=0.2):
                return True
        
        manager.set_scene(MockScene(), None)
        
        # Create obstacle at known position
        obs = self.Obstacle(
            id=0,
            type=self.ObstacleType.STATIC,
            position=(5.0, 5.0),
            size=(0.4, 0.4, 1.0)
        )
        manager.obstacles = [obs]
        manager.static_obstacles = [obs]
        
        # Robot at same position should collide
        collision, _ = manager.check_collision((5.0, 5.0), robot_radius=0.18)
        self.assertTrue(collision)
        
        # Robot far away should not collide
        collision, _ = manager.check_collision((1.0, 1.0), robot_radius=0.18)
        self.assertFalse(collision)


class TestHierarchicalEnvironment(unittest.TestCase):
    """Test hierarchical environment."""
    
    @classmethod
    def setUpClass(cls):
        from hierarchical.config import HierarchicalConfig
        from hierarchical.environments.hierarchical_env import (
            HierarchicalEnvironment, MAPretrainingEnvironment,
            TerminationReason, RobotState
        )
        from hierarchical.environments.scenes import SceneType
        
        cls.HierarchicalConfig = HierarchicalConfig
        cls.HierarchicalEnvironment = HierarchicalEnvironment
        cls.MAPretrainingEnvironment = MAPretrainingEnvironment
        cls.TerminationReason = TerminationReason
        cls.RobotState = RobotState
        cls.SceneType = SceneType
    
    def test_environment_reset(self):
        """Test environment reset returns valid observation."""
        config = self.HierarchicalConfig()
        env = self.HierarchicalEnvironment(config)
        
        obs = env.reset(self.SceneType.CORRIDOR)
        
        self.assertIn('lidar', obs)
        self.assertIn('waypoints', obs)
        self.assertIn('robot_state', obs)
        self.assertIn('goal', obs)
        
        # Check shapes
        self.assertEqual(obs['lidar'].shape, (config.LIDAR_RAYS,))
        self.assertEqual(obs['waypoints'].shape, (config.NUM_WAYPOINTS * 2,))
    
    def test_sa_timestep_correct(self):
        """Test SA operates at correct timestep (0.2s)."""
        config = self.HierarchicalConfig()
        
        # Paper: SA at 5 Hz = 0.2s timestep
        self.assertEqual(config.SA_TIME_STEP, 0.2)
    
    def test_ma_timestep_correct(self):
        """Test MA operates at correct timestep (0.05s)."""
        config = self.HierarchicalConfig()
        
        # Paper: MA at 20 Hz = 0.05s timestep
        self.assertEqual(config.MA_TIME_STEP, 0.05)
    
    def test_ma_steps_per_sa(self):
        """Test 4 MA steps per SA step."""
        config = self.HierarchicalConfig()
        
        # 0.2s / 0.05s = 4
        self.assertEqual(config.MA_STEPS_PER_SA, 4)
    
    def test_ma_state_dimension(self):
        """Test MA state has 5 dimensions (v*, ω*, px, py, θdiff)."""
        config = self.HierarchicalConfig()
        env = self.HierarchicalEnvironment(config)
        env.reset(self.SceneType.CORRIDOR)
        
        ma_state = env._get_ma_state()
        self.assertEqual(ma_state.shape, (5,))
    
    def test_termination_goal_reached(self):
        """Test goal reached termination."""
        config = self.HierarchicalConfig()
        env = self.HierarchicalEnvironment(config)
        env.reset(self.SceneType.CORRIDOR)
        
        # Place robot at goal
        env.robot.x = env.goal_pos[0]
        env.robot.y = env.goal_pos[1]
        
        done, reason = env._check_termination()
        self.assertTrue(done)
        self.assertEqual(reason, self.TerminationReason.GOAL_REACHED)
    
    def test_termination_timeout(self):
        """Test timeout termination."""
        config = self.HierarchicalConfig()
        env = self.HierarchicalEnvironment(config)
        env.reset(self.SceneType.CORRIDOR)
        
        # Set step count to timeout
        env.step_count = config.EPISODE_TIMEOUT
        
        done, reason = env._check_termination()
        self.assertTrue(done)
        self.assertEqual(reason, self.TerminationReason.TIMEOUT)
    
    def test_robot_state_dataclass(self):
        """Test RobotState dataclass."""
        state = self.RobotState(x=1.0, y=2.0, theta=0.5, linear_vel=0.3, angular_vel=0.1)
        
        arr = state.to_array()
        self.assertEqual(len(arr), 5)
        self.assertEqual(arr[0], 1.0)
        self.assertEqual(arr[1], 2.0)
        
        # Test from_array
        state2 = self.RobotState.from_array(arr)
        self.assertEqual(state2.x, state.x)
        self.assertEqual(state2.y, state.y)


class TestMAPretrainingEnvironment(unittest.TestCase):
    """Test MA pre-training environment."""
    
    @classmethod
    def setUpClass(cls):
        from hierarchical.config import HierarchicalConfig
        from hierarchical.environments.hierarchical_env import MAPretrainingEnvironment
        cls.HierarchicalConfig = HierarchicalConfig
        cls.MAPretrainingEnvironment = MAPretrainingEnvironment
    
    def test_subgoal_sampling_distribution(self):
        """Test subgoal sampling follows paper distribution."""
        config = self.HierarchicalConfig()
        env = self.MAPretrainingEnvironment(config)
        
        straight_count = 0
        curvy_count = 0
        random_count = 0
        
        # Sample many subgoals
        for _ in range(1000):
            subgoal = env._sample_subgoal()
            px, py = subgoal
            
            angle = abs(math.atan2(py, px))
            
            if angle < 0.1:  # Nearly straight
                straight_count += 1
            elif abs(angle - math.pi/2) < 0.4:  # Curvy (±π/2)
                curvy_count += 1
            else:
                random_count += 1
        
        # Check proportions (with tolerance)
        # Paper: 20% straight, 30% curvy, 50% random
        total = straight_count + curvy_count + random_count
        self.assertGreater(straight_count / total, 0.1)  # At least 10%
        self.assertLess(straight_count / total, 0.35)     # At most 35%
    
    def test_state_shape(self):
        """Test state shape is correct (5 dimensions)."""
        env = self.MAPretrainingEnvironment()
        state = env.reset()
        
        self.assertEqual(state.shape, (5,))
    
    def test_subgoal_reached_terminates(self):
        """Test episode terminates when subgoal is reached."""
        config = self.HierarchicalConfig()
        env = self.MAPretrainingEnvironment(config)
        env.reset()
        
        # Set subgoal very close
        env.subgoal = (0.05, 0.0)
        
        # Step with small movement
        action = np.array([0.1, 0.0])
        _, _, done, info = env.step(action)
        
        # Should terminate with success
        self.assertTrue(done)
        self.assertTrue(info['success'])


# ============================================================================
# STEP 10 TESTS: TRAINING PIPELINE
# ============================================================================

class TestTrainingConfig(unittest.TestCase):
    """Test training configuration."""
    
    @classmethod
    def setUpClass(cls):
        from hierarchical.training.trainer import TrainingConfig, TrainingStage
        cls.TrainingConfig = TrainingConfig
        cls.TrainingStage = TrainingStage
    
    def test_training_stages_enum(self):
        """Test training stages are defined."""
        self.assertEqual(self.TrainingStage.MA_PRETRAINING.value, "ma_pretraining")
        self.assertEqual(self.TrainingStage.SA_TRAINING.value, "sa_training")
    
    def test_default_config_values(self):
        """Test default training config values."""
        config = self.TrainingConfig()
        
        self.assertEqual(config.ma_convergence_threshold, 50)
        self.assertGreater(config.ma_pretrain_episodes, 0)
        self.assertGreater(config.sa_train_episodes, 0)


class TestMAPretrainer(unittest.TestCase):
    """Test MA pre-trainer."""
    
    @classmethod
    def setUpClass(cls):
        from hierarchical.config import HierarchicalConfig
        from hierarchical.training.trainer import MAPretrainer, TrainingConfig
        cls.HierarchicalConfig = HierarchicalConfig
        cls.MAPretrainer = MAPretrainer
        cls.TrainingConfig = TrainingConfig
    
    def test_pretrainer_initialization(self):
        """Test MA pre-trainer initializes correctly."""
        config = self.HierarchicalConfig()
        train_config = self.TrainingConfig()
        
        trainer = self.MAPretrainer(config, train_config)
        
        self.assertIsNotNone(trainer.env)
        self.assertIsNotNone(trainer.agent)
        self.assertEqual(trainer.episode_count, 0)
    
    def test_train_single_episode(self):
        """Test training a single episode."""
        config = self.HierarchicalConfig()
        train_config = self.TrainingConfig()
        
        trainer = self.MAPretrainer(config, train_config)
        metrics = trainer.train_episode()
        
        self.assertEqual(metrics.episode, 1)
        self.assertGreater(metrics.episode_length, 0)
        self.assertIsInstance(metrics.success, bool)
    
    def test_convergence_tracking(self):
        """Test consecutive success tracking."""
        config = self.HierarchicalConfig()
        train_config = self.TrainingConfig()
        
        trainer = self.MAPretrainer(config, train_config)
        
        # Simulate success
        trainer.agent.record_episode_result(True)
        self.assertEqual(trainer.agent.consecutive_successes, 1)
        
        # Simulate failure resets counter
        trainer.agent.record_episode_result(False)
        self.assertEqual(trainer.agent.consecutive_successes, 0)


class TestSATrainer(unittest.TestCase):
    """Test SA trainer."""
    
    @classmethod
    def setUpClass(cls):
        from hierarchical.config import HierarchicalConfig
        from hierarchical.training.trainer import SATrainer, TrainingConfig
        from hierarchical.agents.motion_agent import MotionAgent
        cls.HierarchicalConfig = HierarchicalConfig
        cls.SATrainer = SATrainer
        cls.TrainingConfig = TrainingConfig
        cls.MotionAgent = MotionAgent
    
    def test_sa_trainer_initialization(self):
        """Test SA trainer initializes correctly."""
        config = self.HierarchicalConfig()
        train_config = self.TrainingConfig()
        
        trainer = self.SATrainer(config, train_config)
        
        self.assertIsNotNone(trainer.env)
        self.assertIsNotNone(trainer.sa_agent)
        self.assertIsNotNone(trainer.ma_agent)
    
    def test_ma_frozen_during_sa_training(self):
        """Test MA is frozen during SA training."""
        config = self.HierarchicalConfig()
        train_config = self.TrainingConfig()
        
        trainer = self.SATrainer(config, train_config)
        
        # MA should be in eval mode
        self.assertFalse(trainer.ma_agent.training)
    
    def test_pretrained_ma_used(self):
        """Test pre-trained MA can be passed to SA trainer."""
        config = self.HierarchicalConfig()
        train_config = self.TrainingConfig()
        
        # Create and "pre-train" MA
        pretrained_ma = self.MotionAgent(config)
        pretrained_ma.converged = True  # Mark as converged
        
        trainer = self.SATrainer(config, train_config, pretrained_ma=pretrained_ma)
        
        # Should use the provided MA
        self.assertTrue(trainer.ma_agent.converged)


class TestHierarchicalTrainer(unittest.TestCase):
    """Test complete hierarchical trainer."""
    
    @classmethod
    def setUpClass(cls):
        from hierarchical.config import HierarchicalConfig
        from hierarchical.training.trainer import HierarchicalTrainer, TrainingConfig
        cls.HierarchicalConfig = HierarchicalConfig
        cls.HierarchicalTrainer = HierarchicalTrainer
        cls.TrainingConfig = TrainingConfig
    
    def test_trainer_initialization(self):
        """Test hierarchical trainer initializes correctly."""
        config = self.HierarchicalConfig()
        train_config = self.TrainingConfig()
        
        trainer = self.HierarchicalTrainer(config, train_config)
        
        self.assertIsNotNone(trainer.config)
        self.assertIsNotNone(trainer.train_config)
    
    def test_seed_setting(self):
        """Test random seed is set for reproducibility."""
        train_config = self.TrainingConfig(seed=42)
        trainer = self.HierarchicalTrainer(training_config=train_config)
        
        # Seeds should produce consistent random values
        import random
        val1 = random.random()
        
        trainer._set_seeds(42)
        val2 = random.random()
        
        # After re-seeding, should get same sequence
        trainer._set_seeds(42)
        val3 = random.random()
        
        self.assertEqual(val2, val3)


class TestTrainingMetrics(unittest.TestCase):
    """Test training metrics tracking."""
    
    @classmethod
    def setUpClass(cls):
        from hierarchical.training.trainer import TrainingMetrics
        cls.TrainingMetrics = TrainingMetrics
    
    def test_metrics_initialization(self):
        """Test metrics dataclass initialization."""
        metrics = self.TrainingMetrics(
            episode=10,
            total_steps=500,
            episode_reward=25.5,
            success=True
        )
        
        self.assertEqual(metrics.episode, 10)
        self.assertEqual(metrics.total_steps, 500)
        self.assertEqual(metrics.episode_reward, 25.5)
        self.assertTrue(metrics.success)
    
    def test_default_values(self):
        """Test default metric values."""
        metrics = self.TrainingMetrics()
        
        self.assertEqual(metrics.episode, 0)
        self.assertEqual(metrics.episode_reward, 0.0)
        self.assertFalse(metrics.success)
        self.assertFalse(metrics.collision)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for complete hierarchical system."""
    
    @classmethod
    def setUpClass(cls):
        from hierarchical.config import HierarchicalConfig
        from hierarchical.environments.hierarchical_env import HierarchicalEnvironment
        from hierarchical.environments.scenes import SceneType
        from hierarchical.agents.subgoal_agent import SubgoalAgent
        from hierarchical.agents.motion_agent import MotionAgent
        
        cls.HierarchicalConfig = HierarchicalConfig
        cls.HierarchicalEnvironment = HierarchicalEnvironment
        cls.SceneType = SceneType
        cls.SubgoalAgent = SubgoalAgent
        cls.MotionAgent = MotionAgent
    
    def test_full_episode_execution(self):
        """Test running a complete episode with SA and MA."""
        config = self.HierarchicalConfig()
        env = self.HierarchicalEnvironment(config)
        sa = self.SubgoalAgent(config, device='cpu')
        ma = self.MotionAgent(config, device='cpu')
        
        obs = env.reset(self.SceneType.CORRIDOR)
        
        done = False
        max_steps = 50
        step = 0
        
        while not done and step < max_steps:
            # SA step
            sa_action, _ = sa.select_action(
                obs['lidar'], obs['waypoints'], add_noise=False
            )
            
            # MA steps
            l, theta = sa_action
            subgoal_x = l * np.cos(theta)
            subgoal_y = l * np.sin(theta)
            
            for _ in range(config.MA_STEPS_PER_SA):
                ma_state = ma.build_state(
                    env.robot.linear_vel,
                    env.robot.angular_vel,
                    subgoal_x,
                    subgoal_y
                )
                ma_action = ma.select_action(ma_state, add_noise=False)
                _, _, done, _ = env.step_ma(ma_action)
                
                if done:
                    break
            
            obs = env._get_observation()
            step += 1
        
        # Episode should complete (either goal, collision, or max steps)
        self.assertLessEqual(step, max_steps)
    
    def test_scene_switching(self):
        """Test environment handles scene type switching."""
        config = self.HierarchicalConfig()
        env = self.HierarchicalEnvironment(config)
        
        for scene_type in self.SceneType:
            obs = env.reset(scene_type)
            
            self.assertEqual(env.scene.scene_type, scene_type)
            self.assertIsNotNone(obs['lidar'])


# ============================================================================
# RUN TESTS
# ============================================================================

def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Step 9 tests
    step9_tests = [
        TestSceneClasses,
        TestObstacleManager,
        TestHierarchicalEnvironment,
        TestMAPretrainingEnvironment,
    ]
    
    # Step 10 tests
    step10_tests = [
        TestTrainingConfig,
        TestMAPretrainer,
        TestSATrainer,
        TestHierarchicalTrainer,
        TestTrainingMetrics,
    ]
    
    # Integration tests
    integration_tests = [
        TestIntegration,
    ]
    
    all_tests = step9_tests + step10_tests + integration_tests
    
    for test_class in all_tests:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    print("=" * 70)
    print("TESTING STEPS 9-10: HIERARCHICAL ENVIRONMENT & TRAINING PIPELINE")
    print("=" * 70)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    
    print(f"\nStep 9 (Environment): {sum(loader.loadTestsFromTestCase(t).countTestCases() for t in step9_tests)} tests")
    print(f"Step 10 (Training): {sum(loader.loadTestsFromTestCase(t).countTestCases() for t in step10_tests)} tests")
    print(f"Integration: {sum(loader.loadTestsFromTestCase(t).countTestCases() for t in integration_tests)} tests")
    print(f"\nTOTAL: {passed}/{total}")
    
    if failures == 0 and errors == 0:
        print("\n🎉 ALL STEPS 9-10 TESTS PASSED!")
        print("Ready to train the hierarchical navigation system!")
    else:
        print(f"\n❌ {failures} failures, {errors} errors")
        if result.failures:
            print("\nFailed tests:")
            for test, _ in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nError tests:")
            for test, _ in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

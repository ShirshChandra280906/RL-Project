"""
Comprehensive Tests for Steps 7-8: Subgoal Agent and Motion Agent

Test Coverage:
- Step 7: Subgoal Agent (DDPG)
  - Network architectures match paper specs
  - Action selection with noise
  - Replay buffer operations
  - DDPG update mechanics
  - A* replan trigger every 3 predictions

- Step 8: Motion Agent (TD3)
  - Network architecture [256, 128, 64, 64]
  - State building (v*, ω*, px, py, θdiff)
  - TD3 twin critics and delayed policy update
  - Subgoal sampling for pre-training
  - Convergence tracking
"""

import sys
import os
import math
import traceback
from typing import List, Tuple

import numpy as np

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not available - cannot run agent tests")
    sys.exit(1)


class TestRunner:
    """Simple test runner with detailed output."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures: List[Tuple[str, str]] = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and track results."""
        self.tests_run += 1
        print(f"\n  ▸ {test_name}...", end=" ")
        
        try:
            test_func()
            self.tests_passed += 1
            print("✓ PASSED")
        except AssertionError as e:
            self.tests_failed += 1
            self.failures.append((test_name, str(e)))
            print(f"✗ FAILED: {e}")
        except Exception as e:
            self.tests_failed += 1
            error_msg = f"{type(e).__name__}: {e}"
            self.failures.append((test_name, error_msg))
            print(f"✗ ERROR: {error_msg}")
            traceback.print_exc()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {self.tests_passed}/{self.tests_run} passed")
        
        if self.failures:
            print("\nFailures:")
            for name, reason in self.failures:
                print(f"  - {name}: {reason[:80]}")
        
        if self.tests_failed == 0:
            print("\n✅ ALL TESTS PASSED!")
        else:
            print(f"\n❌ {self.tests_failed} test(s) failed")
        
        print("=" * 60)
        return self.tests_failed == 0


def test_step7_subgoal_agent():
    """Test Step 7: Subgoal Agent (DDPG)."""
    print("\n" + "=" * 60)
    print("STEP 7: Subgoal Agent (DDPG) Tests")
    print("=" * 60)
    
    runner = TestRunner()
    
    from hierarchical.config import HierarchicalConfig
    from hierarchical.agents.networks import (
        SubgoalActorNetwork, SubgoalCriticNetwork,
        SALidarModule, SAPathModule
    )
    from hierarchical.agents.subgoal_agent import (
        SubgoalAgent, OUNoise, ReplayBuffer, compute_sa_reward
    )
    
    config = HierarchicalConfig()
    
    # Test 7.1: SA Actor network architecture
    def test_sa_actor_architecture():
        actor = SubgoalActorNetwork()
        
        # Test forward pass
        lidar = torch.rand(2, 80)
        waypoints = torch.rand(2, 10)
        action, attention = actor(lidar, waypoints)
        
        assert action.shape == (2, 2), f"Expected (2, 2), got {action.shape}"
        assert attention.shape == (2, 10), f"Expected (2, 10), got {attention.shape}"
    
    runner.run_test("7.1 SA Actor architecture", test_sa_actor_architecture)
    
    # Test 7.2: SA Critic network architecture
    def test_sa_critic_architecture():
        critic = SubgoalCriticNetwork()
        
        lidar = torch.rand(2, 80)
        waypoints = torch.rand(2, 10)
        action = torch.rand(2, 2)
        
        q_value = critic(lidar, waypoints, action)
        
        assert q_value.shape == (2, 1), f"Expected (2, 1), got {q_value.shape}"
    
    runner.run_test("7.2 SA Critic architecture", test_sa_critic_architecture)
    
    # Test 7.3: LiDAR module (attention)
    def test_lidar_module():
        lidar_module = SALidarModule(num_sectors=10, rays_per_sector=8)
        
        lidar = torch.rand(4, 80)
        features, attention = lidar_module(lidar)
        
        assert features.shape == (4, 64), f"Expected (4, 64), got {features.shape}"
        assert attention.shape == (4, 10), f"Expected (4, 10), got {attention.shape}"
        assert torch.allclose(attention.sum(dim=1), torch.ones(4), atol=1e-5), \
            "Attention should sum to 1"
    
    runner.run_test("7.3 LiDAR attention module", test_lidar_module)
    
    # Test 7.4: Path module
    def test_path_module():
        path_module = SAPathModule(num_waypoints=5)
        
        waypoints = torch.rand(4, 10)
        features = path_module(waypoints)
        
        assert features.shape == (4, 32), f"Expected (4, 32), got {features.shape}"
    
    runner.run_test("7.4 Path module [128,64,32]", test_path_module)
    
    # Test 7.5: SA action bounds
    def test_sa_action_bounds():
        actor = SubgoalActorNetwork(
            max_distance=0.6,
            min_distance=0.0,
            max_angle=2*math.pi,
            min_angle=0.0
        )
        
        # Test with various inputs
        for _ in range(10):
            lidar = torch.rand(4, 80)
            waypoints = torch.rand(4, 10)
            action, _ = actor(lidar, waypoints)
            
            # Check bounds
            assert (action[:, 0] >= 0).all(), f"Distance should be >= 0"
            assert (action[:, 0] <= 0.6).all(), f"Distance should be <= 0.6"
            assert (action[:, 1] >= 0).all(), f"Angle should be >= 0"
            assert (action[:, 1] <= 2*math.pi).all(), f"Angle should be <= 2π"
    
    runner.run_test("7.5 SA action bounds", test_sa_action_bounds)
    
    # Test 7.6: OUNoise
    def test_ou_noise():
        noise = OUNoise(action_dim=2)
        
        samples = [noise.sample() for _ in range(100)]
        samples = np.array(samples)
        
        assert samples.shape == (100, 2), f"Expected (100, 2), got {samples.shape}"
        
        # Check noise is not constant
        assert samples.std() > 0.01, "Noise should have variance"
        
        # Reset should return to mean
        noise.reset()
        assert np.allclose(noise.state, 0), "Reset should return to mean"
    
    runner.run_test("7.6 OU Noise", test_ou_noise)
    
    # Test 7.7: Replay buffer
    def test_sa_replay_buffer():
        buffer = ReplayBuffer(capacity=100, state_dim=90, action_dim=2)
        
        for i in range(50):
            buffer.add(
                lidar=np.random.rand(80).astype(np.float32),
                waypoints=np.random.rand(10).astype(np.float32),
                action=np.random.rand(2).astype(np.float32),
                reward=-0.5,
                next_lidar=np.random.rand(80).astype(np.float32),
                next_waypoints=np.random.rand(10).astype(np.float32),
                done=False
            )
        
        assert len(buffer) == 50, f"Expected 50, got {len(buffer)}"
        
        # Test sampling
        batch = buffer.sample(16, torch.device('cpu'))
        assert batch['lidar'].shape == (16, 80)
        assert batch['waypoints'].shape == (16, 10)
        assert batch['actions'].shape == (16, 2)
    
    runner.run_test("7.7 SA Replay buffer", test_sa_replay_buffer)
    
    # Test 7.8: SubgoalAgent creation
    def test_sa_creation():
        agent = SubgoalAgent(config, device='cpu')
        
        assert agent.device == torch.device('cpu')
        assert agent.replan_interval == 3
        assert agent.training == True
    
    runner.run_test("7.8 SubgoalAgent creation", test_sa_creation)
    
    # Test 7.9: SA action selection
    def test_sa_action_selection():
        agent = SubgoalAgent(config, device='cpu')
        
        lidar = np.random.rand(80).astype(np.float32)
        waypoints = np.random.rand(10).astype(np.float32)
        
        action, should_replan = agent.select_action(lidar, waypoints)
        
        assert action.shape == (2,), f"Expected (2,), got {action.shape}"
        assert isinstance(should_replan, bool)
    
    runner.run_test("7.9 SA action selection", test_sa_action_selection)
    
    # Test 7.10: A* replan trigger every 3 predictions
    def test_replan_trigger():
        agent = SubgoalAgent(config, device='cpu')
        agent.prediction_count = 0
        
        lidar = np.random.rand(80).astype(np.float32)
        waypoints = np.random.rand(10).astype(np.float32)
        
        replan_flags = []
        for i in range(9):
            _, should_replan = agent.select_action(lidar, waypoints, add_noise=False)
            replan_flags.append(should_replan)
        
        # Should replan at predictions 3, 6, 9
        expected = [False, False, True, False, False, True, False, False, True]
        assert replan_flags == expected, f"Expected {expected}, got {replan_flags}"
    
    runner.run_test("7.10 A* replan every 3 predictions", test_replan_trigger)
    
    # Test 7.11: DDPG update
    def test_ddpg_update():
        agent = SubgoalAgent(config, device='cpu')
        
        # Fill buffer
        for i in range(100):
            agent.store_transition(
                lidar=np.random.rand(80).astype(np.float32),
                waypoints=np.random.rand(10).astype(np.float32),
                action=np.random.rand(2).astype(np.float32),
                reward=-0.5,
                next_lidar=np.random.rand(80).astype(np.float32),
                next_waypoints=np.random.rand(10).astype(np.float32),
                done=False
            )
        
        # Perform update
        losses = agent.update()
        
        assert 'sa_actor_loss' in losses, "Should have actor loss"
        assert 'sa_critic_loss' in losses, "Should have critic loss"
        assert not math.isnan(losses['sa_actor_loss']), "Actor loss should not be NaN"
        assert not math.isnan(losses['sa_critic_loss']), "Critic loss should not be NaN"
    
    runner.run_test("7.11 DDPG update step", test_ddpg_update)
    
    # Test 7.12: SA reward function
    def test_sa_reward():
        # Test collision
        r_collision = compute_sa_reward(
            d_astar=1.0, d_astar_prev=1.0,
            min_lidar=0.1, collision=True,
            goal_reached=False, config=config
        )
        assert r_collision < -5, f"Collision should give large negative reward"
        
        # Test goal reached
        r_goal = compute_sa_reward(
            d_astar=0.0, d_astar_prev=0.1,
            min_lidar=1.0, collision=False,
            goal_reached=True, config=config
        )
        assert r_goal > 50, f"Goal should give large positive reward"
        
        # Test path progress
        r_progress = compute_sa_reward(
            d_astar=0.8, d_astar_prev=1.0,  # Moved closer
            min_lidar=1.0, collision=False,
            goal_reached=False, config=config
        )
        # Moving closer should give small reward
        assert r_progress != 0, "Path progress should affect reward"
    
    runner.run_test("7.12 SA reward function", test_sa_reward)
    
    # Test 7.13: Subgoal to Cartesian conversion
    def test_subgoal_conversion():
        agent = SubgoalAgent(config, device='cpu')
        
        # Test forward direction
        px, py = agent.subgoal_to_cartesian(l=1.0, theta=0)
        assert abs(px - 1.0) < 0.01, f"px should be ~1.0, got {px}"
        assert abs(py - 0.0) < 0.01, f"py should be ~0.0, got {py}"
        
        # Test left direction (90 degrees)
        px, py = agent.subgoal_to_cartesian(l=1.0, theta=math.pi/2)
        assert abs(px - 0.0) < 0.01, f"px should be ~0.0, got {px}"
        assert abs(py - 1.0) < 0.01, f"py should be ~1.0, got {py}"
    
    runner.run_test("7.13 Subgoal polar to Cartesian", test_subgoal_conversion)
    
    # Test 7.14: Attention weights available
    def test_attention_available():
        agent = SubgoalAgent(config, device='cpu')
        
        lidar = np.random.rand(80).astype(np.float32)
        waypoints = np.random.rand(10).astype(np.float32)
        agent.select_action(lidar, waypoints)
        
        attention = agent.get_attention_weights()
        
        assert attention is not None, "Attention should be available"
        assert attention.shape == (10,), f"Expected (10,), got {attention.shape}"
        assert abs(attention.sum() - 1.0) < 0.01, "Attention should sum to 1"
    
    runner.run_test("7.14 Attention weights accessible", test_attention_available)
    
    return runner


def test_step8_motion_agent():
    """Test Step 8: Motion Agent (TD3)."""
    print("\n" + "=" * 60)
    print("STEP 8: Motion Agent (TD3) Tests")
    print("=" * 60)
    
    runner = TestRunner()
    
    from hierarchical.config import HierarchicalConfig
    from hierarchical.agents.networks import (
        MotionActorNetwork, MotionCriticNetwork
    )
    from hierarchical.agents.motion_agent import (
        MotionAgent, GaussianNoise, MAReplayBuffer,
        SubgoalSampler, compute_ma_reward
    )
    
    config = HierarchicalConfig()
    
    # Test 8.1: MA Actor network architecture [256, 128, 64, 64]
    def test_ma_actor_architecture():
        actor = MotionActorNetwork(
            state_dim=5,
            action_dim=2,
            hidden_layers=[256, 128, 64, 64]
        )
        
        state = torch.rand(4, 5)
        action = actor(state)
        
        assert action.shape == (4, 2), f"Expected (4, 2), got {action.shape}"
        
        # Check layer sizes
        hidden_layers = list(actor.hidden)
        linear_layers = [l for l in hidden_layers if isinstance(l, nn.Linear)]
        assert linear_layers[0].in_features == 5, "Input should be 5"
        assert linear_layers[0].out_features == 256, "First hidden should be 256"
    
    runner.run_test("8.1 MA Actor architecture [256,128,64,64]", test_ma_actor_architecture)
    
    # Test 8.2: MA Critic (twin critics for TD3)
    def test_ma_critic_architecture():
        critic = MotionCriticNetwork(
            state_dim=5,
            action_dim=2,
            hidden_layers=[256, 128, 64, 64]
        )
        
        state = torch.rand(4, 5)
        action = torch.rand(4, 2)
        
        q1, q2 = critic(state, action)
        
        assert q1.shape == (4, 1), f"Q1 shape: expected (4, 1), got {q1.shape}"
        assert q2.shape == (4, 1), f"Q2 shape: expected (4, 1), got {q2.shape}"
        
        # Twin critics should give different values (different networks)
        # After initialization, they might be similar but not identical
    
    runner.run_test("8.2 MA Twin Critics (TD3)", test_ma_critic_architecture)
    
    # Test 8.3: MA action bounds
    def test_ma_action_bounds():
        actor = MotionActorNetwork(
            max_linear_vel=0.5,
            min_linear_vel=0.0,
            max_angular_vel=math.pi/2,
            min_angular_vel=-math.pi/2
        )
        
        for _ in range(10):
            state = torch.rand(4, 5)
            action = actor(state)
            
            assert (action[:, 0] >= 0).all(), "v should be >= 0"
            assert (action[:, 0] <= 0.5).all(), "v should be <= 0.5"
            assert (action[:, 1] >= -math.pi/2).all(), "ω should be >= -π/2"
            assert (action[:, 1] <= math.pi/2).all(), "ω should be <= π/2"
    
    runner.run_test("8.3 MA action bounds", test_ma_action_bounds)
    
    # Test 8.4: State building (v*, ω*, px, py, θdiff)
    def test_state_building():
        agent = MotionAgent(config, device='cpu')
        
        state = agent.build_state(
            prev_v=0.3,
            prev_omega=0.2,
            subgoal_x=0.4,
            subgoal_y=0.3
        )
        
        assert state.shape == (5,), f"Expected (5,), got {state.shape}"
        assert state[0] == 0.3, f"v* should be 0.3, got {state[0]}"
        assert state[1] == 0.2, f"ω* should be 0.2, got {state[1]}"
        assert state[2] == 0.4, f"px should be 0.4, got {state[2]}"
        assert state[3] == 0.3, f"py should be 0.3, got {state[3]}"
        
        # θdiff should be atan2(py, px)
        expected_theta = math.atan2(0.3, 0.4)
        assert abs(state[4] - expected_theta) < 0.01, \
            f"θdiff should be {expected_theta:.3f}, got {state[4]:.3f}"
    
    runner.run_test("8.4 State building (v*,ω*,px,py,θdiff)", test_state_building)
    
    # Test 8.5: MA Replay buffer
    def test_ma_replay_buffer():
        buffer = MAReplayBuffer(capacity=100, state_dim=5, action_dim=2)
        
        for i in range(50):
            buffer.add(
                state=np.random.rand(5).astype(np.float32),
                action=np.random.rand(2).astype(np.float32),
                reward=-0.3,
                next_state=np.random.rand(5).astype(np.float32),
                done=False
            )
        
        assert len(buffer) == 50
        
        batch = buffer.sample(16, torch.device('cpu'))
        assert batch['states'].shape == (16, 5)
        assert batch['actions'].shape == (16, 2)
    
    runner.run_test("8.5 MA Replay buffer", test_ma_replay_buffer)
    
    # Test 8.6: MotionAgent creation
    def test_ma_creation():
        agent = MotionAgent(config, device='cpu')
        
        assert agent.device == torch.device('cpu')
        assert agent.policy_delay == 2, "TD3 policy delay should be 2"
        assert not agent.converged
    
    runner.run_test("8.6 MotionAgent creation", test_ma_creation)
    
    # Test 8.7: MA action selection
    def test_ma_action_selection():
        agent = MotionAgent(config, device='cpu')
        
        state = np.random.rand(5).astype(np.float32)
        action = agent.select_action(state)
        
        assert action.shape == (2,), f"Expected (2,), got {action.shape}"
        assert action[0] >= 0 and action[0] <= 0.5, "v out of bounds"
        assert action[1] >= -math.pi/2 and action[1] <= math.pi/2, "ω out of bounds"
    
    runner.run_test("8.7 MA action selection", test_ma_action_selection)
    
    # Test 8.8: TD3 update with delayed policy
    def test_td3_update():
        agent = MotionAgent(config, device='cpu')
        agent.policy_delay = 2
        
        # Fill buffer
        for i in range(100):
            agent.store_transition(
                state=np.random.rand(5).astype(np.float32),
                action=np.random.rand(2).astype(np.float32),
                reward=-0.3,
                next_state=np.random.rand(5).astype(np.float32),
                done=False
            )
        
        # First update - no actor update
        losses1 = agent.update()
        assert 'ma_critic_loss' in losses1
        
        # Second update - should have actor update
        losses2 = agent.update()
        assert 'ma_critic_loss' in losses2
        assert 'ma_actor_loss' in losses2, "Actor should update on 2nd step"
    
    runner.run_test("8.8 TD3 delayed policy update", test_td3_update)
    
    # Test 8.9: MA reward function
    def test_ma_reward():
        # Subgoal reached
        r_reach = compute_ma_reward(
            distance_to_subgoal=0.05,
            subgoal_reached=True,
            config=config
        )
        assert r_reach > 1, f"Reaching subgoal should give positive reward"
        
        # Far from subgoal
        r_far = compute_ma_reward(
            distance_to_subgoal=0.5,
            subgoal_reached=False,
            config=config
        )
        assert r_far < 0, "Far from subgoal should give negative reward"
        
        # Closer is better
        r_close = compute_ma_reward(
            distance_to_subgoal=0.1,
            subgoal_reached=False,
            config=config
        )
        assert r_close > r_far, "Closer should be better than far"
    
    runner.run_test("8.9 MA reward function", test_ma_reward)
    
    # Test 8.10: Subgoal sampling for pre-training
    def test_subgoal_sampling():
        sampler = SubgoalSampler(config)
        
        samples = [sampler.sample() for _ in range(100)]
        
        # Check distance range (0, 0.7]
        for x, y in samples:
            dist = math.sqrt(x**2 + y**2)
            assert 0 < dist <= 0.7 + 0.01, f"Distance {dist} out of range"
        
        # Check we get variety of directions
        angles = [math.atan2(y, x) for x, y in samples]
        angle_range = max(angles) - min(angles)
        assert angle_range > math.pi, "Should sample various directions"
    
    runner.run_test("8.10 Subgoal sampling for pre-training", test_subgoal_sampling)
    
    # Test 8.11: Convergence tracking
    def test_convergence_tracking():
        agent = MotionAgent(config, device='cpu')
        agent.convergence_threshold = 50
        
        assert not agent.is_converged()
        
        # Record successes
        for i in range(49):
            agent.record_episode_result(True)
        assert not agent.is_converged(), "Should not converge at 49"
        
        agent.record_episode_result(True)
        assert agent.is_converged(), "Should converge at 50"
    
    runner.run_test("8.11 Convergence tracking (50 episodes)", test_convergence_tracking)
    
    # Test 8.12: Convergence reset on failure
    def test_convergence_reset():
        agent = MotionAgent(config, device='cpu')
        
        for i in range(30):
            agent.record_episode_result(True)
        assert agent.consecutive_successes == 30
        
        agent.record_episode_result(False)
        assert agent.consecutive_successes == 0, "Should reset on failure"
    
    runner.run_test("8.12 Convergence reset on failure", test_convergence_reset)
    
    # Test 8.13: Gaussian noise
    def test_gaussian_noise():
        noise = GaussianNoise(action_dim=2, sigma=0.1)
        
        samples = [noise.sample() for _ in range(100)]
        samples = np.array(samples)
        
        assert samples.shape == (100, 2)
        assert abs(samples.mean()) < 0.1, "Mean should be near 0"
        assert 0.05 < samples.std() < 0.2, f"Std should be near sigma, got {samples.std()}"
    
    runner.run_test("8.13 Gaussian exploration noise", test_gaussian_noise)
    
    # Test 8.14: Save and load
    def test_save_load():
        agent = MotionAgent(config, device='cpu')
        
        # Make some updates
        for i in range(100):
            agent.store_transition(
                state=np.random.rand(5).astype(np.float32),
                action=np.random.rand(2).astype(np.float32),
                reward=-0.3,
                next_state=np.random.rand(5).astype(np.float32),
                done=False
            )
        agent.update()
        
        # Save
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            path = f.name
        agent.save(path)
        
        # Create new agent and load
        agent2 = MotionAgent(config, device='cpu')
        agent2.load(path)
        
        # Check weights match
        for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
            assert torch.allclose(p1, p2), "Loaded weights should match"
        
        # Cleanup
        os.remove(path)
    
    runner.run_test("8.14 Save and load agent", test_save_load)
    
    return runner


def main():
    """Run all tests."""
    print("=" * 60)
    print(" STEPS 7-8: Subgoal Agent & Motion Agent Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Step 7: Subgoal Agent
    runner7 = test_step7_subgoal_agent()
    if not runner7.print_summary():
        all_passed = False
    
    # Step 8: Motion Agent
    runner8 = test_step8_motion_agent()
    if not runner8.print_summary():
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    total_run = runner7.tests_run + runner8.tests_run
    total_passed = runner7.tests_passed + runner8.tests_passed
    
    print(f"Step 7 (Subgoal Agent/DDPG): {runner7.tests_passed}/{runner7.tests_run}")
    print(f"Step 8 (Motion Agent/TD3):   {runner8.tests_passed}/{runner8.tests_run}")
    print(f"─" * 40)
    print(f"TOTAL: {total_passed}/{total_run}")
    
    if all_passed:
        print("\n🎉 ALL STEPS 7-8 TESTS PASSED!")
        print("\nAgents implemented:")
        print("  ✓ Subgoal Agent (DDPG) - attention-based collision avoidance")
        print("  ✓ Motion Agent (TD3) - smooth velocity control")
        print("\nReady to proceed to Step 9: Hierarchical Environment")
    else:
        print("\n❌ Some tests failed. Please fix issues before proceeding.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

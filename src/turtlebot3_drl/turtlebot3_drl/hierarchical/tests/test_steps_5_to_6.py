"""
Comprehensive Tests for Steps 5-6: LiDAR Preprocessor and Attention Module

Test Coverage:
- Step 5: LiDAR Preprocessor
  - Min-pooling from 360 to 80 rays
  - Normalization to [0, 1]
  - Sector conversion (10 sectors × 8 rays)
  - Edge cases handling (inf values, zeros)

- Step 6: Attention Module
  - Embedding network dimensions
  - Feature network dimensions
  - Score network dimensions
  - Softmax attention (sums to 1)
  - Output dimension verification
  - Gradient flow through attention
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
    print("⚠ PyTorch not available - skipping neural network tests")


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
                print(f"  - {name}: {reason}")
        
        if self.tests_failed == 0:
            print("\n✅ ALL TESTS PASSED!")
        else:
            print(f"\n❌ {self.tests_failed} test(s) failed")
        
        print("=" * 60)
        return self.tests_failed == 0


def test_step5_lidar_preprocessor():
    """Test Step 5: LiDAR Preprocessor."""
    print("\n" + "=" * 60)
    print("STEP 5: LiDAR Preprocessor Tests")
    print("=" * 60)
    
    runner = TestRunner()
    
    from hierarchical.preprocessing.lidar_processor import LidarProcessor
    
    # Create processor
    processor = LidarProcessor(
        input_rays=360,
        output_rays=80,
        max_range=3.5
    )
    
    # Test 5.1: Min-pooling downsampling
    def test_min_pooling():
        # Create test data: 360 rays
        lidar_360 = np.ones(360) * 2.0
        # Set some rays to have smaller values
        lidar_360[0:10] = 0.5  # First sector should have min=0.5
        lidar_360[45:55] = 1.0  # Another sector
        
        result = processor.process(lidar_360)
        
        assert result.shape == (80,), f"Expected shape (80,), got {result.shape}"
        # First 2 output rays should be ~0.5 (min of first 9 rays)
        assert abs(result[0] - 0.5) < 0.1, f"First ray should be ~0.5, got {result[0]}"
    
    runner.run_test("5.1 Min-pooling (360→80)", test_min_pooling)
    
    # Test 5.2: Downsampling ratio
    def test_downsampling_ratio():
        pool_size = processor.pool_size
        assert pool_size == 4 or pool_size == 5, f"Pool size should be 4 or 5, got {pool_size}"
        
        # 360/80 = 4.5, so we should handle this properly
        lidar_360 = np.random.uniform(0.5, 3.0, 360)
        result = processor.process(lidar_360)
        
        assert result.shape == (80,), f"Output should be exactly 80 rays"
    
    runner.run_test("5.2 Downsampling ratio correct", test_downsampling_ratio)
    
    # Test 5.3: Normalization
    def test_normalization():
        lidar_360 = np.array([0.0] * 120 + [1.75] * 120 + [3.5] * 120)  # 360 rays
        normalized = processor.process_normalized(lidar_360)
        
        # After min-pooling, check the normalized values
        assert normalized.shape == (80,), "Output should be 80 rays"
        assert normalized.min() >= 0.0, f"Min should be >= 0, got {normalized.min()}"
        assert normalized.max() <= 1.0, f"Max should be <= 1, got {normalized.max()}"
    
    runner.run_test("5.3 Normalization to [0, 1]", test_normalization)
    
    # Test 5.4: Inf handling
    def test_inf_handling():
        lidar_360 = np.ones(360) * 2.0
        lidar_360[0:10] = float('inf')  # No obstacle detected
        
        result = processor.process(lidar_360)
        
        assert not np.isinf(result).any(), "Output should have no inf values"
        assert not np.isnan(result).any(), "Output should have no NaN values"
        # Inf should be replaced with max_range (clamped to clip_range)
        assert result.max() <= processor.clip_range, f"Max value should be <= clip_range, got {result.max()}"
    
    runner.run_test("5.4 Inf handling", test_inf_handling)
    
    # Test 5.5: Zero/close obstacle handling
    def test_close_obstacles():
        lidar_360 = np.zeros(360)  # All obstacles at robot
        result = processor.process(lidar_360)
        
        assert result.min() >= 0.0, f"Min should be >= 0, got {result.min()}"
        assert not np.isnan(result).any(), "Should handle zeros without NaN"
    
    runner.run_test("5.5 Close obstacle handling", test_close_obstacles)
    
    # Test 5.6: Sector conversion
    def test_sector_conversion():
        lidar_80 = np.random.uniform(0.1, 1.0, 80)
        sectors = processor.to_sectors(lidar_80)
        
        assert sectors.shape == (10, 8), f"Expected (10, 8), got {sectors.shape}"
        
        # Verify data integrity
        flat = sectors.flatten()
        assert np.allclose(flat, lidar_80), "Sector data should match original"
    
    runner.run_test("5.6 Sector conversion (10×8)", test_sector_conversion)
    
    # Test 5.7: Full pipeline
    def test_full_pipeline():
        lidar_360 = np.random.uniform(0.2, 3.0, 360)
        lidar_360[100:120] = float('inf')  # Some inf values
        
        # Use process_normalized to get [0, 1] range
        result = processor.process_normalized(lidar_360)
        
        assert result.shape == (80,), f"Expected (80,), got {result.shape}"
        assert result.min() >= 0.0, f"Min should be >= 0, got {result.min()}"
        assert result.max() <= 1.0, f"Max should be <= 1, got {result.max()}"
        assert not np.isnan(result).any(), "No NaN allowed"
        assert not np.isinf(result).any(), "No inf allowed"
    
    runner.run_test("5.7 Full pipeline (normalized)", test_full_pipeline)
    
    # Test 5.8: Reproducibility
    def test_reproducibility():
        lidar_360 = np.random.uniform(0.5, 2.5, 360)
        
        result1 = processor.process(lidar_360.copy())
        result2 = processor.process(lidar_360.copy())
        
        assert np.allclose(result1, result2), "Processing should be deterministic"
    
    runner.run_test("5.8 Reproducibility", test_reproducibility)
    
    if TORCH_AVAILABLE:
        from hierarchical.preprocessing.lidar_processor import LidarProcessorTorch
        
        # Test 5.9: PyTorch processor
        def test_torch_processor():
            torch_processor = LidarProcessorTorch()
            
            lidar_batch = torch.rand(4, 360) * 3.5
            result = torch_processor.process(lidar_batch)
            
            assert result.shape == (4, 80), f"Expected (4, 80), got {result.shape}"
            # Values should be in [0, clip_range]
            assert (result >= 0).all(), f"Values should be >= 0"
            assert (result <= torch_processor.clip_range).all(), f"Values should be <= clip_range"
        
        runner.run_test("5.9 PyTorch processor", test_torch_processor)
        
        # Test 5.10: Torch-NumPy consistency
        def test_torch_numpy_consistency():
            torch_processor = LidarProcessorTorch()
            
            lidar_np = np.random.uniform(0.5, 2.5, 360)
            lidar_torch = torch.from_numpy(lidar_np).unsqueeze(0).float()
            
            result_np = processor.process(lidar_np)
            result_torch = torch_processor.process(lidar_torch)
            
            # Results should be similar (not exact due to implementation differences)
            diff = np.abs(result_np - result_torch[0].numpy())
            assert diff.mean() < 0.1, f"NumPy and PyTorch results should be similar, diff={diff.mean()}"
        
        runner.run_test("5.10 NumPy-PyTorch consistency", test_torch_numpy_consistency)
    
    return runner


def test_step6_attention_module():
    """Test Step 6: Attention Module."""
    print("\n" + "=" * 60)
    print("STEP 6: Attention Module Tests")
    print("=" * 60)
    
    runner = TestRunner()
    
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available - skipping attention tests")
        return runner
    
    from hierarchical.preprocessing.attention import (
        LidarAttention,
        LidarAttentionEfficient,
        PathModule,
        CombinedFeatureExtractor,
        make_mlp
    )
    
    # Test 6.1: make_mlp helper
    def test_make_mlp():
        mlp = make_mlp([10, 32, 16])
        
        # Should have 2 linear layers and 1 activation
        assert len(mlp) == 3, f"Expected 3 layers (lin, relu, lin), got {len(mlp)}"
        assert isinstance(mlp[0], nn.Linear), "First layer should be Linear"
        assert isinstance(mlp[1], nn.ReLU), "Second should be ReLU"
        assert isinstance(mlp[2], nn.Linear), "Third should be Linear"
        
        x = torch.rand(4, 10)
        y = mlp(x)
        assert y.shape == (4, 16), f"Expected (4, 16), got {y.shape}"
    
    runner.run_test("6.1 make_mlp helper", test_make_mlp)
    
    # Test 6.2: LidarAttention output shape
    def test_attention_output_shape():
        attention = LidarAttention(
            num_sectors=10,
            rays_per_sector=8,
            feature_dim=64
        )
        
        lidar = torch.rand(4, 80)
        output, weights = attention(lidar)
        
        assert output.shape == (4, 64), f"Expected output (4, 64), got {output.shape}"
        assert weights.shape == (4, 10), f"Expected weights (4, 10), got {weights.shape}"
    
    runner.run_test("6.2 Attention output shape", test_attention_output_shape)
    
    # Test 6.3: Attention weights sum to 1
    def test_attention_softmax():
        attention = LidarAttention()
        
        lidar = torch.rand(8, 80)
        _, weights = attention(lidar)
        
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5), \
            f"Attention should sum to 1, got sums: {sums}"
    
    runner.run_test("6.3 Softmax sums to 1", test_attention_softmax)
    
    # Test 6.4: Attention weights in [0, 1]
    def test_attention_range():
        attention = LidarAttention()
        
        lidar = torch.rand(4, 80)
        _, weights = attention(lidar)
        
        assert (weights >= 0).all(), f"Weights should be >= 0, min: {weights.min()}"
        assert (weights <= 1).all(), f"Weights should be <= 1, max: {weights.max()}"
    
    runner.run_test("6.4 Attention in [0, 1]", test_attention_range)
    
    # Test 6.5: Embedding network architecture
    def test_embedding_architecture():
        attention = LidarAttention()
        
        # Check embedding network layers
        embed_layers = list(attention.embedding_net.children())
        
        # Should be: Linear(8→512), ReLU, Linear(512→256), ReLU, Linear(256→128)
        assert len(embed_layers) == 5, f"Expected 5 layers in embedding, got {len(embed_layers)}"
        
        # Check dimensions
        first_linear = embed_layers[0]
        assert first_linear.in_features == 8, f"Input should be 8, got {first_linear.in_features}"
        assert first_linear.out_features == 512, f"First hidden should be 512, got {first_linear.out_features}"
    
    runner.run_test("6.5 Embedding architecture [8→512→256→128]", test_embedding_architecture)
    
    # Test 6.6: Feature network architecture
    def test_feature_architecture():
        attention = LidarAttention()
        
        feat_layers = list(attention.feature_net.children())
        
        # Should be: Linear(128→256), ReLU, Linear(256→128), ReLU, Linear(128→64)
        assert len(feat_layers) == 5, f"Expected 5 layers in feature net, got {len(feat_layers)}"
        
        last_linear = [l for l in feat_layers if isinstance(l, nn.Linear)][-1]
        assert last_linear.out_features == 64, f"Output should be 64, got {last_linear.out_features}"
    
    runner.run_test("6.6 Feature architecture [128→256→128→64]", test_feature_architecture)
    
    # Test 6.7: Score network architecture
    def test_score_architecture():
        attention = LidarAttention()
        
        score_layers = list(attention.score_net.children())
        
        # Should be: Linear(128→128), ReLU, Linear(128→64), ReLU, Linear(64→1)
        linear_layers = [l for l in score_layers if isinstance(l, nn.Linear)]
        last_linear = linear_layers[-1]
        assert last_linear.out_features == 1, f"Score output should be 1, got {last_linear.out_features}"
    
    runner.run_test("6.7 Score architecture [128→128→64→1]", test_score_architecture)
    
    # Test 6.8: Efficient attention equivalence
    def test_efficient_attention():
        efficient = LidarAttentionEfficient()
        
        lidar = torch.rand(4, 80)
        output, weights = efficient(lidar)
        
        assert output.shape == (4, 64), f"Expected (4, 64), got {output.shape}"
        assert weights.shape == (4, 10), f"Expected (4, 10), got {weights.shape}"
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5), \
            "Efficient attention should also sum to 1"
    
    runner.run_test("6.8 Efficient attention", test_efficient_attention)
    
    # Test 6.9: PathModule
    def test_path_module():
        path_module = PathModule(input_dim=10, output_dim=32)
        
        waypoints = torch.rand(4, 10)  # 5 waypoints × 2 coords
        output = path_module(waypoints)
        
        assert output.shape == (4, 32), f"Expected (4, 32), got {output.shape}"
    
    runner.run_test("6.9 PathModule", test_path_module)
    
    # Test 6.10: CombinedFeatureExtractor
    def test_combined_extractor():
        extractor = CombinedFeatureExtractor(
            lidar_rays=80,
            num_waypoints=5,
            lidar_feature_dim=64,
            path_feature_dim=32
        )
        
        lidar = torch.rand(4, 80)
        waypoints = torch.rand(4, 10)
        
        combined, attention = extractor(lidar, waypoints, return_attention=True)
        
        assert combined.shape == (4, 96), f"Expected (4, 96), got {combined.shape}"
        assert extractor.output_dim == 96, f"Output dim should be 96"
    
    runner.run_test("6.10 CombinedFeatureExtractor", test_combined_extractor)
    
    # Test 6.11: Gradient flow
    def test_gradient_flow():
        attention = LidarAttention()
        
        lidar = torch.rand(2, 80, requires_grad=True)
        output, _ = attention(lidar)
        
        loss = output.sum()
        loss.backward()
        
        assert lidar.grad is not None, "Gradient should flow to input"
        assert not torch.isnan(lidar.grad).any(), "Gradients should not be NaN"
        assert lidar.grad.abs().sum() > 0, "Gradients should be non-zero"
    
    runner.run_test("6.11 Gradient flow", test_gradient_flow)
    
    # Test 6.12: Combined gradient flow
    def test_combined_gradient_flow():
        extractor = CombinedFeatureExtractor()
        
        lidar = torch.rand(2, 80, requires_grad=True)
        waypoints = torch.rand(2, 10, requires_grad=True)
        
        combined, _ = extractor(lidar, waypoints)
        loss = combined.sum()
        loss.backward()
        
        assert lidar.grad is not None, "Gradient should flow to LiDAR input"
        assert waypoints.grad is not None, "Gradient should flow to waypoints"
    
    runner.run_test("6.12 Combined gradient flow", test_combined_gradient_flow)
    
    # Test 6.13: No NaN or Inf in outputs
    def test_numerical_stability():
        attention = LidarAttention()
        
        # Test with various inputs
        test_inputs = [
            torch.zeros(2, 80),       # All zeros
            torch.ones(2, 80),        # All ones
            torch.rand(2, 80) * 100,  # Large values
            torch.rand(2, 80) * 0.01, # Small values
        ]
        
        for i, lidar in enumerate(test_inputs):
            output, weights = attention(lidar)
            assert not torch.isnan(output).any(), f"Test {i}: Output has NaN"
            assert not torch.isinf(output).any(), f"Test {i}: Output has Inf"
            assert not torch.isnan(weights).any(), f"Test {i}: Weights have NaN"
    
    runner.run_test("6.13 Numerical stability", test_numerical_stability)
    
    # Test 6.14: Batch independence
    def test_batch_independence():
        attention = LidarAttention()
        
        lidar1 = torch.rand(1, 80)
        lidar2 = torch.rand(1, 80)
        lidar_batch = torch.cat([lidar1, lidar2], dim=0)
        
        # Process individually
        out1, _ = attention(lidar1)
        out2, _ = attention(lidar2)
        
        # Process as batch
        out_batch, _ = attention(lidar_batch)
        
        assert torch.allclose(out_batch[0], out1[0], atol=1e-5), \
            "Batch processing should match individual"
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5), \
            "Batch processing should match individual"
    
    runner.run_test("6.14 Batch independence", test_batch_independence)
    
    # Test 6.15: Parameter count reasonable
    def test_parameter_count():
        attention = LidarAttention()
        
        total_params = sum(p.numel() for p in attention.parameters())
        
        # Paper doesn't specify exact count, but should be reasonable
        # Estimate: ~300-500K parameters for the attention module
        assert 100_000 < total_params < 1_000_000, \
            f"Parameter count should be reasonable, got {total_params:,}"
        
        print(f"    (Parameter count: {total_params:,})")
    
    runner.run_test("6.15 Parameter count", test_parameter_count)
    
    return runner


def test_integration():
    """Test integration between LiDAR processor and attention."""
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS")
    print("=" * 60)
    
    runner = TestRunner()
    
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available - skipping integration tests")
        return runner
    
    from hierarchical.preprocessing import (
        LidarProcessor,
        LidarProcessorTorch,
        LidarAttention,
        CombinedFeatureExtractor
    )
    
    # Test I.1: Full LiDAR pipeline
    def test_full_lidar_pipeline():
        # Raw LiDAR → Preprocessor → Attention
        processor = LidarProcessor()
        attention = LidarAttention()
        
        # Simulate raw LiDAR (360 rays, some inf)
        lidar_raw = np.random.uniform(0.3, 3.5, 360)
        lidar_raw[50:70] = float('inf')
        
        # Preprocess
        lidar_processed = processor.process(lidar_raw)
        
        # Convert to tensor and add batch dim
        lidar_tensor = torch.from_numpy(lidar_processed).unsqueeze(0).float()
        
        # Attention
        features, weights = attention(lidar_tensor)
        
        assert features.shape == (1, 64), f"Expected (1, 64), got {features.shape}"
        assert not torch.isnan(features).any(), "Features should not have NaN"
    
    runner.run_test("I.1 Full LiDAR pipeline", test_full_lidar_pipeline)
    
    # Test I.2: Batch processing pipeline
    def test_batch_pipeline():
        processor = LidarProcessorTorch()
        attention = LidarAttention()
        
        # Batch of raw LiDAR
        lidar_raw = torch.rand(8, 360) * 3.5
        
        # Process
        lidar_processed = processor.process(lidar_raw)
        features, weights = attention(lidar_processed)
        
        assert features.shape == (8, 64), f"Expected (8, 64), got {features.shape}"
        assert weights.shape == (8, 10), f"Expected (8, 10), got {weights.shape}"
    
    runner.run_test("I.2 Batch processing pipeline", test_batch_pipeline)
    
    # Test I.3: Combined with waypoints
    def test_combined_pipeline():
        processor = LidarProcessorTorch()
        extractor = CombinedFeatureExtractor()
        
        # Inputs
        lidar_raw = torch.rand(4, 360) * 3.5
        waypoints = torch.rand(4, 10)  # 5 waypoints × 2 coords
        
        # Process LiDAR
        lidar_processed = processor.process(lidar_raw)
        
        # Combined features
        combined, attention = extractor(lidar_processed, waypoints, return_attention=True)
        
        assert combined.shape == (4, 96), f"Expected (4, 96), got {combined.shape}"
    
    runner.run_test("I.3 Combined with waypoints", test_combined_pipeline)
    
    # Test I.4: Sector attention visualization
    def test_attention_visualization():
        processor = LidarProcessor()
        attention = LidarAttention()
        
        # Create scan with obstacle in front
        lidar_raw = np.ones(360) * 3.5  # No obstacles
        lidar_raw[170:190] = 0.5  # Obstacle in front (sector ~4-5)
        
        lidar_processed = processor.process(lidar_raw)
        lidar_tensor = torch.from_numpy(lidar_processed).unsqueeze(0).float()
        
        _, weights = attention(lidar_tensor)
        
        # Just verify we can get weights
        assert weights.shape == (1, 10), "Should have weights for 10 sectors"
        # Note: We can't guarantee which sector gets most attention without training
    
    runner.run_test("I.4 Attention visualization", test_attention_visualization)
    
    return runner


def main():
    """Run all tests."""
    print("=" * 60)
    print(" STEPS 5-6: LiDAR Preprocessor & Attention Module Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Step 5: LiDAR Preprocessor
    runner5 = test_step5_lidar_preprocessor()
    if not runner5.print_summary():
        all_passed = False
    
    # Step 6: Attention Module
    runner6 = test_step6_attention_module()
    if not runner6.print_summary():
        all_passed = False
    
    # Integration tests
    runner_int = test_integration()
    if not runner_int.print_summary():
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    total_run = runner5.tests_run + runner6.tests_run + runner_int.tests_run
    total_passed = runner5.tests_passed + runner6.tests_passed + runner_int.tests_passed
    
    print(f"Step 5 (LiDAR Preprocessor): {runner5.tests_passed}/{runner5.tests_run}")
    print(f"Step 6 (Attention Module):   {runner6.tests_passed}/{runner6.tests_run}")
    print(f"Integration Tests:           {runner_int.tests_passed}/{runner_int.tests_run}")
    print(f"─" * 40)
    print(f"TOTAL: {total_passed}/{total_run}")
    
    if all_passed:
        print("\n🎉 ALL STEPS 5-6 TESTS PASSED!")
        print("\nReady to proceed to Step 7: Subgoal Agent (DDPG)")
    else:
        print("\n❌ Some tests failed. Please fix issues before proceeding.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

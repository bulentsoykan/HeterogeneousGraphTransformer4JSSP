"""
Test script to verify installation and basic functionality.

Run with: python tests/test_installation.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False

    try:
        import torch_geometric
        print(f"  ✓ PyTorch Geometric {torch_geometric.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch Geometric import failed: {e}")
        return False

    try:
        import gymnasium as gym
        print(f"  ✓ Gymnasium {gym.__version__}")
    except ImportError as e:
        print(f"  ✗ Gymnasium import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False

    try:
        from src.JSSP_Env import JSSPEnv
        print("  ✓ JSSP Environment")
    except ImportError as e:
        print(f"  ✗ JSSP Environment import failed: {e}")
        return False

    try:
        from src.Policy import HGTPolicy
        print("  ✓ HGT Policy")
    except ImportError as e:
        print(f"  ✗ HGT Policy import failed: {e}")
        return False

    try:
        from src.PPO import PPOTrainer
        print("  ✓ PPO Trainer")
    except ImportError as e:
        print(f"  ✗ PPO Trainer import failed: {e}")
        return False

    try:
        from benchmarks import get_benchmark_instance
        print("  ✓ Benchmark instances")
    except ImportError as e:
        print(f"  ✗ Benchmark instances import failed: {e}")
        return False

    return True


def test_environment():
    """Test JSSP environment creation and basic operation."""
    print("\nTesting JSSP Environment...")

    try:
        from src.JSSP_Env import JSSPEnv
        import numpy as np

        # Create simple environment
        env = JSSPEnv(num_jobs=3, num_machines=3)
        print("  ✓ Environment created")

        # Reset
        state, info = env.reset()
        print("  ✓ Environment reset")

        # Check state structure
        assert 'op' in state.node_types
        assert ('op', 'precedes', 'op') in state.edge_types
        assert ('op', 'competes', 'op') in state.edge_types
        print("  ✓ State structure correct")

        # Take action
        action_mask = state['op'].action_mask.numpy()
        valid_actions = np.where(action_mask > 0)[0]
        action = valid_actions[0]

        next_state, reward, terminated, truncated, info = env.step(action)
        print("  ✓ Step execution successful")

        return True

    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy():
    """Test HGT policy creation and forward pass."""
    print("\nTesting HGT Policy...")

    try:
        import torch
        from src.JSSP_Env import JSSPEnv
        from src.Policy import create_hgt_policy

        # Create environment and get state
        env = JSSPEnv(num_jobs=3, num_machines=3)
        state, _ = env.reset()
        print("  ✓ Environment initialized")

        # Create policy
        policy = create_hgt_policy(
            node_feature_dim=3,
            hidden_dim=64,
            embedding_dim=32,
            num_hgt_layers=2,
            num_heads=2,
        )
        print("  ✓ Policy created")

        # Forward pass
        policy.eval()
        with torch.no_grad():
            action, log_prob, entropy, value = policy.get_action_and_value(state)

        print("  ✓ Forward pass successful")
        print(f"    - Action: {action.item()}")
        print(f"    - Value: {value.item():.4f}")

        return True

    except Exception as e:
        print(f"  ✗ Policy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo():
    """Test PPO trainer."""
    print("\nTesting PPO Trainer...")

    try:
        import torch
        from src.JSSP_Env import JSSPEnv
        from src.Policy import create_hgt_policy
        from src.PPO import PPOTrainer, PPOConfig

        # Create environment
        env = JSSPEnv(num_jobs=3, num_machines=3)
        print("  ✓ Environment initialized")

        # Create policy
        policy = create_hgt_policy(
            hidden_dim=32,
            embedding_dim=16,
            num_hgt_layers=2,
        )
        print("  ✓ Policy created")

        # Create trainer
        config = PPOConfig(batch_size=16)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = PPOTrainer(policy=policy, config=config, device=device)
        print(f"  ✓ Trainer created (device: {device})")

        # Collect rollout
        rollout_stats = trainer.collect_rollout(env, num_episodes=2)
        print("  ✓ Rollout collection successful")
        print(f"    - Mean makespan: {rollout_stats['mean_makespan']:.2f}")

        # Update
        update_stats = trainer.update()
        print("  ✓ Policy update successful")
        print(f"    - Policy loss: {update_stats['policy_loss']:.4f}")

        return True

    except Exception as e:
        print(f"  ✗ PPO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmarks():
    """Test benchmark instances."""
    print("\nTesting Benchmark Instances...")

    try:
        from benchmarks import get_benchmark_instance, FT06, FT10, FT20

        # Test FT06
        instance = get_benchmark_instance('FT06')
        assert instance['num_jobs'] == 6
        assert instance['num_machines'] == 6
        assert instance['optimal_makespan'] == 55
        print("  ✓ FT06 loaded correctly")

        # Test FT10
        instance = get_benchmark_instance('FT10')
        assert instance['num_jobs'] == 10
        assert instance['num_machines'] == 10
        assert instance['optimal_makespan'] == 930
        print("  ✓ FT10 loaded correctly")

        # Test FT20
        instance = get_benchmark_instance('FT20')
        assert instance['num_jobs'] == 20
        assert instance['num_machines'] == 5
        assert instance['optimal_makespan'] == 1165
        print("  ✓ FT20 loaded correctly")

        return True

    except Exception as e:
        print(f"  ✗ Benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("HGT-Scheduler Installation Test")
    print("=" * 80)

    all_passed = True

    # Run tests
    all_passed &= test_imports()
    all_passed &= test_environment()
    all_passed &= test_policy()
    all_passed &= test_ppo()
    all_passed &= test_benchmarks()

    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests passed! Installation is successful.")
        print("\nYou can now:")
        print("  1. Run example: python example_usage.py")
        print("  2. Start training: python train.py --instance FT06 --total_timesteps 100000")
        print("  3. Read documentation: cat README.md")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("  1. Verify all dependencies are installed: pip install -r requirements.txt")
        print("  2. Check PyTorch installation: python -c 'import torch; print(torch.__version__)'")
        print("  3. Check PyTorch Geometric: python -c 'import torch_geometric; print(torch_geometric.__version__)'")

    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())

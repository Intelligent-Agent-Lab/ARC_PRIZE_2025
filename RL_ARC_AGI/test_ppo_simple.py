#!/usr/bin/env python3
"""
Simple test script for PPO training on ArcAgiGrid environment.
This script runs a minimal training loop to verify everything works.
"""

import os
import sys
import random
import numpy as np
import torch

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arc_agi_grid_env import ArcAgiGridEnv
from env_wrappers import create_wrapped_env
from ppo_agent import PPOAgent


def test_environment_creation():
    """Test that the environment can be created and wrapped properly."""
    print("Testing environment creation...")
    
    try:
        # Create base environment
        base_env = ArcAgiGridEnv(
            training_challenges_json='../datasets/arc-agi_training_challenges.json',
            training_solutions_json='../datasets/arc-agi_training_solutions.json',
            evaluation_challenges_json='../datasets/arc-agi_evaluation_challenges.json',
            evaluation_solutions_json='../datasets/arc-agi_evaluation_solutions.json',
            test_challenges_json=None
        )
        
        # Wrap environment
        env = create_wrapped_env(base_env, normalize=True)
        
        print(f"✓ Environment created successfully!")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset(
            seed=42,
            options={'mode': 'train', 'task_id': None, 'reset_sol_grid': 'padding'}
        )
        print(f"✓ Environment reset successful!")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful!")
        print(f"  Action: {action}, Reward: {reward}")
        
        return env
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return None


def test_agent_creation(env):
    """Test that the PPO agent can be created."""
    print("\nTesting agent creation...")
    
    try:
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        agent = PPOAgent(
            input_size=obs_size,
            action_size=action_size,
            hidden_size=256,  # Smaller for testing
            learning_rate=3e-4
        )
        
        print(f"✓ PPO Agent created successfully!")
        print(f"  Device: {agent.device}")
        print(f"  Input size: {obs_size}")
        print(f"  Action size: {action_size}")
        
        return agent
        
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return None


def test_action_selection(env, agent):
    """Test agent action selection."""
    print("\nTesting action selection...")
    
    try:
        obs, info = env.reset(
            seed=42,
            options={'mode': 'train', 'task_id': None, 'reset_sol_grid': 'padding'}
        )
        
        # Test action selection
        action, log_prob, value = agent.select_action(obs)
        print(f"✓ Action selection successful!")
        print(f"  Action: {action}, Log prob: {log_prob:.3f}, Value: {value:.3f}")
        
        # Test multiple steps
        for step in range(5):
            action, log_prob, value = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(obs, action, log_prob, reward, value, terminated or truncated)
            
            print(f"  Step {step+1}: action={action}, reward={reward:.3f}")
            
            if terminated or truncated:
                break
        
        print("✓ Multi-step execution successful!")
        return True
        
    except Exception as e:
        print(f"✗ Action selection failed: {e}")
        return False


def test_training_update(agent):
    """Test a single training update."""
    print("\nTesting training update...")
    
    try:
        if len(agent.observations) == 0:
            print("  No data stored, skipping update test")
            return True
        
        # Perform update
        metrics = agent.update(next_value=0.0, ppo_epochs=1, mini_batch_size=4)
        
        print(f"✓ Training update successful!")
        if metrics:
            print(f"  Metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training update failed: {e}")
        return False


def run_mini_training(env, agent, steps=50):
    """Run a very short training loop."""
    print(f"\nRunning mini training loop ({steps} steps)...")
    
    try:
        obs, info = env.reset(
            seed=42,
            options={'mode': 'train', 'task_id': None, 'reset_sol_grid': 'padding'}
        )
        
        total_reward = 0
        episode_count = 0
        
        for step in range(steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, log_prob, reward, value, done)
            total_reward += reward
            obs = next_obs
            
            if done:
                episode_count += 1
                print(f"  Episode {episode_count} completed, reward: {total_reward:.3f}")
                obs, info = env.reset(
                    seed=random.randint(0, 1000),
                    options={'mode': 'train', 'task_id': None, 'reset_sol_grid': 'padding'}
                )
                total_reward = 0
        
        # Perform update
        metrics = agent.update(next_value=0.0, ppo_epochs=2, mini_batch_size=8)
        
        print(f"✓ Mini training completed!")
        print(f"  Episodes completed: {episode_count}")
        if metrics:
            print(f"  Final metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mini training failed: {e}")
        return False


def main():
    print("=== PPO ArcAgiGrid Test Suite ===\n")
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test environment
    env = test_environment_creation()
    if env is None:
        print("\n✗ Environment test failed. Exiting.")
        return
    
    # Test agent
    agent = test_agent_creation(env)
    if agent is None:
        print("\n✗ Agent test failed. Exiting.")
        return
    
    # Test action selection
    if not test_action_selection(env, agent):
        print("\n✗ Action selection test failed. Exiting.")
        return
    
    # Test training update
    if not test_training_update(agent):
        print("\n✗ Training update test failed. Exiting.")
        return
    
    # Run mini training
    if not run_mini_training(env, agent, steps=100):
        print("\n✗ Mini training test failed. Exiting.")
        return
    
    print("\n🎉 All tests passed! PPO is ready for full training.")
    print("\nTo start full training, run:")
    print("  python train_ppo.py --num_updates 100 --rollout_steps 512")


if __name__ == '__main__':
    main()
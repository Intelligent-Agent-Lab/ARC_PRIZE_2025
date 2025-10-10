import gymnasium as gym
import sys
import os

# Add current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the registration
import __init__

def test_environment_registration():
    """Test that the ArcAgiGrid environment can be created successfully."""
    try:
        # Create the environment
        env = gym.make('ArcAgiGrid-v0')
        print("✓ Environment 'ArcAgiGrid-v0' registered successfully!")
        
        # Test basic functionality
        observation, info = env.reset(seed=42)
        print("✓ Environment reset successfully!")
        print(f"Observation keys: {observation.keys()}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test one step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step executed! Reward: {reward}")
        
        env.close()
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_environment_registration()
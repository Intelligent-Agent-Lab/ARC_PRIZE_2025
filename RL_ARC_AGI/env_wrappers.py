import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple


class SequenceObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that extracts only the 'current_grid_seq' from dict observations.
    Converts the dict observation space to a Box observation space.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Check if the environment has dict observation space
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("Environment must have Dict observation space")
        
        if 'current_grid_seq' not in env.observation_space.spaces:
            raise ValueError("Environment observation must contain 'current_grid_seq' key")
        
        # Set new observation space to just the sequence
        self.observation_space = env.observation_space.spaces['current_grid_seq']
    
    def observation(self, observation: Dict[str, Any]) -> np.ndarray:
        """Extract current_grid_seq from dict observation."""
        return observation['current_grid_seq']


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that normalizes observations to [0, 1] range.
    Useful for neural networks that expect normalized inputs.
    """
    
    def __init__(self, env, max_value: float = 10.0):
        super().__init__(env)
        self.max_value = max_value
        
        # Update observation space bounds
        if isinstance(self.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=self.observation_space.shape,
                dtype=np.float32
            )
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize observation to [0, 1] range."""
        return observation.astype(np.float32) / self.max_value


class RewardShapingWrapper(gym.RewardWrapper):
    """
    Wrapper that applies reward shaping for better learning dynamics.
    """
    
    def __init__(self, env, correct_step_reward: float = 0.01, completion_bonus: float = 1.0):
        super().__init__(env)
        self.correct_step_reward = correct_step_reward
        self.completion_bonus = completion_bonus
    
    def reward(self, reward: float) -> float:
        """Apply reward shaping."""
        # The original environment already provides:
        # - 0.01 for correct steps
        # - -1 for incorrect steps (termination)
        # - 1.01 for successful completion
        
        # We can add additional shaping here if needed
        return reward


class ActionMaskingWrapper(gym.Wrapper):
    """
    Wrapper that can mask invalid actions (optional for future use).
    Currently, all actions (0-10) are valid for ArcAgiGrid.
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def get_action_mask(self) -> np.ndarray:
        """Return mask of valid actions (all True for ArcAgiGrid)."""
        return np.ones(self.action_space.n, dtype=bool)


def create_wrapped_env(base_env, normalize: bool = False, reward_shaping: bool = False) -> gym.Env:
    """
    Factory function to create a properly wrapped ArcAgiGrid environment.
    
    Args:
        base_env: The base ArcAgiGrid environment
        normalize: Whether to normalize observations to [0, 1]
        reward_shaping: Whether to apply additional reward shaping
    
    Returns:
        Wrapped environment ready for PPO training
    """
    env = base_env
    
    # Extract sequence observations
    # env = SequenceObservationWrapper(env)
    
    # Normalize observations if requested
    if normalize:
        env = NormalizeObservationWrapper(env, max_value=10.0)
    
    # Apply reward shaping if requested
    if reward_shaping:
        env = RewardShapingWrapper(env)
    
    return env
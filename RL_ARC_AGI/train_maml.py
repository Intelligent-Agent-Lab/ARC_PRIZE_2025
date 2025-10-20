# %%
import os
import sys
import random
import time
import math
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import deque
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from network.mlp import ActorCritic_MLP
from network.vit import ActorCritic_ViT
from env.arc_agi_grid_env_coord import cmap
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from env.meta_preprocess import generate_meta_dataset
from env.meta_arc_agi_grid_env_coord import SelectedMetaArcAgiGridEnv, \
                                    load_challenges_and_solutions, \
                                        make_meta_task_env
from env.utils import convert_int_to_dict, vectorized_convert_int_to_dict
from matplotlib import colors
from pathlib import Path

# %%
config = OmegaConf.load('./config/maml_ppo.yaml')
config

num_steps = config.environment.num_steps
total_timesteps = config.training.total_timesteps
ppo_epochs = config.training.ppo_epochs


# %%
# %%

class Agent(nn.Module):
    """Vision Transformer based Actor-Critic network for PPO with grid observations."""
    
    def __init__(self, cfg,):
        super().__init__()
        if cfg.network.type == 'vit':
            self.ac_network = ActorCritic_ViT()
    
    def forward(self, x):
        """Forward pass returning CLS token representation."""
        # Patch embedding
        action_logits, state_value = self.ac_network(x)
        return action_logits, state_value 
       
    def get_value(self, x):
        _, state_value = self.forward(x)
        return state_value

    def get_action_and_value(self, x, action=None, obs_info=None):
        action_logits, state_value = self.forward(x)
        
        # Apply action masking if obs_info is provided
        if obs_info is not None:
            action_mask = self._create_action_mask(x, obs_info)
            # Set invalid actions to very large negative value
            action_logits = action_logits + action_mask
        # Note: During policy updates, obs_info is None (this is normal PPO behavior)
        
        probs = Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), state_value
    
device = 'cuda'
agent = Agent(config).to(device)
    
# %%
training_challenges_json = "../datasets/arc-agi_training_challenges.json"
training_solutions_json = "../datasets/arc-agi_training_solutions.json"
evaluation_challenges_json = "../datasets/arc-agi_evaluation_challenges.json"
evaluation_solutions_json = "../datasets/arc-agi_evaluation_solutions.json"
test_challenges_json = "../datasets/arc-agi_test_challenges.json"

training_challenges, training_solutions, evaluation_challenges, \
evaluation_solutions, test_challenges = load_challenges_and_solutions(
                                        training_challenges_json,
                                        training_solutions_json,
                                        evaluation_challenges_json,
                                        evaluation_solutions_json,
                                        test_challenges_json,
                                    )
# %%
from env.meta_preprocess import generate_meta_dataset
meta_train_dataset = generate_meta_dataset(training_challenges, training_solutions)
meta_eval_dataset = generate_meta_dataset(evaluation_challenges, evaluation_solutions)
meta_test_dataset = generate_meta_dataset(test_challenges, None)
# %%
from typing import Dict, Any, List

# %%
from env.meta_arc_agi_grid_env_coord import make_meta_task_env
# %%
print(len(meta_train_dataset['794b24be']['train_data']))
print(len(meta_train_dataset['794b24be']['test_data']))
num_adapt_envs = len(meta_train_dataset['794b24be']['train_data'])
num_eval_envs = len(meta_train_dataset['794b24be']['test_data'])

from tensordict import TensorDict
num_pairs = 90

# %%
def setup_buffer(num_pairs):
    """Setup buffer for rollout data."""
    obs = torch.zeros((num_steps, num_pairs) + (30, 120)).to(device)
    actions = torch.zeros((num_steps, num_pairs)).to(device)
    logprobs = torch.zeros((num_steps, num_pairs)).to(device)
    rewards = torch.zeros((num_steps, num_pairs)).to(device)
    dones = torch.zeros((num_steps, num_pairs)).to(device)
    values = torch.zeros((num_steps, num_pairs)).to(device)
    buffer = TensorDict({'obs': obs,
                        'actions': actions,
                        'logprobs': logprobs,
                        'rewards': rewards,
                        'dones': dones,
                        'values': values,
                        })
    return buffer
        
def rollout(num_pairs, 
            envs, 
            agent,
            buffer,
            next_obs,
            next_done,
            episode_returns,
            episode_lengths,
            ):
    for step in range(num_steps):
        buffer['obs'][step] = next_obs
        buffer['dones'][step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            # DDP 래퍼 자체를 직접 호출하는 것으로 변경
            # 이전처럼 내부 모듈을 직접 호출하면 forward 훅을 우회하는 작업임
            action_logits, value = agent(next_obs)
            buffer['values'][step] = value.flatten()
            # Sample action and get log probability
            probs = Categorical(logits=action_logits)
            action = probs.sample()
            logprob = probs.log_prob(action)

        buffer['actions'][step] = action
        buffer['logprobs'][step] = logprob

        # Execute actions in vectorized environment
        dict_action = vectorized_convert_int_to_dict(action.cpu())
        next_obs, reward, terminations, truncations, infos = envs.step(dict_action)
        
        # Store current infos for next step
        next_done = np.logical_or(terminations, truncations)
        buffer['rewards'][step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        # Update episode tracking
        episode_returns += reward
        episode_lengths += 1
        
        # Log episode statistics when episodes end
        for i in range(num_pairs):
            if terminations[i] or truncations[i]:
                # Episode ended - log stats
                # final_returns[i] = (episode_returns[i])
                # final_lengths[i] = (episode_lengths[i])

                # Reset tracking for this environment
                episode_returns[i] = 0.0
                episode_lengths[i] = 0
    
    # Bootstrap value for the next state
    with torch.no_grad():
        _, next_value = agent(next_obs)
        next_value = next_value.reshape(1, -1)
    return next_value

def adapt(mode, task_id, init_policy):
    adapt_train_envs = make_meta_task_env(meta_train_dataset,
                                    meta_eval_dataset,
                                    meta_test_dataset,
                                    mode=mode,
                                    phase='train',
                                    task_id=task_id,
                                    rand_init=False,
                                    )
    next_value = rollout(adapt_train_envs, init_policy)
    
def evaluation(mode, task_id, adapted_policy):
    eval_test_envs = make_meta_task_env(meta_train_dataset,
                                    meta_eval_dataset,
                                    meta_test_dataset,
                                    mode=mode,
                                    phase='test',
                                    task_id=task_id,
                                    rand_init=False,
                                    )
    next_value = rollout(eval_test_envs, adapted_policy)
    
# %%

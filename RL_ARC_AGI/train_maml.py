# %%
import os
import sys
import random
import time
import copy
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
from collections import deque, OrderedDict
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
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from env.meta_preprocess import generate_meta_dataset
from env.meta_arc_agi_grid_env_coord import SelectedMetaArcAgiGridEnv, \
                                    load_challenges_and_solutions, \
                                        make_meta_task_env
from env.utils import convert_int_to_dict, vectorized_convert_int_to_dict
from matplotlib import colors
from pathlib import Path
from network.vit import ViTPolicy, ViTValue

# %%
config = OmegaConf.load('./config/maml_ppo.yaml')
num_steps = 9
total_timesteps = config.training.total_timesteps
ppo_epochs = config.training.ppo_epochs
gamma = config.training.gamma
gae_lambda = config.training.gae_lambda
eps_clip = config.training.eps_clip
num_meta_iterations = 1000000
device = 'cuda'
# %%
class PolicyNetwork(nn.Module):
    """정책 네트워크: 상태를 입력받아 행동 분포를 출력"""
    def __init__(self,):
        super(PolicyNetwork, self).__init__()
        self.network = ViTPolicy(grid_size=(30, 120), patch_size=15, embed_dim=256, 
                 num_heads=4, num_layers=6, mlp_ratio=4, vocab_size=12, 
                 action_size=9000, dropout=0.1)
        
    def forward(self, obs):
        action_logit = self.network(obs)
        return action_logit
    
    def get_action(self, obs):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()  # Shape: [batch_size]
        log_prob = dist.log_prob(action)  # Shape: [batch_size]
        return action, log_prob

    def get_log_prob(self, obs, action):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        # Squeeze action if it has extra dimension
        if action.dim() > 1:
            action = action.squeeze(-1)
        log_prob = dist.log_prob(action)  # Shape: [batch_size]
        return log_prob
    
    def _extract_current_info(self, obs_info, i):
        """Helper to extract info for batch index i."""
        if obs_info is None:
            return None

        current_info = {}
        if isinstance(obs_info, dict):
            for key in ['active_shape', 'active_color']:
                if key in obs_info and len(obs_info[key]) > i:
                    val = obs_info[key][i]
                    if hasattr(val, 'tolist'):
                        current_info[key] = val.tolist()
                    elif isinstance(val, (list, tuple)):
                        current_info[key] = list(val)
                    else:
                        current_info[key] = val
        elif isinstance(obs_info, list) and len(obs_info) > i:
            current_info = obs_info[i]

        return current_info

    def _create_action_mask(self, obs, obs_info):
        """Create action mask based on observation and info (vectorized version)."""
        batch_size = obs.shape[0] if obs.dim() > 1 else 1
        if batch_size == 1:
            obs = obs.unsqueeze(0)

        action_mask = torch.zeros((batch_size, 9000), device=obs.device)

        if obs_info is None:
            print("Warning: obs_info is None, no action masking applied")
            return action_mask.squeeze(0) if batch_size == 1 else action_mask

        for i in range(batch_size):
            current_obs = obs[i]  # Shape: (30, 120)
            solution_area = current_obs[:, 90:]  # Shape: (30, 30)

            # Extract current info
            current_info = self._extract_current_info(obs_info, i)
            active_shape = np.array(current_info['active_shape'])
            width = np.sum(active_shape[0, :]).item()   # 첫 번째 행의 1 개수
            height = np.sum(active_shape[:, 0]).item()  # 첫 번째 열의 1 개수
            target_rows, target_cols = height, width
            
            active_color = np.array(current_info['active_color'])
            valid_colors = set(np.where(active_color == 1)[0].tolist())
                        
            # Create position and color masks vectorized
            if target_rows > 0 and target_cols > 0:
                # Valid positions (30x30 grid)
                valid_positions = torch.zeros(30, 30, dtype=torch.bool, device=obs.device)
                valid_positions[:target_rows, :target_cols] = (solution_area[:target_rows, :target_cols] == 11)

                # Create position indices grid (row * 30 + col)
                row_indices = torch.arange(30, device=obs.device).view(-1, 1).expand(30, 30)
                col_indices = torch.arange(30, device=obs.device).view(1, -1).expand(30, 30)
                position_indices = row_indices * 30 + col_indices

                # Vectorized masking for each color
                for color in range(10):
                    color_offset = color * 900
                    color_action_indices = color_offset + position_indices

                    if color not in valid_colors:
                        # Mask all positions for invalid colors
                        action_mask[i, color_action_indices.flatten()] = -1e9
                    else:
                        # Mask only invalid positions for valid colors
                        invalid_pos_mask = ~valid_positions
                        if invalid_pos_mask.any():
                            invalid_actions = color_action_indices[invalid_pos_mask]
                            action_mask[i, invalid_actions] = -1e9

            # Basic logging without excessive counters
            if hasattr(self, 'mask_log_counter'):
                self.mask_log_counter += 1
            else:
                self.mask_log_counter = 1
                
            # Action masking stats logging disabled for clean output
            # if self.mask_log_counter % 100 == 0:  # Log every 100 calls
            #     valid_actions = total_actions - masked_actions
            #     mask_ratio = masked_actions / total_actions
            #     print(f"Action Masking Stats: {valid_actions}/{total_actions} valid ({1-mask_ratio:.1%})")
            # except Exception as e:
            #     print(f"Warning: Could not create action mask for batch {i}: {e}")
            #     continue

        return action_mask.squeeze(0) if batch_size == 1 else action_mask

class ValueNetwork(nn.Module):
    """가치 네트워크: 상태 가치 함수 V(s)"""
    def __init__(self,):
        super(ValueNetwork, self).__init__()
        self.network = ViTValue(grid_size=(30, 120), patch_size=15, embed_dim=256, 
                 num_heads=4, num_layers=6, mlp_ratio=4, vocab_size=12, 
                 action_size=9000, dropout=0.1)
        
    def forward(self, obs):
        value = self.network(obs)
        return value

    
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
meta_train_dataset = generate_meta_dataset(training_challenges, training_solutions)
meta_eval_dataset = generate_meta_dataset(evaluation_challenges, evaluation_solutions)
meta_test_dataset = generate_meta_dataset(test_challenges, None)
# %%
from typing import Dict, Any, List
import torch.nn.utils as torch_utils

# %%
from tensordict import TensorDict

# %%
def functional_call(module, parameters_dict, x, method_name='forward'):
    """Functional call to a module with different parameters.
    This is more efficient than deepcopy for MAML."""
    old_params = {}
    for name, param in module.named_parameters():
        old_params[name] = param.data
        param.data = parameters_dict[name]

    # Call the method
    if method_name == 'forward':
        result = module(x)
    elif method_name == 'get_log_prob':
        result = module.get_log_prob(x[0], x[1])
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Restore old parameters
    for name, param in module.named_parameters():
        param.data = old_params[name]

    return result

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
        
def rollout(
            meta_iter,
            phase,
            num_pairs: int,
            envs,
            policy: PolicyNetwork,
            critic: ValueNetwork,
            buffer: TensorDict,
            next_obs,
            next_done,
            infos,
            track_episode_returns: bool = False,
            ):
    """
    Perform rollout in the environment.

    Args:
        track_episode_returns: If True, return episode returns for each environment

    Returns:
        next_value: Bootstrap value for next state
        episode_returns_list: (Optional) List of episode returns if track_episode_returns=True
    """
    episode_returns_list = [] if track_episode_returns else None

    for step in range(num_steps):
        buffer['obs'][step] = torch.tensor(next_obs)
        buffer['dones'][step] = torch.tensor(next_done)

        # ALGO LOGIC: action logic
        with torch.no_grad():
            # Get action and log probability from policy
            obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)
            action, logprob = policy.get_action(obs_tensor)
            action_logits = policy(obs_tensor)
            if infos is not None:
                # Call the mask creation method from the underlying module
                action_mask = policy._create_action_mask(torch.tensor(next_obs).to(device), infos)
                action_logits = action_logits + action_mask
            probs = Categorical(logits=action_logits)
            action = probs.sample()
            logprob = probs.log_prob(action)
            value = critic(obs_tensor)
            buffer['values'][step] = value.flatten().cpu()

        buffer['actions'][step] = action.cpu()
        buffer['logprobs'][step] = logprob.cpu()

        # Execute actions in vectorized environment
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

        # Store current infos for next step
        next_done = np.logical_or(terminations, truncations)
        buffer['rewards'][step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        # Track episode returns when episodes end
        if track_episode_returns:
            for i in range(num_pairs):
                if terminations[i] or truncations[i]:
                    episode_return = infos['episode_returns'][i]
                    episode_returns_list.append({
                        'env_idx': i,
                        'episode_return': float(episode_return),
                        'step': step
                    })
                    if episode_return > 0.0:
                        print(f"{meta_iter}-{phase}-{i}-th env: episode_return: {episode_return:.4f}")

    # Bootstrap value for the next state
    with torch.no_grad():
        next_value = critic(torch.tensor(next_obs).to(device))
        next_value = next_value.reshape(1, -1)

    if track_episode_returns:
        return next_value, episode_returns_list
    return next_value


def compute_gae(buffer, next_value):
    """Generalized Advantage Estimation (GAE) 계산
    
    A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t+1}δ_{T-1}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    rewards = buffer['rewards']
    values = buffer['values']
    dones = buffer['dones']
    
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value.flatten()
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.stack(advantages).reshape(-1,)
    values = values.reshape(-1,)
    returns = advantages + values
    returns = returns.reshape(-1,)
    
    return advantages, returns
    
# %%
meta_batch_size = 5
lr_inner = 0.1
lr_outer = 0.001
num_inner_steps = 1  # Inner loop PPO epochs - reduced for efficiency
first_order = True  # Use first-order MAML for faster computation

# Initialize wandb
try:
    run = wandb.init(
        project="ARC-AGI-MAML-PPO",
        config={
            "meta_batch_size": meta_batch_size,
            "lr_inner": lr_inner,
            "lr_outer": lr_outer,
            "num_inner_steps": num_inner_steps,
            "first_order": first_order,
            "num_steps": num_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "eps_clip": eps_clip,
            "embed_dim": 256,
            "num_heads": 4,
            "num_layers": 6,
        },
        name=f"maml_ppo_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    print(f"Wandb initialized successfully: {run.name}")
    print(f"Wandb project: {run.project}, URL: {run.url}")
except Exception as e:
    print(f"Warning: Failed to initialize wandb: {e}")
    import traceback
    traceback.print_exc()

meta_policy = PolicyNetwork().cuda()
meta_critic = ValueNetwork().cuda()
meta_policy_optimizer = optim.Adam(meta_policy.parameters(), lr=lr_outer)
meta_critic_optimizer = optim.Adam(meta_critic.parameters(), lr=lr_outer)

for meta_iter in range(num_meta_iterations):
    meta_policy_loss = 0
    meta_critic_loss = 0
    meta_policy_optimizer.zero_grad()
    meta_critic_optimizer.zero_grad()

    # Initialize tracking variables for this iteration
    eval_episode_returns = None
    train_episode_returns = None

    # task_batch = random.sample(list(meta_train_dataset.keys()), meta_batch_size)
    # task 하나만 학습 잘 되는지 확인
    task_batch = ['794b24be']
    for task_id in task_batch:
        # adaptation phase
        adapt_train_envs = make_meta_task_env(meta_train_dataset,
                                    meta_eval_dataset,
                                    meta_test_dataset,
                                    mode='meta_train',
                                    phase='train',
                                    task_id=task_id,
                                    rand_init=False,
                                    )
        num_pairs = len(adapt_train_envs.envs)
        adapt_buffer = setup_buffer(num_pairs)

        # Initialize task-specific parameters from meta-parameters
        task_policy_params = OrderedDict()
        task_critic_params = OrderedDict()
        for name, param in meta_policy.named_parameters():
            task_policy_params[name] = param.clone()
        for name, param in meta_critic.named_parameters():
            task_critic_params[name] = param.clone()

        next_obs, infos = adapt_train_envs.reset()
        next_done = np.array([False for i in range(num_pairs)])

        # Create temporary policy/critic for rollout - using deepcopy
        # We need this for environment interaction
        task_policy = copy.deepcopy(meta_policy)
        task_critic = copy.deepcopy(meta_critic)

        # Load initial parameters
        for name, param in task_policy.named_parameters():
            param.data = task_policy_params[name]
        for name, param in task_critic.named_parameters():
            param.data = task_critic_params[name]

        # Training rollout with tracking
        rollout_result = rollout(
                            meta_iter,
                            "train",
                            num_pairs,
                             adapt_train_envs,
                            task_policy,
                            task_critic,
                            adapt_buffer,
                            next_obs,
                            next_done,
                            infos,
                            track_episode_returns=True,
                            )
        next_value, train_episode_returns = rollout_result
        print(f"{meta_iter}: start to adapt parameters")
        advantages, returns = compute_gae(adapt_buffer, next_value)
        print(f"Before normalization - advantages mean: {advantages.mean().item():.4f}, std: {advantages.std().item():.4f}")
        print(f"Advantages range: [{advantages.min().item():.4f}, {advantages.max().item():.4f}]")
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        obs = adapt_buffer['obs'].reshape(-1, 30, 120)
        actions = adapt_buffer['actions'].reshape(-1)  # Shape: [batch_size], not [batch_size, 1]

        # Inner loop adaptation - using parameter dictionaries for efficiency
        for inner_step in range(num_inner_steps):
            # Update task_policy with current parameters
            for name, param in task_policy.named_parameters():
                param.data = task_policy_params[name]

            # Policy adaptation
            log_probs = task_policy.get_log_prob(obs, actions)
            if inner_step == 0:
                old_log_probs = log_probs.detach()
            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Gradient 계산 - first-order MAML uses create_graph=False
            policy_grads = torch.autograd.grad(
                policy_loss,
                task_policy_params.values(),
                create_graph=not first_order,  # False for first-order MAML
                allow_unused=True
            )

            # Inner loop 파라미터 업데이트
            new_policy_params = OrderedDict()
            for (name, param), grad in zip(task_policy_params.items(), policy_grads):
                if grad is not None:
                    # First-order MAML: Use the original meta-parameter + gradient update
                    # This maintains gradient flow from meta-parameter to adapted parameter
                    if first_order:
                        # Get the corresponding meta parameter
                        meta_param = None
                        for meta_name, meta_p in meta_policy.named_parameters():
                            if meta_name == name:
                                meta_param = meta_p
                                break
                        # Use meta_param - lr * grad (detached gradient)
                        new_policy_params[name] = meta_param - lr_inner * grad.detach()
                    else:
                        # Second-order: keep full gradient flow
                        new_policy_params[name] = param - lr_inner * grad
                else:
                    new_policy_params[name] = param

            task_policy_params = new_policy_params

        # Value function 업데이트
        for inner_step in range(num_inner_steps):
            # Update task_critic with current parameters
            for name, param in task_critic.named_parameters():
                param.data = task_critic_params[name]

            value_pred = task_critic(obs).squeeze()
            value_loss = nn.MSELoss()(value_pred, returns)

            value_grads = torch.autograd.grad(
                value_loss,
                task_critic_params.values(),
                create_graph=not first_order,
                allow_unused=True
            )

            new_critic_params = OrderedDict()
            for (name, param), grad in zip(task_critic_params.items(), value_grads):
                if grad is not None:
                    if first_order:
                        # Get the corresponding meta parameter
                        meta_param = None
                        for meta_name, meta_p in meta_critic.named_parameters():
                            if meta_name == name:
                                meta_param = meta_p
                                break
                        # Use meta_param - lr * grad (detached gradient)
                        new_critic_params[name] = meta_param - lr_inner * grad.detach()
                    else:
                        # Second-order: keep full gradient flow
                        new_critic_params[name] = param - lr_inner * grad
                else:
                    new_critic_params[name] = param

            task_critic_params = new_critic_params

        # Load adapted parameters into models for evaluation
        for name, param in task_policy.named_parameters():
            param.data = task_policy_params[name]
        for name, param in task_critic.named_parameters():
            param.data = task_critic_params[name]

        # evaluation phase - compute meta-loss on query set
        print(f"{meta_iter}: start evaluation")
        eval_test_envs = make_meta_task_env(meta_train_dataset,
                                        meta_eval_dataset,
                                        meta_test_dataset,
                                        mode='meta_train',
                                        phase='test',
                                        task_id=task_id,
                                        rand_init=False,
                                        )
        num_pairs = len(eval_test_envs.envs)
        eval_buffer = setup_buffer(num_pairs)
        next_obs, infos = eval_test_envs.reset()
        next_done = np.array([False for i in range(num_pairs)])

        # Track episode returns during evaluation
        rollout_result = rollout(
                            meta_iter,
                            "test",
                            num_pairs,
                            eval_test_envs,
                            task_policy,
                            task_critic,
                            eval_buffer,
                            next_obs,
                            next_done,
                            infos,
                            track_episode_returns=True,
                            )
        next_value, eval_episode_returns = rollout_result

        advantages, returns = compute_gae(eval_buffer, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_eval = eval_buffer['obs'].reshape(-1, 30, 120)
        actions_eval = eval_buffer['actions'].reshape(-1)  # Shape: [batch_size], not [batch_size, 1]

        # Compute meta-loss on adapted policy
        # Use log probs from the adapted policy
        log_probs = task_policy.get_log_prob(obs_eval, actions_eval)

        # For MAML, we don't need PPO clipping on the meta-loss
        # We just want the policy gradient on the adapted policy
        task_policy_loss = -(log_probs * advantages).mean()
        print(f"task_policy_loss: {task_policy_loss.item()}")
        # Value loss on query set
        value_pred = task_critic(obs_eval).squeeze()
        task_value_loss = nn.MSELoss()(value_pred, returns)

        # Accumulate meta-gradients
        meta_policy_loss += task_policy_loss / meta_batch_size
        meta_critic_loss += task_value_loss / meta_batch_size

    # Compute gradients for meta-update
    meta_policy_loss.backward()
    meta_critic_loss.backward()

    # Check gradient norms
    policy_grad_norm = torch.nn.utils.clip_grad_norm_(meta_policy.parameters(), float('inf'))
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(meta_critic.parameters(), float('inf'))
    print(f"Policy grad norm: {policy_grad_norm:.4f}, Critic grad norm: {critic_grad_norm:.4f}")

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(meta_policy.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(meta_critic.parameters(), 0.5)

    # meta-update
    meta_policy_optimizer.step()
    meta_critic_optimizer.step()

    # Logging
    print(f"Meta-iter {meta_iter}: Policy Loss = {meta_policy_loss.item():.4f}, Value Loss = {meta_critic_loss.item():.4f}")

    # Wandb logging - wrap in try-except for debugging
    try:
        wandb_log_dict = {
            "meta_iteration": meta_iter,
            "meta_policy_loss": meta_policy_loss.item(),
            "meta_value_loss": meta_critic_loss.item(),
            "policy_grad_norm": policy_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
        }

        # Log training episode returns
        if train_episode_returns is not None and len(train_episode_returns) > 0:
            for ep_return_info in train_episode_returns:
                env_idx = ep_return_info['env_idx']
                episode_return = ep_return_info['episode_return']
                wandb_log_dict[f"train/env_{env_idx}_episode_return"] = episode_return

            # Also log mean and max episode returns for training
            train_returns = [ep['episode_return'] for ep in train_episode_returns]
            wandb_log_dict["train/mean_episode_return"] = float(np.mean(train_returns))
            wandb_log_dict["train/max_episode_return"] = float(np.max(train_returns))
            wandb_log_dict["train/min_episode_return"] = float(np.min(train_returns))
            wandb_log_dict["train/num_episodes"] = len(train_returns)
            print(f"Train episodes: {len(train_returns)}, mean_return: {np.mean(train_returns):.4f}")
        else:
            print(f"No training episodes completed this iteration")

        # Log evaluation episode returns for each environment
        if eval_episode_returns is not None and len(eval_episode_returns) > 0:
            for ep_return_info in eval_episode_returns:
                env_idx = ep_return_info['env_idx']
                episode_return = ep_return_info['episode_return']
                wandb_log_dict[f"eval/env_{env_idx}_episode_return"] = episode_return

            # Also log mean and max episode returns for evaluation
            eval_returns = [ep['episode_return'] for ep in eval_episode_returns]
            wandb_log_dict["eval/mean_episode_return"] = float(np.mean(eval_returns))
            wandb_log_dict["eval/max_episode_return"] = float(np.max(eval_returns))
            wandb_log_dict["eval/min_episode_return"] = float(np.min(eval_returns))
            wandb_log_dict["eval/num_episodes"] = len(eval_returns)
            print(f"Eval episodes: {len(eval_returns)}, mean_return: {np.mean(eval_returns):.4f}")
        else:
            print(f"No evaluation episodes completed this iteration")

        # Log to wandb
        print(f"Logging to wandb with {len(wandb_log_dict)} metrics...")
        wandb.log(wandb_log_dict, step=meta_iter)
        print(f"Successfully logged iteration {meta_iter} to wandb")

    except Exception as e:
        print(f"Error logging to wandb: {e}")
        import traceback
        traceback.print_exc()

    # Save checkpoint
    if meta_iter % 100 == 0:
        checkpoint_dir = Path("./ckpts/maml_ppo")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'meta_iter': meta_iter,
            'meta_policy_state_dict': meta_policy.state_dict(),
            'meta_critic_state_dict': meta_critic.state_dict(),
            'meta_policy_optimizer_state_dict': meta_policy_optimizer.state_dict(),
            'meta_critic_optimizer_state_dict': meta_critic_optimizer.state_dict(),
            'config': {
                'lr_inner': lr_inner,
                'lr_outer': lr_outer,
                'meta_batch_size': meta_batch_size,
                'num_inner_steps': num_inner_steps,
                'first_order': first_order,
            }
        }, checkpoint_dir / f"checkpoint_iter_{meta_iter}.pt")
        print(f"Checkpoint saved at iteration {meta_iter}")

    # Memory cleanup
    torch.cuda.empty_cache()

# Finish wandb run
try:
    print("Finishing wandb run...")
    wandb.finish()
    print("Wandb run finished successfully")
except Exception as e:
    print(f"Error finishing wandb: {e}")

# %%

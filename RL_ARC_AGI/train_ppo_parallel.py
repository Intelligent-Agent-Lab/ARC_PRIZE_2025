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
from arc_agi_grid_env_coord import cmap
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from arc_agi_grid_env_coord import ArcAgiGridEnvCoord, create_arc_env_coord, \
                                    action_converter, vectorized_action_converter, \
                                    load_challenges_and_solutions, \
                                    preprocess_data
from matplotlib import colors
from pathlib import Path


def log_action_details(action_tensor, infos, step_num, max_logs=5):
    """Log detailed information about selected actions."""
    if step_num % 50 == 0:  # Log every 50 steps to avoid spam
        num_envs = min(len(action_tensor), max_logs)
        print(f"\n=== Step {step_num} Action Details ===")
        
        for env_idx in range(num_envs):
            action_int = action_tensor[env_idx].item()
            action_dict = action_converter(action_int)
            
            color = action_dict['color']
            coordinate = action_dict['coordinate']
            row, col = coordinate
            
            # Get info for this environment - handle vectorized environment structure
            color_candidate = 'Unknown'
            size_candidate = 'Unknown'
            
            if infos:
                if isinstance(infos, dict):
                    # Vectorized environment returns flattened arrays
                    if 'size_candidate' in infos:
                        if len(infos['size_candidate']) > env_idx:
                            val = infos['size_candidate'][env_idx]
                            size_candidate = val.tolist() if hasattr(val, 'tolist') else val
                    if 'color_candidate' in infos:
                        if len(infos['color_candidate']) > env_idx:
                            val = infos['color_candidate'][env_idx]
                            color_candidate = val.tolist() if hasattr(val, 'tolist') else val
                elif isinstance(infos, list) and len(infos) > env_idx:
                    # If infos is a list, get the element at env_idx
                    info = infos[env_idx] if infos[env_idx] else {}
                    if isinstance(info, dict):
                        color_candidate = info.get('color_candidate', 'Unknown')
                        size_candidate = info.get('size_candidate', 'Unknown')
            
            print(f"  Env {env_idx}: Action {action_int} -> Color {color} at ({row}, {col})")
            print(f"    Valid colors: {color_candidate}")
            print(f"    Target size: {size_candidate}")
            
            # Check if color is valid
            if isinstance(color_candidate, list) and color in color_candidate:
                valid_color = "✓"
            elif isinstance(color_candidate, list):
                valid_color = "✗"
            else:
                valid_color = "?"
            print(f"    Color validity: {valid_color}")


def make_env(fixed_task: bool, 
             fixed_pair_idx: bool, 
             task_id: str, 
             pair_idx: int,
             training_challenges,
             training_solutions,
             evaluation_challenges,
             evaluation_solutions,
             test_challenges,
             train_task_img_dict,
             eval_task_img_dict):
    """Create and wrap environment for vectorized training."""
    def thunk():
        env = create_arc_env_coord(
                fixed_task=fixed_task, 
                fixed_pair_idx=fixed_pair_idx,
                task_id=task_id,
                pair_idx=pair_idx,
                training_challenges=training_challenges,
                training_solutions=training_solutions, 
                evaluation_challenges=evaluation_challenges,
                evaluation_solutions=evaluation_solutions,
                test_challenges=test_challenges,
                train_task_img_dict=train_task_img_dict,
                eval_task_img_dict=eval_task_img_dict,
            )
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


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
            try:
                current_obs = obs[i]  # Shape: (30, 180)
                solution_area = current_obs[:, 150:]  # Shape: (30, 30)

                # Extract current info
                current_info = self._extract_current_info(obs_info, i)
                size_candidate = current_info.get('size_candidate', [4, 4]) if current_info else [4, 4]
                valid_colors = set(current_info.get('color_candidate', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) if current_info else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

                target_rows, target_cols = size_candidate[0], size_candidate[1]

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
            except Exception as e:
                print(f"Warning: Could not create action mask for batch {i}: {e}")
                continue

        return action_mask.squeeze(0) if batch_size == 1 else action_mask

    def _extract_current_info(self, obs_info, i):
        """Helper to extract info for batch index i."""
        if obs_info is None:
            return None

        current_info = {}
        if isinstance(obs_info, dict):
            for key in ['size_candidate', 'color_candidate']:
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


class ArcAgiVectorizedTrainer:
    """Trainer class for PPO with vectorized ArcAgiGrid environments."""
    
    def __init__(self, config: DictConfig, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        self.setup_environment()
        self.setup_agent()
        self.setup_logging()
        self.setup_storage()
        
    def setup_environment(self):
        """Setup the vectorized training environments."""
        # Load data
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
        train_task_img_dict, _, train_img_shape_colors, _ = preprocess_data(training_challenges, training_solutions)
        eval_task_img_dict, _, eval_img_shape_colors, _= preprocess_data(evaluation_challenges, evaluation_solutions)

        self.task_id_list = list(self.config.environment.task_id_list)
        self.seed = self.config.environment.seed
        self.fixed_task = self.config.environment.fixed_task
        self.fixed_pair_idx = self.config.environment.fixed_pair_idx
        self.task_id = self.config.environment.task_id
        self.pair_idx = self.config.environment.pair_idx
        
        self.num_envs = int(self.config.environment.num_envs)
        self.num_steps = self.config.environment.num_steps
        
        # Get size_candidate and color_candidate from config if available
        self.size_candidate = getattr(self.config.environment, 'size_candidate', [4, 4])
        self.color_candidate = getattr(self.config.environment, 'color_candidate', [0, 1, 4, 5, 9])
        
        # Create vectorized environment
        self.envs = gym.vector.SyncVectorEnv([
            make_env(
                fixed_task=self.fixed_task,
                fixed_pair_idx=self.fixed_pair_idx,
                task_id=self.task_id,
                pair_idx=self.pair_idx,
                training_challenges=training_challenges,
                training_solutions=training_solutions,
                evaluation_challenges=evaluation_challenges,
                evaluation_solutions=evaluation_solutions,
                test_challenges=test_challenges,
                train_task_img_dict=train_task_img_dict,
                eval_task_img_dict=eval_task_img_dict
            ) for i in range(self.num_envs)
        ])

        if self.rank == 0:
            print(f"Vectorized environments created successfully!")
            print(f"Number of environments per rank: {self.num_envs}")
            print(f"Actual total environments: {self.num_envs * self.world_size}")
            print(f"Single observation space: {self.envs.single_observation_space}")
            print(f"Single action space: {self.envs.single_action_space}")
        
    def setup_agent(self):
        """Setup the DDP agent for the current process (rank)."""
        self.device = torch.device(f"cuda:{self.rank}")

        # Create agent and move to the correct device
        agent = Agent(self.config).to(self.device)
        # Wrap the agent with DDP
        self.agent = DDP(agent, device_ids=[self.rank])

        # Create optimizer for the DDP-wrapped model
        self.optimizer = optim.Adam(
            self.agent.module.parameters(),  # Use .module to access original model
            lr=self.config.training.learning_rate,
            eps=1e-5
        )

        # Calculate mini-batch size per process
        self.mini_batch_size = int(self.config.environment.num_steps * self.config.environment.num_envs / self.config.training.num_minibatches)

        if self.rank == 0:
            print(f"PPO DDP Agent created on device: {self.device}")
            print(f"Mini batch size per GPU: {self.mini_batch_size}")


        
    def setup_logging(self):
        """Setup logging and metrics tracking for the main process."""
        # These deques will live on each process, but only rank 0 will aggregate and log
        self.episode_returns = deque(maxlen=self.config.environment.num_envs)
        self.episode_lengths = deque(maxlen=self.config.environment.num_envs)
        self.success_rate = deque(maxlen=self.config.environment.num_envs)
        
        self.tensorboard_writer = None
        # Logging and directory creation should only happen on the main process
        if self.rank == 0:
            # Create save directory
            os.makedirs(self.config.logging.save_dir, exist_ok=True)
            
            run_name = f"ppo_vectorized_{self.task_id}_{int(time.time())}"
            
            # Initialize wandb logger
            if self.config.logging.use_wandb:
                wandb.init(
                    project=self.config.logging.wandb_project,
                    config=OmegaConf.to_container(self.config, resolve=True),
                    name=run_name,
                    sync_tensorboard=True,
                    monitor_gym=True,
                    save_code=True
                )
            
            # Initialize tensorboard logger
            if self.config.logging.use_tensorboard:
                self.tensorboard_writer = SummaryWriter(f"runs/{run_name}")
                self.tensorboard_writer.add_text(
                    "hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(self.config, resolve=True).items()])),
                )
    
    def setup_storage(self):
        """Setup storage for rollout data."""
        self.obs = torch.zeros((self.num_steps, self.num_envs) + (30, 180)).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        
        # Manual episode tracking
        self.current_episode_returns = np.zeros(self.num_envs)
        self.current_episode_lengths = np.zeros(self.num_envs)
    
    def visualize_grid(self, iteration: int, w=0.5, first=False):
        """Visualize current grid state and log to wandb/tensorboard."""
        try:
            norm = colors.Normalize(vmin=0, vmax=11)
            # Get current grid from first environment
            info_dict = self.envs.envs[0]._get_info()
            target_grid = info_dict['target_grid_img']
            current_grid = info_dict['current_grid_img']
            timestep = info_dict['timestep']
            task_id = info_dict['task_id']
            test_input_idx = info_dict['test_input_idx']
            
            test_sol_current_mat = current_grid[:, 120:]
            test_sol_target_mat = target_grid[:, 120:]
                
            if first:
                plt.imshow(test_sol_target_mat, cmap=cmap, norm=norm)
                plt.grid(True, which='both', color='lightgrey', linewidth=1.0)
                plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
                plt.grid(visible=True, which='both', color='#666666', linewidth=w)
                plt.xticks([x-0.5 for x in range(1 + len(test_sol_target_mat[0]))])
                plt.yticks([x-0.5 for x in range(1 + len(test_sol_target_mat))])
                plt.tick_params(axis='both', color='none', length=0)
                plt.title(f'task: #{task_id}  test_input_idx: #{test_input_idx}', fontsize=12, color='#000000')
                figure_folder_path = Path("./figures")
                if not figure_folder_path.exists():
                    figure_folder_path.mkdir(parents=True)
                plt.savefig(f"./figures/target.png")
                plt.close()

            plt.imshow(test_sol_current_mat, cmap=cmap, norm=norm)
            plt.grid(True, which='both', color='lightgrey', linewidth=1.0)
            plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
            plt.grid(visible=True, which='both', color='#666666', linewidth=w)
            plt.xticks([x-0.5 for x in range(1 + len(test_sol_current_mat[0]))])
            plt.yticks([x-0.5 for x in range(1 + len(test_sol_current_mat))])
            plt.tick_params(axis='both', color='none', length=0)
            plt.title(f'task: #{task_id}  test_input_idx: #{test_input_idx}  iter: #{iteration}  timestep: #{timestep}', 
                     fontsize=12, color='#000000')
            figure_folder_path = Path("./figures")
            if not figure_folder_path.exists():
                figure_folder_path.mkdir(parents=True)
            plt.savefig(f"./figures/{iteration}_current.png")
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not visualize grid at iteration {iteration}: {e}")
        
    def collect_rollouts(self, next_obs, next_done, iteration: int):
        """Collect rollout data for training, ensuring DDP wrapper is always called."""
        for step in range(self.num_steps):
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # DDP 래퍼 자체를 직접 호출하는 것으로 변경
                # 이전처럼 내부 모듈을 직접 호출하면 forward 훅을 우회하는 작업임
                action_logits, value = self.agent(next_obs)
                self.values[step] = value.flatten()

                # Use current infos for action masking (from previous step or reset)
                current_infos = getattr(self, 'current_infos', None)
                if current_infos is not None:
                    # Call the mask creation method from the underlying module
                    action_mask = self.agent.module._create_action_mask(next_obs, current_infos)
                    action_logits = action_logits + action_mask
                else:
                    print(f"WARNING: current_infos is None at step {step}!")

                # Sample action and get log probability
                probs = Categorical(logits=action_logits)
                action = probs.sample()
                logprob = probs.log_prob(action)

            self.actions[step] = action
            self.logprobs[step] = logprob

            # Execute actions in vectorized environment
            dict_action = vectorized_action_converter(action.cpu())
            next_obs, reward, terminations, truncations, infos = self.envs.step(dict_action)
            
            # Log action details disabled for clean output
            # log_action_details(action, infos, step + iteration * self.num_steps)

            # Store current infos for next step
            self.current_infos = infos
            next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
            
            # Update episode tracking
            self.current_episode_returns += reward
            self.current_episode_lengths += 1
            
            # Log episode statistics when episodes end
            for i in range(self.num_envs):
                if terminations[i] or truncations[i]:
                    self.episode_returns.append(self.current_episode_returns[i])
                    self.episode_lengths.append(self.current_episode_lengths[i])
                    self.success_rate.append(1.0 if self.current_episode_returns[i] > 10.0 else 0.0)
                    
                    self.current_episode_returns[i] = 0.0
                    self.current_episode_lengths[i] = 0
        
        # Bootstrap value for the next state
        with torch.no_grad():
            # --- Correct DDP Forward Pass for Value ---
            _, next_value = self.agent(next_obs)
            next_value = next_value.reshape(1, -1)
            # --- End Correct DDP Forward Pass ---
            
        return next_value

    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation."""
        with torch.no_grad():
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.config.training.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.config.training.gamma * self.config.training.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values
            
        return advantages, returns

    def update_agent(self, advantages, returns, global_step):
        """Update the agent using PPO."""
        # Flatten the batch
        batch_size = self.num_envs * self.num_steps
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        grad_acc = max(1, getattr(self.config.training, "gradient_accumulation_steps", 1))

        for epoch in range(self.config.training.ppo_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_inds = b_inds[start:end]
                
                mini_batch_idx = start // self.mini_batch_size
                if mini_batch_idx % grad_acc == 0:
                    self.optimizer.zero_grad()
                
                action_logits, newvalue = self.agent(b_obs[mb_inds])
                
                # compute probs, logprob, and entropy
                probs = Categorical(logits=action_logits)
                newlogprob = probs.log_prob(b_actions.long()[mb_inds])
                entropy = probs.entropy()

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.training.eps_clip).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.config.training.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.training.eps_clip, 1 + self.config.training.eps_clip)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.training.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.training.eps_clip,
                        self.config.training.eps_clip,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.training.entropy_coef * entropy_loss + v_loss * self.config.training.value_coef
                
                if hasattr(self.config.training, 'gradient_accumulation_steps'):
                    loss = loss / grad_acc

                # --- Correct Gradient Accumulation ---
                # 새로운 미니배치가 들어오기 바로 직전(마지막 누적일때)
                # 축적된 그래디언트가 gpu 간 동기화되고 평균 계산
                is_sync_step = (mini_batch_idx + 1) % grad_acc == 0
                
                # Last accumulation step or last micro-batch in epoch, sync gradients
                if is_sync_step:
                    loss.backward()
                else:
                    # Accumulation step, don't sync
                    with self.agent.no_sync():
                        loss.backward()
                
                if is_sync_step:
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.training.max_grad_norm)
                    self.optimizer.step()
                # --- End Correct Gradient Accumulation ---

            if self.config.training.target_kl is not None and approx_kl > self.config.training.target_kl:
                break

        # Calculate explained variance                    
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'old_approx_kl': old_approx_kl.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': np.mean(clipfracs),
            'explained_variance': explained_var
        }

    def train(self):
        """Main training loop."""
        if self.rank == 0:
            print("Starting PPO training...")
            print(f"Configuration: {self.config}")

        # Calculate training parameters
        batch_size_per_rank = self.num_envs * self.num_steps
        total_batch_size = batch_size_per_rank * self.world_size
        num_iterations = self.config.training.total_timesteps // total_batch_size

        if self.rank == 0:
            print(f"Total Timesteps: {self.config.training.total_timesteps}, World Size: {self.world_size}")
            print(f"Batch Size per Rank: {batch_size_per_rank}, Total Batch Size: {total_batch_size}")
            print(f"Number of Iterations: {num_iterations}")

        global_step = 0
        start_time = time.time()
        best_mean_reward = -float('inf')
        first_vis = True

        reset_options = {
            'size_candidate': self.size_candidate,
            'color_candidate': self.color_candidate
        }
        next_obs, infos = self.envs.reset(seed=self.seed + self.rank, options=reset_options)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        # Store initial infos for first action
        self.current_infos = infos

        for iteration in range(1, num_iterations + 1):
            # Annealing learning rate
            if self.config.training.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lr_now = frac * self.config.training.learning_rate
                self.optimizer.param_groups[0]["lr"] = lr_now

            # Collect rollouts
            self.agent.eval()
            next_value = self.collect_rollouts(next_obs, next_done, iteration)

            # Compute GAE
            advantages, returns = self.compute_gae(next_value)
            
            # Train agent
            self.agent.train()
            training_metrics = self.update_agent(advantages, returns, global_step)
            
            # Update global step
            global_step += total_batch_size

            # Logging
            if iteration % self.config.logging.log_interval == 0:
                # 1. Aggregate metrics from all processes
                local_reward = np.mean(self.episode_returns) if self.episode_returns else 0
                local_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                local_success_rate = np.mean(self.success_rate) if self.success_rate else 0
                
                metrics_tensor = torch.tensor([local_reward, local_length, local_success_rate], dtype=torch.float32, device=self.device)
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
                
                # Aggregate training_metrics dictionary
                metrics_keys = sorted(training_metrics.keys())
                local_metrics_values = torch.tensor([training_metrics[k] for k in metrics_keys], dtype=torch.float32, device=self.device)
                dist.all_reduce(local_metrics_values, op=dist.ReduceOp.AVG)

                if self.rank == 0:
                    # 2. Log aggregated metrics on rank 0
                    mean_reward = metrics_tensor[0].item()
                    mean_length = metrics_tensor[1].item()
                    success_rate = metrics_tensor[2].item()
                    
                    agg_training_metrics = {k: v.item() for k, v in zip(metrics_keys, local_metrics_values)}
                    
                    sps = int(global_step / (time.time() - start_time))
                    lr_now = self.optimizer.param_groups[0]['lr']

                    # --- Detailed Print Block ---
                    print(f"\nIteration {iteration}/{num_iterations}")
                    print(f"Global step: {global_step}")
                    print(f"Global Mean reward: {mean_reward:.3f}")
                    print(f"Global Mean episode length: {mean_length:.1f}")
                    print(f"Global Success rate: {success_rate:.3f}")
                    print(f"SPS: {sps}")
                    print(f"Learning rate: {lr_now:.6f}")
                    
                    if agg_training_metrics:
                        print(f"Policy loss: {agg_training_metrics['policy_loss']:.4f}")
                        print(f"Value loss: {agg_training_metrics['value_loss']:.4f}")
                        print(f"Entropy loss: {agg_training_metrics['entropy_loss']:.4f}")
                        print(f"Explained variance: {agg_training_metrics['explained_variance']:.4f}")

                    # --- Detailed WandB Log Block ---
                    if self.config.logging.use_wandb:
                        log_dict = {
                            'charts/learning_rate': lr_now,
                            'charts/episodic_return': mean_reward,
                            'charts/episodic_length': mean_length,
                            'charts/SPS': sps,
                            'charts/success_rate': success_rate,
                            'losses/policy_loss': agg_training_metrics['policy_loss'],
                            'losses/value_loss': agg_training_metrics['value_loss'],
                            'losses/entropy': agg_training_metrics['entropy_loss'],
                            'losses/old_approx_kl': agg_training_metrics['old_approx_kl'],
                            'losses/approx_kl': agg_training_metrics['approx_kl'],
                            'losses/clipfrac': agg_training_metrics['clipfrac'],
                            'losses/explained_variance': agg_training_metrics['explained_variance']
                        }
                        wandb.log(log_dict, step=global_step)
                    
                    # --- Detailed Tensorboard Log Block ---
                    if self.tensorboard_writer:
                        self.tensorboard_writer.add_scalar('charts/learning_rate', lr_now, global_step)
                        self.tensorboard_writer.add_scalar('charts/episodic_return', mean_reward, global_step)
                        self.tensorboard_writer.add_scalar('charts/episodic_length', mean_length, global_step)
                        self.tensorboard_writer.add_scalar('charts/SPS', sps, global_step)
                        self.tensorboard_writer.add_scalar('charts/success_rate', success_rate, global_step)
                        self.tensorboard_writer.add_scalar('losses/policy_loss', agg_training_metrics['policy_loss'], global_step)
                        self.tensorboard_writer.add_scalar('losses/value_loss', agg_training_metrics['value_loss'], global_step)
                        self.tensorboard_writer.add_scalar('losses/entropy', agg_training_metrics['entropy_loss'], global_step)
                        self.tensorboard_writer.add_scalar('losses/old_approx_kl', agg_training_metrics['old_approx_kl'], global_step)
                        self.tensorboard_writer.add_scalar('losses/approx_kl', agg_training_metrics['approx_kl'], global_step)
                        self.tensorboard_writer.add_scalar('losses/clipfrac', agg_training_metrics['clipfrac'], global_step)
                        self.tensorboard_writer.add_scalar('losses/explained_variance', agg_training_metrics['explained_variance'], global_step)

            # --- VISUALIZATION AND CHECKPOINTING (RANK 0 ONLY) ---
            if self.rank == 0:
                # Visualization
                if iteration % self.config.logging.visualize_period == 0:
                    self.visualize_grid(iteration, first=first_vis)
                    first_vis = False
                
                # Save checkpoint
                if iteration % self.config.logging.save_interval == 0 and iteration > 0:
                    torch.save(self.agent.module.state_dict(), os.path.join(self.config.logging.save_dir, f'checkpoint_{iteration}.pth'))
                    print(f"Checkpoint saved at iteration {iteration}")

                    # Save best model based on aggregated reward
                    # Note: mean_reward is only calculated on log intervals. We need to re-calculate for saving best model.
                    current_agg_reward_tensor = torch.tensor([np.mean(self.episode_returns) if self.episode_returns else -float('inf')], dtype=torch.float32, device=self.device)
                    dist.all_reduce(current_agg_reward_tensor, op=dist.ReduceOp.AVG)
                    current_agg_reward = current_agg_reward_tensor.item()

                    if current_agg_reward > best_mean_reward:
                        best_mean_reward = current_agg_reward
                        torch.save(self.agent.module.state_dict(), os.path.join(self.config.logging.save_dir, 'best_model.pth'))
                        print(f"New best model saved! Global mean reward: {best_mean_reward:.3f}")

        # --- FINALIZATION (RANK 0 ONLY) ---
        if self.rank == 0:
            print("Training completed!")
            final_path = os.path.join(self.config.logging.save_dir, 'final_model.pth')
            torch.save(self.agent.module.state_dict(), final_path)
            print(f"Final model saved: {final_path}")
            
            if self.config.logging.use_wandb:
                wandb.finish()
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
        
        # All processes must wait here before closing envs
        dist.barrier()
        self.envs.close()


def train_rank(rank: int, world_size: int, cfg: DictConfig) -> None:
    """A single process's training entry point."""
    # DDP environment setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Isolate printing to the main process
    if rank == 0:
        print("Vectorized PPO Training configuration:")
        print(OmegaConf.to_yaml(cfg))
    
    # Set a common seed for model initialization (important for DDP)
    torch.manual_seed(cfg.environment.seed)
    torch.backends.cudnn.deterministic = True

    # Set different seeds for environment randomization
    random.seed(cfg.environment.seed + rank)
    np.random.seed(cfg.environment.seed + rank)
    
    # Create trainer and start training
    trainer = ArcAgiVectorizedTrainer(cfg, rank, world_size)
    trainer.train()

    # Cleanup
    dist.destroy_process_group()

@hydra.main(version_base=None, config_path="config", config_name="ppo_vector_env")
def main(cfg: DictConfig) -> None:
    """Spawns DDP training processes."""
    world_size = torch.cuda.device_count()
    # Ensure world_size is not zero
    if world_size == 0:
        print("No GPUs detected. DDP training requires at least one GPU.")
        return
        
    mp.spawn(train_rank,
             args=(world_size, cfg),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main()

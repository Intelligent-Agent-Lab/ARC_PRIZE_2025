# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import math
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from arc_agi_grid_env_coord import ArcAgiGridEnvCoord, create_arc_env_coord, \
                                    action_converter, \
                                    load_challenges_and_solutions, \
                                    preprocess_data, vectorized_action_converter
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "arc_agi_ppo"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


training_challenges_json="../datasets/arc-agi_training_challenges.json"
training_solutions_json="../datasets/arc-agi_training_solutions.json" 
evaluation_challenges_json="../datasets/arc-agi_evaluation_challenges.json"
evaluation_solutions_json="../datasets/arc-agi_evaluation_solutions.json"
test_challenges_json="../datasets/arc-agi_test_challenges.json"

training_challenges, training_solutions, evaluation_challenges, \
evaluation_solutions, test_challenges = load_challenges_and_solutions(
                                        training_challenges_json,
                                        training_solutions_json,
                                        evaluation_challenges_json,
                                        evaluation_solutions_json,
                                        test_challenges_json,
                                    )

train_task_img_dict, _ = preprocess_data(training_challenges, training_solutions)
eval_task_img_dict, _ = preprocess_data(evaluation_challenges, evaluation_solutions)


def make_env(fixed_task: bool, 
             fixed_pair_idx: bool, 
             task_id: str, 
             pair_idx: int,
             ):
    def thunk():
        env = create_arc_env_coord(
                fixed_task=fixed_task, 
                fixed_pair_idx=fixed_pair_idx,
                task_id=task_id,
                pair_idx=pair_idx,
                training_challenges=training_challenges,
                training_solutions=training_solutions, 
                evaluation_challenges=evaluation_solutions,
                evaluation_solutions=evaluation_solutions,
                test_challenges=test_challenges,
                train_task_img_dict=train_task_img_dict,
                eval_task_img_dict=eval_task_img_dict,
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PatchEmbedding(nn.Module):
    """Patch embedding layer for ViT adapted to 30x180 input."""
    
    def __init__(self, grid_size=(30, 180), patch_size=15, embed_dim=768, vocab_size=12):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.n_patches = (grid_size[0] // patch_size) * (grid_size[1] // patch_size)
        
        # Token embedding for discrete values (0-10)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Patch projection
        self.patch_projection = nn.Linear(patch_size * patch_size * embed_dim, embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Position embedding
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Convert to patches: [batch, n_patches, patch_size, patch_size]
        patches = []
        for i in range(0, self.grid_size[0], self.patch_size):
            for j in range(0, self.grid_size[1], self.patch_size):
                patch = x[:, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        
        # Stack patches: [batch, n_patches, patch_size, patch_size]
        patches = torch.stack(patches, dim=1)
        
        # Token embedding: [batch, n_patches, patch_size, patch_size, embed_dim]
        # Clamp values to valid token range [0, vocab_size-1] and convert to long
        patches_clamped = torch.clamp(patches, 0, 11).long()
        patches = self.token_embedding(patches_clamped)
        
        # Flatten patches: [batch, n_patches, patch_size * patch_size * embed_dim]
        patches = patches.view(batch_size, self.n_patches, -1)
        
        # Project patches: [batch, n_patches, embed_dim]
        patches = self.patch_projection(patches)
        
        # Add class token: [batch, 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate: [batch, n_patches + 1, embed_dim]
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # Add position embedding
        x = x + self.position_embedding
        
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        output = self.proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class Agent(nn.Module):
    """Vision Transformer based Actor-Critic network for PPO with grid observations."""
    
    def __init__(self, grid_size=(30, 180), patch_size=15, embed_dim=768, 
                 num_heads=12, num_layers=12, mlp_ratio=4, vocab_size=12, 
                 action_size=9900, dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(grid_size, patch_size, embed_dim, vocab_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, action_size)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
    
    def forward(self, x):
        """Forward pass returning both action probabilities and state value."""
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Use CLS token (first token) for classification
        cls_token = x[:, 0]
        
        # Actor and critic heads
        return cls_token
    
    def get_value(self, x):
        cls_token = self.forward(x)
        return self.critic(cls_token)

    def get_action_and_value(self, x, action=None):
        cls_token = self.forward(x)
        action_logits = self.actor(cls_token)
        probs = Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(cls_token)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    task_id = '3cd86f4f'
    run_name = f"{'ARC_AGI_2'}__task_id_{task_id}_paird_idx_{0}_{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(fixed_task=True, 
                    fixed_pair_idx=True, 
                    task_id='3cd86f4f', 
                    pair_idx=0,) for i in range(args.num_envs)],
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (30, 180)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs,)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            dict_action = vectorized_action_converter(action.cpu())
            # print(dict_action)
            next_obs, reward, terminations, truncations, infos = envs.step(dict_action)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            print(infos.keys())
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
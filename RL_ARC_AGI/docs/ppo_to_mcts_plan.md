# ARC AGI PPO→MCTS Migration Notes

## 1. Single Source of Truth: Current PPO + ViT Stack

### 1.1 Task and Environment Loading
- `train_ppo_parallel.py` bootstraps dataset access by loading the ARC-AGI JSON challenge/solution files and precomputing XYXYXY image pairs through `preprocess_data` before instantiating the vectorized grid environments.【F:RL_ARC_AGI/train_ppo_parallel.py†L254-L303】
- The coordinate-aware environment (`create_arc_env_coord`) yields a `gymnasium` compatible interface with observations drawn directly from the mutable grid tensor maintained by `ArcAgiGridEnvCoord` while exposing metadata such as candidate colors/sizes for masking.【F:RL_ARC_AGI/arc_agi_grid_env_coord.py†L300-L328】

### 1.2 Observation and Action Surfaces
- Each observation is a `30×180` integer grid composed of the current canvas and contextual panels; the environment returns a raw copy so the agent can normalize it downstream.【F:RL_ARC_AGI/arc_agi_grid_env_coord.py†L313-L328】
- The discrete action space factorizes into a color (10 options) and a `(row, col)` coordinate on the `30×30` output panel, giving `10 × 30 × 30 = 9,000` legal combinations that the trainer flattens into a single categorical index for PPO.【F:RL_ARC_AGI/train_ppo_parallel.py†L122-L200】【F:RL_ARC_AGI/arc_agi_grid_env_coord.py†L300-L304】
- `vectorized_action_converter` bridges the integer logits to environment dictionaries, while `_create_action_mask` in the vectorized `Agent` suppresses colors or coordinates disallowed by the per-task metadata before sampling from the categorical policy.【F:RL_ARC_AGI/train_ppo_parallel.py†L436-L477】【F:RL_ARC_AGI/train_ppo_parallel.py†L122-L217】

### 1.3 Dynamics and Rewards
- Stepping applies the chosen color to the solution canvas, marking termination with a `-1` penalty if the color or cell is invalid, rewarding perfect completion with `+1`, and giving a small shaping reward for progress; `truncated` indicates success paths where the board is filled correctly.【F:RL_ARC_AGI/arc_agi_grid_env_coord.py†L431-L487】

### 1.4 Policy/Value Backbone and Normalization
- `PPOAgent` wraps a ViT-based actor-critic (`ActorCritic_ViT`) that embeds the grid into patches, runs twelve transformer blocks, and projects the CLS token into policy logits and a scalar value estimate, matching the flattened 9,000-action categorical output expected by PPO.【F:RL_ARC_AGI/ppo_agent.py†L45-L121】【F:RL_ARC_AGI/network/vit.py†L7-L202】
- Observation and reward running statistics are maintained for normalization prior to computing advantages, aligning PPO with the discrete, high-variance signal of the puzzle tasks.【F:RL_ARC_AGI/ppo_agent.py†L13-L111】【F:RL_ARC_AGI/ppo_agent.py†L214-L259】

### 1.5 Rollout Collection and Optimization Loop
- `ArcAgiVectorizedTrainer.collect_rollouts` steps a synchronized vector of environments, queries the policy with masking, executes actions until done, and stores transitions for GAE bootstrapping before handing the trajectory back to the PPO updater.【F:RL_ARC_AGI/train_ppo_parallel.py†L436-L522】
- The trainer periodically visualizes grids, logs to TensorBoard/W&B, evaluates aggregated metrics across devices, and iterates PPO epochs with clipped objectives and entropy bonuses supplied by the PPO updater.【F:RL_ARC_AGI/train_ppo_parallel.py†L640-L815】【F:RL_ARC_AGI/train_ppo_parallel.py†L524-L636】

## 2. Migration Blueprint: AlphaZero-style MCTS

### 2.1 Goals and Guiding Principles
- Replace on-policy PPO rollouts with search-guided self-play trajectories that pair every encountered grid with an MCTS-improved policy target (`π`) and terminal value (`z`).
- Reuse the ViT backbone as the shared policy/value head, retaining action masking to respect puzzle constraints while exposing prior probabilities needed by MCTS.

### 2.2 State and Action Interfaces for Search
- **Canonical state**: Continue using the `30×180` grid but formalize a `GameState` wrapper around `ArcAgiGridEnvCoord` that freezes the numpy array, hash key, available actions, and terminal flag so the tree can clone without side effects.
- **Action encoding**: Keep the single-index mapping (color × row × col) for tree nodes while feeding mask-aware priors from the network to skip invalid children.
- **Terminal evaluation**: Treat `terminated` (failure) and `truncated` (successful completion) as absorbing leaves with utility `z ∈ {+1, -1}` and optionally include intermediate shaping rewards only for logging, not for training targets.

### 2.3 Neural Inference Service
- Factor out a `PolicyValueNet` module that wraps the existing ViT forward pass and returns `(masked_policy, value)` tensors for batches of game states; expose a fast `infer(states)` API for batched evaluation from MCTS.
- Ensure observation normalization mirrors the PPO setup (reuse `RunningMeanStd` or freeze statistics) so training targets remain consistent.

### 2.4 MCTS Core Implementation
- Implement a `PUCTNode` structure storing visit counts `N`, mean action values `Q`, prior probabilities `P`, and child references keyed by flattened action indices.
- Create a search loop (`MCTS.run(root_state)`) that:
  1. Repeatedly selects down the tree using `Q + c_{puct} · P · √(∑ N_parent)/(1 + N_child)` while honoring action masks.
  2. Expands new leaf nodes by querying the ViT inference service for priors/value, attaching only valid-action children.
  3. Backpropagates the negated value (since ARC puzzles are single-agent, the negation step can be omitted or treated as maximizing utility for the same player).
- Include temperature-controlled action sampling from visit counts (e.g., softmax over `N^1/τ`) to diversify early moves.

### 2.5 Self-play Data Generation Loop
- For each selected ARC task, instantiate a fresh environment, wrap it as a `GameState`, and run MCTS per move to choose the next action; record `(state, π, z)` triples where `π` is the normalized visit count vector.
- Upon episode completion, set `z = +1` for solved puzzles and `z = -1` for failures; propagate the final outcome back to all intermediate steps before storing the batch in a replay buffer.
- Maintain a FIFO or prioritized buffer to sample minibatches for training, mirroring AlphaZero’s dataset curation.

### 2.6 Network Training and Optimization
- Swap the PPO loss with a composite AlphaZero loss: cross-entropy between predicted policy logits and target `π`, mean-squared error between predicted value and `z`, plus optional L2 regularization.
- Use the same optimizer/scheduler scaffolding already in place for PPO, adjusting learning-rate schedules to match the larger, replay-based dataset.
- Periodically freeze evaluation checkpoints and gate self-play policy updates via `arena` comparisons (optional but helps stability).

### 2.7 Integration and Tooling
- Introduce configuration toggles (Hydra/W&B) to switch between PPO and MCTS modes while reusing logging, visualization, and dataset loading code.
- Add unit tests for `GameState` cloning, action masking within the search, and deterministic rollouts on fixed tasks to ensure the search/planning stack respects ARC constraints before training at scale.

### 2.8 Incremental Migration Strategy
1. **Refactor** the ViT module into a reusable `PolicyValueNet` class without altering PPO behavior (guarded by tests).
2. **Prototype** a minimal MCTS on top of a frozen PPO policy to validate environment integration and visit-count sampling.
3. **Implement** the self-play dataset writer and AlphaZero loss, train on a small subset, and verify convergence on held-out puzzles.
4. **Scale** to the full training set, add evaluation harnesses, and retire PPO-specific components once MCTS performance surpasses the baseline.

### 2.9 Compute Strategy: CPU Search, GPU Inference, and Multi-GPU Training
- **Search placement**: Run the tree policy/backup loops on CPU threads (e.g., a `ThreadPoolExecutor` or Ray actors) so that hundreds of light-weight simulations can execute concurrently without being bottlenecked by GPU launch latency. The CPU search workers only maintain node statistics and enqueue observation tensors for batched inference.
- **Inference service**: Funnel expansion requests from CPU workers to a shared GPU inference queue that collates states into batches before calling the ViT `PolicyValueNet`. This keeps the GPU saturated while decoupling search step latency from neural evaluation throughput.
- **Training backend**: Persist self-play trajectories to disk or a shared replay buffer and launch a separate trainer process wrapped in PyTorch DDP/FSDP across multiple GPUs. Each rank samples mini-batches, computes the AlphaZero policy/value loss, and synchronizes gradients via `DistributedDataParallel`, reusing the existing optimizer configuration with per-rank gradient scaling.
- **Parameter updates**: Periodically broadcast the newest checkpoint from the multi-GPU trainer back to the CPU/GPU inference workers (e.g., via shared filesystem or RPC) so that self-play uses up-to-date weights while keeping search fully CPU-hosted. This mirrors AlphaZero's producer-consumer loop and is straightforward to implement with the current experiment management tooling.

This roadmap keeps the current data loading and network assets intact while layering the planning machinery required for an AlphaZero-style agent over the ARC puzzle domain.

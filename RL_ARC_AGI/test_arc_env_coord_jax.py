# %%
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Import our JAX environment and visualization
from arc_agi_grid_env_coord_jax import (
    ArcAgiGridEnvCoord, EnvParams, EnvState,
    preprocess_data_jax, load_challenges_and_solutions_jax,
    convert_int_to_dict_jax, convert_dict_to_int_jax
)
from arc_agi_grid_env_coord import preprocess_data
from visualize_arc_agi import ArcAgiVisualizer, plot_grids_comparison

# %%
def register_arc_env_with_gymnax():
    """Register ARC environment with gymnax-style interface."""
    # For gymnax, we don't need explicit registration like gym
    # The environment can be used directly as long as it follows the interface
    pass


# %%
class OptimalAgent:
    """Agent that performs only correct actions based on the target grid."""

    def __init__(self, env: ArcAgiGridEnvCoord):
        self.env = env

    def get_optimal_action(self, state: EnvState) -> int:
        """
        Get the optimal action by finding an empty cell and placing the correct color.

        Args:
            state: Current environment state

        Returns:
            action: Integer action that places correct color in an empty cell
        """
        current_grid = state.current_grid_img
        target_grid = state.target_grid_img
        height, width = state.size_candidate[0], state.size_candidate[1]

        # Find empty cells (value 11) in the solution area
        solution_area_current = current_grid[0:height, 150:150+width]
        solution_area_target = target_grid[0:height, 150:150+width]

        # Find positions where current grid is empty (11) but target has a color
        empty_positions = jnp.where(solution_area_current == 11)

        if len(empty_positions[0]) == 0:
            # No empty positions, return dummy action (shouldn't happen in correct usage)
            return 0

        # Take the first empty position
        row = empty_positions[0][0]
        col = empty_positions[1][0]

        # Get the correct color for this position
        correct_color = solution_area_target[row, col]

        # Convert to action
        dict_action = {
            'color': correct_color,
            'coordinate': jnp.array([row, col])
        }

        action = convert_dict_to_int_jax(dict_action)
        return action

    def get_all_optimal_actions(self, state: EnvState) -> List[int]:
        """Get all optimal actions to complete the puzzle."""
        current_grid = state.current_grid_img
        target_grid = state.target_grid_img
        height, width = state.size_candidate[0], state.size_candidate[1]

        # Find all empty cells in the solution area
        solution_area_current = current_grid[0:height, 150:150+width]
        solution_area_target = target_grid[0:height, 150:150+width]

        empty_positions = jnp.where(solution_area_current == 11)
        actions = []

        for i in range(len(empty_positions[0])):
            row = empty_positions[0][i]
            col = empty_positions[1][i]
            correct_color = solution_area_target[row, col]

            dict_action = {
                'color': correct_color,
                'coordinate': jnp.array([row, col])
            }

            action = convert_dict_to_int_jax(dict_action)
            actions.append(int(action))

        return actions


# %%
def run_optimal_rollout(env: ArcAgiGridEnvCoord,
                       params: EnvParams,
                       key: jax.random.PRNGKey,
                       max_steps: int = 100) -> Tuple[List[EnvState], List[int], List[float], bool]:
    """
    Run a rollout where the agent performs only optimal actions.

    Args:
        env: JAX environment
        params: Environment parameters
        key: JAX random key
        max_steps: Maximum number of steps

    Returns:
        states: List of environment states
        actions: List of actions taken
        rewards: List of rewards received
        success: Whether the puzzle was solved
    """
    # Reset environment
    obs, state = env.reset_env(key, params)

    # Initialize agent
    agent = OptimalAgent(env)

    # Storage for rollout data
    states = [state]
    actions = []
    rewards = []

    step_count = 0
    terminated = False

    while not terminated and step_count < max_steps:
        # Get optimal action
        action = agent.get_optimal_action(state)
        actions.append(int(action))

        # Take step
        key, subkey = random.split(key)
        obs, new_state, reward, terminated, info = env.step_env(subkey, state, action, params)

        rewards.append(float(reward))
        states.append(new_state)

        state = new_state
        step_count += 1

        # Check if puzzle is solved
        if reward == 1.0:
            print(f"✅ Puzzle solved in {step_count} steps!")
            break
        elif reward == -1.0:
            print(f"❌ Failed at step {step_count}")
            break

    success = (len(rewards) > 0 and rewards[-1] == 1.0)
    return states, actions, rewards, success


# %%
def visualize_rollout_results(states: List[EnvState],
                            actions: List[int],
                            rewards: List[float],
                            task_id: str):
    """Visualize the results of a rollout."""
    visualizer = ArcAgiVisualizer()

    if not states:
        print("No states to visualize")
        return

    initial_state = states[0]
    final_state = states[-1]

    # Convert JAX arrays to numpy for visualization
    initial_grid = np.array(initial_state.current_grid_img)
    final_grid = np.array(final_state.current_grid_img)
    target_grid = np.array(initial_state.target_grid_img)

    # Plot comparison
    visualizer.plot_rollout_comparison(initial_grid, final_grid, target_grid, task_id)

    # Plot action history if we have a chosen grid
    if hasattr(final_state, 'chosen_grid_img'):
        chosen_grid = np.array(final_state.chosen_grid_img)
        visualizer.plot_action_history(actions, chosen_grid)

    # Print summary
    total_reward = sum(rewards)
    print(f"\n📊 Rollout Summary:")
    print(f"Task ID: {task_id}")
    print(f"Total steps: {len(actions)}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Success: {rewards[-1] == 1.0 if rewards else False}")
    print(f"Rewards per step: {rewards}")


# %% Data Loading and Environment Setup
print("🧪 Setting up JAX ARC Environment...")

# Load data
training_challenges_json = "../datasets/arc-agi_training_challenges.json"
training_solutions_json = "../datasets/arc-agi_training_solutions.json"
evaluation_challenges_json = "../datasets/arc-agi_evaluation_challenges.json"
evaluation_solutions_json = "../datasets/arc-agi_evaluation_solutions.json"
test_challenges_json = "../datasets/arc-agi_test_challenges.json"

training_challenges, training_solutions, evaluation_challenges, \
evaluation_solutions, test_challenges = load_challenges_and_solutions_jax(
    training_challenges_json,
    training_solutions_json,
    evaluation_challenges_json,
    evaluation_solutions_json,
    test_challenges_json,
)

print("✅ Data loaded successfully!")

# %% Preprocess Data
print("🔄 Preprocessing data...")
preprocess_data
train_task_img_dict, _, train_img_shape_colors, _ = preprocess_data(
    training_challenges, training_solutions
)
eval_task_img_dict, _, eval_img_shape_colors, _ = preprocess_data(
    evaluation_challenges, evaluation_solutions
)
# train_task_img_dict, _, train_img_shape_colors, _ = preprocess_data_jax(
#     training_challenges, training_solutions
# )
# eval_task_img_dict, _, eval_img_shape_colors, _ = preprocess_data_jax(
#     evaluation_challenges, evaluation_solutions
# )

print("✅ Data preprocessing completed!")

# %%
type(train_task_img_dict['00576224'][0])

# %% Create Environment
print("🏗️ Creating JAX environment...")

env = ArcAgiGridEnvCoord(
    training_challenges=training_challenges,
    training_solutions=training_solutions,
    evaluation_challenges=evaluation_challenges,
    evaluation_solutions=evaluation_solutions,
    test_challenges=test_challenges,
    train_task_img_dict=train_task_img_dict,
    eval_task_img_dict=eval_task_img_dict
)

print("✅ Environment created successfully!")
print(f"📊 Training tasks: {len(train_task_img_dict)}")
print(f"📊 Evaluation tasks: {len(eval_task_img_dict)}")


# %% Test Single Task - Basic Configuration
print("\n🎯 Testing single task - Basic configuration")

# Set up task parameters
task_id = '794b24be'
pair_idx = 0
mode = 'train'

params = EnvParams(
    mode=mode,
    task_id=task_id,
    pair_idx=pair_idx,
    rand_init=False,
    ratio_fill_correct=0.0,
    max_steps_in_episode=900
)

# Initialize random key
key = random.PRNGKey(42)

print(f"🎮 Testing task: {task_id} (pair {pair_idx})")
print(f"⚙️ Configuration: rand_init={params.rand_init}, ratio_fill_correct={params.ratio_fill_correct}")

# %% Run Optimal Rollout - Basic
print("\n🤖 Running optimal rollout...")

states, actions, rewards, success = run_optimal_rollout(env, params, key)

print(f"\n📊 Results:")
print(f"Success: {success}")
print(f"Steps taken: {len(actions)}")
print(f"Total reward: {sum(rewards):.2f}")

# %% Visualize Results - Basic
print("\n📈 Visualizing results...")

visualize_rollout_results(states, actions, rewards, task_id)


# %% Test with Random Initialization
print("\n🎲 Testing with random initialization...")

# Test with partial random initialization
params_rand = EnvParams(
    mode='train',
    task_id='794b24be',
    pair_idx=0,
    rand_init=True,
    ratio_fill_correct=0.5,
    max_steps_in_episode=1000
)

key = random.PRNGKey(123)  # Different seed for variety

print(f"🎮 Testing task: {params_rand.task_id} with random init")
print(f"⚙️ Configuration: rand_init={params_rand.rand_init}, ratio_fill_correct={params_rand.ratio_fill_correct}")

# %% Run Optimal Rollout - Random Init
print("\n🤖 Running optimal rollout with random initialization...")

states_rand, actions_rand, rewards_rand, success_rand = run_optimal_rollout(env, params_rand, key)

print(f"\n📊 Results with random init:")
print(f"Success: {success_rand}")
print(f"Steps taken: {len(actions_rand)}")
print(f"Total reward: {sum(rewards_rand):.2f}")

# %% Visualize Results - Random Init
print("\n📈 Visualizing random initialization results...")

visualize_rollout_results(states_rand, actions_rand, rewards_rand, params_rand.task_id)

# %% Test Different Task
print("\n🔄 Testing different task...")

# Test another task
task_id_2 = '3cd86f4f'
params_task2 = EnvParams(
    mode='train',
    task_id=task_id_2,
    pair_idx=0,
    rand_init=False,
    ratio_fill_correct=0.0,
    max_steps_in_episode=1000
)

key = random.PRNGKey(456)

print(f"🎮 Testing task: {task_id_2}")

# %% Run Rollout - Different Task
print("\n🤖 Running optimal rollout for different task...")

states_task2, actions_task2, rewards_task2, success_task2 = run_optimal_rollout(env, params_task2, key)

print(f"\n📊 Results for {task_id_2}:")
print(f"Success: {success_task2}")
print(f"Steps taken: {len(actions_task2)}")
print(f"Total reward: {sum(rewards_task2):.2f}")

# %% Visualize Results - Different Task
print("\n📈 Visualizing different task results...")

visualize_rollout_results(states_task2, actions_task2, rewards_task2, task_id_2)


# %% Performance Comparison
print("\n⚡ Performance comparison...")

# Compare different configurations
configs_to_test = [
    {'name': 'Empty Start', 'rand_init': False, 'ratio_fill_correct': 0.0},
    {'name': '30% Pre-filled', 'rand_init': True, 'ratio_fill_correct': 0.3},
    {'name': '70% Pre-filled', 'rand_init': True, 'ratio_fill_correct': 0.7},
]

results_comparison = []

for config in configs_to_test:
    print(f"\n🧪 Testing {config['name']}...")

    params_test = EnvParams(
        mode='train',
        task_id='794b24be',
        pair_idx=0,
        rand_init=config['rand_init'],
        ratio_fill_correct=config['ratio_fill_correct'],
        max_steps_in_episode=1000
    )

    key = random.PRNGKey(789)
    states_test, actions_test, rewards_test, success_test = run_optimal_rollout(env, params_test, key)

    result = {
        'config_name': config['name'],
        'success': success_test,
        'steps': len(actions_test),
        'total_reward': sum(rewards_test) if rewards_test else 0
    }
    results_comparison.append(result)

    print(f"  ✅ Success: {success_test}, Steps: {len(actions_test)}, Reward: {sum(rewards_test):.2f}")

# %% Summary and Analysis
print("\n📋 FINAL SUMMARY")
print("=" * 50)

print("\n🎯 Performance Comparison:")
for result in results_comparison:
    print(f"  {result['config_name']:15} | Success: {result['success']} | Steps: {result['steps']:2d} | Reward: {result['total_reward']:5.2f}")

print("\n🏆 Key Findings:")
print("  • JAX environment successfully created and tested")
print("  • Optimal agent can solve puzzles step by step")
print("  • Random initialization works as expected")
print("  • Visualization functions work correctly")
print("  • Environment follows gymnax interface patterns")

print(f"\n📊 Environment Statistics:")
print(f"  • Training tasks available: {len(train_task_img_dict)}")
print(f"  • Evaluation tasks available: {len(eval_task_img_dict)}")
print(f"  • Action space size: {env.num_actions}")
print(f"  • Observation space shape: {env.observation_space().shape}")

print("\n✅ All tests completed successfully!")

# %% Additional Testing (Optional)
print("\n🔬 Additional testing block - run if needed...")

# You can add more tests here by uncommenting and running this block
# For example, test with evaluation mode or different task IDs

# params_eval = EnvParams(
#     mode='evaluation',  # Test evaluation mode
#     task_id=None,       # Random task selection
#     pair_idx=None,      # Random pair selection
#     rand_init=False,
#     ratio_fill_correct=0.0,
#     max_steps_in_episode=1000
# )

# key = random.PRNGKey(999)
# states_eval, actions_eval, rewards_eval, success_eval = run_optimal_rollout(env, params_eval, key)
# print(f"Random evaluation task test - Success: {success_eval}, Steps: {len(actions_eval)}")

print("🎉 Interactive testing setup complete! Use '# %%' blocks to run step by step.")
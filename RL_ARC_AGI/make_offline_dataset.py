# %%
import gymnasium as gym
from arc_agi_grid_env_coord import create_arc_env_coord, preprocess_data, load_challenges_and_solutions
from arc_agi_grid_env_coord import convert_dict_to_int, vectorized_convert_dict_to_int
import numpy as np
import h5py
from omegaconf import OmegaConf


# %%
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
train_task_img_dict, _, train_img_shape_colors, _ = preprocess_data(training_challenges, training_solutions)
eval_task_img_dict, _, eval_img_shape_colors, _= preprocess_data(evaluation_challenges, evaluation_solutions)

# %%
train_img_shape_colors
# %%
train_img_shape_colors['e9fc42f2']

# %%
# config = OmegaConf.load('./config/ppo_vector_env.yaml')
# task_id_list = list(config.environment.task_id_list)
# seed = config.environment.seed
# fixed_task = config.environment.fixed_task
# fixed_pair_idx = config.environment.fixed_pair_idx
# task_id = config.environment.task_id
# pair_idx = config.environment.pair_idx
# num_envs = config.environment.num_envs
# num_steps = config.environment.num_steps
# Get size_candidate and color_candidate from config if available

# %%
for task_id in train_task_img_dict.keys():
    img_shape_color_list = train_img_shape_colors[task_id]
    for pair_idx in range(len(train_task_img_dict[task_id])):
        row = img_shape_color_list[pair_idx].row
        col = img_shape_color_list[pair_idx].col
        size_candidate = [row, col]
        color_candidate = img_shape_color_list[pair_idx].color
        print(img_shape_color_list[pair_idx])
        print(task_id, pair_idx, size_candidate, color_candidate)
        env = make_env(fixed_task=True,
                        fixed_pair_idx=True,
                        task_id=task_id,
                        pair_idx=pair_idx,
                        training_challenges=training_challenges,
                        training_solutions=training_solutions,
                        evaluation_challenges=evaluation_challenges,
                        evaluation_solutions=evaluation_solutions,
                        test_challenges=test_challenges,
                        train_task_img_dict=train_task_img_dict,
                        eval_task_img_dict=eval_task_img_dict,
                    )()
        options = {'mode': 'train',
                'task_id': task_id,
                'pair_idx': pair_idx,
                'reset_sol_grid': 'padding',
                'size_candidate': size_candidate,
                'color_candidate': color_candidate,
                }
        state, info = env.reset(seed=1,
                            options=options,)
        done = False

        test_sol = env.unwrapped._target_grid_img[:30, 150:]
        # size_candidate = env.unwrapped.size_candidate
        total_reward = 0
        dataset = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        
        traj_len = size_candidate[0] * size_candidate[1]
        t = 0
        while not done:
            row = t // size_candidate[0]
            col = t % size_candidate[1]
            dict_action = {'color': test_sol[row, col],
                    'coordinate': [row, col]
                    }
            action = convert_dict_to_int(dict_action)
            next_state, reward, terminated, truncated, _ = env.step(dict_action)
            done = np.logical_or(terminated, truncated)
            dataset['states'].append(state)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['next_states'].append(next_state)
            dataset['dones'].append(done)
            state = next_state
            print(f"timestep {t}: {reward}, {terminated}, {truncated}")
            total_reward += reward
            if done:
                # env.plot_current_task_and_sol()
                env.plot_current_grid()
                print(env.unwrapped.size_candidate)
                break
            t += 1

        print(round(total_reward, 2))
        # HDF5로 저장
        with h5py.File(f'./offline_dataset/arcagi_{task_id}_{pair_idx}.h5', 'w') as f:
            for key, value in dataset.items():
                f.create_dataset(key, data=np.array(value))
    
# %%

# %%
import gymnasium as gym
from arc_agi_grid_env_coord import create_arc_env_coord, preprocess_data, load_challenges_and_solutions
from arc_agi_grid_env_coord import convert_dict_to_int, vectorized_convert_dict_to_int, make_env
import numpy as np
import h5py
from omegaconf import OmegaConf
import random
import minari
from minari import DataCollector
from pathlib import Path
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
# task_ids = random.sample(list(train_task_img_dict.keys()), 10)
# task_ids.extend(['3cd86f4f', '794b24be'])
# task_ids = list(set(task_ids))
# task_ids
task_ids = list(train_task_img_dict.keys())[:10]
task_ids
# %%
seed = 42
env = make_env(
        training_challenges=training_challenges,
        training_solutions=training_solutions,
        evaluation_challenges=evaluation_challenges,
        evaluation_solutions=evaluation_solutions,
        test_challenges=test_challenges,
        train_task_img_dict=train_task_img_dict,
        eval_task_img_dict=eval_task_img_dict,
    )()
env = DataCollector(env, record_infos=False)
# %%
# for task_id in train_task_img_dict.keys():
count = 1
for task_id in task_ids:
    img_shape_color_list = train_img_shape_colors[task_id]
    for pair_idx in range(len(train_task_img_dict[task_id])):
        # pair_idx = 1
        print(img_shape_color_list[pair_idx])
        row = img_shape_color_list[pair_idx].row
        col = img_shape_color_list[pair_idx].col
        size_candidate = [row, col]
        color_candidate = img_shape_color_list[pair_idx].color
        options = {'mode': 'train',
        'task_id': task_id,
        'pair_idx': pair_idx,
        'size_candidate': size_candidate,
        'color_candidate': color_candidate,
        }
        # num_episodes = max(row*col, 100)
        num_episodes = 1
        for p in range(num_episodes):
            state, info = env.reset(seed=1,
                                options=options,)
            done = False
            test_sol = env.unwrapped._target_grid_img[:30, 150:]
            indices = np.where(test_sol != 10,)
            seed += 1
            non_mask_coord_lst = list(zip(indices[0], indices[1]))
            random.shuffle(non_mask_coord_lst)
            total_reward = 0
            traj_len = len(non_mask_coord_lst)
            t = 0
            for coord in non_mask_coord_lst:
                row = coord[0]
                col = coord[1]
                dict_action = {'color': test_sol[row, col],
                        'coordinate': [row, col]
                        }
                action = convert_dict_to_int(dict_action)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = np.logical_or(terminated, truncated)
                state = next_state
                # print(f"timestep {t}: {dict_action}, {reward}, {terminated}, {truncated}")
                total_reward += reward
                t += 1
                if done:
                    # env.plot_current_task_and_sol()
                    # env.plot_current_grid()
                    break
            success_reward = (((traj_len-1) * 0.05 + 1))
            total_reward = np.round(total_reward, 2)
            is_success = (total_reward == success_reward)
            print(f"Count: {count}, Task: {task_id}, Pair: {pair_idx}, Episode: {p}. Sucess: {is_success}, Total Reward: {total_reward}, Traj Length: {traj_len}")
            count += 1
        # break
dataset = env.create_dataset(
        # dataset_id=f"arc_agi/{task_id}-v0",
        # dataset_id=f"arc_agi/{task_id}_{pair_idx}-v1",
        dataset_id=f"arc_agi/arc_agi_{count}-v0",
        algorithm_name="ground_truth",
        code_permalink="https://github.com/Farama-Foundation/Minari",
        author="Kukjin Kim",
        author_email="ye20013@gmail.com"
        )
    # break


# %%

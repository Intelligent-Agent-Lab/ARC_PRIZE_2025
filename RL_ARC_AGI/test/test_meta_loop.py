# %%
from env.meta_arc_agi_grid_env_coord import ActiveShapeColorOntHot, MetaArcAgiGridEnvCoord, SelectedMetaArcAgiGridEnv, load_challenges_and_solutions
import gymnasium as gym
# %%
training_challenges_json = "../../datasets/arc-agi_training_challenges.json"
training_solutions_json = "../../datasets/arc-agi_training_solutions.json"
evaluation_challenges_json = "../../datasets/arc-agi_evaluation_challenges.json"
evaluation_solutions_json = "../../datasets/arc-agi_evaluation_solutions.json"
test_challenges_json = "../../datasets/arc-agi_test_challenges.json"

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

adapt_train_envs = make_meta_task_env(meta_train_dataset,
                                meta_eval_dataset,
                                meta_test_dataset,
                                mode='meta_train',
                                phase='train',
                                task_id='794b24be',
                                rand_init=False,
                                )
eval_test_envs = make_meta_task_env(meta_train_dataset,
                                meta_eval_dataset,
                                meta_test_dataset,
                                mode='meta_train',
                                phase='test',
                                task_id='794b24be',
                                rand_init=False,
                                )
# %%
len(adapt_train_envs)

# %%
len(adapt_train_envs.envs)
# %%
len(eval_test_envs.envs)
# %%
print(adapt_train_envs.envs[0].pair_idx)
print(adapt_train_envs.envs[1].pair_idx)
print(adapt_train_envs.envs[2].pair_idx)
# %%
obs, info = adapt_train_envs.reset()
# %%

# %%
from env.visualize_arc_agi import ArcAgiVisualizer
v = ArcAgiVisualizer()
pair_idx = 0
# %%
v.plot_current_grid(obs, '794b24be', pair_idx)
# %%
target_grid_img = info['target_grid_img'][pair_idx]
target_grid_img
# %%
v.plot_target_grid(target_grid_img, '794b24be', pair_idx)
# %%

# %%
from env.meta_arc_agi_grid_env_coord import convert_dict_to_int, convert_int_to_dict
import numpy as np 
coord_list = np.where(env.active_shape == 1)
coord_list
non_mask_coord_lst = list(zip(coord_list[0], coord_list[1]))
# %%

#  %%
t = 0
for coord in non_mask_coord_lst:
    row = coord[0]
    col = coord[1]
    dict_action = {'color': test_sol[row, col],
            'coordinate': [row, col]
            }
    int_action = convert_dict_to_int(dict_action)
    # if t > 898:
    #     int_action = env.action_space.sample()
    #     dict_action = convert_int_to_dict(int_action)
    #     print(f"last action: {dict_action}")
    print(dict_action)
    next_obs, reward, terminated, truncated, info = env.step(int_action)
    t += 1
    print(f"timestep {t}: {reward}, {terminated}, {truncated}")
    total_reward += reward
    if terminated or truncated: 
        break
# %%
v.plot_current_grid(next_obs, '794b24be', 0)


# %%

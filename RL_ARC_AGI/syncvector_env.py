# %%
import gymnasium as gym 
from arc_agi_grid_env_coord import ArcAgiGridEnvCoord, \
                                    ArcAgiWrapper, \
                                    create_arc_env_coord, \
                                    load_challenges_and_solutions, \
                                    preprocess_data
from gymnasium.envs.registration import register, registry
import torch 

# %%
training_challenges_json="../datasets/arc-agi_training_challenges.json"
training_solutions_json="../datasets/arc-agi_training_solutions.json" 
evaluation_challenges_json="../datasets/arc-agi_evaluation_challenges.json"
evaluation_solutions_json="../datasets/arc-agi_evaluation_solutions.json"
test_challenges_json="../datasets/arc-agi_test_challenges.json"

# %%
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


# %%
env = create_arc_env_coord(
        fixed_task=True, 
        fixed_pair_idx=True,
        task_id='794b24be',
        pair_idx=None,
        training_challenges=training_challenges,
        training_solutions=training_solutions, 
        evaluation_challenges=evaluation_solutions,
        evaluation_solutions=evaluation_solutions,
        test_challenges=test_challenges,
        train_task_img_dict=train_task_img_dict,
        eval_task_img_dict=eval_task_img_dict,
    )
# %%
# options = {'mode': 'train',
#            'task_id': '794b24be', # 794b24be, 3cd86f4f
#            'reset_sol_grid': 'padding',}
obs, info = env.reset(seed=12,)
                        # options=options)
test_sol = env.unwrapped._target_grid_img[:30,150:]
total_reward = 0
env.plot_target_grid()
# %%
test_sol.shape
# %%
env.plot_current_grid()
# %%
for t in range(900):
    row = t // 30
    col = t % 30
    action = {'color': test_sol[row, col],
              'coordinate': [row, col]
              }
    
    if t > 898:
        action = env.action_space.sample()
        print(f"last action: {action}")
    print(action)
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"timestep {t}: {reward}, {terminated}, {truncated}")
    total_reward += reward
    if terminated or truncated: 
        break
env.plot_current_grid()
# %%
def make_env(fixed_task: bool, 
             fixed_pair_idx: bool, 
             task_id: str, 
             pair_idx: int,
             ):
    def thunk():
        env = create_arc_env_coord(
                fixed_task=True, 
                fixed_pair_idx=True,
                task_id='794b24be',
                pair_idx=None,
                training_challenges=training_challenges,
                training_solutions=training_solutions, 
                evaluation_challenges=evaluation_solutions,
                evaluation_solutions=evaluation_solutions,
                test_challenges=test_challenges,
                train_task_img_dict=train_task_img_dict,
                eval_task_img_dict=eval_task_img_dict,
            )
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

# %%
num_envs = 4
envs = gym.vector.SyncVectorEnv(
    [make_env(True, True, '794b24be', 0) for i in range(num_envs)],
)
# %%
seed = 4
options = {'mode': 'train',
           'task_id': '794b24be', # 794b24be, 3cd86f4f
           'reset_sol_grid': 'padding',}
next_obs, _ = envs.reset(seed=seed,)
# # %%
next_obs.shape
num_rollout_steps = 128
for step in range(num_rollout_steps):
    action = envs.action_space.sample()
    next_obs, reward, terminations, truncations, infos = envs.step(action)
    print(action)
    print(next_obs)
    # break


# %%
import matplotlib.pyplot as plt
from matplotlib import colors

cmap = colors.ListedColormap(
    ['#000000', # 0: black
     '#0074D9', # 1: blue
     '#FF4136', # 2: red
     '#2ECC40', # 3: green
     '#FFDC00', # 4: yello
     '#8B00FF', # 5: gray
     '#F012BE', # 6: magenta
     '#FF851B', # 7: oragne
     '#7FDBFF', # 8: sky
     '#870C25', # 9: brwon
     '#AAAAAA', # 10: mask
     '#FFFFFF', # 11: empty
     ])
norm = colors.Normalize(vmin=0, vmax=11)
for i in range(4):
    w = 0.5
    input_matrix = next_obs[i]
    plt.imshow(input_matrix, cmap=cmap, norm=norm)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    '''Grid:'''
    plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
    plt.tick_params(axis='both', color='none', length=0)
    '''sub title:'''
    plt.show()

# %%
import numpy as np 
for i in range(0, 4):
    if i == 3:
        break
    print(np.unique(next_obs[i] == next_obs[i+1]))
# %%
np.unique(next_obs[1] == next_obs[2])
# %%
np.unique(next_obs[1] == next_obs[2])
# %%
plt.imshow(next_obs[0][:3, :3], cmap=cmap, norm=norm)

# %%
plt.imshow(next_obs[1][:3, :3], cmap=cmap, norm=norm)
# %%
plt.imshow(next_obs[2][:3, :3], cmap=cmap, norm=norm)
# %%
plt.imshow(next_obs[3][:3, :3], cmap=cmap, norm=norm)

# %%
envs.envs[0].plot_target_grid()
# %%
envs.envs[1].plot_target_grid()
# %%
envs.envs[2].plot_target_grid()
# %%
envs.envs[3].plot_target_grid()

# %%

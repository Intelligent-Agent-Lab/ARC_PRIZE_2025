# %%
import gymnasium as gym 
from arc_agi_grid_env_coord import ArcAgiGridEnvCoord, ArcAgiWrapper, create_arc_env_coord
from gymnasium.envs.registration import register, registry

# %%
env = create_arc_env_coord(
        training_challenges_json="../datasets/arc-agi_training_challenges.json",
        training_solutions_json="../datasets/arc-agi_training_solutions.json", 
        evaluation_challenges_json="../datasets/arc-agi_evaluation_challenges.json",
        evaluation_solutions_json="../datasets/arc-agi_evaluation_solutions.json",
        test_challenges_json="../datasets/arc-agi_test_challenges.json"
    )

# 794b24be train input pair 10개
# 8dab14c2 test input 4개
# 3cd86f4f
options = {'mode': 'train',
           'task_id': '794b24be',
           'reset_sol_grid': 'padding',}
obs, info = env.reset(seed=1,
                      options=options,)
# obs, info = env.reset(seed=1,
#                         mode='train',
#                         task_id='794b24be',
#                         reset_sol_grid='random')

# %%
env.print_train_task_info('794b24be')
# %%
env.plot_current_task_and_sol()
# %%
# 264363fd
env.plot_padded_task(task_id='794b24be', i=0)

# %%
env.plot_current_grid()
# %%
env.plot_target_grid()

# %%
env.plot_one_task(mode='train', 
                task_id='794b24be')
# %%
env.plot_original_task(task_id='794b24be',
            train_or_test='test',
            i=0,
            input_or_output='output',
            )

# %%
options = {'mode': 'train',
           'task_id': '794b24be', # 794b24be, 3cd86f4f
           'reset_sol_grid': 'padding',}
obs, info = env.reset(seed=12,
                        options=options)
test_sol = env.unwrapped._target_grid_seq[4500:]
total_reward = 0

# %%
env.plot_target_grid()
# %%
env.plot_current_grid()
# %%
for t in range(900):
    row = t // 30
    col = t % 30
    action = {'color': test_sol[t],
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

print(round(total_reward, 2))
# %%
env.plot_current_grid()

# %%
env.train_task_img_dict['3cd86f4f'][0].shape
# %%
len(env.train_task_img_dict['794b24be'])

# %%

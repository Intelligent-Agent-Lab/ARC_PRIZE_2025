# %%
import gymnasium as gym 
from arc_agi_grid_env_coord import ArcAgiGridEnvCoord, ArcAgiWrapper, create_arc_env_coord
from arc_agi_grid_env_coord import create_arc_env_coord, preprocess_data, load_challenges_and_solutions
from arc_agi_grid_env_coord import convert_dict_to_int, vectorized_convert_dict_to_int, make_env
from gymnasium.envs.registration import register, registry

# %%


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
env = make_env(training_challenges=training_challenges,
                training_solutions=training_solutions,
                evaluation_challenges=evaluation_challenges,
                evaluation_solutions=evaluation_solutions,
                test_challenges=test_challenges,
                train_task_img_dict=train_task_img_dict,
                eval_task_img_dict=eval_task_img_dict,
                )()
# %%
# 3cd86f4f
# 794b24be train input pair 10개
# 8dab14c2 test input 4개
task_id = '794b24be'
pair_idx = 0
options = {'mode': 'train',
        'task_id': task_id,
        'pair_idx': pair_idx,
        'rand_init': False,
        }
# %%
obs, info = env.reset(seed=42,
                      options=options,)
# %%
env.print_train_task_info(task_id)
# %%
env.plot_current_task_and_sol()
# %%
# 264363fd
env.plot_padded_task(task_id=task_id, i=0)

# %%
env.plot_current_grid()
# %%
env.plot_target_grid()

# %%
env.plot_one_task(mode='train', 
                task_id=task_id)
# %%
env.plot_original_task(task_id=task_id,
            train_or_test='test',
            i=pair_idx,
            input_or_output='output',
            )
# %%
task_id = '8dab14c2'
pair_idx = 0
# options = {'mode': 'train',
#         'task_id': task_id,
#         'pair_idx': pair_idx,
#         'rand_init': True,
#         'ratio_fill_correct': 0.5,
#         'ratio_fill_incorrect': 0.0,
#         }
options = {'mode': 'train',
        'task_id': task_id,
        'pair_idx': pair_idx,
        'rand_init': True,
        'ratio_fill_correct': 0.5,
        'ratio_fill_incorrect': 0.5,
        }
# %%
obs, info = env.reset(seed=42,
                      options=options,)
# %%
env.print_train_task_info(task_id)
# %%
env.plot_current_task_and_sol()
# %%
# 264363fd
env.plot_padded_task(task_id=task_id, i=0)

# %%
env.plot_current_grid()
# %%
env.plot_target_grid()

# %%
env.plot_one_task(mode='train', 
                task_id=task_id)
# %%
env.plot_original_task(task_id=task_id,
            train_or_test='test',
            i=pair_idx,
            input_or_output='output',
            )

# %%
num_envs = 4
envs = gym.vector.SyncVectorEnv([
    make_env(
        training_challenges=training_challenges,
        training_solutions=training_solutions,
        evaluation_challenges=evaluation_challenges,
        evaluation_solutions=evaluation_solutions,
        test_challenges=test_challenges,
        train_task_img_dict=train_task_img_dict,
        eval_task_img_dict=eval_task_img_dict
    ) for i in range(num_envs)
])
task_id = '3cd86f4f' # 8dab14c2, 794b24be, 3cd86f4f
# options = {'mode': 'train',
#         'task_id': task_id,
#         'pair_idx': pair_idx,
#         'rand_init': False,
#         'ratio_fill_correct': 0.0,
#         'ratio_fill_incorrect': 0.0,
#         }
# options = {'mode': 'train',
#         'task_id': task_id,
#         'pair_idx': None,
#         'rand_init': False,
#         'ratio_fill_correct': 0.0,
#         'ratio_fill_incorrect': 0.0,
#         }
options = {'mode': 'train',
        'task_id': task_id,
        'pair_idx': None,
        'rand_init': True,
        'ratio_fill_correct': 0.5,
        'ratio_fill_incorrect': 0.5,
        }
# %%
obs, info = envs.reset(seed=42, options=options)
# %%
envs.envs[0].plot_current_grid()
# %%
envs.envs[0].plot_target_grid()
# %%
envs.envs[1].plot_current_grid()
# %%
envs.envs[1].plot_target_grid()
# %%
envs.envs[2].plot_current_grid()
# %%
envs.envs[2].plot_target_grid()
# %%
envs.envs[3].plot_current_grid()
# %%
envs.envs[3].plot_target_grid()
# %%
print(infoinfos)
# %%

# %%
import gymnasium as gym 
from arc_agi_grid_env_coord import ArcAgiGridEnvCoord, ArcAgiWrapper, create_arc_env_coord
from arc_agi_grid_env_coord import create_arc_env_coord, preprocess_data, load_challenges_and_solutions
from arc_agi_grid_env_coord import convert_dict_to_int, vectorized_convert_dict_to_int, make_env
from gymnasium.envs.registration import register, registry

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

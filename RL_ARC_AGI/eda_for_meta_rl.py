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
task_id_list = list(train_task_img_dict.keys())
max_train_input_output_pairs = -1
for task_id, xyxyxy_pairs in train_task_img_dict.items():
    max_train_input_output_pairs = max(len(xyxyxy_pairs), max_train_input_output_pairs)
print(max_train_input_output_pairs)
# %%
task_id_list = list(training_challenges.keys())
max_train_input_output_pairs = -1
max_test_input_output_pairs = -1
min_train_input_output_pairs = 100
min_test_input_output_pairs = 100
for task_id, train_test_dict in training_challenges.items():
    train_input_ouput_arrays = train_test_dict['train']
    test_input_ouput_arrays = train_test_dict['test']
    max_train_input_output_pairs = max(len(train_input_ouput_arrays), max_train_input_output_pairs)
    max_test_input_output_pairs = max(len(test_input_ouput_arrays), max_test_input_output_pairs)
    min_train_input_output_pairs = min(len(train_input_ouput_arrays), min_train_input_output_pairs)
    min_test_input_output_pairs = min(len(test_input_ouput_arrays), min_test_input_output_pairs)
    
print(max_train_input_output_pairs)
print(min_train_input_output_pairs)
print(max_test_input_output_pairs)
print(min_test_input_output_pairs)
# %%
task_id_list = list(test_challenges.keys())
max_train_input_output_pairs = -1
max_test_input_output_pairs = -1
min_train_input_output_pairs = 100
min_test_input_output_pairs = 100
for task_id, train_test_dict in test_challenges.items():
    train_input_ouput_arrays = train_test_dict['train']
    test_input_ouput_arrays = train_test_dict['test']
    max_train_input_output_pairs = max(len(train_input_ouput_arrays), max_train_input_output_pairs)
    max_test_input_output_pairs = max(len(test_input_ouput_arrays), max_test_input_output_pairs)
    min_train_input_output_pairs = min(len(train_input_ouput_arrays), min_train_input_output_pairs)
    min_test_input_output_pairs = min(len(test_input_ouput_arrays), min_test_input_output_pairs)
    
print(max_train_input_output_pairs)
print(min_train_input_output_pairs)
print(max_test_input_output_pairs)
print(min_test_input_output_pairs)

# %%


# test code in-file
# %%
from env.meta_preprocess import load_challenges_and_solutions, generate_meta_dataset

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
task_id_list = list(training_challenges.keys())
max_train_input_output_pairs = -1
max_test_input_output_pairs = -1
min_train_input_output_pairs = 100
min_test_input_output_pairs = 100
max_train_pair_id = None
max_test_pair_id = None
min_train_pair_id = None
min_test_pair_id = None
for task_id, train_test_dict in training_challenges.items():
    train_input_ouput_arrays = train_test_dict['train']
    test_input_ouput_arrays = train_test_dict['test']
    if len(train_input_ouput_arrays) > max_train_input_output_pairs:
        max_train_input_output_pairs = len(train_input_ouput_arrays)
        max_train_pair_id = task_id
    if len(test_input_ouput_arrays) > max_test_input_output_pairs:
        max_test_input_output_pairs = len(test_input_ouput_arrays)
        max_test_pair_id = task_id
    if len(train_input_ouput_arrays) < min_train_input_output_pairs:
        min_train_input_output_pairs = len(train_input_ouput_arrays)
        min_train_pair_id = task_id
    if len(test_input_ouput_arrays) < min_test_input_output_pairs:
        min_test_input_output_pairs = len(test_input_ouput_arrays)
        min_test_pair_id = task_id

# %%
print(max_train_input_output_pairs)
print(max_train_pair_id)
print(min_train_input_output_pairs)
print(min_train_pair_id)
print(max_test_input_output_pairs)
print(max_test_pair_id)
print(min_test_input_output_pairs)
print(min_test_pair_id)

# %%
print(len(training_challenges['794b24be']['train']))
# %%
print(len(training_challenges['794b24be']['test']))

# %%
meta_train_dataset = generate_meta_dataset(training_challenges, training_solutions)
meta_eval_dataset = generate_meta_dataset(evaluation_challenges, evaluation_solutions)
meta_test_dataset = generate_meta_dataset(test_challenges, None)
# %%
meta_train_dataset['794b24be']['train_data']
print(len(meta_train_dataset['794b24be']['train_data']))
# %%
print(len(meta_train_dataset['794b24be']['train_info']))
# %%
print(len(meta_train_dataset['794b24be']['train_index_map']))
# %%
print((meta_train_dataset['794b24be']['train_index_map']))


# %%
meta_train_dataset['794b24be']['test_data']
print(len(meta_train_dataset['794b24be']['test_data']))
# %%
meta_train_dataset['794b24be']['test_info']
print(len(meta_train_dataset['794b24be']['test_info']))
# %%
print((meta_train_dataset['794b24be']['test_index_map']))

# %%
from env.visualize_arc_agi import ArcAgiVisualizer
visualizer = ArcAgiVisualizer()
# %%
task_id = '794b24be'
pair_idx = 1
meta_train_xyxy_ex = meta_train_dataset[task_id]['train_data'][pair_idx]
meta_train_xyxy_ex
visualizer.plot_target_grid(meta_train_xyxy_ex, task_id, pair_idx)
visualizer.plot_one_task(training_challenges,
                            training_solutions,
                            task_id, 
                            mode='train',
                        )
visualizer.plot_padded_task(meta_train_dataset,
                            task_id,
                            'train',
                            pair_idx,
                            )

# %%
meta_test_task_id = list(meta_test_dataset.keys())[0]
print(meta_test_task_id)
# %%
visualizer.plot_padded_task(meta_test_dataset,
                            meta_test_task_id,
                            'test',
                            pair_idx,
                            )
# %%
print(len(test_challenges[meta_test_task_id]['train']))
print(len(test_challenges[meta_test_task_id]['test']))
print(len(meta_test_dataset[meta_test_task_id]['train_data']))
print(len(meta_test_dataset[meta_test_task_id]['test_data']))

# %%
print(meta_train_dataset['794b24be']['train_info'][0])
# %%
print(meta_train_dataset['794b24be']['test_info'][0])
# %%
print(meta_test_dataset[meta_test_task_id]['train_index_map'])
# %%
print(meta_test_dataset[meta_test_task_id]['test_index_map'])
# %%



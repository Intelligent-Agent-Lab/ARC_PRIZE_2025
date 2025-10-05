# %%
from meta_arc_agi_grid_env_coord import MetaArcAgiGridEnvCoord, load_challenges_and_solutions
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
# %%
from meta_preprocess import generate_meta_dataset
meta_train_dataset = generate_meta_dataset(training_challenges, training_solutions)
meta_eval_dataset = generate_meta_dataset(evaluation_challenges, evaluation_solutions)
meta_test_dataset = generate_meta_dataset(test_challenges, None)
# %%
env = MetaArcAgiGridEnvCoord(meta_train_dataset,
                             meta_eval_dataset,
                             meta_test_dataset)
# %%
seed = 35
options = {'mode': 'meta_train',
           'phase': 'train',
           'task_id': '794b24be',
           'pair_idx': 0,
           'rand_init': True,}

obs, info = env.reset(seed, options)
# %%
from visualize_arc_agi import ArcAgiVisualizer
v = ArcAgiVisualizer()

# %%
v.plot_current_grid(obs, '794b24be', 0)
# %%

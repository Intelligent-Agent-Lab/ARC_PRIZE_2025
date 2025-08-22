from gymnasium.envs.registration import register

register(
    id='ArcAgiGrid-v0',
    entry_point='arc_agi_grid_env:ArcAgiGridEnv',
    max_episode_steps=900,
    kwargs={
        'training_challenges_json': '../datasets/arc-agi_training_challenges.json',
        'training_solutions_json': '../datasets/arc-agi_training_solutions.json',
        'evaluation_challenges_json': '../datasets/arc-agi_evaluation_challenges.json',
        'evaluation_solutions_json': '../datasets/arc-agi_evaluation_solutions.json',
        'test_challenges_json': None,
    }
)
import spatialreasoners as sr

if __name__ == '__main__':
    # 🧠 Enhanced spatial reasoning training with DiT
    sr.run_training(overrides=[
        "experiment=mnist_sudoku",           # Use spatial reasoning experiment
        "trainer.max_epochs=100",            # Extended training for spatial patterns
        "data_loader.train.batch_size=24",   # Optimized batch size for spatial learning
        "variable_mapper.name=image",        # Override to use image variable mapper
        "time_sampler=sequential",           # Sequential time sampling for spatial reasoning
        "wandb.tags=[spatial_reasoning, enhanced_dit, sequential_sampling]"  # Track spatial reasoning experiments
    ])
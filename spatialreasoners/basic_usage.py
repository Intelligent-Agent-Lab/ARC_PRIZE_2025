import spatialreasoners as sr

if __name__ == '__main__':
    # ⚙️ Customize training parameters
    sr.run_training(overrides=[
        "experiment=mnist_sudoku",    # Use specific experiment
        "trainer.max_epochs=50",      # Train for 50 epochs
        "data_loader.train.batch_size=32"  # Adjust batch size
    ])
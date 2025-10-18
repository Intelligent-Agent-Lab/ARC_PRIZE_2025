import io
import random

import chex
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import jax.numpy as jnp
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image

# Define color map
arc_cmap = ListedColormap(
    [
        "#000000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#AAAAAA",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
    ]
)
arc_norm = Normalize(vmin=0, vmax=9)


def display_grid(ax: plt.Axes, grid: chex.Array, grid_shape: chex.Array) -> None:
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    rows = min(rows, grid.shape[0])
    cols = min(cols, grid.shape[1])

    ax.imshow(
        grid[:rows, :cols],
        cmap=arc_cmap,
        interpolation="nearest",
        aspect="equal",
        norm=arc_norm,
        origin="lower",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Add shape information below the grid
    ax.text(0.5, -0.02, f"{rows}x{cols}", transform=ax.transAxes, ha="center", va="top")


def ax_to_pil(ax):
    fig = ax.figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close(fig)  # Close the figure to free memory
    buf.close()
    return pil_image


def display_function_examples(grids, shapes, num_pairs=None, seed=None) -> tuple:
    if seed is not None:
        random.seed(seed)
    if num_pairs is None:
        num_pairs = grids.shape[1]
    num_pairs = min(max(num_pairs, 1), grids.shape[1])  # Ensure num_pairs between 1 and number of pairs

    b = random.randint(0, grids.shape[0] - 1)  # Choose a random batch
    _, axs = plt.subplots(2, num_pairs, figsize=(5 * num_pairs, 2 * num_pairs))

    for i in range(num_pairs):
        input_grid = grids[b, i, :, :, 0]
        output_grid = grids[b, i, :, :, 1]
        input_grid_shape = shapes[b, i, :, 0]
        output_grid_shape = shapes[b, i, :, 1]

        # Display input grid
        display_grid(axs[0, i], input_grid, input_grid_shape)

        # Display output grid
        display_grid(axs[1, i], output_grid, output_grid_shape)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    return grids[b], shapes[b]  # Return the input grids and shapes for the batch


def visualize_json_submission(
    challenges: dict[str, list], generations: dict[str, list], solutions: dict[str, list], num_tasks: int = 5
) -> plt.Figure:
    keys = list(generations.keys())[:num_tasks]
    num_tasks = len(keys)
    # Find the maximum number of (train + test) pairs throughout all tasks
    max_num_pairs = max(
        len(challenge["train"]) + len(challenge["test"])
        for task_id, challenge in challenges.items()
        if task_id in generations
    )
    height_ratios = num_tasks * [1, 1, 1, 0.2]  # Input, True Output, Predicted Output, Line
    fig, axs = plt.subplots(
        4 * num_tasks - 1,
        max_num_pairs,
        figsize=(3 * max_num_pairs, 9.6 * num_tasks - 0.6),
        height_ratios=height_ratios[:-1],
        dpi=min(100, int(100 / (num_tasks / 60))),  # Limit the total number of pixels
    )

    for task_index, key in enumerate(keys):
        challenge_list, solution_list, generation_list = challenges[key], solutions[key], generations[key]
        num_test_grids = len(challenge_list["test"])
        for test_index, (challenge, solution, generation) in enumerate(
            zip(challenge_list["test"], solution_list, generation_list)
        ):
            input_grid = np.array(challenge["input"])
            output_grid = np.array(solution)
            prediction_grid = np.array(generation["attempt_1"])

            display_grid(axs[4 * task_index, test_index], input_grid, (30, 30))
            axs[4 * task_index, test_index].set_title("Input")

            display_grid(axs[4 * task_index + 1, test_index], output_grid, (30, 30))
            axs[4 * task_index + 1, test_index].set_title("Output")

            display_grid(axs[4 * task_index + 2, test_index], prediction_grid, (30, 30))
            axs[4 * task_index + 2, test_index].set_title("Prediction")

        for train_task_index, train_task in enumerate(challenge_list["train"]):
            input_grid = np.array(train_task["input"])
            output_grid = np.array(train_task["output"])

            display_grid(axs[4 * task_index, num_test_grids + train_task_index], input_grid, (30, 30))
            axs[4 * task_index, num_test_grids + train_task_index].set_title("Input")

            display_grid(axs[4 * task_index + 1, num_test_grids + train_task_index], output_grid, (30, 30))
            axs[4 * task_index + 1, num_test_grids + train_task_index].set_title("Output")

            axs[4 * task_index + 2, num_test_grids + train_task_index].axis("off")

        for col in range(num_test_grids + len(challenge_list["train"]), max_num_pairs):
            for row in range(4 * task_index, 4 * task_index + 3):
                axs[row, col].axis("off")

        # Draw a line to separate tasks
        if task_index < len(keys) - 1:
            for i in range(max_num_pairs):
                axs[4 * task_index + 3, i].axis("off")
                axs[4 * task_index + 3, i].axhline(0.5, color="black", linewidth=4)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.25)  # Increased hspace for shape labels

    return fig


def visualize_dataset_generation(
    dataset: chex.Array,
    grid_shapes: chex.Array,
    predicted_grids: chex.Array,
    predicted_shapes: chex.Array,
    num_tasks: int,
) -> plt.Figure:
    num_pairs = dataset.shape[1]
    height_ratios = num_tasks * [1, 1, 1, 0.2]  # Input, True Output, Predicted Output, Line
    fig, axs = plt.subplots(
        4 * num_tasks - 1,
        num_pairs,
        figsize=(3 * num_pairs, 9.6 * num_tasks - 0.6),
        height_ratios=height_ratios[:-1],
    )

    for task_index in range(num_tasks):

        for i in range(num_pairs):
            input_grid = dataset[task_index, i, :, :, 0]
            output_grid = dataset[task_index, i, :, :, 1]
            prediction_grid = predicted_grids[task_index, i]

            input_shape = grid_shapes[task_index, i, :, 0]
            output_shape = grid_shapes[task_index, i, :, 1]
            prediction_shape = predicted_shapes[task_index, i]

            display_grid(axs[4 * task_index, i], input_grid, input_shape)
            axs[4 * task_index, i].set_title("Input")

            display_grid(axs[4 * task_index + 1, i], output_grid, output_shape)
            axs[4 * task_index + 1, i].set_title("Output")

            display_grid(axs[4 * task_index + 2, i], prediction_grid, prediction_shape)
            axs[4 * task_index + 2, i].set_title("Prediction")

        # Draw a line to separate tasks
        if task_index < num_tasks - 1:
            for i in range(num_pairs):
                axs[4 * task_index + 3, i].axis("off")
                axs[4 * task_index + 3, i].axhline(0.5, color="black", linewidth=4)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.25)  # Increased hspace for shape labels

    return fig


def visualize_heatmap(data, proportion):
    # Ensure the inputs are JAX arrays
    data = jnp.asarray(data)
    proportion = jnp.asarray(proportion)

    # Create a new figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot the original data
    im1 = ax1.imshow(data, cmap="coolwarm")
    # make the color bar in the range [0, 1]
    im1.set_clim(0, 1)
    ax1.set_title("Original Data")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(which="major", color="w", linestyle="-", linewidth=0.5)
    fig.colorbar(im1, ax=ax1, label="Value")

    # Plot the proportion data
    im2 = ax2.imshow(proportion, cmap="viridis")
    ax2.set_title("Proportion in Dataset")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(which="major", color="w", linestyle="-", linewidth=0.5)
    fig.colorbar(im2, ax=ax2, label="Proportion")

    # Adjust layout and return the figure
    plt.tight_layout()
    return fig


def visualize_tsne(latents, program_ids, perplexity=30, max_iter=1000, random_state=42):
    """
    Create a t-SNE visualization of latent embeddings, colored by program IDs with distinct colors,
    a legend, and numbered points.

    Args:
    latents (jnp.array): Array of latent embeddings with shape (B, latent_embedding_size)
    program_ids (jnp.array or list): Integer values for each latent, used for coloring
    perplexity (int or float): Perplexity parameter for t-SNE (default: 30)
    max_iter (int): Number of iterations for t-SNE (default: 1000)
    random_state (int): Random state for reproducibility (default: 42)

    Returns:
    fig (matplotlib.figure.Figure): Figure object containing the t-SNE plot
    """
    # Convert JAX array to NumPy array
    latents_np = np.array(latents).astype(float)

    # Ensure perplexity is a scalar float
    perplexity = float(perplexity)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=random_state)
    try:
        if np.all(latents_np == latents_np[0]):
            # If all latents are the same, t-SNE will fail
            return None
        embeddings_2d = tsne.fit_transform(latents_np)
    except Exception as e:
        print(f"Error during t-SNE: {e}")
        print(f"Shape of latents: {latents_np.shape}")
        print(f"Data type of latents: {latents_np.dtype}")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 12))

    # Get unique program IDs and assign a color to each
    unique_ids = np.unique(program_ids)
    num_colors = len(unique_ids)

    # Use a color palette that ensures distinct colors
    color_palette = sns.color_palette("husl", n_colors=num_colors)
    color_map = dict(zip(unique_ids, color_palette))

    # Plot each program ID with its assigned color and add numbering
    for id in unique_ids:
        mask = np.array(program_ids) == id
        points = embeddings_2d[mask]
        ax.scatter(points[:, 0], points[:, 1], c=[color_map[id]], label=f"Program {id}", alpha=0.7, s=50)

        # Add numbering to each point
        for point in points:
            ax.annotate(str(id), point, xytext=(3, 3), textcoords="offset points", fontsize=8, alpha=0.8)

    ax.set_title("t-SNE Visualization of Latent Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # Adjust layout to prevent legend from being cut off
    plt.tight_layout()

    return fig

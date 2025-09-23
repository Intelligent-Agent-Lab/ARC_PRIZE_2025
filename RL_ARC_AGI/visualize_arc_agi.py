import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap, Normalize
import jax.numpy as jnp
from typing import Dict, Any, Optional


# Color map for ARC AGI visualization
cmap = colors.ListedColormap([
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#8B00FF',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: sky
    '#870C25',  # 9: brown
    '#AAAAAA',  # 10: mask
    '#FFFFFF',  # 11: empty
])
norm = colors.Normalize(vmin=0, vmax=11)


class ArcAgiVisualizer:
    """Visualization utilities for ARC AGI environments and rollouts."""

    def __init__(self):
        self.cmap = cmap
        self.norm = norm

    def plot_chosen_grid(self, chosen_grid_img: np.ndarray):
        """Plot the chosen grid showing how many times each cell was selected."""
        plt.imshow(chosen_grid_img, cmap='viridis')
        plt.colorbar()

        # Display values on each cell
        for i in range(30):
            for j in range(30):
                plt.text(j, i, chosen_grid_img[i, j], ha='center', va='center', color='white')

    def plot_current_grid(self, current_grid_img: np.ndarray, task_id: str, test_input_idx: int, w: float = 0.5):
        """Plot the current state of the grid."""
        fs = 12
        test_sol_current_mat = current_grid_img[:, 150:]
        plt.imshow(test_sol_current_mat, cmap=self.cmap, norm=self.norm)
        plt.grid(True, which='both', color='lightgrey', linewidth=1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])

        # Grid styling
        plt.grid(visible=True, which='both', color='#666666', linewidth=w)
        plt.xticks([x-0.5 for x in range(1 + len(test_sol_current_mat[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(test_sol_current_mat))])
        plt.tick_params(axis='both', color='none', length=0)

        # Title
        plt.title(f'task: {task_id}   #{test_input_idx}', fontsize=fs, color='#000000')

    def plot_target_grid(self, target_grid_img: np.ndarray, task_id: str, test_input_idx: int, w: float = 0.5):
        """Plot the target grid."""
        fs = 12
        test_sol_target_mat = target_grid_img[:, 150:]
        plt.imshow(test_sol_target_mat, cmap=self.cmap, norm=self.norm)
        plt.grid(True, which='both', color='lightgrey', linewidth=1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])

        # Grid styling
        plt.grid(visible=True, which='both', color='#666666', linewidth=w)
        plt.xticks([x-0.5 for x in range(1 + len(test_sol_target_mat[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(test_sol_target_mat))])
        plt.tick_params(axis='both', color='none', length=0)

        # Title
        plt.title(f'task: {task_id}   #{test_input_idx}', fontsize=fs, color='#000000')

    def plot_one_task(self, challenges: Dict, solutions: Dict, task_id: str, mode: str = 'train', size: float = 2.5, w1: float = 0.9):
        """Plot one complete task with train and test examples."""
        if mode == 'train':
            task = challenges[task_id]
            task_solutions = solutions[task_id]
        elif mode == 'evaluation' or mode == 'eval':
            task = challenges[task_id]
            task_solutions = solutions[task_id]
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

        titleSize = 16
        num_train = len(task['train'])
        num_test = len(task['test'])
        wn = num_train + num_test
        fig, axs = plt.subplots(2, wn, figsize=(size*wn, 2*size))
        plt.suptitle(f'Task #{task_id}', fontsize=titleSize, fontweight='bold', y=1, color='#eeeeee')

        # Train examples
        for j in range(num_train):
            self._plot_one(axs[0, j], j, task, 'train', 'input', w=w1)
            self._plot_one(axs[1, j], j, task, 'train', 'output', w=w1)

        # Test examples
        for k in range(num_test):
            self._plot_one(axs[0, j+k+1], k, task, 'test', 'input', w=w1)
            task['test'][k]['output'] = task_solutions[k]
            self._plot_one(axs[1, j+k+1], k, task, 'test', 'output', w=w1)

        # Styling
        axs[1, j+1].set_xticklabels([])
        axs[1, j+1].set_yticklabels([])
        axs[1, j+1] = plt.figure(1).add_subplot(111)
        axs[1, j+1].set_xlim([0, wn])

        # Separators
        colorSeparator = 'white'
        for m in range(1, wn):
            axs[1, j+1].plot([m, m], [0, 1], '--', linewidth=1, color=colorSeparator)
        axs[1, j+1].plot([num_train, num_train], [0, 1], '-', linewidth=3, color=colorSeparator)
        axs[1, j+1].axis("off")

        # Frame and background
        fig.patch.set_linewidth(5)
        fig.patch.set_edgecolor('black')
        fig.patch.set_facecolor('#444444')
        plt.tight_layout()
        print(f'#{task_id}')  # for fast and convenient search
        plt.show()

    def _plot_one(self, ax, i: int, task: Dict, train_or_test: str, input_or_output: str, w: float = 0.8):
        """Plot one input/output example."""
        fs = 12
        input_matrix = task[train_or_test][i][input_or_output]
        ax.imshow(input_matrix, cmap=self.cmap, norm=self.norm)
        ax.grid(True, which='both', color='lightgrey', linewidth=1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
        ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])

        # Grid styling
        ax.grid(visible=True, which='both', color='#666666', linewidth=w)
        ax.tick_params(axis='both', color='none', length=0)

        # Subtitle
        ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs, color='#dddddd')

    def plot_original_task(self, challenges: Dict, task_id: str, train_or_test: str, i: int,
                          input_or_output: str, mode: str = 'train', w: float = 0.8):
        """Plot original task without padding."""
        fs = 12
        if mode == 'train':
            task = challenges[task_id]
        elif mode == 'evaluation' or mode == 'eval':
            task = challenges[task_id]

        input_matrix = task[train_or_test][i][input_or_output]
        plt.imshow(input_matrix, cmap=self.cmap, norm=self.norm)
        plt.grid(True, which='both', color='lightgrey', linewidth=1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])

        # Grid styling
        plt.grid(visible=True, which='both', color='#666666', linewidth=w)
        plt.xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(input_matrix))])
        plt.tick_params(axis='both', color='none', length=0)

        # Title
        plt.title(f'task: {task_id}  {train_or_test} {input_or_output}  #{i}', fontsize=fs, color='#000000')

    def plot_padded_task(self, train_task_img_dict: Dict, task_id: str, i: int, w: float = 0.5):
        """Plot padded task."""
        fs = 12
        task = train_task_img_dict[task_id]
        input_matrix = task[i]
        plt.figure(figsize=(100, 200))
        plt.imshow(input_matrix, cmap=self.cmap, norm=self.norm)
        plt.grid(True, which='both', color='lightgrey', linewidth=1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])

        # Grid styling
        plt.grid(visible=True, which='both', color='#666666', linewidth=w)
        plt.xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(input_matrix))])
        plt.tick_params(axis='both', color='none', length=0)

        # Title
        plt.title(f'task: {task_id}   #{i}', fontsize=fs, color='#000000')

    def plot_rollout_comparison(self, initial_state: np.ndarray, final_state: np.ndarray,
                               target_state: np.ndarray, task_id: str, figsize: tuple = (15, 5)):
        """Plot comparison of initial, final, and target states."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Initial state
        axes[0].imshow(initial_state[:, 150:], cmap=self.cmap, norm=self.norm)
        axes[0].set_title('Initial State')
        axes[0].grid(True, color='#666666', linewidth=0.5)
        axes[0].set_xticks([x-0.5 for x in range(1 + initial_state[:, 150:].shape[1])])
        axes[0].set_yticks([x-0.5 for x in range(1 + initial_state[:, 150:].shape[0])])
        axes[0].tick_params(axis='both', color='none', length=0)

        # Final state
        axes[1].imshow(final_state[:, 150:], cmap=self.cmap, norm=self.norm)
        axes[1].set_title('Final State')
        axes[1].grid(True, color='#666666', linewidth=0.5)
        axes[1].set_xticks([x-0.5 for x in range(1 + final_state[:, 150:].shape[1])])
        axes[1].set_yticks([x-0.5 for x in range(1 + final_state[:, 150:].shape[0])])
        axes[1].tick_params(axis='both', color='none', length=0)

        # Target state
        axes[2].imshow(target_state[:, 150:], cmap=self.cmap, norm=self.norm)
        axes[2].set_title('Target State')
        axes[2].grid(True, color='#666666', linewidth=0.5)
        axes[2].set_xticks([x-0.5 for x in range(1 + target_state[:, 150:].shape[1])])
        axes[2].set_yticks([x-0.5 for x in range(1 + target_state[:, 150:].shape[0])])
        axes[2].tick_params(axis='both', color='none', length=0)

        plt.suptitle(f'Rollout Results - Task {task_id}')
        plt.tight_layout()
        plt.show()

    def plot_action_history(self, action_history: list, chosen_grid: np.ndarray, figsize: tuple = (12, 6)):
        """Plot action history and final chosen grid."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Action history
        if action_history:
            actions = np.array(action_history)
            colors = actions // 900
            coordinates = actions % 900
            rows = coordinates // 30
            cols = coordinates % 30

            axes[0].scatter(cols, rows, c=colors, cmap='tab10', s=50, alpha=0.7)
            axes[0].set_title('Action History (Color-coded by action color)')
            axes[0].set_xlabel('Column')
            axes[0].set_ylabel('Row')
            axes[0].invert_yaxis()
            axes[0].grid(True, alpha=0.3)

        # Chosen grid heatmap
        im = axes[1].imshow(chosen_grid, cmap='viridis')
        axes[1].set_title('Cell Selection Frequency')
        plt.colorbar(im, ax=axes[1])

        # Add text annotations for non-zero values
        for i in range(chosen_grid.shape[0]):
            for j in range(chosen_grid.shape[1]):
                if chosen_grid[i, j] > 0:
                    axes[1].text(j, i, int(chosen_grid[i, j]), ha='center', va='center', color='white')

        plt.tight_layout()
        plt.show()


# Convenience functions for easy access
def create_visualizer() -> ArcAgiVisualizer:
    """Create a new ArcAgiVisualizer instance."""
    return ArcAgiVisualizer()


def plot_task(challenges: Dict, solutions: Dict, task_id: str, mode: str = 'train'):
    """Quick function to plot a task."""
    visualizer = create_visualizer()
    visualizer.plot_one_task(challenges, solutions, task_id, mode)


def plot_grids_comparison(current_grid: np.ndarray, target_grid: np.ndarray, task_id: str):
    """Quick function to compare current and target grids."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Current grid
    axes[0].imshow(current_grid[:, 150:], cmap=cmap, norm=norm)
    axes[0].set_title('Current Grid')
    axes[0].grid(True, color='#666666', linewidth=0.5)

    # Target grid
    axes[1].imshow(target_grid[:, 150:], cmap=cmap, norm=norm)
    axes[1].set_title('Target Grid')
    axes[1].grid(True, color='#666666', linewidth=0.5)

    plt.suptitle(f'Grid Comparison - Task {task_id}')
    plt.tight_layout()
    plt.show()


def visualize_rollout_states(states: list, task_id: str, max_states: int = 10):
    """Visualize a sequence of states from a rollout."""
    if not states:
        print("No states to visualize")
        return

    n_states = min(len(states), max_states)
    cols = min(5, n_states)
    rows = (n_states + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_states):
        row = i // cols
        col = i % cols

        if hasattr(states[i], 'current_grid_img'):
            grid = states[i].current_grid_img[:, 150:]
        else:
            grid = states[i][:, 150:]

        axes[row, col].imshow(grid, cmap=cmap, norm=norm)
        axes[row, col].set_title(f'Step {i}')
        axes[row, col].grid(True, color='#666666', linewidth=0.5)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

    # Hide unused subplots
    for i in range(n_states, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.suptitle(f'Rollout States - Task {task_id}')
    plt.tight_layout()
    plt.show()
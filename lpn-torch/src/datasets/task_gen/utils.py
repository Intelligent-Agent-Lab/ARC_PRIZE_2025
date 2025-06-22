import functools
import random
import signal
from typing import Any, Optional, Tuple, Callable, TypeVar, TypeAlias

import matplotlib.pyplot as plt

from src.visualization import arc_cmap, arc_norm


def plot_task(task: list[dict] | dict, title: str = None, figsize_factor: float = 3) -> None:
    height = 2
    if isinstance(task, dict):
        task = [task]
    width = len(task)
    figure_size = (width * figsize_factor, height * figsize_factor)
    figure, axes = plt.subplots(height, width, figsize=figure_size, squeeze=False)
    for column, example in enumerate(task):
        axes[0, column].imshow(example["input"], cmap=arc_cmap, norm=arc_norm, origin="lower")
        axes[0, column].text(
            0.5,
            -0.02,
            "{}x{}".format(*example["input"].shape),
            transform=axes[0, column].transAxes,
            ha="center",
            va="top",
        )
        axes[1, column].imshow(example["output"], cmap=arc_cmap, norm=arc_norm, origin="lower")
        axes[1, column].text(
            0.5,
            -0.02,
            "{}x{}".format(*example["output"].shape),
            transform=axes[1, column].transAxes,
            ha="center",
            va="top",
        )
        axes[0, column].axis("off")
        axes[1, column].axis("off")

    if title is not None:
        figure.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def is_grid(grid: Any) -> bool:
    """Check if the input is a valid grid in terms of structure and values."""
    if not isinstance(grid, tuple):
        return False
    return (
        all(
            isinstance(row, tuple)
            and all(isinstance(pixel, int) for pixel in row)
            and all(0 <= pixel <= 9 for pixel in row)
            and len(row) == len(grid[0])
            for row in grid
        )
        and 1 <= len(grid) <= 30
        and 1 <= len(grid[0]) <= 30
    )


T = TypeVar("T")
OptionalRandomState: TypeAlias = Optional[Tuple]


def run_with_timeout(
    func: Callable[[], T], timeout: int
) -> Callable[[OptionalRandomState], Tuple[Optional[T], OptionalRandomState, Optional[Exception]]]:
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function execution timed out after {timeout} seconds")

    @functools.wraps(func)
    def wrapper(random_state: OptionalRandomState = None):
        result = None
        exception = None

        # Store the original SIGINT and SIGALRM handlers
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigalrm_handler = signal.getsignal(signal.SIGALRM)

        def handle_interrupt(signum, frame):
            # Restore the original handlers
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGALRM, original_sigalrm_handler)
            raise KeyboardInterrupt()

        try:
            # Set up the alarm signal
            signal.signal(signal.SIGALRM, timeout_handler)
            # Set up the interrupt handler
            signal.signal(signal.SIGINT, handle_interrupt)
            signal.alarm(timeout)

            if random_state is not None:
                random.setstate(random_state)
            result = func()
            random_state = random.getstate() if random_state is not None else None
        except KeyboardInterrupt as e:
            raise e
        except TimeoutError as e:
            # Timeout occurred
            if random_state is not None:
                random.setstate(random_state)
                random.seed(random.randint(0, 2**32 - 1))
                random_state = random.getstate()
            exception = e
        except Exception as e:
            # Some other exception occurred
            random_state = random.getstate() if random_state is not None else None
            exception = e
        finally:
            # Cancel the alarm and restore the original SIGINT and SIGALRM handlers
            signal.alarm(0)
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGALRM, original_sigalrm_handler)
        return result, random_state, exception  # Move return here

    return wrapper


class EMA:
    def __init__(self, start: float, smoothing: float = 0.3, return_inverse: bool = False, eps: float = 1e-8):
        self.alpha = max(eps, min(smoothing, 1 - eps))
        self.eps = eps
        self.last_value = start
        self.diff = 0
        self.calls = 0
        self.return_inverse = return_inverse

    def __call__(self, x: float) -> float:
        beta = 1 - self.alpha
        self.diff = self.alpha * (x - self.last_value) + beta * self.diff
        self.last_value = x
        self.calls += 1
        if self.return_inverse:
            return (1 - beta**self.calls) / (self.diff + self.eps)
        else:
            return self.diff / (1 - beta**self.calls)

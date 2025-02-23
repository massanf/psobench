import numpy as np
from matplotlib.axes import Axes
from typing import Any
from pso import PSO


def scatter_progress(pso: PSO, ax: Axes, label: str) -> Any:
    """Plot the fitness of each particle over time as a scatter plot onto ax.

    Args:
        pso (PSO): _description_
        ax (Axes): _description_
        label (str): _description_

    Returns:
        Any: _description_
    """
    pso.load_full()
    x = []
    y = []
    for i, iteration in enumerate(pso.iterations):
        for particle in iteration.particles:
            x.append(i)
            y.append(particle.fitness)

    scatter = ax.scatter(x, y, s=0.01, rasterized=True)

    ax.scatter(
        x[0] + 10000,
        y[0],
        s=10.0,
        label=label,
        color=scatter.get_facecolor()[0],
    )

    ax.set_yscale("log")
    return ax


def plot_progress(
    pso: PSO, ax: Axes, label: str, color: str, width: float
) -> Any:
    """Plot the fitness of the best particle over time as a line plot onto ax.

    Args:
        pso (PSO): _description_
        ax (Axes): _description_
        label (str): _description_
        color (str): _description_
        width (float): _description_

    Returns:
        Any: _description_
    """
    pso.load_full()
    y = []
    for i, iteration in enumerate(pso.iterations):
        min_fitness = 1e20
        for particle in iteration.particles:
            min_fitness = min(min_fitness, particle.fitness)
        y.append(min_fitness)

    window_size = 40
    smoothed_percentages = np.convolve(
        y, np.ones(window_size) / window_size, mode="valid"
    )

    # Original x-axis and new x-axis after smoothing
    original_x = np.arange(len(y))
    smoothed_x = np.arange(len(smoothed_percentages))

    # Interpolate the smoothed data back to the original size
    interpolated_y = np.interp(
        original_x, smoothed_x, smoothed_percentages
    )

    ax.plot(
        original_x,
        interpolated_y,
        label=label,
        linewidth=width,
        color=color,
    )
    ax.set_yscale("log")
    return ax

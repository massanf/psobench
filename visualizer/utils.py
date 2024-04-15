from typing import List, Any
import pathlib
from pso import PSO
import matplotlib.pyplot as plt  # type: ignore
from grid_searches import GridSearches
from tests import Tests
from matplotlib.animation import FuncAnimation  # type: ignore
import pathlib
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from constants import HOME, DATA, GRAPHS

def plot_and_fill(ax: Any, iterations: List[List[float]],
                  log=True, label="", alpha=0.15) -> None:
    t = np.linspace(0, len(iterations), len(iterations))
    top = []
    bottom = []
    avg = []
    for iteration in iterations:
        top.append(max(iteration))
        bottom.append(min(iteration))
        avg.append(np.average(iteration))

    if label != "":
        line, = ax.plot(t, avg, label=label)
    else:
        line, = ax.plot(t, avg)

    ax.fill_between(t, top, bottom, color=line.get_color(), alpha=alpha)
    ax.set_xlim(0, len(iterations))

    if log:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    return ax

def generate_gridmap_collage(path: pathlib.Path):
    optimizer = GridSearches(DATA, GRAPHS, path)
    optimizer.heatmap_collage("grid_search.png", True, True)

def generate_summary(name: pathlib.Path):
    data = HOME / "data" / name
    graphs = HOME / "graphs" / name
    graphs.mkdir(parents=True, exist_ok=True)
    tests = Tests(data)
    fig, axs = plt.subplots(5, 6, figsize=(15, 10), dpi=300)
    axs = tests.plot_all(axs)
    plt.tight_layout()
    print(f"Saving: {graphs/ f'progress_comparison.png'}")
    plt.savefig(graphs / f"progress_comparison.png")

def generate_overview(name: pathlib.Path, skip=1, end=500):
    data = HOME / "data" / name 
    graphs = HOME / "graphs" / name 
    graphs.mkdir(parents=True, exist_ok=True)
    pso = PSO(data)
    pso.load_full()
    pso.overview(False, graphs)
    if pso.has_mass():
        pso.animate_mass(graphs / "mass.gif", skip, 0, end)
    print(f"Saving: {graphs / 'animation.gif'}")
    pso.animate_particles(graphs/ "animation.gif", skip, 0, end)

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
from matplotlib import cycler
from constants import HOME, DATA, GRAPHS

def plot_and_fill(ax: Any, iterations: List[List[float]],
                  log=True, label="", alpha=0.15) -> None:
    t = np.linspace(0, len(iterations), len(iterations))
    top = []
    btm = []
    mid = []
    for iteration in iterations:
        btm.append(np.quantile(iteration, 0.25))
        mid.append(np.quantile(iteration, 0.5))
        top.append(np.quantile(iteration, 0.75))

    if label != "":
        if "tiled" in label: 
            # We're expecting 'tiled' to come right after the untiled one.
            # This will break if this rule is not followed.
            line, = ax.plot(t, mid, label=label, linestyle='--', color=ax.get_lines()[-1].get_color(), linewidth=1)
        else:
            line, = ax.plot(t, mid, label=label, linestyle='-', linewidth=1)
    else:
        line, = ax.plot(t, mid)

    ax.fill_between(t, top, btm, color=line.get_color(), alpha=alpha)
    ax.set_xlim(0, len(iterations))

    if log:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    return ax

def generate_gridmap_collage(path: pathlib.Path):
    optimizer = GridSearches(DATA, GRAPHS, path)
    optimizer.heatmap_collage("grid_search.png", True, True)

def generate_progress_comparison(name: pathlib.Path):
    color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#a65628', '#984ea3',
                '#999999', '#e41a1c', '#dede00']
    color_cycler = cycler(color=color_cycle)
    plt.rc('axes', prop_cycle=(color_cycler))
    data = HOME / "data" / name
    graphs = HOME / "graphs" / name
    graphs.mkdir(parents=True, exist_ok=True)
    tests = Tests(data)
    fig, axs = plt.subplots(5, 6, figsize=(15, 10), dpi=300)
    axs = tests.plot_all(axs)
    plt.tight_layout()
    print(f"Saving: {graphs/ f'progress_comparison.png'}")
    plt.savefig(graphs / f"progress_comparison.png")

def generate_entropy_comparison(name: pathlib.Path):
    data = HOME / "data" / name
    graphs = HOME / "graphs" / name
    graphs.mkdir(parents=True, exist_ok=True)
    tests = Tests(data)
    fig, axs = plt.subplots(5, 6, figsize=(15, 10), dpi=300)
    axs = tests.plot_all_entropy(axs)
    plt.tight_layout()
    print(f"Saving: {graphs/ f'entropy_comparison.png'}")
    plt.savefig(graphs / f"entropy_comparison.png")

def generate_overview(name: pathlib.Path, skip=1, end=500, animate_mass=False):
    data = HOME / "data" / name 
    graphs = HOME / "graphs" / name 
    graphs.mkdir(parents=True, exist_ok=True)
    pso = PSO(data)
    pso.load_full()
    pso.overview(False, graphs)
    if pso.has_mass() and animate_mass:
        pso.animate_mass(graphs / "mass.gif", skip, 0, end)
    print(f"Saving: {graphs / 'animation.gif'}")
    pso.animate_particles(graphs/ "animation.gif", skip, 0, end)

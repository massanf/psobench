import pathlib
from pso import PSO
from attempts import Attempts
from tqdm import tqdm
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import matplotlib.pyplot as plt  # type: ignore
from typing import Any
from grid_search import GridSearch
import matplotlib.image as mpimg
from grid_searches import GridSearches

HOME = pathlib.Path(__file__).parent.parent
DATA = HOME / "data"
GRAPHS = HOME / "graphs"

class Tests:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.data = {}

        for pso_type in self.path.glob("*"):
            self.data[pso_type.name] = {}
            for function in pso_type.glob("*"):
                self.data[pso_type.name][function.name] = Attempts(function)

    def plot_best_global_progress(self, axs, pso_type: str):
        axs = axs.flatten()
        for i, function in enumerate(tqdm(sorted(self.data[pso_type]))):
            attempts = self.data[pso_type][function]
            axs[i].yaxis.set_major_locator(LogLocator(base=10.0))
            axs[i].yaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=False))
            axs[i].plot(attempts.average_global_best_progress(), label=pso_type)
            axs[i].set_title(function)
            if i == 0:
                axs[i].legend()
        return axs

    def plot_all(self, axs):
        for pso_type in sorted(self.data):
            axs = self.plot_best_global_progress(axs, pso_type)
        return axs


# tests = Tests(DATA / "test")
# fig, axs = plt.subplots(5, 6, figsize=(12, 10), dpi=300)
# axs = tests.plot_all(axs)
# plt.tight_layout()
# plt.savefig(GRAPHS / f"progress_comparison.png")


# gsa_path = pathlib.Path("grid_search") / f"gsa_30"
# gsa = GridSearches(DATA, GRAPHS, gsa_path)
# gsa.heatmap_collage("grid_search.png", True, True)

# tiled_gsa_path = pathlib.Path("grid_search") / f"tiled_gsa_30"
# tiled_gsa = GridSearches(DATA, GRAPHS, tiled_gsa_path)
# tiled_gsa.heatmap_collage("grid_search.png", True, True)

# pso_path = pathlib.Path("grid_search") / f"pso_30"
# pso = GridSearches(DATA, GRAPHS, pso_path)
# pso.heatmap_collage("grid_search.png", False, False)

# fig, axs = plt.subplots(5, 6, figsize=(12, 10), dpi=300)
# axs = gsa.plot_best_global_progresses(axs)
# axs = pso.plot_best_global_progresses(axs)
# # axs = tiled_gsa.plot_best_global_progress(axs)
# plt.tight_layout()
# plt.savefig(GRAPHS / f"progress_comparison.png")

# NAME = pathlib.Path("test") / "tiled_gsa"
# DATA = HOME / "data" / NAME
# GRAPHS = HOME / "graphs" / NAME
# pso = PSO(DATA)
# pso.load_full()
# pso.overview(False, GRAPHS)
# pso.animate(GRAPHS / "animation.gif", 1, 0, 50)

NAME = pathlib.Path("test") / "gsa"
DATA = HOME / "data" / NAME
GRAPHS = HOME / "graphs" / NAME
pso = PSO(DATA)
pso.load_full()
pso.overview(False, GRAPHS)
pso.animate(GRAPHS / "animation.gif", 1, 0, 50)

# NAME = pathlib.Path("test") / "pso"
# DATA = HOME / "data" / NAME
# GRAPHS = HOME / "graphs" / NAME
# pso = PSO(DATA)
# pso.load_full()
# pso.overview(False, GRAPHS)
# pso.animate(GRAPHS / "animation.gif", 1, 0, 500)

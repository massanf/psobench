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
from tests import Tests

HOME = pathlib.Path(__file__).parent.parent
DATA = HOME / "data"
GRAPHS = HOME / "graphs"

tests = Tests(DATA / "test" / "30")
fig, axs = plt.subplots(5, 6, figsize=(12, 10), dpi=300)
axs = tests.plot_all(axs)
plt.tight_layout()
plt.savefig(GRAPHS / f"progress_comparison.png")

# pso_path = pathlib.Path("grid_search") / "30" / "gsa"
# pso = GridSearches(DATA, GRAPHS, pso_path)
# pso.heatmap_collage("grid_search.png", True, True)

# gsa_path = pathlib.Path("grid_search") / "30" / "igsa"
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


# NAME = pathlib.Path("test") / "30" / "pso" / "CEC2017_F01" / "0"
# DATA = HOME / "data" / NAME
# GRAPHS = HOME / "graphs" / NAME
# pso = PSO(DATA)
# pso.load_full()
# pso.overview(False, GRAPHS)
# pso.animate(GRAPHS / "animation.gif", 1, 0, 500)

# NAME = pathlib.Path("test") / "30" / "gsa" / "CEC2017_F01" / "0"
# DATA = HOME / "data" / NAME
# GRAPHS = HOME / "graphs" / NAME
# pso = PSO(DATA)
# pso.load_full()
# pso.overview(False, GRAPHS)
# pso.animate(GRAPHS / "animation.gif", 1, 0, 50)

# NAME = pathlib.Path("test") / "30" / "igsa" / "CEC2017_F01" / "0"
# DATA = HOME / "data" / NAME
# GRAPHS = HOME / "graphs" / NAME
# pso = PSO(DATA)
# pso.load_full()
# pso.overview(False, GRAPHS)
# pso.animate(GRAPHS / "animation.gif", 1, 0, 500)


# NAME = pathlib.Path("test") / "pso"
# DATA = HOME / "data" / NAME
# GRAPHS = HOME / "graphs" / NAME
# pso = PSO(DATA)
# pso.load_full()
# pso.overview(False, GRAPHS)
# pso.animate(GRAPHS / "animation.gif", 1, 0, 500)

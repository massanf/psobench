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

def generate_gridmap_collage(path: pathlib.Path):
    optimizer = GridSearches(DATA, GRAPHS, path)
    optimizer.heatmap_collage("grid_search.png", True, True)

def generate_summary(path: pathlib.Path):
    tests = Tests(path)
    fig, axs = plt.subplots(5, 6, figsize=(12, 10), dpi=300)
    axs = tests.plot_all(axs)
    plt.tight_layout()
    plt.savefig(GRAPHS / f"progress_comparison.png")

def generate_overview(name: pathlib.Path, skip=1, end=500):
    data = HOME / "data" / name 
    graphs = HOME / "graphs" /name 
    pso = PSO(data)
    pso.load_full()
    pso.overview(False, graphs)
    pso.animate(GRAPHS / "animation.gif", 1, 0, 500)

generate_summary(DATA / "test" / "30")

# gsa_path = pathlib.Path("grid_search") / "30" / "gsa"
# generate_gridmap_collage(gsa_path)

# igsa_path = pathlib.Path("grid_search") / "30" / "igsa"
# generate_gridmap_collage(igsa_path)

# pso_path = pathlib.Path("grid_search") / "30" / "pso"
# generate_gridmap_collage(pso_path)

# generate_overview(pathlib.Path("test") / "30" / "igsa" / "CEC2017_F01" / "0", 1, 500)

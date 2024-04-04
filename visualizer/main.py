import pathlib
from pso import PSO
from tqdm import tqdm
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import matplotlib.pyplot as plt  # type: ignore
from typing import Any
from gridsearch import GridSearch
import matplotlib.image as mpimg

HOME = pathlib.Path(__file__).parent.parent
DATA = HOME / "data"
GRAPHS = HOME / "graphs"

class GridSearches:
    def __init__(self, path: pathlib.Path):
        self.path = path

        self.data = DATA / path
        self.graphs = GRAPHS / path

        self.data_paths = []
        self.graph_paths = []
        for function_dir in self.data.glob("*"):
            self.data_paths.append(function_dir)
            self.graph_paths.append(self.graphs / function_dir.name)

        self.data_paths = sorted(self.data_paths, key=lambda path: path.name)
        self.graph_paths = sorted(self.graph_paths, key=lambda path: path.name)

    def draw_heatmap(self, log_1: bool, log_2: bool):
        for idx, function_dir in enumerate(self.data_paths):
            graph_dir = self.graphs / function_dir.name
            graph_dir.mkdir(parents=True, exist_ok=True)
            grid = GridSearch(function_dir)
            grid.draw_heatmap(graph_dir, log_1, log_2)
            grid.plot_best_global_progress(graph_dir)

    def heatmap_collage(self, filename: str, log_1: bool, log_2: bool):
        self.draw_heatmap(log_1, log_2)
        n_rows = 5
        n_cols = 6

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 8), dpi=500)
        axs = axs.flatten()

        for i, ax in enumerate(axs):
            if i < len(self.graph_paths):
                img = mpimg.imread(self.graph_paths[i] / filename)
                ax.imshow(img)
            ax.axis('off')

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(self.graphs / filename, bbox_inches='tight', pad_inches=0.1)

    def plot_best_global_progress(self, axs) -> Any:
        axs = axs.flatten()
        for i, ax in enumerate(tqdm(axs)):
            if i >= len(self.data_paths):
                axs[i].set_visible(False)
                continue
            grid = GridSearch(self.data_paths[i])
            axs[i].yaxis.set_major_locator(LogLocator(base=10.0))
            axs[i].yaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=False))
            axs[i].plot(grid.best_global_progress(), label=self.path.parent.name)
            axs[i].set_title(grid.name)
            axs[i].legend()
        return axs

# dim = 30

# gsa_path = pathlib.Path(f"gsa_{dim}") / "grid_search"
# gsa = GridSearches(gsa_path)
# # gsa.heatmap_collage("grid_search.png", True, True)

# pso_path = pathlib.Path(f"pso_{dim}") / "grid_search"
# pso = GridSearches(pso_path)
# # pso.heatmap_collage("grid_search.png", False, False)

# fig, axs = plt.subplots(5, 6, figsize=(12, 10), dpi=300)
# axs = gsa.plot_best_global_progress(axs)
# axs = pso.plot_best_global_progress(axs)
# plt.tight_layout()
# plt.savefig(GRAPHS / f"progress_comparison_{dim}.png")

NAME = pathlib.Path("test") / "gsa"
DATA = HOME / "data" / NAME
GRAPHS = HOME / "graphs" / NAME
pso = PSO(DATA)
pso.load_full()
pso.overview(False, GRAPHS)
pso.animate(GRAPHS / "animation.gif", 1, 0, 500)

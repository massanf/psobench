import math
import pathlib
import matplotlib.pyplot as plt  # type: ignore
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

        for idx, function_dir in enumerate(self.data_paths):
            graph_dir = self.graphs / function_dir.name
            graph_dir.mkdir(parents=True, exist_ok=True)
            grid = GridSearch(function_dir)
            grid.plot(graph_dir)

        self.graph_paths = sorted(self.graph_paths, key=lambda path: path.name)

    def cec17_summary(self):
        n_rows = 5
        n_cols = 6

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 8), dpi=500)
        axs = axs.flatten()

        for i, ax in enumerate(axs):
            if i < len(self.graph_paths):
                img = mpimg.imread(self.graph_paths[i] / "grid_search.png")
                ax.imshow(img)
            ax.axis('off')

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(self.graphs / 'grid_searches.png', bbox_inches='tight', pad_inches=0.1)

gsa_path = pathlib.Path("gsa") / "grid_search"
gsa = GridSearches(gsa_path)
gsa.cec17_summary()

# NAME = pathlib.Path("test") / "gsa"
# DATA = HOME / "data" / NAME
# GRAPHS = HOME / "graphs" / NAME
# pso = PSO(DATA)
# pso.load_full()
# pso.overview(False, GRAPHS)
# pso.animate(GRAPHS / "animation.mp4", 1, 0, 200)

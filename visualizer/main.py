import pathlib
from gridsearch import GridSearch
from pso import PSO

HOME = pathlib.Path(__file__).parent.parent
search_parent = HOME / "data" / "base_gsa_all"

# graph_grandparent = HOME / "graphs" / search_parent.name
# graph_grandparent.mkdir(parents=True, exist_ok=True)

# for search in search_parent.glob("*"):
#     graph_parent = graph_grandparent / search.name
#     graph_parent.mkdir(parents=True, exist_ok=True)
#     grid = GridSearch(search)
#     grid.animate(graph_parent)

NAME = pathlib.Path("test") / "gsa"
DATA = HOME / "data" / NAME
GRAPHS = HOME / "graphs" / NAME
pso = PSO(DATA)
pso.load_full()
pso.overview(False, GRAPHS)
pso.animate(GRAPHS / "animation.mp4", 1, 0, 200)

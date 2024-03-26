import pathlib
from gridsearch import GridSearch
from pso import PSO

HOME = pathlib.Path(__file__).parent.parent

# search_parent = HOME / "data" / "base_pso_all"

# graph_grandparent = HOME / "graphs" / search_parent.name
# graph_grandparent.mkdir(parents=True, exist_ok=True)

# for search in search_parent.glob("*"):
#     graph_parent = graph_grandparent / search.name
#     graph_parent.mkdir(parents=True, exist_ok=True)
#     grid = GridSearch(search)
#     grid.animate(graph_parent)

pso = PSO(HOME / "data" / "test")
pso.load_full()
pso.animate(HOME / "graphs" / "animation_g_1.gif", 10)

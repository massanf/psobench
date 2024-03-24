import pathlib
from gridsearch import GridSearch

HOME = pathlib.Path(__file__).parent.parent


grid = GridSearch(HOME / "data" / "base_pso_test5" / "CEC2017_F1",
                  "dim", "particle_count")
grid.animate(HOME / "graphs" / "CEC2017_F1.gif")

# pso = PSO(HOME / "data" / "base_pso_test3" / "g_1_d_2")
# pso.load_full()
# pso.animate(HOME / "graphs" / "animation_g_1.gif", 10)

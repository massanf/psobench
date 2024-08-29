import questionary
import utils
import glob
import utils
from pso import PSO
import pathlib
import numpy as np
from constants import DATA


graph_type = questionary.select(
    "Select graph type:",
    choices=[
        'single',
        'gridmaps',
        'collage',
    ]).ask()

if graph_type == 'collage':
    test_options = [folder.name for folder in DATA.iterdir()
                    if folder.is_dir()]
    tests = questionary.checkbox(
        "Select tests:",
        choices=test_options
    ).ask()

    type_options = ["bar", "progress", "entropy"]
    types = questionary.checkbox(
        "Select types:",
        choices=type_options
    ).ask()

    for test in tests:
        for path in sorted((DATA / test).glob("*")):
            if "entropy" in types:
                utils.generate_entropy_comparison(
                    pathlib.Path(test) / path.name)
            if "progress" in types:
                utils.generate_progress_comparison(
                    pathlib.Path(test) / path.name)
            if "bar" in types:
                utils.generate_final_results(pathlib.Path(test) / path.name)


# final_results
# progress comparison
# test_name = "test"

# grid search
# for path in (DATA / "grid_search" / "50").glob("*"):
#     utils.generate_gridmap_collage(path)

# gsa_path = pathlib.Path("grid_search") / "50" / "gsa_MinMax"
# utils.generate_gridmap_collage(gsa_path)

# gsa_path = pathlib.Path("grid_search") / "50" / "gsa_Sigmoid4_tiled"
# utils.generate_gridmap_collage(gsa_path)

# gsa_path = pathlib.Path("grid_search") / "50" / "gsa_Rank"
# utils.generate_gridmap_collage(gsa_path)

# gsa_path = pathlib.Path("grid_search") / "50" / "gsa_Rank_tiled"
# utils.generate_gridmap_collage(gsa_path)

# gsa_path = pathlib.Path("grid_search") / "50" / "gsa_Sigmoid"
# utils.generate_gridmap_collage(gsa_path)

# gsa_path = pathlib.Path("grid_search") / "50" / "gsa_Sigmoid_tiled"
# utils.generate_gridmap_collage(gsa_path)
# gsa_path = pathlib.Path("grid_search") / "50" / "gsa_Rank"
# utils.generate_gridmap_collage(gsa_path)

# igsa_path = pathlib.Path("grid_search") / "50" / "igsa"
# utils.generate_gridmap_collage(igsa_path)

# pso_path = pathlib.Path("grid_search") / "50" / "gsa_Sigmoid_tiled"
# utils.generate_gridmap_collage(pso_path)

# pso_path = pathlib.Path("grid_search") / "50" / "gsa_Rank_tiled"
# utils.generate_gridmap_collage(pso_path)

# overview
# utils.generate_overview(pathlib.Path(test_name) / "50" / "gsa_MinMax" / "CEC2017_F30" / "0", 1, 500)
# utils.generate_overview(pathlib.Path(test_name) / "50" / "gsa_Sigmoid_tiled" / "CEC2017_F30" / "0", 1, 500)


# for i in range(1, 31):
#     if i == 2:
#         continue
#     pso = PSO(DATA / pathlib.Path(test_name) / "50" / "gsa_MinMax" / f"CEC2017_F{i:02}" / "0")
#     pso.load_full()
#     pos = min(pso.iterations[-1].particles, key=lambda x: x.fitness).pos
#     val = np.std(pos)
#     val1 = np.sqrt(val * val / 50.)

#     pso = PSO(DATA / pathlib.Path(test_name) / "50" / "gsa_MinMax_tiled" / f"CEC2017_F{i:02}" / "0")
#     pso.load_full()
#     pos = min(pso.iterations[-1].particles, key=lambda x: x.fitness).pos
#     val = np.std(pos)
#     val2 = np.sqrt(val * val / 50.)

#     print(f"[CEC2017_F{i:02}] gM: {val1:.3f}, gMt: {val2:.3f}")

# utils.generate_overview(pathlib.Path("test") / "50" / "tiledgsa" / "CEC2017_F10" / "0", 1, 300)
# utils.generate_overview(pathlib.Path("test_1000") / "100" / "igsa" / "CEC2017_F30" / "0", 1, 300)
# utils.generate_overview(pathlib.Path("test_1000") / "100" / "tiledigsa" / "CEC2017_F30" / "0", 1, 300)
# generate_overview(pathlib.Path("test") / "30" / "igsa" / "CEC2017_F05" / "0", 1, 1000)

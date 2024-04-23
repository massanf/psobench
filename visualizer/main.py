import utils
import pathlib
from constants import DATA

test_name = "test_50"
for path in (DATA / test_name).glob("*"):
    # utils.generate_entropy_comparison(pathlib.Path("test") / path.name)
    # print(path)
    utils.generate_progress_comparison(pathlib.Path(test_name) / path.name)

# gsa_path = pathlib.Path("grid_search") / "50" / "gsa"
# utils.generate_gridmap_collage(gsa_path)

# igsa_path = pathlib.Path("grid_search") / "50" / "igsa"
# utils.generate_gridmap_collage(igsa_path)

# pso_path = pathlib.Path("grid_search") / "50" / "tiledgsa"
# utils.generate_gridmap_collage(pso_path)

# pso_path = pathlib.Path("grid_search") / "50" / "tiledigsa"
# utils.generate_gridmap_collage(pso_path)

utils.generate_overview(pathlib.Path("test_50") / "10" / "gsa" / "CEC2017_F10" / "0", 1, 300)
# utils.generate_overview(pathlib.Path("test_1000") / "30" / "tiledgsa" / "CEC2017_F10" / "0", 1, 300)
# utils.generate_overview(pathlib.Path("test_1000") / "100" / "igsa" / "CEC2017_F30" / "0", 1, 300)
# utils.generate_overview(pathlib.Path("test_1000") / "100" / "tiledigsa" / "CEC2017_F30" / "0", 1, 300)
# generate_overview(pathlib.Path("test") / "30" / "igsa" / "CEC2017_F05" / "0", 1, 1000)

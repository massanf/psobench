import utils
import pathlib
from constants import DATA

for path in (DATA / "test").glob("*"):
    print(path)
    utils.generate_summary(pathlib.Path("test") / path.name)

# gsa_path = pathlib.Path("grid_search") / "30" / "gsa"
# generate_gridmap_collage(gsa_path)

# igsa_path = pathlib.Path("grid_search") / "30" / "igsa"
# generate_gridmap_collage(igsa_path)

# pso_path = pathlib.Path("grid_search") / "30" / "pso"
# generate_gridmap_collage(pso_path)

# generate_overview(pathlib.Path("test") / "30" / "gsa" / "CEC2017_F05" / "0", 1, 1000)
# generate_overview(pathlib.Path("test") / "30" / "igsa" / "CEC2017_F05" / "0", 1, 1000)

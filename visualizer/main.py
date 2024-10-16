import questionary
import random
import utils
import matplotlib.pyplot as plt  # type: ignore
import utils
from pso import PSO
import pathlib
import os
from constants import DATA, GRAPHS


graph_type = questionary.select(
    "Select graph type:",
    choices=[
        'single',
        'gridmaps',
        'animation',
        'collage',
    ]).ask()


def get_paths(level: str):
    results = []
    test_options = [folder.name for folder in DATA.iterdir()
                    if folder.is_dir()]
    tests = questionary.checkbox(
        "Select tests:",
        choices=test_options
    ).ask()

    for test in tests:
        if level == "tests":
            results.append(test)
            continue

        dim_options = [str(x) for x in sorted([int(folder.name) for folder in (DATA / test).iterdir()
                              if folder.is_dir()])]
        dims = questionary.checkbox(
            f"Select dimensions ({test}):", choices=dim_options).ask()


        for dim in dims:
            if level == "dims":
                results.append(dim)
                continue

            optimizer_options = sorted([folder.name for folder in (DATA / test / dim).iterdir()
                                        if folder.is_dir()])
            optimizers = questionary.checkbox(
                f"Select problems ({pathlib.Path(test) / dim}):", choices=optimizer_options).ask()

            for optimizer in optimizers:
                if level == "optimizer":
                    results.append(optimizer)
                    continue

                problem_options = sorted([folder.name for folder in (DATA / test / dim / optimizer).iterdir()
                                          if folder.is_dir()])
                problems = questionary.checkbox(
                    f"Select problems ({pathlib.Path(test) / dim / optimizer}):", choices=problem_options).ask()

                for problem in problems:
                    if level == "problem":
                        results.append(problem)
                        continue

                    attempt_options = sorted([folder.name for folder in (DATA / test / dim / optimizer / problem).iterdir()
                                              if folder.is_dir()])
                    attempts = questionary.checkbox(
                        f"Select attempts ({pathlib.Path(test) / dim / optimizer / problem}):", choices=attempt_options).ask()

                    for attempt in attempts:
                        if level == "attempts":
                            results.append(
                            pathlib.Path(test) / dim / optimizer / problem / attempt)
    return results


if graph_type == 'single':
    attempts = get_paths("attempts")
    plt.close()
    plt.cla()
    plt.rcdefaults()
    plt.yscale("log")

    for attempt in attempts:
        data = DATA / attempt
        graphs = GRAPHS / attempt
        graphs.mkdir(parents=True, exist_ok=True)
        pso = PSO(data)
        pso.load_full()
        plt.plot(pso.global_best_fitness_progress(), label=attempt)

    plt.legend()
    plt.gca().autoscale(axis='y', tight=False)
    if not (GRAPHS / "custom_singles").exists():
        os.makedirs(GRAPHS / "custom_singles")
    filestem = questionary.text("Filename:").ask()
    if filestem == "":
        filestem = str(random.randint(100000, 999999))
    filepath = GRAPHS / "custom_singles" / f"fitness_over_time_{filestem}.png"
    print(f"Saving: {filepath}")
    plt.savefig(filepath)
    plt.close()


if graph_type == 'gridmaps':
    raise NotImplementedError()


if graph_type == 'animation':
    for attempt in get_paths("attempts"):
        pso = PSO(DATA / attempt)
        pso.load_full()
        skip = int(questionary.text(
            "Skip count: ", default="1").ask())
        end = int(questionary.text(
            "End: ", default=str(len(pso.iterations))).ask())
        pso.animate_particles(
            GRAPHS / attempt / "animation.gif", skip, 0, end)


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
        dim_options = [folder.name for folder in (DATA / test).iterdir()
                       if folder.is_dir()]
        dims = questionary.checkbox(
            "Select dimensions:", choices=dim_options).ask()
        for dim in dims:
            if "entropy" in types:
                utils.generate_entropy_comparison(
                    pathlib.Path(test) / dim)
            if "progress" in types:
                utils.generate_progress_comparison(
                    pathlib.Path(test) / dim)
            if "bar" in types:
                utils.generate_final_results(pathlib.Path(test) / dim)


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

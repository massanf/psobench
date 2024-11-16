import questionary
from scipy.stats import pearsonr
import datetime
import utils
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import utils
from pso import PSO
import pathlib
import matplotlib.animation as animation
import os
from constants import DATA, GRAPHS


graph_type = questionary.select(
    "Select graph type:",
    choices=[
        'single',
        'grid',
        'animation',
        'collage',
        'last best distance',
        'rmt',
    ]).ask()

def marchenko_pastur_pdf(x, q, sigma=1.0):
    lambda_min = sigma**2 * (1 - np.sqrt(1 / q))**2
    lambda_max = sigma**2 * (1 + np.sqrt(1 / q))**2
    return np.where(
        (x >= lambda_min) & (x <= lambda_max),
        q / (2 * np.pi * sigma**2) * np.sqrt((lambda_max - x) * (x - lambda_min)) / x,
        0,
    )


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
                results.append(pathlib.Path(test) / dim)
                continue

            optimizer_options = sorted([folder.name for folder in (DATA / test / dim).iterdir()
                                        if folder.is_dir()])
            optimizers = questionary.checkbox(
                f"Select problems ({pathlib.Path(test) / dim}):", choices=optimizer_options).ask()

            for optimizer in optimizers:
                if level == "optimizers":
                    results.append(pathlib.Path(test) / dim / optimizer)
                    continue

                problem_options = sorted([folder.name for folder in (DATA / test / dim / optimizer).iterdir()
                                          if folder.is_dir()])
                problems = questionary.checkbox(
                    f"Select problems ({pathlib.Path(test) / dim / optimizer}):", choices=problem_options).ask()

                for problem in problems:
                    if level == "problems":
                        results.append(pathlib.Path(test) /
                                       dim / optimizer / problem)
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
    fig, ax = plt.subplots()

    for attempt in attempts:
        data = DATA / attempt
        graphs = GRAPHS / attempt
        graphs.mkdir(parents=True, exist_ok=True)
        pso = PSO(data)
        pso.load_full()
        utils.plot_and_fill_best_worst(ax=ax, btm=pso.global_best_fitness_progress(
        ), top=pso.global_worst_fitness_progress(), log=True, label=attempt)

    plt.legend()
    plt.gca().autoscale(axis='y', tight=False)
    if not (GRAPHS / "custom_singles").exists():
        os.makedirs(GRAPHS / "custom_singles")
    filestem = questionary.text("Filename:").ask()
    if filestem == "":
        filestem = str(int(datetime.datetime.now().timestamp()))
    filepath = GRAPHS / "custom_singles" / f"fitness_over_time_{filestem}.png"
    print(f"Saving: {filepath}")
    plt.savefig(filepath)
    plt.close()


if graph_type == 'grid':
    gsa_paths = get_paths(level="optimizers")
    for path in gsa_paths:
        utils.generate_gridmap_collage(path)

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

if graph_type == 'rmt':
    for problem in get_paths("problems"):
        eigenvalues = [[] for _ in range (1000)]
        d = 0
        n = 0
        attempts = sorted([folder.name for folder in (DATA / problem).iterdir() if folder.is_dir()])
        for attempt in attempts:
            print(attempt)
            pso = PSO(DATA / problem / attempt)
            pso.load_full()
            diffs = []

            for i in range(0, len(pso.iterations)):
                positions = []
                for particle in pso.iterations[i].particles:
                    x = particle.pos
                    x -= np.mean(x)
                    x /= np.std(x)
                    positions.append(x)

                cov_matrix = np.cov(positions)

                # eigenvalues.append([np.real(val) for val in np.linalg.eigvals(cov_matrix)])
                for value in np.linalg.eigvals(cov_matrix):
                    if np.linalg.norm(value) > 0.00001:
                        eigenvalues[i].append(np.real(value))

                d = len(positions[0])
                n = len(positions)

        # Prepare for animation
        fig, ax = plt.subplots()
        q = d / n  # Parameter for Marchenko-Pastur distribution
        x = np.linspace(0, max([max(ev) if ev else 0 for ev in eigenvalues]) * 1.1, 500)
        theoretical_pdf = marchenko_pastur_pdf(x, q) * 5
        
        def animate(i):
            ax.clear()
            if eigenvalues[i]:
                ax.hist(eigenvalues[i], bins=50, density=True, alpha=0.7, label="Empirical Spectrum")
                ax.plot(x, theoretical_pdf, 'r-', label="Marchenko-Pastur")
                ax.set_title(f"Iteration {i+1}")
                ax.legend()
            else:
                ax.text(0.5, 0.5, f"No data at iteration {i+1}", ha='center', va='center')
        
        ani = animation.FuncAnimation(fig, animate, frames=1000, interval=100)
        ani.save('eigenvalue_animation.gif', writer='pillow')
        plt.close()
        # plt.hist(eigenvalues[999], bins=50, density=True, label="Empirical Spectrum")
        # # plt.xlim(3000, 35000)
        # 
        # q = d / n
        # x = np.linspace(0, 20, 500)
        # theoretical_pdf = marchenko_pastur_pdf(x, q) * 5 
        # plt.plot(x, theoretical_pdf, label="Marchenko-Pastur")

        # plt.legend()
        # plt.show()


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

import questionary
import datetime
import utils
import sys
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import utils
from pso import PSO
import pathlib
import matplotlib.animation as animation
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Pool
from tqdm import tqdm
from typing import Any
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


def wigner_dyson(s):
    """Wigner-Dyson distribution for GOE."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


def marchenko_pastur_pdf(x, q, sigma=1.0):
    lambda_min = sigma**2 * (1 - np.sqrt(1 / q))**2
    lambda_max = sigma**2 * (1 + np.sqrt(1 / q))**2
    return np.where(
        (x >= lambda_min) & (x <= lambda_max),
        q / (2 * np.pi * sigma**2 + sys.float_info.epsilon) *
        np.sqrt(abs((lambda_max - x) * (x - lambda_min))) / (x + sys.float_info.epsilon),
        0,
    )


def get_singles_filepath(name: str, extension: str):
    if not (GRAPHS / "custom_singles").exists():
        os.makedirs(GRAPHS / "custom_singles")
    filestem = questionary.text("Filename:").ask()
    if filestem == "":
        filestem = str(int(datetime.datetime.now().timestamp()))
    filepath = GRAPHS / "custom_singles" / f"{name}_{filestem}.{extension}"
    print(f"Saving: {filepath}")
    return filepath


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
    # plt.yscale("log")
    fig, ax = plt.subplots()

    for attempt in attempts:
        data = DATA / attempt
        graphs = GRAPHS / attempt
        graphs.mkdir(parents=True, exist_ok=True)
        pso = PSO(data)
        # utils.plot_and_fill_best_worst(ax=ax, btm=pso.global_best_fitness_progress(
        # ), top=pso.global_worst_fitness_progress(), log=True, label=attempt)
        pso.scatter_progress(ax=ax, label=attempt)

    plt.legend()
    plt.gca().autoscale(axis='y', tight=False)
    plt.savefig(get_singles_filepath("fitness_over_time", "png"))
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


def process_attempt(args):
    attempt, problem, analysis_type, lower, upper = args
    pso = PSO(DATA / problem / attempt)
    pso.load_full()
    eigenvalues_local = [[] for _ in range(1000)]
    spacings_local = [[] for _ in range(1000)]
    d_local = 0
    n_local = 0

    all_positions = []

    for i in range(len(pso.iterations)):
        positions = []
        fitness = []

        for particle in pso.iterations[i].particles:
            positions.append(particle.pos)
            fitness.append(particle.fitness)

        positions = np.array(positions)
        fitness = np.array(fitness)

        if analysis_type == "position":
            positions -= np.mean(positions, axis=0)
            positions /= np.std(positions, axis=0)
            n_local = positions.shape[0]
            cov_matrix = np.dot(positions.T, positions) / n_local
        elif analysis_type == "velocity":
            all_positions.append(positions)
            if len(all_positions) > 1:
                velocities = all_positions[-1] - all_positions[-2]
                velocities -= np.mean(velocities, axis=0)
                velocities /= np.std(velocities, axis=0)
                n_local = velocities.shape[0]
                cov_matrix = np.cov(velocities, rowvar=False) / n_local
            else:
                continue
        # elif analysis_type == "fitness_weighted":
        #     mean_position = np.sum(positions.T * fitness, axis=1)
        #     centered_positions = positions - mean_position
        #     cov_matrix = np.zeros((positions.shape[1], positions.shape[1]))
        #     for j in range(len(positions)):
        #         cov_matrix += fitness[j] * \
        #             np.outer(centered_positions[j], centered_positions[j])
        else:
            raise ValueError()
        
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-12]

        eigenvalues_local[i] = eigenvalues.tolist()

        # Compute eigenvalue spacings
        sorted_eigenvalues = np.sort(eigenvalues)
        spacings = np.diff(sorted_eigenvalues)
        spacings_local[i] = spacings.tolist()

        d_local = len(positions[0])
        n_local = len(positions)
    return eigenvalues_local, spacings_local, d_local, n_local

if graph_type == 'rmt':
    analysis_contents = questionary.checkbox(
        "Select analysis content:",
        choices=[
            'position',
            'velocity',
            'fitness_weighted',
        ]).ask()
    analysis_type = questionary.select(
        "Select analysis type:",
        choices=[
            'ESD',
            'ESD sub',
            'ESA',
            'ESD sub'
        ]).ask()

    for problem in get_paths("problems"):
        for analysis_content in analysis_contents:
            lower = 0
            upper = 100
            if 'sub' in analysis_type:
                lower = int(questionary.text(
                    "Lower: ", default="0").ask())
                upper = int(questionary.text(
                    "Upper: ", default="100").ask())

            attempts = sorted([folder.name for folder in (
                DATA / problem).iterdir() if folder.is_dir()])
            args_list = [(attempt, problem, analysis_content, lower, upper)
                         for attempt in attempts]

            with Pool() as pool:
                with tqdm(total=len(args_list), desc="Calc...") as pbar:
                    results = []
                    for result in pool.imap(process_attempt, args_list):
                        results.append(result)
                        pbar.update(1)

            iterations = len(results[0][0])
            eigenvalues = [[] for _ in range(iterations)]
            spacings = [[] for _ in range(iterations)]
            d = n = 0

            start= int(questionary.text(
                "Start: ", default="0").ask())
            skip = int(questionary.text(
                "Skip count: ", default="1").ask())
            end = int(questionary.text(
                "End: ", default=str(iterations)).ask())
            frames = list(range(start, end, skip))
            
            # Combine results from all attempts
            for eigenvalues_local, spacings_local, d_local, n_local in results:
                for i in range(len(eigenvalues_local)):
                    eigenvalues[i].extend(eigenvalues_local[i])
                    spacings[i].extend(spacings_local[i])
                d = d_local
                n = n_local

            if 'ESA' in analysis_type:
                # Prepare for Spacing Density Animation
                fig, ax = plt.subplots()
                max_evs = max([max(evs) if evs else 0 for evs in spacings]) * 1.1
                x = np.linspace(0, max_evs, 500)
                theoretical_pdf = wigner_dyson(x)
                progress_bar = tqdm(total=len(frames), desc="Graphing ESA")

                def animate_spacing(i) -> Any:
                    ax.clear()
                    progress_bar.update(1)
                    if spacings[i]:
                        ax.hist(spacings[i], bins=50, density=True,
                                alpha=0.7, label="Empirical Spacing Density")
                        ax.plot(x, theoretical_pdf, 'r-', label="Wigner-Dyson")
                        ax.set_ylim(0, 2.0)
                        ax.set_title(f"Iteration {i+1}")
                        ax.set_xlabel("Spacing")
                        ax.set_ylabel("Density")
                        ax.legend()
                    else:
                        ax.text(
                            0.5, 0.5, f"No data at iteration {i}", ha='center', va='center')

                ani_spacing = animation.FuncAnimation(
                    fig, animate_spacing, frames=frames, interval=100)
                ani_spacing.save(GRAPHS / problem /
                                 'esa_{analysis_type}_{lower}_{upper}.gif', writer='pillow')
                plt.close()

            if 'ESD' in analysis_type:
                # Prepare for ESD Animation
                fig, ax = plt.subplots()
                q = d / n  # Marchenko-Pastur parameter
                max_ev = max([max(ev) if ev else 0 for ev in eigenvalues]) * 1.1
                x = np.linspace(0, max_ev, 500)
                theoretical_pdf = marchenko_pastur_pdf(x, q) * 5 * (upper - lower) / 100
                progress_bar = tqdm(total=len(frames), desc="Graphing ESD")

                def animate(i) -> Any:
                    ax.clear()
                    progress_bar.update(1)
                    if eigenvalues[i]:
                        ax.hist(eigenvalues[i], bins=50, density=True,
                                alpha=0.7, label="Empirical Spectrum")
                        ax.plot(x, theoretical_pdf, 'r-', label="Marchenko-Pastur")
                        # ax.set_ylim(0,1)
                        ax.set_title(f"Iteration {i}")
                        ax.legend()
                    else:
                        ax.text(
                            0.5, 0.5, f"No data at iteration {i+1}", ha='center', va='center')

                ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
                ani.save(GRAPHS / problem /
                         f'esd_{analysis_content}_{lower}_{upper}.gif', writer='pillow')
                plt.close()
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

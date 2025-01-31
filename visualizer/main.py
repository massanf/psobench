import questionary
from matplotlib.colors import LogNorm
import pickle
import matplotlib.cm as cm
from joblib import Memory
import hashlib
import datetime
import utils
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import utils
import rmt
from pso import PSO
import pathlib
import matplotlib.ticker as mticker
import umap
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Pool
from tqdm import tqdm
from typing import Any
from constants import DATA, GRAPHS, CACHE

graph_type = questionary.select(
    "Select graph type:",
    choices=[
        'single',
        'multiple',
        'grid',
        'animation',
        'animation_frame',
        'collage',
        'last best distance',
        'rmt',
        'rmt_model',
        'umap',
        'fitness histogram animation',
        'dwd or adp'
    ]).ask()

# Define paths
CACHE.mkdir(parents=True, exist_ok=True)

plt.rcParams['text.usetex'] = True


def compute_attempt_checksum(attempt_path):
    """
    Compute a lightweight checksum based on file metadata for an attempt.
    """
    hash_sha256 = hashlib.sha256()
    for file_path in sorted(attempt_path.iterdir()):
        if file_path.is_file():
            stat = file_path.stat()
            # Incorporate file name, size, and modification time
            hash_sha256.update(file_path.name.encode())
            hash_sha256.update(str(stat.st_size).encode())
            hash_sha256.update(str(stat.st_mtime).encode())
    return hash_sha256.hexdigest()


def load_attempt_cache(problem, attempt, analysis_content):
    """
    Load cached result and checksum for a given problem and attempt.
    """
    cache_file = CACHE / problem / f"{attempt}_{analysis_content}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None, None


def save_attempt_cache(problem, attempt, checksum, result, analysis_content):
    """
    Save result and checksum to cache for a given problem and attempt.
    """
    problem_cache_dir = CACHE / problem
    problem_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = problem_cache_dir / f"{attempt}_{analysis_content}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump((checksum, result), f)


def process_attempt_cached(args):
    """
    Wrapper function to handle caching for a single attempt.
    """
    attempt, problem, analysis_content = args
    attempt_path = DATA / problem / attempt
    checksum = compute_attempt_checksum(attempt_path)

    cached_checksum, cached_result = load_attempt_cache(
        problem, attempt, analysis_content)

    if cached_checksum == checksum and cached_result is not None:
        return cached_result
    else:
        try:
            # Process the attempt since cache is invalid or doesn't exist
            result = rmt.process_attempt(args)
            save_attempt_cache(problem, attempt, checksum,
                               result, analysis_content)
            return result
        except Exception as e:
            print(f"Error on {problem} {attempt}: {e}")
            return None


def get_singles_filepath(name: str, extension: str):
    if not (GRAPHS / "custom_singles").exists():
        os.makedirs(GRAPHS / "custom_singles")
    filestem = questionary.text("Filename:").ask()
    if filestem == "":
        filestem = str(int(datetime.datetime.now().timestamp()))
    filepath = GRAPHS / "custom_singles" / f"{name}_{filestem}.{extension}"
    print(f"Saving: {filepath}")
    return filepath


PROD = True

if graph_type == 'single':
    plt.close()
    plt.cla()
    plt.rcdefaults()
    plt.rcParams['text.usetex'] = True

    from cycler import cycler
    # linestyle_cycler=(cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c"]) *
    #                   cycler('linestyle', ['-', '--', ':', '-.']))

    if PROD:
        plt.rcParams['font.family'] = 'Times New Roman'
    attempts = utils.get_paths("attempts")
    name_suffix = questionary.select("Name:", ["single", "pso_gsa"]).ask()
    attempt_path = pathlib.Path()

    iterations = []

    fig, ax = plt.subplots()

    for attempt in attempts:
        attempt_path = attempt
        attempt_path = attempt
        data = DATA / attempt
        graphs = GRAPHS / attempt
        graphs.mkdir(parents=True, exist_ok=True)
        pso = PSO(data)
        try:
            name = str(int(attempt.parent.parent.name.split("_")[2])) + "\\%"
        except Exception:
            name = "Default"

        # utils.plot_and_fill_best_worst(ax=ax, btm=pso.global_best_fitness_progress(),
        #                                top=pso.global_worst_fitness_progress(), log=True, label=attempt)
        name_dict = {"pso": "PSO", "ogsa": "GSA"}
        pso.scatter_progress(
            ax=ax, label=name_dict[attempt.parent.parent.name.split("_")[0]])
        iterations.append(len(pso.iterations))

    # ax.yaxis.set_minor_formatter(ticker.ScalarFormatter(True, useMathText=True))
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(True, useMathText=True))
    # ax.ticklabel_format(style='sci', scilimits=(0, 0), useOffset=True, useMathText=True, axis='y')

    # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
    # ax.yaxis.set_minor_locator(ticker.NullLocator())
    # ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10.0))

    legend = plt.legend(
        prop={'size': 'small'},
        edgecolor='black',    # Set the edge color to black
        fancybox=False        # Remove the rounded corners
    )
    legend.get_frame().set_linewidth(0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.xlim(0, max(iterations))
    plt.gca().autoscale(axis='y', tight=False)
    graphs = GRAPHS / "multiples" / "progress" / attempt_path.parent.name
    graphs.mkdir(parents=True, exist_ok=True)

    if PROD:
        plt.savefig(graphs / f"fitness_over_time_{name_suffix}.svg",
                    bbox_inches="tight", pad_inches=0.05, dpi=300)
    else:
        plt.savefig(get_singles_filepath("fitness_over_time", "png"))
    plt.close()

if graph_type == 'multiple':
    plt.close()
    plt.cla()
    plt.rcdefaults()
    plt.rcParams['text.usetex'] = True

    from cycler import cycler
    # linestyle_cycler=(cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c"]) *
    #                   cycler('linestyle', ['-', '--', ':', '-.']))
    # linestyle_cycler = (cycler('linestyle', ['-', '--', ':', '-.', '--', ':', '-.']))
    linestyle_cycler = (cycler('linestyle', [
        '-',
        (0, (1, 1)),
        (0, (5, 1, 1, 1, 1, 1, 1, 1)),
        (0, (5, 1, 1, 1, 1, 1)),
        (0, (5, 1, 1, 1)),
        (0, (5, 1))
    ]))
    plt.rc('axes', prop_cycle=linestyle_cycler)

    if PROD:
        plt.rcParams['font.family'] = 'Times New Roman'
    attempts = utils.get_paths("attempts")

    fig, ax = plt.subplots(figsize=(4, 3))

    iterations = []
    color_dict = {
        "Default": "#1f77b4",
        "5\\%": "#2ECC71",
        "10\\%": "#27AE60",
        "25\\%": "#16A085",
        "50\\%": "#147D6F",
        "100\\%": "#126E82"
    }

    attempt_path = pathlib.Path()

    for attempt in attempts:
        attempt_path = attempt
        data = DATA / attempt
        pso = PSO(data)
        try:
            name = str(int(attempt.parent.parent.name.split("_")[2])) + "\\%"
        except Exception:
            name = "Default"

        pso.plot_progress(
            ax=ax, label=name, color=color_dict[name], width=1.4 if name == "Default" else 0.8)
        iterations.append(len(pso.iterations))

    utils.style_legend(plt)
    plt.xlabel("Iteration")
    plt.ylabel("Fitnesss")
    plt.xlim(0, max(iterations))

    plt.gca().autoscale(axis='y', tight=False)

    graphs = GRAPHS / "multiples" / "progress" / attempt_path.parent.name
    graphs.mkdir(parents=True, exist_ok=True)
    if PROD:
        name = questionary.select("Name:", ["elites", "pso_gsa"]).ask()
        plt.savefig(graphs / f"fitness_over_time_{name}.svg",
                    bbox_inches="tight", pad_inches=0.05, dpi=300)
    else:
        plt.savefig(get_singles_filepath("fitness_over_time", "png"))
    plt.close()


if graph_type == 'umap':
    plt.close()
    plt.cla()
    plt.rcdefaults()
    plt.rcParams['text.usetex'] = True

    attempts = utils.get_paths("attempts")
    fig, ax = plt.subplots()
    iterations = []
    for attempt in attempts:
        data = DATA / attempt
        graphs = GRAPHS / attempt
        graphs.mkdir(parents=True, exist_ok=True)
        pso = PSO(data)
        pso.load_full()

        frames = []
        folder = GRAPHS / "umap_test" / attempt
        folder.mkdir(parents=True, exist_ok=True)
        for idx, iteration in enumerate(pso.iterations):
            if idx != 900:
                continue
            print(idx)
            positions = []
            fitness = []
            for particle in iteration.particles:
                positions.append(particle.pos)
                fitness.append(particle.fitness)

            # Convert positions and fitness to numpy arrays for compatibility with UMAP
            positions = np.array(positions)
            fitness = np.array(fitness)

            # Replace zero or negative fitness values to avoid issues with log scaling
            fitness[fitness <= 0] = np.min(fitness[fitness > 0]) * 1e-1

            # UMAP for dimensionality reduction
            embedding = umap.UMAP().fit_transform(positions)

            # Scatter plot with log scale for color
            plt.scatter(embedding[:, 0], embedding[:, 1],
                        c=fitness, cmap=cm.viridis, norm=LogNorm())
            plt.colorbar(label='Fitness (log scale)')

            # Save the plot
            plt.savefig(folder / f'umap_{idx}.png')
            plt.close()

if graph_type == 'grid':
    gsa_paths = utils.get_paths(level="optimizers")
    for path in gsa_paths:
        utils.generate_gridmap_collage(path)

if graph_type == 'animation':
    for attempt in utils.get_paths("attempts"):
        pso = PSO(DATA / attempt)
        pso.load_full()
        skip = int(questionary.text(
            "Skip count: ", default="1").ask())
        end = int(questionary.text(
            "End: ", default=str(len(pso.iterations))).ask())
        pso.animate_particles(
            GRAPHS / attempt / "animation.gif", skip, 0, end)

if graph_type == 'animation_frame':
    for attempt in utils.get_paths("attempts"):
        pso = PSO(DATA / attempt)
        pso.load_full()
        # frame = int(questionary.text(
        #    "Frame: ", default=str(len(pso.iterations))).ask())
        frames = [1, 2, 5, 10, 50, 100, 500, 1000]
        for frame in frames:
            pso.animate_particles_frame(
                GRAPHS / attempt / "animation.gif", frame - 1)


if graph_type == 'fitness histogram animation':
    for attempt in utils.get_paths("attempts"):
        pso = PSO(DATA / attempt)
        pso.load_full()
        skip = int(questionary.text(
            "Skip count: ", default="1").ask())
        end = int(questionary.text(
            "End: ", default=str(len(pso.iterations))).ask())
        pso.animate_fitness_histogram(
            GRAPHS / attempt / "fitness.gif", skip, 0, end)

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
    analysis_contents = questionary.checkbox(
        "Select analysis content:",
        choices=[
            'position',
            'velocity',
            '_fitness_weighted',
        ]).ask()
    analysis_type = questionary.select(
        "Select analysis type:",
        choices=[
            'eigenvalues',
            'eigenvalues_collage',
            'ESD',
            'ESA',
        ]).ask()

    visualization_type = ""
    frames = None
    visualize_iteration = 0
    if 'eigenvalues' not in analysis_type:
        visualization_type = questionary.select(
            "Select visualization type:",
            choices=[
                "animation",
                "image"
            ]).ask()
        if visualization_type == "animation":
            start = int(questionary.text(
                "Start: ", default="0").ask())
            skip = int(questionary.text(
                "Skip count: ", default="1").ask())
            end = int(questionary.text(
                "End: ", default=str(1000)).ask())
            frames = list(range(start, end, skip))
        else:
            visualize_iteration = int(questionary.text(
                "Iteration: "
            ).ask())
    else:
        graph_type = questionary.select(
            "Type:",
            choices=[
                'Different functions',
                'Different algorithms',
            ]).ask()

    # linestyle_cycler=(cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c"]) *
    #                   cycler('linestyle', ['-', '--', ':', '-.']))
    problem_path = pathlib.Path()
    problems = utils.get_paths("problems")

    for analysis_content in analysis_contents:
        # from cycler import cycler
        # linestyle_cycler = (cycler('linestyle', [
        #     '-',
        #     (0, (1, 1)),
        #     (0, (5, 1, 1, 1, 1, 1, 1, 1)),
        #     (0, (5, 1, 1, 1, 1, 1)),
        #     (0, (5, 1, 1, 1)),
        #     (0, (5, 1))
        # ]))
        # plt.rc('axes', prop_cycle=linestyle_cycler)

        if graph_type == 'Different algorithms':
            plt.figure(figsize=(4, 3))
        else:
            plt.figure(figsize=(4, 3))

        for problem in problems:
            plt.rcParams['text.usetex'] = True
            print(problem)
            problem_path = problem
            # Make folder.
            (GRAPHS / problem).mkdir(parents=True, exist_ok=True)

            # Do RMT Calculations.
            attempts = sorted([folder.name for folder in (
                DATA / problem).iterdir() if folder.is_dir()], key=lambda x: int(x))

            args_list = [(attempt, problem, analysis_content)
                         for attempt in attempts]

            with Pool() as pool:
                with tqdm(total=len(args_list), desc="Calc...") as pbar:
                    results = []
                    for result in pool.imap(process_attempt_cached, args_list):
                        if result is not None:
                            results.append(result)
                        pbar.update(1)

            iterations = len(results[0][0])

            eigenvalues = [[] for _ in range(iterations)]
            max_eigenvalues = [[] for _ in range(iterations)]
            spacings = [[] for _ in range(iterations)]
            means = [[] for _ in range(iterations)]
            stds = [[] for _ in range(iterations)]
            d = n = 0

            # Combine results from all attempts
            for eigenvalues_local, spacings_local, d_local, n_local in results:
                for i in range(0, len(eigenvalues_local)):
                    eigenvalues[i].extend(eigenvalues_local[i])
                    max_eigenvalues[i].append(max(eigenvalues_local[i]))
                    spacings[i].extend(spacings_local[i])
                d = d_local
                n = n_local
            q = d / n

            lambda_plus = (1 + q**0.5)**2
            lambda_minus = (1 - q**0.5)**2
            average_of_largest_eigenvalues = np.average(
                max_eigenvalues, axis=1)

            # Plot.
            if 'eigenvalues' in analysis_type:
                rmt.plot_percentage_of_outside_eigenvalues(
                    eigenvalues, lambda_plus, lambda_minus, problem, analysis_content,
                    ('collage' not in analysis_type), graph_type)
                # rmt.plot_average_of_largest_eigenvalues(max_eigenvalues, lambda_plus, problem, analysis_content)
                # rmt.plot_all_max_eigenvalue_evolutions(max_eigenvalues, problem, analysis_content)

            # Animations.
            if visualization_type == "animation":
                if 'ESD' in analysis_type:
                    rmt.animate_esd(q, eigenvalues, frames,
                                    problem, analysis_content)
                if 'ESA' in analysis_type:
                    rmt.animate_esa(spacings, frames,
                                    problem, analysis_content)
            else:
                if 'ESD' in analysis_type:
                    rmt.plot_esd(q, eigenvalues, problem,
                                 analysis_content, visualize_iteration)
                if 'ESA' in analysis_type:
                    rmt.plot_esa(spacings, problem,
                                 analysis_content, visualize_iteration)

        if 'collage' in analysis_type:
            assert 'eigenvalues' in analysis_type
            utils.style_legend(plt)
            # plt.ylim(-2, 50)
            if graph_type == "Different functions":
                parent = GRAPHS / "multiples" / "rmt" / \
                    "functions" / (problem_path.parent).name
                parent.mkdir(parents=True, exist_ok=True)
                path = parent / f"deviation_from_MP_{analysis_content}.svg"
            else:
                parent = GRAPHS / "multiples" / "rmt" / "algorithms" / problem_path.name
                parent.mkdir(parents=True, exist_ok=True)
                path = parent / f"deviation_from_MP_{analysis_content}.svg"
            plt.savefig(path, bbox_inches="tight", pad_inches=0.05)
            plt.close()

if graph_type == 'rmt_model':
    analysis_type = questionary.select(
        "Select analysis type:",
        choices=[
            'eigenvalues',
            'eigenvalues_collage',
            'ESD',
            'ESA',
        ]).ask()

    visualization_type = ""
    frames = None
    visualize_iteration = 0

    # linestyle_cycler=(cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c"]) *
    #                   cycler('linestyle', ['-', '--', ':', '-.']))

    from cycler import cycler
    linestyle_cycler = (cycler('linestyle', [
        '-',
        (0, (1, 1)),
        (0, (5, 1, 1, 1, 1, 1, 1, 1)),
        (0, (5, 1, 1, 1, 1, 1)),
        (0, (5, 1, 1, 1)),
        (0, (5, 1))
    ]))
    plt.rc('axes', prop_cycle=linestyle_cycler)

    if graph_type == 'Different algorithms':
        plt.figure(figsize=(4, 3))
    else:
        plt.figure(figsize=(4, 3))

    plt.rcParams['text.usetex'] = True
    # Make folder.
    (GRAPHS / "rmt_model").mkdir(parents=True, exist_ok=True)

    # Do RMT Calculations.
    iterations = 1

    eigenvalues = [[] for _ in range(iterations)]
    max_eigenvalues = [[] for _ in range(iterations)]
    spacings = [[] for _ in range(iterations)]
    means = [[] for _ in range(iterations)]
    stds = [[] for _ in range(iterations)]
    d = n = 0

    # Combine results from all attempts
    n_local = 50
    d_local = int(n_local / 2)
    matrix = np.random.randn(n_local, d_local)
    cov_matrix = np.cov(matrix, rowvar=False)

    eigenvalues_local, _ = np.linalg.eig(cov_matrix)

    sorted_eigenvalues = np.sort(eigenvalues_local)[::-1]
    
    # 4) Keep only top `rank` eigenvalues
    significant_eigenvalues = sorted_eigenvalues[:None]
    
    # 5) Sort them again in ascending order for unfolding
    ascending_eigs = np.sort(significant_eigenvalues)
    
    # 6) Unfold them
    unfolded_eigs = rmt.unfold_eigenvalues(ascending_eigs, poly_degree=6)
    
    # 7) Compute nearest-neighbor spacings in the unfolded variable
    spacings_local = np.diff(unfolded_eigs)

    # for eigenvalues_local, spacings_local, d_local, n_local in results:
    #     for i in range(0, len(eigenvalues_local)):
    #         eigenvalues[i].extend(eigenvalues_local[i])
    #         max_eigenvalues[i].append(max(eigenvalues_local[i]))
    #         spacings[i].extend(spacings_local[i])
    #     d = d_local
    #     n = n_local

    # for eigenvalues_here, d_local, n_local in [(eigenvalues_local, n_local, d_local)]:
    #     for i in range(0, len(eigenvalues_here)):
    #         eigenvalues[i].extend(eigenvalues_here[i])
    #     d = d_local
    #     n = n_local
    eigenvalues = [eigenvalues_local]
    spacings = [spacings_local]

    q = d_local / n_local

    lambda_plus = (1 + q**0.5)**2
    lambda_minus = (1 - q**0.5)**2
    average_of_largest_eigenvalues = np.average(
        max_eigenvalues, axis=1)

    # Animations.
    if 'ESD' in analysis_type:
        rmt.plot_esd_model(q, eigenvalues, n_local)

    if 'ESA' in analysis_type:
        rmt.plot_esa_model(spacings, n_local)

if graph_type == 'dwd or adp':
    analysis_contents = questionary.checkbox(
        "Select analysis content:",
        choices=[
            'position',
            'velocity',
            '_fitness_weighted',
        ]).ask()
    analysis_type = questionary.select(
        "Select analysis type:",
        choices=[
            'dwd',
            'adp',
        ]).ask()

    problem_path = pathlib.Path()
    problems = utils.get_paths("attempts")

    for analysis_content in analysis_contents:
        plt.figure(figsize=(6, 4.5))
        for problem in problems:
            plt.rcParams['text.usetex'] = True
            problem_path = problem
            
            plt.rcParams['font.family'] = 'Times New Roman'
            pso = PSO(DATA / problem)
            pso.load_full()
            number = int(problem.parent.name.split("F")[1])
            name = r"$F_{" + str(number) + "}$"
            print(pso.dwd(analysis_content))

            if analysis_type == "dwd":
                plt.plot(pso.dwd(analysis_content), linewidth=0.8, label=name)
            else:
                plt.plot(pso.adp(analysis_content), linewidth=0.8, label=name)

        plt.xlim(0, 1000)
        plt.xlabel("Iteration")
        if analysis_type == "dwd":
            plt.ylabel("Dimension-wise Diversity")
        else:
            plt.ylabel("Averaged Population Density")
        utils.style_legend(plt)
        parent = GRAPHS / "multiples" / analysis_type / (problem_path.parent.parent).name
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"{analysis_type}_{analysis_content}.svg"
        plt.yscale("log")

        plt.savefig(path, bbox_inches="tight", pad_inches=0.05)
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

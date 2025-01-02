import questionary
import pickle
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
        'fitness histogram animation'
    ]).ask()

# Define paths
CACHE_DIR = DATA / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

def load_attempt_cache(problem, attempt):
    """
    Load cached result and checksum for a given problem and attempt.
    """
    cache_file = CACHE_DIR / problem / f"{attempt}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None, None

def save_attempt_cache(problem, attempt, checksum, result):
    """
    Save result and checksum to cache for a given problem and attempt.
    """
    problem_cache_dir = CACHE_DIR / problem
    problem_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = problem_cache_dir / f"{attempt}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump((checksum, result), f)

def process_attempt_cached(args):
    """
    Wrapper function to handle caching for a single attempt.
    """
    attempt, problem, analysis_content = args
    attempt_path = DATA / problem / attempt
    checksum = compute_attempt_checksum(attempt_path)
    
    cached_checksum, cached_result = load_attempt_cache(problem, attempt)
    
    if cached_checksum == checksum and cached_result is not None:
        return cached_result
    else:
        # Process the attempt since cache is invalid or doesn't exist
        result = rmt.process_attempt(args)
        save_attempt_cache(problem, attempt, checksum, result)
        return result

def get_singles_filepath(name: str, extension: str):
    if not (GRAPHS / "custom_singles").exists():
        os.makedirs(GRAPHS / "custom_singles")
    filestem = questionary.text("Filename:").ask()
    if filestem == "":
        filestem = str(int(datetime.datetime.now().timestamp()))
    filepath = GRAPHS / "custom_singles" / f"{name}_{filestem}.{extension}"
    print(f"Saving: {filepath}")
    return filepath

if graph_type == 'single':
    plt.close()
    plt.cla()
    plt.rcdefaults()

    prod = False
    if prod:
        plt.rcParams['font.family'] = 'Times New Roman'
    attempts = utils.get_paths("attempts")
    fig, ax = plt.subplots()
    iterations = []

    for attempt in attempts:
        data = DATA / attempt
        graphs = GRAPHS / attempt
        graphs.mkdir(parents=True, exist_ok=True)
        pso = PSO(data)
        # utils.plot_and_fill_best_worst(ax=ax, btm=pso.global_best_fitness_progress(
        # ), top=pso.global_worst_fitness_progress(), log=True, label=attempt)
        pso.scatter_progress(ax=ax, label=attempt)
        iterations.append(len(pso.iterations))

    if not prod:
        plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.xlim(0, max(iterations))
    plt.gca().autoscale(axis='y', tight=False)
    if prod:
        plt.savefig(get_singles_filepath("fitness_over_time", "svg"),
                     bbox_inches="tight", pad_inches=0.05, dpi=300)
    else:
        plt.savefig(get_singles_filepath("fitness_over_time", "png"))
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
            start= int(questionary.text(
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

    from cycler import cycler
    linestyle_cycler=(cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c"]) * 
                      cycler('linestyle', ['-', '--', ':', '-.']))
    plt.rc('axes', prop_cycle=linestyle_cycler)

    problem_name = ""
    problems = utils.get_paths("problems")

    for analysis_content in analysis_contents:
        for problem in problems:
            problem_name = problem.name
            # Make folder.
            (GRAPHS / problem).mkdir(parents=True, exist_ok=True)

            # Do RMT Calculations.
            attempts = sorted([folder.name for folder in (
                DATA / problem).iterdir() if folder.is_dir()], key=lambda x: int(x))[:100]

            args_list = [(attempt, problem, analysis_content)
                         for attempt in attempts]

            with Pool() as pool:
                with tqdm(total=len(args_list), desc="Calc...") as pbar:
                    results = []
                    for result in pool.imap(process_attempt_cached, args_list):
                        results.append(result)
                        pbar.update(1)

            iterations = len(results[0][0])

            eigenvalues = [[] for _ in range(iterations)]
            max_eigenvalues = [[] for _ in range (iterations)]
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
            average_of_largest_eigenvalues = np.average(max_eigenvalues, axis=1) 
            
            # Plot.
            if 'eigenvalues' in analysis_type:
                rmt.plot_percentage_of_outside_eigenvalues(eigenvalues, lambda_plus, lambda_minus, problem, analysis_content, save=('collage' not in analysis_type))
                # rmt.plot_average_of_largest_eigenvalues(max_eigenvalues, lambda_plus, problem, analysis_content)
                # rmt.plot_all_max_eigenvalue_evolutions(max_eigenvalues, problem, analysis_content) 

            # Animations.
            if visualization_type == "animation":
                if 'ESD' in analysis_type:
                    rmt.animate_esd(q, eigenvalues, frames, problem, analysis_content)
                if 'ESA' in analysis_type:
                    rmt.animate_esa(spacings, frames, problem, analysis_content)
            else:
                if 'ESD' in analysis_type:
                    rmt.plot_esd(q, eigenvalues, problem, analysis_content, visualize_iteration)
                if 'ESA' in analysis_type:
                    rmt.plot_esa(spacings, problem, analysis_content, visualize_iteration)

        if 'collage' in analysis_type:
            plt.legend()
            plt.ylim(0, 25)
            path = get_singles_filepath(name=f"{analysis_content}_{problem_name}", extension='svg')
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

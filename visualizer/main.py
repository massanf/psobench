import questionary
import pathlib
import matplotlib.pyplot as plt  # type: ignore
import rmt
import utils
import numpy as np
from pso import PSO
from graphers import scatter_progress, plot_progress
from tqdm import tqdm  # type: ignore
from multiprocessing import Pool
from constants import DATA, GRAPHS, CACHE
from cache import process_attempt_cached

# -----------------------------
# Graph Generation Functions
# Each function encapsulates the code for one graph type.
# -----------------------------


def generate_single_graph() -> None:
    """Generate a single graph plot."""
    plt.close("all")
    plt.cla()
    plt.rcdefaults()
    utils.configure_matplotlib()

    PROD = True
    attempts = utils.get_paths("attempts")
    name_suffix = questionary.select("Name:", ["single", "pso_gsa"]).ask()
    iterations = []
    fig, ax = plt.subplots()

    for attempt in attempts:
        data = DATA / attempt
        graphs_dir = GRAPHS / attempt
        graphs_dir.mkdir(parents=True, exist_ok=True)
        pso = PSO(data)

        name_dict = {"pso": "PSO", "ogsa": "GSA"}
        name = attempt.parent.parent.name.split("_")[0]
        scatter_progress(pso, ax=ax, label=name_dict[name])
        iterations.append(len(pso.iterations))

    legend = plt.legend(
        prop={"size": "small"}, edgecolor="black", fancybox=False
    )
    legend.get_frame().set_linewidth(0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.xlim(0, max(iterations))
    plt.gca().autoscale(axis="y", tight=False)

    # Save graph
    output_dir = GRAPHS / "multiples" / "progress" / attempt.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if PROD:
        plt.savefig(
            output_dir / f"fitness_over_time_{name_suffix}.svg",
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=300,
        )
    else:
        plt.savefig(utils.get_singles_filepath("fitness_over_time", "png"))
    plt.close()


def generate_multiple_graphs() -> None:
    """Generate multiple graphs in one figure."""
    plt.close("all")
    plt.cla()
    plt.rcdefaults()
    utils.configure_matplotlib()

    from cycler import cycler

    linestyle_cycler = cycler(
        "linestyle",
        [
            "-",
            (0, (1, 1)),
            (0, (5, 1, 1, 1, 1, 1, 1, 1)),
            (0, (5, 1, 1, 1, 1, 1)),
            (0, (5, 1, 1, 1)),
            (0, (5, 1)),
        ],
    )
    plt.rc("axes", prop_cycle=linestyle_cycler)

    PROD = True
    if PROD:
        plt.rcParams["font.family"] = "Times New Roman"
    attempts = utils.get_paths("attempts")
    fig, ax = plt.subplots(figsize=(4, 3))
    iterations = []
    color_dict = {
        "Default": "#1f77b4",
        "5\\%": "#2ECC71",
        "10\\%": "#27AE60",
        "25\\%": "#16A085",
        "50\\%": "#147D6F",
        "100\\%": "#126E82",
    }

    for attempt in attempts:
        data = DATA / attempt
        pso = PSO(data)
        try:
            name = str(int(attempt.parent.parent.name.split("_")[2])) + "\\%"
        except Exception:
            name = "Default"
        plot_progress(
            pso,
            ax=ax,
            label=name,
            color=color_dict.get(name, "#1f77b4"),
            width=1.4 if name == "Default" else 0.8,
        )
        iterations.append(len(pso.iterations))

    utils.style_legend(plt)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.xlim(0, max(iterations))
    plt.gca().autoscale(axis="y", tight=False)

    output_dir = GRAPHS / "multiples" / "progress" / attempt.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if PROD:
        selected_name = questionary.select(
            "Name:", ["elites", "pso_gsa"]
        ).ask()
        plt.savefig(
            output_dir / f"fitness_over_time_{selected_name}.svg",
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=300,
        )
    else:
        plt.savefig(utils.get_singles_filepath("fitness_over_time", "png"))
    plt.close()


def generate_grid() -> None:
    """Generate a grid collage using gridmap."""
    gsa_paths = utils.get_paths(level="optimizers")
    for path in gsa_paths:
        utils.generate_gridmap_collage(path)


def generate_animation() -> None:
    """Generate particle animation for each attempt."""
    for attempt in utils.get_paths("attempts"):
        pso = PSO(DATA / attempt)
        pso.load_full()
        skip = int(questionary.text("Skip count: ", default="1").ask())
        end = int(
            questionary.text("End: ", default=str(len(pso.iterations))).ask()
        )
        pso.animate_particles(GRAPHS / attempt / "animation.gif", skip, 0, end)


def generate_animation_frame() -> None:
    """Generate animations for specific frames."""
    for attempt in utils.get_paths("attempts"):
        pso = PSO(DATA / attempt)
        pso.load_full()
        frames = [1, 2, 5, 10, 50, 100, 500, 1000]
        for frame in frames:
            pso.animate_particles_frame(
                GRAPHS / attempt / "animation.gif", frame - 1
            )


def generate_collage() -> None:
    """Generate a collage of multiple result types."""
    test_options = [f.name for f in DATA.iterdir() if f.is_dir()]
    tests = questionary.checkbox("Select tests:", choices=test_options).ask()
    type_options = ["bar", "progress", "entropy"]
    types = questionary.checkbox("Select types:", choices=type_options).ask()

    for test in tests:
        dim_options = [f.name for f in (DATA / test).iterdir() if f.is_dir()]
        dims = questionary.checkbox(
            "Select dimensions:", choices=dim_options
        ).ask()
        for dim in dims:
            path = pathlib.Path(test) / dim
            if "entropy" in types:
                utils.generate_entropy_comparison(path)
            if "progress" in types:
                utils.generate_progress_comparison(path)
            if "bar" in types:
                utils.generate_final_results(path)


def generate_rmt() -> None:
    """Generate RMT analysis plots and animations."""
    analysis_contents = questionary.checkbox(
        "Select analysis content:",
        choices=["position", "velocity", "_fitness_weighted"],
    ).ask()
    analysis_type = questionary.select(
        "Select analysis type:",
        choices=["eigenvalues", "eigenvalues_collage", "ESD", "ESA"],
    ).ask()

    visualization_type = ""
    visualize_iteration = 0
    if "eigenvalues" not in analysis_type:
        visualization_type = questionary.select(
            "Select visualization type:", choices=["animation", "image"]
        ).ask()
        if visualization_type == "animation":
            start = int(questionary.text("Start: ", default="0").ask())
            skip = int(questionary.text("Skip count: ", default="1").ask())
            end = int(questionary.text("End: ", default=str(1000)).ask())
            frames = list(range(start, end, skip))
        else:
            visualize_iteration = int(questionary.text("Iteration: ").ask())
    else:
        graph_type_choice = questionary.select(
            "Type:", choices=["Different functions", "Different algorithms"]
        ).ask()

    problems = utils.get_paths("problems")
    for analysis_content in analysis_contents:
        plt.figure(figsize=(4, 3))
        for problem in problems:
            utils.configure_matplotlib()
            print(problem)
            (GRAPHS / problem).mkdir(parents=True, exist_ok=True)
            attempts = sorted(
                [
                    folder.name
                    for folder in (DATA / problem).iterdir()
                    if folder.is_dir()
                ],
                key=lambda x: int(x),
            )
            args_list = [
                (attempt, problem, analysis_content) for attempt in attempts
            ]
            with Pool() as pool:
                with tqdm(total=len(args_list), desc="Calc...") as pbar:
                    results = []
                    for result in pool.imap(process_attempt_cached, args_list):
                        if result is not None:
                            results.append(result)
                        pbar.update(1)

            iterations = len(results[0][0])
            eigenvalues = np.array([[] for _ in range(iterations)])
            max_eigenvalues = np.array([[] for _ in range(iterations)])
            spacings = np.array([[] for _ in range(iterations)])
            d = n = 0
            for eigen_local, spacings_local, d_local, n_local in results:
                for i in range(len(eigen_local)):
                    eigenvalues[i].extend(eigen_local[i])
                    max_eigenvalues[i].append(max(eigen_local[i]))
                    spacings[i].extend(spacings_local[i])
                d = d_local
                n = n_local
            q = d / n
            lambda_plus = (1 + q**0.5) ** 2
            lambda_minus = (1 - q**0.5) ** 2

            if "eigenvalues" in analysis_type:
                rmt.plot_percentage_of_outside_eigenvalues(
                    eigenvalues,
                    lambda_plus,
                    lambda_minus,
                    problem,
                    analysis_content,
                    ("collage" not in analysis_type),
                    (
                        graph_type_choice
                        if "graph_type_choice" in locals()
                        else ""
                    ),
                )
            if visualization_type == "animation":
                if "ESD" in analysis_type:
                    rmt.animate_esd(
                        q, eigenvalues, frames, problem, analysis_content
                    )
                if "ESA" in analysis_type:
                    rmt.animate_esa(
                        spacings, frames, problem, analysis_content
                    )
            else:
                if "ESD" in analysis_type:
                    rmt.plot_esd(
                        q,
                        eigenvalues,
                        problem,
                        analysis_content,
                        visualize_iteration,
                    )
                if "ESA" in analysis_type:
                    rmt.plot_esa(
                        spacings,
                        problem,
                        analysis_content,
                        visualize_iteration,
                    )
        if "collage" in analysis_type:
            utils.style_legend(plt)
            if graph_type_choice == "Different functions":
                parent = (
                    GRAPHS
                    / "multiples"
                    / "rmt"
                    / "functions"
                    / (problem.parent).name
                )
            else:
                parent = (
                    GRAPHS / "multiples" / "rmt" / "algorithms" / problem.name
                )
            parent.mkdir(parents=True, exist_ok=True)
            path = parent / f"deviation_from_MP_{analysis_content}.svg"
            plt.savefig(path, bbox_inches="tight", pad_inches=0.05)
            plt.close()

# -----------------------------
# Dispatcher Dictionary
# Maps graph types to corresponding functions.
# -----------------------------


GRAPH_DISPATCH = {
    "single": generate_single_graph,
    "multiple": generate_multiple_graphs,
    "grid": generate_grid,
    "animation": generate_animation,
    "animation_frame": generate_animation_frame,
    "collage": generate_collage,
    "rmt": generate_rmt,
}

# -----------------------------
# Main Function
# -----------------------------


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    selected_graph_type = questionary.select(
        "Select graph type:",
        choices=list(GRAPH_DISPATCH.keys()),
    ).ask()
    if selected_graph_type in GRAPH_DISPATCH:
        GRAPH_DISPATCH[selected_graph_type]()
    else:
        print("Invalid graph type selected.")


if __name__ == "__main__":
    main()

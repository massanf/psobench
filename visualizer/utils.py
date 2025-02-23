from typing import List, Dict
import questionary
import pathlib
import matplotlib.pyplot as plt  # type: ignore
from typing import Any
from matplotlib.axes import Axes
from grid_searches import GridSearches
from tests import Tests
import os
import datetime
import numpy as np
from cycler import cycler
from constants import HOME, DATA, GRAPHS


def style_legend(plt: Any) -> None:
    plt.rcParams["text.usetex"] = True
    legend = plt.legend(
        prop={"size": "small"}, edgecolor="black", fancybox=False
    )
    legend.get_frame().set_linewidth(0.8)


def plot_and_fill(
    ax: Axes,
    iterations: List[List[float]],
    log: bool = True,
    label: str = "",
    alpha: float = 0.15,
) -> Axes:
    t = np.linspace(0, len(iterations), len(iterations))
    top = []
    btm = []
    mid = []
    for iteration in iterations:
        btm.append(np.quantile(iteration, 0.25))
        mid.append(np.quantile(iteration, 0.5))
        top.append(np.quantile(iteration, 0.75))

    if label != "":
        (line,) = ax.plot(t, mid, label=label, linestyle="-", linewidth=1)
    else:
        (line,) = ax.plot(t, mid)

    ax.fill_between(t, top, btm, color=line.get_color(), alpha=alpha)
    ax.set_xlim(0, len(iterations))

    if log:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    return ax


def plot_and_fill_best_worst(
    ax: Axes,
    btm: List[float],
    top: List[float],
    log: bool = True,
    label: str = "",
    alpha: float = 0.15,
) -> Axes:
    t = np.linspace(0, len(btm), len(top))

    line = ax.plot(t, btm)
    color = line[0].get_color()
    ax.fill_between(t, top, btm, alpha=alpha, color=color, label=label)
    ax.set_xlim(0, len(top))

    if log:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    return ax


def generate_gridmap_collage(
    path: pathlib.Path, average_mse: bool = False
) -> None:
    optimizer = GridSearches(DATA, GRAPHS, path)
    optimizer.heatmap_collage("grid_search.png", True, True)
    if average_mse:
        print("Average MSE: ", optimizer.average_mse(True, True))


def generate_progress_comparison(name: pathlib.Path) -> None:
    color_cycle = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]
    color_cycler = cycler(color=color_cycle)
    plt.rc("axes", prop_cycle=(color_cycler))
    data = HOME / "data" / name
    graphs = HOME / "graphs" / name
    graphs.mkdir(parents=True, exist_ok=True)
    tests = Tests(data)
    fig, axs = plt.subplots(5, 6, figsize=(15, 10), dpi=300)
    axs = tests.plot_all(axs)
    plt.tight_layout()
    print(f"Saving: {graphs/ f'progress_comparison.png'}")
    plt.savefig(graphs / "progress_comparison.png")


def generate_final_results(name: pathlib.Path) -> None:
    data = HOME / "data" / name
    graphs = HOME / "graphs" / name
    graphs.mkdir(parents=True, exist_ok=True)
    tests = Tests(data)
    fig, axs = plt.subplots(5, 6, figsize=(15, 10), dpi=300)
    axs = tests.plot_final_results(axs)
    plt.tight_layout()
    print(f"Saving: {graphs/ f'final_results.png'}")
    plt.savefig(graphs / "final_results.png")


def get_final_results(name: pathlib.Path) -> Dict[str, Dict[str, List[float]]]:
    data = HOME / "data" / name
    graphs = HOME / "graphs" / name
    graphs.mkdir(parents=True, exist_ok=True)
    tests = Tests(data)
    return tests.get_final_results()


def generate_entropy_comparison(name: pathlib.Path) -> None:
    data = HOME / "data" / name
    graphs = HOME / "graphs" / name
    graphs.mkdir(parents=True, exist_ok=True)
    tests = Tests(data)
    fig, axs = plt.subplots(5, 6, figsize=(15, 10), dpi=300)
    axs = tests.plot_all_entropy(axs)
    plt.tight_layout()
    print(f"Saving: {graphs/ f'entropy_comparison.png'}")
    plt.savefig(graphs / "entropy_comparison.png")


def shorter_names(names: List[str]) -> List[str]:
    # ["gsa_MinMax", "gsa_MinMax_tiled"] -> ["gM", "gMt"]
    result = []
    for name in names:
        ans = ""
        words = name.split("_")
        for word in words:
            ans += word[0]
            for i in range(1, len(word)):
                if word[i].isdigit():
                    ans += word[i]
        result.append(ans)
    return result


def get_paths(level: str) -> List[pathlib.Path]:
    results = []
    test_options = [
        folder.name for folder in DATA.iterdir() if folder.is_dir()
    ]
    tests = questionary.checkbox("Select tests:", choices=test_options).ask()

    for test in tests:
        if level == "tests":
            results.append(test)
            continue

        dim_options = [
            str(x)
            for x in sorted(
                [
                    int(folder.name)
                    for folder in (DATA / test).iterdir()
                    if folder.is_dir()
                ]
            )
        ]
        dims = questionary.checkbox(
            f"Select dimensions ({test}):", choices=dim_options
        ).ask()

        for dim in dims:
            if level == "dims":
                results.append(pathlib.Path(test) / dim)
                continue

            optimizer_options = sorted(
                [
                    folder.name
                    for folder in (DATA / test / dim).iterdir()
                    if folder.is_dir()
                ]
            )
            optimizers = questionary.checkbox(
                f"Select problems ({pathlib.Path(test) / dim}):",
                choices=optimizer_options,
            ).ask()

            for optimizer in optimizers:
                if level == "optimizers":
                    results.append(pathlib.Path(test) / dim / optimizer)
                    continue

                problem_options = sorted(
                    [
                        folder.name
                        for folder in (DATA / test / dim / optimizer).iterdir()
                        if folder.is_dir()
                    ]
                )
                path = pathlib.Path(test) / dim / optimizer
                problems = questionary.checkbox(
                    f"Select problems ({path}):",
                    choices=problem_options,
                ).ask()

                for problem in problems:
                    if level == "problems":
                        results.append(
                            pathlib.Path(test) / dim / optimizer / problem
                        )
                        continue

                    attempt_options = sorted(
                        [
                            folder.name
                            for folder in (
                                DATA / test / dim / optimizer / problem
                            ).iterdir()
                            if folder.is_dir()
                        ]
                    )
                    path = pathlib.Path(test) / dim / optimizer / problem
                    attempts = questionary.checkbox(
                        f"Select attempts ({path}):",
                        choices=attempt_options,
                    ).ask()

                    for attempt in attempts:
                        if level == "attempts":
                            results.append(
                                pathlib.Path(test)
                                / dim
                                / optimizer
                                / problem
                                / attempt
                            )
    return results


def get_singles_filepath(name: str, extension: str) -> pathlib.Path:
    if not (GRAPHS / "custom_singles").exists():
        os.makedirs(GRAPHS / "custom_singles")
    filestem = questionary.text("Filename:").ask()
    if filestem == "":
        filestem = str(int(datetime.datetime.now().timestamp()))
    filepath = GRAPHS / "custom_singles" / f"{name}_{filestem}.{extension}"
    print(f"Saving: {filepath}")
    return filepath


def configure_matplotlib() -> None:
    """Central configuration for matplotlib."""
    plt.rcParams["text.usetex"] = True
    # Set any additional configuration common to all graphs here.
    # For production, you might set a preferred font.
    PROD = True
    if PROD:
        plt.rcParams["font.family"] = "Times New Roman"

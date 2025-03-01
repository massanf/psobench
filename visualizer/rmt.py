import matplotlib.pyplot as plt  # type: ignore
import warnings
from pso import PSO
import matplotlib.animation as animation
from typing import Any, List
from tqdm import tqdm  # type: ignore
import numpy as np
import pathlib
from constants import DATA, GRAPHS
from numpy.typing import NDArray
import utils


def wigner_dyson(s: NDArray[np.float64], beta: int) -> NDArray[np.float64]:
    """
    Wigner-Dyson distribution for random matrix ensembles.

    Parameters:
    - s (float or np.ndarray): Normalized spacing between eigenvalues.
    - beta (int): Symmetry index (1 for GOE, 2 for GUE, 4 for GSE).

    Returns:
    - float or np.ndarray: Wigner-Dyson distribution for the given beta.
    """
    # Define the constants for normalization
    if beta == 1:  # GOE
        a = np.pi / 2
        b = np.pi / 4
    elif beta == 2:  # GUE
        a = 32 / (np.pi**2)
        b = 4 / np.pi
    elif beta == 4:  # GSE
        a = 2**18 / (3**6 * np.pi**3)
        b = 64 / (9 * np.pi)
    else:
        raise ValueError(
            "Unsupported beta value. Use beta = 1 (GOE), 2 (GUE), or 4 (GSE)."
        )

    # Calculate the Wigner-Dyson distribution
    return np.asarray(a * (s**beta) * np.exp(-b * s**2))


def poisson(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Poisson distribution for nearest-neighbor spacings."""
    return np.exp(-s)


def marchenko_pastur_pdf(
    x: NDArray[np.float64], q: float, sigma: float = 1.0
) -> tuple[NDArray[np.float64], float, float]:
    lambda_min = sigma**2 * (1 - np.sqrt(q)) ** 2
    lambda_max = sigma**2 * (1 + np.sqrt(q)) ** 2

    pdf = np.zeros_like(x)  # Initialize PDF with zeros
    valid = (x >= lambda_min) & (x <= lambda_max)  # Valid range for MP

    # Compute the PDF only for valid x
    pdf[valid] = (
        1
        / (2 * np.pi * q * sigma**2)
        * np.sqrt((lambda_max - x[valid]) * (x[valid] - lambda_min))
        / x[valid]
    )
    return pdf, lambda_min, lambda_max


def compute_cov_matrix(
    data: NDArray[np.float64], cov_type: str = "dimension"
) -> NDArray[np.float64]:
    """Standardizes data and computes the covariance matrix."""
    cov_matrix: NDArray[np.float64]
    if cov_type == "dimension":
        # Standardize columns (features)
        data -= np.mean(data, axis=0, keepdims=True)
        data /= np.std(data, axis=0, keepdims=True) + 1e-12
        n_local = data.shape[0]
        cov_matrix = np.dot(data.T, data) / n_local

    elif cov_type == "particle":
        # Standardize rows (particles)
        data -= np.mean(data, axis=0, keepdims=True)
        data /= np.std(data, axis=0, keepdims=True) + 1e-12

        d_local = data.shape[1]
        cov_matrix = np.dot(data, data.T) / d_local

    else:
        raise ValueError(f"Unknown covariance type: {cov_type}")

    return cov_matrix


def unfold_eigenvalues(
    eigenvals: NDArray[np.float64], poly_degree: int = 6
) -> NDArray[np.float64]:
    """
    Unfolds a sorted set of eigenvalues by fitting a polynomial (or spline)
    to the cumulative level count. The result is a transformed set of
    'unfolded' eigenvalues with (ideally) unit mean spacing throughout.

    Args:
        eigenvals (np.ndarray): 1D array of eigenvalues, sorted in ascending
                                order.
        poly_degree (int): Degree of polynomial for the unfolding fit.

    Returns:
        x (np.ndarray): Unfolded eigenvalues.
    """
    # Number of levels
    n = len(eigenvals)

    # Indices of the levels, e.g. 1, 2, ..., n
    level_indices = np.arange(1, n + 1)

    # Fit a polynomial: level_index = p(eigenvalue)
    # i.e. we fit p(\lambda) ~ i
    with warnings.catch_warnings():
        p = np.polyfit(eigenvals, level_indices, deg=poly_degree)

    # Create a callable polynomial
    poly = np.poly1d(p)

    # Now map each eigenvalue λ_i to x_i = poly(λ_i).
    # This 'x' is the smoothed "count" of levels below λ_i.
    x = poly(eigenvals)

    return x


def process_eigenvalues(
    cov_matrix: NDArray[np.float64], poly_degree: int = 6
) -> tuple:
    """
    Computes eigenvalues and spacings, accounting for numerical precision and
    matrix rank.
    Args:
        cov_matrix (np.ndarray): Covariance matrix.
        rank (int, optional): Maximum expected rank of the covariance matrix.
                              If None, it will be inferred from the matrix
                              dimensions.

    Returns:
        tuple: (eigenvalues, spacings)
    """
    # 1) Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    eigenvalues = np.real(eigenvalues)

    # 2) Decide on rank
    rank = min(cov_matrix.shape)

    # 3) Sort eigenvalues (descending)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    # 4) Keep only top `rank` eigenvalues
    significant_eigenvalues = sorted_eigenvalues[:rank]

    # 5) Sort them again in ascending order for unfolding
    ascending_eigs = np.sort(significant_eigenvalues)

    # 6) Unfold them
    unfolded_eigs = unfold_eigenvalues(ascending_eigs, poly_degree=poly_degree)

    # 7) Compute nearest-neighbor spacings in the unfolded variable
    spacings = np.diff(unfolded_eigs)

    return ascending_eigs, spacings


def process_attempt(
    args: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, int]:
    attempt, problem, analysis_type = args
    pso = PSO(DATA / problem / attempt)
    pso.load_full()

    eigenvalues_local = []
    spacings_local = []
    all_positions = []

    for i in range(len(pso.iterations)):
        positions = np.array([p.pos for p in pso.iterations[i].particles])

        if analysis_type == "position":
            cov_matrix = compute_cov_matrix(positions)
            rank = min(positions.shape)

        elif analysis_type == "velocity":
            all_positions.append(positions)
            if len(all_positions) > 1:
                velocities = all_positions[-1] - all_positions[-2]
                cov_matrix = compute_cov_matrix(velocities)
                rank = min(velocities.shape)
            else:
                continue
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        eigenvalues, spacings = process_eigenvalues(cov_matrix, rank)
        eigenvalues_local.append(eigenvalues)
        spacings_local.append(spacings)

    return (
        np.array(eigenvalues_local),
        np.array(spacings_local),
        positions.shape[1],
        positions.shape[0],
    )


def plot_percentage_of_outside_eigenvalues(
    eigenvalues: NDArray[np.float64],
    lambda_plus: float,
    lambda_minus: float,
    problem: pathlib.Path,
    analysis_content: str,
    save: bool,
    graph_type: str,
) -> None:
    """Plots the percentage of eigenvalues that exceed the upper bound
    over time.

    Args:
        eigenvalues (NDArray[np.float64]): Array of eigenvalues of all
            iterations, with shape (num_iterations, num_eigenvalues).
        lambda_plus (float): Upper bound for eigenvalues.
        lambda_minus (float): Lower bound for eigenvalues.
        problem (pathlib.Path): Path to the problem folder.
        analysis_content (str): Name of the content ("position" or "velocity).
        save (bool): Weather to save the plot or not.
        graph_type (str): Graph type.

    """
    percentages_of_large_eigenvalues = []

    prod = True
    if prod:
        plt.rcParams["font.family"] = "Times New Roman"

    for iteration_eigenvalues in eigenvalues:
        total = len(iteration_eigenvalues)
        count = sum(
            x > lambda_plus or x < lambda_minus for x in iteration_eigenvalues
        )
        if total != 0:
            percentage = (count / total) * 100
            percentages_of_large_eigenvalues.append(percentage)

    if graph_type == "Different algorithms":
        try:
            name = str(int(problem.parent.name.split("_")[2])) + "\\%"
        except Exception:
            name = "Default"
        color_dict = {
            "Default": "#1f77b4",
            "5\\%": "#2ECC71",
            "10\\%": "#27AE60",
            "25\\%": "#16A085",
            "50\\%": "#147D6F",
            "100\\%": "#126E82",
        }
        color = color_dict[name]
    else:
        number = int(problem.name.split("F")[1])
        name = r"$F_{" + str(number) + "}$"

    window_size = 20  # Adjust as needed
    smoothed_percentages = np.convolve(
        percentages_of_large_eigenvalues,
        np.ones(window_size) / window_size,
        mode="valid",
    )

    # Original x-axis and new x-axis after smoothing
    original_x = np.arange(len(percentages_of_large_eigenvalues))
    smoothed_x = np.arange(len(smoothed_percentages))

    # Interpolate the smoothed data back to the original size
    interpolated_y = np.interp(original_x, smoothed_x, smoothed_percentages)

    if graph_type == "Different algorithms":
        assert color is not None
        (line,) = plt.plot(
            original_x,
            interpolated_y,
            linewidth=1.4 if name == "Default" else 0.8,
            label=name,
            color=color,
        )
    else:
        (line,) = plt.plot(
            original_x, interpolated_y, linewidth=0.8, label=name
        )

    if not prod:
        plt.title("Percentage of Eigenvalues Exceeding MP Upper Bound")

    plt.xlim(0, len(eigenvalues))
    plt.xlabel("Iteration")
    plt.ylabel("Percentage (\\%)")
    # plt.ylim(0, 20)

    if save:
        if prod:
            plt.savefig(
                GRAPHS / problem / f"esa_{analysis_content}_cnt.svg",
                bbox_inches="tight",
                pad_inches=0.05,
            )
        else:
            plt.savefig(
                GRAPHS / problem / f"esa_{analysis_content}_cnt.png",
                bbox_inches="tight",
                pad_inches=0.05,
            )
        plt.close()


def plot_average_of_largest_eigenvalues(
    max_eigenvalues: NDArray[np.float64],
    lambda_plus: float,
    problem: pathlib.Path,
    analysis_content: str,
) -> None:
    average_of_largest_eigenvalues = np.average(max_eigenvalues, axis=1)
    plt.plot(
        range(len(average_of_largest_eigenvalues)),
        average_of_largest_eigenvalues,
        linestyle="-",
        label="Percentage > λ+",
    )
    plt.axhline(
        y=lambda_plus,
        color="gray",
        linestyle="--",
        linewidth=0.5,
        label="λ+ Threshold",
    )
    plt.title("Average of Largest Eigenvalues")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 15)
    utils.style_legend(plt)
    plt.grid()
    plt.savefig(GRAPHS / problem / f"esa_{analysis_content}_max.png")
    plt.close()


def plot_all_max_eigenvalue_evolutions(
    max_eigenvalues: NDArray[np.float64],
    problem: pathlib.Path,
    analysis_content: str,
) -> None:
    (GRAPHS / problem / "max").mkdir(parents=True, exist_ok=True)
    for i in range(0, len(max_eigenvalues[0])):
        plt.plot(range(len(max_eigenvalues)), np.array(max_eigenvalues)[:, i])
        plt.title("Max Eigenvalues for Each Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Percentage (%)")
        plt.grid()
        plt.savefig(
            GRAPHS / problem / "max" / f"esa_{analysis_content}_max_{i}.png"
        )
        plt.close()


def plot_esd(
    q: float,
    eigenvalues: NDArray[np.float64],
    problem: pathlib.Path,
    analysis_content: str,
    iteration: int,
) -> None:
    """Draws the empirical spectral density (ESD) of a particular iteration.

    Args:
        q (float): The `q` parameter for the Marchenko-Pastur distribution.
        eigenvalues (NDArray[np.float64]): Eigenvalues.
        problem (pathlib.Path): Path to the problem folder.
        analysis_content (str): Name of the content ("position" or "velocity).
        iteration (int): Iteration number.
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["text.usetex"] = True

    _, ax = plt.subplots(figsize=(4, 3))
    max_ev = max([max(ev) if ev else 0 for ev in eigenvalues]) * 1.1
    x = np.linspace(0, max_ev, 10000)
    theoretical_pdf, lambda_min, lambda_max = marchenko_pastur_pdf(x, q)
    bin_edges = list(np.arange(0, 10 + 0.5, 0.1))
    if "gsa" in problem.parent.name:
        color = "#1f77b4"
    else:
        color = "#ff7f0e"

    if eigenvalues[iteration]:
        valid_idx = (x >= lambda_min) & (x <= lambda_max)
        ax.plot(
            x[valid_idx],
            theoretical_pdf[valid_idx],
            color="black",
            linestyle="-",
            label="Marchenko-Pastur",
            linewidth=1,
        )
        ax.hist(
            eigenvalues[iteration],
            bins=bin_edges,
            density=True,
            alpha=0.7,
            color=color,
        )
        ax.set_ylim(0, 2.0)
        ax.set_xlim(0, 5.0)
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        utils.style_legend(plt)
    else:
        ax.text(
            0.5,
            0.5,
            f"No data at iteration {iteration+1}",
            ha="center",
            va="center",
        )

    plt.savefig(
        GRAPHS / problem / f"esd_{analysis_content}_{iteration}.svg",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()


def plot_esa(
    spacings: NDArray[np.float64],
    problem: pathlib.Path,
    analysis_content: str,
    iteration: int,
) -> None:
    """Draws the eigenvalue spacing analysis(ESA) of a particular iteration.

    Args:
        spacings (NDArray[np.float64]): Eigenvalue spacings of a particular
                                        iteration.
        problem (pathlib.Path): Path to the problem folder.
        analysis_content (str): Name of the content ("position" or "velocity).
        iteration (int): Iteration number.
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(4, 3))
    standardized_spacings = np.array(spacings)
    max_evs = max([max(evs) for evs in standardized_spacings]) * 1.1
    x = np.linspace(0, max_evs, 500)
    wigner_dyson_goe_pdf = wigner_dyson(x, 1)
    poisson_pdf = poisson(x)
    bin_edges = list(np.arange(0, 10 + 0.5, 0.1))
    if "gsa" in problem.parent.name:
        color = "#1f77b4"
    else:
        color = "#ff7f0e"

    ax.hist(
        standardized_spacings[iteration],
        bins=bin_edges,
        density=True,
        alpha=0.7,
        color=color,
    )
    ax.plot(
        x,
        wigner_dyson_goe_pdf,
        color="black",
        linestyle="--",
        label="Wigner-Dyson (GOE)",
        linewidth=1,
    )
    ax.plot(x, poisson_pdf, color="black", linestyle="-", label="Poisson")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 5)
    ax.set_xlabel("Spacing")
    ax.set_ylabel("Density")
    utils.style_legend(plt)

    save_path = GRAPHS / problem / f"esa_{analysis_content}_{iteration}.svg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def animate_esd(
    q: float,
    eigenvalues: NDArray[np.float64],
    frames: List[int],
    problem: pathlib.Path,
    analysis_content: str,
) -> None:
    """Animates the empirical spectral density (ESD) of a particular iteration.

    Args:
        q (float): The `q` parameter for the Marchenko-Pastur distribution.
        eigenvalues (NDArray[np.float64]): Eigenvalues.
        frames (tuple[int, int, int]): Tuple of frames to animate:
                                       [start, end, skip].
        problem (pathlib.Path): Path to the problem folder.
        analysis_content (str): Name of the content ("position" or "velocity).
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(4.5, 3.3753))
    max_ev = max([max(ev) if ev else 0 for ev in eigenvalues]) * 1.1
    x = np.linspace(0, max_ev, 500)
    theoretical_pdf, _, _ = marchenko_pastur_pdf(x, q)
    progress_bar = tqdm(total=len(frames), desc="Graphing ESD")
    bin_edges = list(np.arange(0, 10 + 0.5, 0.1))
    if "gsa" in problem.parent.name:
        color = "#1f77b4"
    else:
        color = "#ff7f0e"

    def animate(i: int) -> Any:
        ax.clear()
        progress_bar.update(1)
        if eigenvalues[i]:

            utils.style_legend(plt)
            ax.plot(
                x,
                theoretical_pdf,
                color="black",
                linestyle="-",
                label="Marchenko-Pastur",
            )
            ax.hist(
                eigenvalues[i],
                bins=bin_edges,
                density=True,
                alpha=0.7,
                color=color,
            )
            ax.set_ylim(0, 1.5)
            ax.set_xlim(0, 10)
            ax.set_title(f"Iteration {i}")
            utils.style_legend(plt)
        else:
            ax.text(
                0.5,
                0.5,
                f"No data at iteration {i+1}",
                ha="center",
                va="center",
            )

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
    ani.save(
        GRAPHS / problem / f"esd_{analysis_content}.gif",
        writer="pillow",
        dpi=300,
    )
    plt.close()


def animate_esa(
    spacings: NDArray[np.float64],
    frames: List[int],
    problem: pathlib.Path,
    analysis_type: str,
) -> None:
    """Animates the eigenvalue spacing analysis (ESA) of a particular
       iteration.

    Args:
        spacings (NDArray[np.float64]): Eigenvalue spacings of a particular
                                        iteration.
        frames (tuple[int, int, int]): Tuple of frames to animate:
                                       [start, end, skip].
        problem (pathlib.Path): Path to the problem folder.
        analysis_content (str): Name of the content ("position" or "velocity).
    """

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(4.5, 3.3753))
    standardized_spacings = np.array(spacings)
    for i in range(len(standardized_spacings)):
        standardized_spacings[i] /= np.average(standardized_spacings[i])
    max_evs = max([max(evs) for evs in standardized_spacings]) * 1.1
    x = np.linspace(0, max_evs, 500)
    wigner_dyson_goe_pdf = wigner_dyson(x, 1)
    # wigner_dyson_gue_pdf = wigner_dyson(x, 2)
    # wigner_dyson_gse_pdf = wigner_dyson(x, 4)
    poisson_pdf = poisson(x)
    progress_bar = tqdm(total=len(frames), desc="Graphing ESA")
    bin_edges = list(np.arange(0.1, 10 + 0.5, 0.1))

    if "gsa" in problem.parent.name:
        color = "#1f77b4"
    else:
        color = "#ff7f0e"

    def animate_spacing(i: int) -> Any:
        ax.clear()
        progress_bar.update(1)

        # KDE calculation
        # data = standardized_spacings[i]
        # kde = gaussian_kde(data)
        # kde.set_bandwidth(bw_method=0.1)
        # kde_values = kde(x)

        # ax.plot(x, kde_values, label="Empirical Spacing Density (KDE)")
        # plt.fill_between(x, kde_values, alpha=0.3)
        ax.hist(
            standardized_spacings[i],
            bins=bin_edges,
            density=True,
            alpha=0.7,
            color=color,
        )
        ax.plot(
            x,
            wigner_dyson_goe_pdf,
            color="black",
            linestyle="--",
            label="Wigner-Dyson (GOE)",
        )
        ax.plot(x, poisson_pdf, color="black", linestyle="-", label="Poisson")
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, 6)
        ax.set_title(f"Iteration {i}")
        ax.set_xlabel("Spacing")
        ax.set_ylabel("Density")
        utils.style_legend(plt)

    # frames = list(range(1, 994, 1))
    ani_spacing = animation.FuncAnimation(
        fig, animate_spacing, frames=frames, interval=100
    )

    # Create directories if they don't exist
    save_path = GRAPHS / problem / f"esa_{analysis_type}.gif"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ani_spacing.save(save_path, writer="pillow", dpi=300)
    plt.close()

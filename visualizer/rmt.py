import matplotlib.pyplot as plt  # type: ignore
from scipy.stats import gaussian_kde
from pso import PSO
import matplotlib.animation as animation
from typing import Any
from tqdm import tqdm
import numpy as np
from constants import DATA, GRAPHS

def wigner_dyson(s, beta):
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
        raise ValueError("Unsupported beta value. Use beta = 1 (GOE), 2 (GUE), or 4 (GSE).")
    
    # Calculate the Wigner-Dyson distribution
    return a * (s**beta) * np.exp(-b * s**2)

def poisson(s):
    """Poisson distribution for nearest-neighbor spacings."""
    return np.exp(-s)

def marchenko_pastur_pdf(x, q, sigma=1.0):
    lambda_min = sigma**2 * (1 - np.sqrt(q))**2
    lambda_max = sigma**2 * (1 + np.sqrt(q))**2

    pdf = np.zeros_like(x)  # Initialize PDF with zeros
    valid = (x >= lambda_min) & (x <= lambda_max)  # Valid range for MP

    # Compute the PDF only for valid x
    pdf[valid] = (
        1 / (2 * np.pi * q * sigma**2)
        * np.sqrt((lambda_max - x[valid]) * (x[valid] - lambda_min))
        / x[valid]
    )
    return pdf

def compute_cov_matrix(data, cov_type="dimension"):
    """Standardizes data and computes the covariance matrix."""
    if cov_type == "dimension":
        # Standardize columns (features)
        data -= np.mean(data, axis=0, keepdims=True)  # Center each feature
        data /= (np.std(data, axis=0, keepdims=True) + 1e-12)  # Scale each feature
        n_local = data.shape[0]
        cov_matrix = np.dot(data.T, data) / n_local

    elif cov_type == "particle":
        # Standardize rows (particles)
        data -= np.mean(data, axis=0, keepdims=True)  # Center each particle
        data /= (np.std(data, axis=0, keepdims=True) + 1e-12)  # Scale each particle

        d_local = data.shape[1]
        cov_matrix = np.dot(data, data.T) / d_local

    else:
        raise ValueError(f"Unknown covariance type: {cov_type}")

    return cov_matrix

def process_eigenvalues(cov_matrix, rank=None):
    """
    Computes eigenvalues and spacings, accounting for numerical precision and matrix rank.
    Args:
        cov_matrix (np.ndarray): Covariance matrix.
        rank (int, optional): Maximum expected rank of the covariance matrix. If None, 
                              it will be inferred from the matrix dimensions.

    Returns:
        tuple: (eigenvalues, spacings)
    """
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    eigenvalues = np.real(eigenvalues)  # Keep real part (discard numerical imaginary noise)
    
    if rank is None:
        rank = min(cov_matrix.shape)  # Maximum possible rank based on matrix dimensions
    
    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Keep only the top `rank` eigenvalues
    significant_eigenvalues = sorted_eigenvalues[:rank]
    
    # Filter eigenvalues based on relative tolerance to the largest eigenvalue
    # rel_tol = 1e-15  # Relative tolerance
    # max_eigenvalue = significant_eigenvalues[0]
    # significant_eigenvalues = significant_eigenvalues[significant_eigenvalues > rel_tol * max_eigenvalue]

    # Compute spacings between consecutive eigenvalues
    spacings = np.diff(np.sort(significant_eigenvalues))
    
    return significant_eigenvalues.tolist(), spacings.tolist()

def process_attempt(args):
    attempt, problem, analysis_type = args
    pso = PSO(DATA / problem / attempt)
    pso.load_full()

    eigenvalues_local = []
    spacings_local = []
    means = []
    stds = []
    all_positions = []

    for i in range(len(pso.iterations)):
        positions = np.array([particle.pos for particle in pso.iterations[i].particles])

        if analysis_type == "position":
            stds.append(float(np.mean(np.std(positions, axis=0))))
            cov_matrix = compute_cov_matrix(positions)
            rank = min(positions.shape)

        elif analysis_type == "velocity":
            all_positions.append(positions)
            if len(all_positions) > 1:
                velocities = all_positions[-1] - all_positions[-2]
                cov_matrix = compute_cov_matrix(velocities, cov_type)
                rank = min(velocities.shape)
            else:
                continue
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        eigenvalues, spacings = process_eigenvalues(cov_matrix, rank)
        eigenvalues_local.append(eigenvalues)
        spacings_local.append(spacings)

    return eigenvalues_local, spacings_local, positions.shape[1], positions.shape[0]

def plot_percentage_of_large_eigenvalues(eigenvalues, lambda_plus, problem, analysis_content):
    percentages_of_large_eigenvalues = []

    for iteration_eigenvalues in eigenvalues:
        total = len(iteration_eigenvalues)
        count = sum(x > lambda_plus for x in iteration_eigenvalues)
        if total != 0:
            percentage = (count / total) * 100
            percentages_of_large_eigenvalues.append(percentage)

    plt.plot(range(len(percentages_of_large_eigenvalues)),
             percentages_of_large_eigenvalues, linestyle='-', label='Percentage > λ+')
    plt.axhline(y=0, color='gray', linestyle='--',
                linewidth=0.5, label='λ+ Threshold')
    plt.title("Percentage of Eigenvalues Exceeding MP Upper Bound")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 15)
    plt.legend()
    plt.grid()
    plt.savefig(GRAPHS / problem / f'esa_{analysis_content}_cnt.png')
    plt.close()

def plot_average_of_largest_eigenvalues(max_eigenvalues, lambda_plus, problem, analysis_content):
    average_of_largest_eigenvalues = np.average(max_eigenvalues, axis=1) 
    plt.plot(range(len(average_of_largest_eigenvalues)),
             average_of_largest_eigenvalues, linestyle='-', label='Percentage > λ+')
    plt.axhline(y=lambda_plus, color='gray', linestyle='--',
                linewidth=0.5, label='λ+ Threshold')
    plt.title("Average of Largest Eigenvalues")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 15)
    plt.legend()
    plt.grid()
    plt.savefig(GRAPHS / problem / f'esa_{analysis_content}_max.png')
    plt.close()

def plot_all_max_eigenvalue_evolutions(max_eigenvalues, problem, analysis_content):
    # Plot the results
    (GRAPHS / problem / "max").mkdir(parents=True, exist_ok=True)
    for i in range(0, len(max_eigenvalues[0])):
        plt.plot(range(len(max_eigenvalues)), np.array(max_eigenvalues)[:, i])
        plt.title("Max Eigenvalues for Each Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Percentage (%)")
        plt.grid()
        plt.savefig(GRAPHS / problem / "max" / f'esa_{analysis_content}_max_{i}.png')
        plt.close()

def plot_mean_average(means, problem, analysis_content):
    plt.plot(np.mean(means, axis=1), label=f"{problem} ({analysis_content})")
    plt.title(f'{analysis_content} (mean)')
    plt.yscale("log")
    plt.legend()
    plt.savefig(GRAPHS / problem / f'esa_{analysis_content}_mean.png')
    plt.close()

def plot_std_average(stds, problem, analysis_content):
    plt.plot(np.mean(stds, axis=1), label=f"{problem} ({analysis_content})")
    plt.title(f'{analysis_content} (std)')
    plt.yscale("log")
    plt.legend()
    plt.savefig(GRAPHS / problem / f'esa_{analysis_content}_std.png')
    plt.close()

def animate_esd(q, eigenvalues, frames, problem, analysis_content):
    fig, ax = plt.subplots()
    max_ev = max([max(ev) if ev else 0 for ev in eigenvalues]) * 1.1
    x = np.linspace(0, max_ev, 500)
    theoretical_pdf = marchenko_pastur_pdf(x, q)
    progress_bar = tqdm(total=len(frames), desc="Graphing ESD")

    def animate(i) -> Any:
        ax.clear()
        progress_bar.update(1)
        if eigenvalues[i]:
            ax.plot(x, theoretical_pdf, color="black", linestyle='-', label="Marchenko-Pastur")
            # data = eigenvalues[i]
            # kde = gaussian_kde(data)
            # kde_values = kde(x)
            # ax.plot(x, kde_values, label=f"{problem} ({analysis_content})")
            # plt.fill_between(x, kde_values, alpha=0.3)
            ax.hist(eigenvalues[i], bins=50, density=True, alpha=0.7, edgecolor="gray")
            ax.set_ylim(0, 1.5)
            ax.set_xlim(0, 10)
            ax.set_title(f"Iteration {i}")
            ax.legend()
        else:
            ax.text(
                0.5, 0.5, f"No data at iteration {i+1}", ha='center', va='center')

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
    ani.save(GRAPHS / problem /
             f'esd_{analysis_content}.gif', writer='pillow')
    plt.close()

def animate_esa(spacings, frames, problem, analysis_type):
    fig, ax = plt.subplots()
    standardized_spacings = np.array(spacings)
    for i in range(len(standardized_spacings)):
        standardized_spacings[i] /= np.average(standardized_spacings[i])
    standardized_spacings = [row[row <= 10] for row in standardized_spacings]
    max_evs = max([max(evs) for evs in standardized_spacings]) * 1.1
    x = np.linspace(0, max_evs, 500)
    wigner_dyson_goe_pdf = wigner_dyson(x, 1)
    poisson_pdf = poisson(x)
    progress_bar = tqdm(total=len(frames), desc="Graphing ESA")

    def animate_spacing(i) -> Any:
        ax.clear()
        progress_bar.update(1)

        # KDE calculation
        # data = standardized_spacings[i]
        # kde = gaussian_kde(data)
        # kde.set_bandwidth(bw_method=0.1)
        # kde_values = kde(x)

        # ax.plot(x, kde_values, label="Empirical Spacing Density (KDE)")
        # plt.fill_between(x, kde_values, alpha=0.3)
        ax.hist(standardized_spacings[i], bins=50, density=True, alpha=0.7, edgecolor="gray")
        ax.plot(x, wigner_dyson_goe_pdf, color="black", linestyle='--', label="Wigner-Dyson (GOE)")
        ax.plot(x, poisson_pdf, color="black", linestyle='-', label="Poisson")
        ax.set_ylim(0, 1.5)
        ax.set_xlim(0, 10)
        ax.set_title(f"Iteration {i}")
        ax.set_xlabel("Spacing")
        ax.set_ylabel("Density")
        ax.legend()
        
    ani_spacing = animation.FuncAnimation(
        fig, animate_spacing, frames=frames, interval=100)
    
    # Create directories if they don't exist
    save_path = GRAPHS / problem / f'esa_{analysis_type}.gif'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    ani_spacing.save(save_path, writer='pillow')
    plt.close()

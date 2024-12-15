import matplotlib.pyplot as plt  # type: ignore
from pso import PSO
import matplotlib.animation as animation
from typing import Any
from tqdm import tqdm
import numpy as np
from constants import DATA, GRAPHS

def wigner_dyson(s):
    """Wigner-Dyson distribution for GOE."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

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

def compute_cov_matrix(data, cov_type):
    """Standardizes data and computes the covariance matrix."""
    if cov_type == "dimension":
        # Standardize columns (features)
        data -= np.mean(data, axis=0, keepdims=True)  # Center each feature
        data /= (np.std(data, axis=0, keepdims=True) + 1e-12)  # Scale each feature
        n_local = data.shape[0]
        cov_matrix = np.dot(data.T, data) / n_local

    elif cov_type == "particle":
        # Standardize rows (particles)
        data -= np.mean(data, axis=1, keepdims=True)  # Center each particle
        data /= (np.std(data, axis=1, keepdims=True) + 1e-12)  # Scale each particle

        d_local = data.shape[1]
        cov_matrix = np.dot(data, data.T) / d_local

    else:
        raise ValueError(f"Unknown covariance type: {cov_type}")

    return cov_matrix

def process_eigenvalues(cov_matrix):
    """Computes eigenvalues and spacings."""
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-12]

    sorted_eigenvalues = np.sort(eigenvalues)
    spacings = np.diff(sorted_eigenvalues)

    return eigenvalues.tolist(), spacings.tolist()

def process_attempt(args):
    attempt, problem, analysis_type, cov_type = args
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
            cov_matrix = compute_cov_matrix(positions, cov_type)

        elif analysis_type == "velocity":
            all_positions.append(positions)
            if len(all_positions) > 1:
                velocities = all_positions[-1] - all_positions[-2]
                cov_matrix = compute_cov_matrix(velocities, cov_type)
            else:
                continue
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        eigenvalues, spacings = process_eigenvalues(cov_matrix)
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
            ax.plot(x, theoretical_pdf, 'r-', label="Marchenko-Pastur")
            ax.hist(eigenvalues[i], bins=50, density=True,
                # weights=[1 / len(attempts)] * len(eigenvalues[i]),
                alpha=0.7, label=f"{problem} ({analysis_content})")
            ax.set_ylim(0, 1.2)
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
    standardized_spacings = spacings
    for i in range(len(standardized_spacings)):
        standardized_spacings[i] /= np.average(standardized_spacings[i])
    max_evs = max([max(evs) for evs in standardized_spacings]) * 1.1
    x = np.linspace(0, max_evs, 500)
    wigner_dyson_pdf = wigner_dyson(x)
    poisson_pdf = poisson(x)
    progress_bar = tqdm(total=len(frames), desc="Graphing ESA")

    def animate_spacing(i) -> Any:
        ax.clear()
        progress_bar.update(1)
        ax.hist(standardized_spacings[i], bins=50, density=True,
                alpha=0.7, label="Empirical Spacing Density")
        ax.plot(x, wigner_dyson_pdf, 'r-', label="Wigner-Dyson")
        ax.plot(x, poisson_pdf, 'r-', label="Poisson")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Iteration {i}")
        ax.set_xlabel("Spacing")
        ax.set_ylabel("Density")
        ax.legend()
        
    ani_spacing = animation.FuncAnimation(
        fig, animate_spacing, frames=frames, interval=100)
    ani_spacing.save(GRAPHS / problem /
                     f'esa_{analysis_type}.gif', writer='pillow')
    plt.close()

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

def process_attempt(args):
    attempt, problem, analysis_type = args
    pso = PSO(DATA / problem / attempt)
    pso.load_full()
    eigenvalues_local = []
    spacings_local = []
    means = []
    stds = []
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
            stds.append(float(np.mean(np.std(positions, axis=0))))

            positions -= np.mean(positions, axis=0, keepdims=True)
            positions /= np.std(positions, axis=0, keepdims = True)

            n_local = positions.shape[0]
            cov_matrix = np.dot(positions, positions.T) / n_local

            d_local = len(positions[0])
            n_local = len(positions)
            
        elif analysis_type == "velocity":
            all_positions.append(positions)
            if len(all_positions) > 1:
                velocities = all_positions[-1] - all_positions[-2]
                velocities -= np.mean(velocities, axis=0, keepdims=True)
                velocities /= np.std(velocities, axis=0, keepdims=True)
                means.append(float(np.mean(np.abs(np.mean(velocities, axis=0)))))
                stds.append(float(np.mean(np.std(velocities, axis=0))))

                n_local = velocities.shape[0]
                cov_matrix = np.dot(velocities, velocities.T) / n_local

                d_local = len(velocities[0])
                n_local = len(velocities)

            else:
                continue
            else:
            raise ValueError()
        
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-12]

        eigenvalues_local.append(eigenvalues.tolist())

        # Compute eigenvalue spacings
        sorted_eigenvalues = np.sort(eigenvalues)
        spacings = np.diff(sorted_eigenvalues)
        spacings_local.append(spacings.tolist())

    return eigenvalues_local, spacings_local, means, stds, d_local, n_local

def plot_percentage_of_large_eigenvalues(eigenvalues, lambda_plus, problem, analysis_content):
    # Initialize the list to store percentages
    percentages_of_large_eigenvalues = []

    # Calculate percentage of eigenvalues > lambda_plus for each iteration
    for iteration_eigenvalues in eigenvalues:
        total = len(iteration_eigenvalues)
        count = sum(x > lambda_plus for x in iteration_eigenvalues)
        if total != 0:
            percentage = (count / total) * 100
            percentages_of_large_eigenvalues.append(percentage)

    # Plot the results
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
                     'esa_{analysis_type}.gif', writer='pillow')
    plt.close()

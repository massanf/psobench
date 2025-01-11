import pathlib
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation  # type: ignore
from scipy.spatial import distance_matrix  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import concurrent.futures
from tqdm import tqdm  # type: ignore
from cycler import cycler
import utils
import math
import numpy as np
import os
import json
from particle import Particle
import re
from iteration import Iteration
from typing import List, Dict, Any, Callable, cast, Iterable
from matplotlib.artist import Artist


class PSO:
    def __init__(self, experiment_path: pathlib.Path) -> None:
        self.experiment_path = experiment_path
        self.fully_loaded = False

        # Config
        with open(experiment_path / "config.json", 'r') as file:
            config: Dict[str, Any] = json.load(file)
        self.config = config

        # Summary
        with open(experiment_path / "summary.json", 'r') as file:
            summary: Dict[str, Any] = json.load(file)
        self.summary = summary

    def load_full(self) -> None:
        # Data
        if not (self.experiment_path / "data.json").exists():
            raise ValueError("Full data not exported.")
        self.fully_loaded = True
        with open(self.experiment_path / "data.json", 'r') as file:
            data: List[Dict[str, Any]] = json.load(file)
        iterations = []
        for iteration_data in data:
            global_best_fitness = iteration_data["global_best_fitness"]
            particles = []
            for particle in iteration_data["particles"]:
                particles.append(Particle(particle))
            iterations.append(Iteration(global_best_fitness, particles))
        self.iterations = iterations

    def load_additional(self) -> None:
        # Data
        if not (self.experiment_path / "additional_data.json").exists():
            raise ValueError("Additional data not exported.")
        self.fully_loaded = True
        with open(self.experiment_path / "additional_data.json", 'r') as file:
            data: List[Dict[str, Any]] = json.load(file)
        iterations = []
        for iteration_data in data:
            particles = []
            for particle in iteration_data["particles"]:
                dict = {}
                for key in particle.keys():
                    dict[key] = particle[key]
                particles.append(dict)
            iterations.append(particles)
        self.additional_data_iterations = iterations

    def unload(self) -> None:
        # Data
        self.fully_loaded = False
        del self.iterations

    def global_best_fitness_progress(self) -> List[float]:
        global_best_fitness = self.summary["global_best_fitness"]
        if not (isinstance(global_best_fitness, list) and all(isinstance(i, float)
                                            for i in global_best_fitness)):
            raise ValueError("Incorrect dictionary type.")
        match = re.match(r"CEC2017_F(\d+)", self.config["problem"]["name"])
        solution = 0
        if match:
            solution = 100 * int(match.group(1))
        result = [] 
        for fitness in global_best_fitness:
            result.append(fitness - solution)
        return result

    def global_worst_fitness_progress(self) -> List[float]:
        global_worst_fitness = self.summary["global_worst_fitness"]
        if not (isinstance(global_worst_fitness, list) and all(isinstance(i, float)
                                            for i in global_worst_fitness)):
            raise ValueError("Incorrect dictionary type.")
        match = re.match(r"CEC2017_F(\d+)", self.config["problem"]["name"])
        solution = 0
        if match:
            solution = 100 * int(match.group(1))
        result = [] 
        for fitness in global_worst_fitness:
            result.append(fitness - solution)
        return result

    def evaluation_count(self) -> int:
        result = self.summary["evaluation_count"]
        if isinstance(result, int):
            return result
        raise ValueError("Incorrect dictionary type.")

    def global_best_fitness(self, idx: int = -1) -> float:
        progress = self.summary["global_best_fitness"]
        if idx != -1 and not (idx >= 0 and idx < len(progress)):
            raise ValueError("Index is out of bounds.")
        result = progress[idx]
        if isinstance(result, float):
            return result
        raise ValueError("Unexpected `global_best_fitness` value")

    def global_worst_fitness(self, idx: int = -1) -> float:
        progress = self.summary["global_worst_fitness"]
        if idx != -1 and not (idx >= 0 and idx < len(progress)):
            raise ValueError("Index is out of bounds.")
        result = progress[idx]
        if isinstance(result, float):
            return result
        raise ValueError("Unexpected `global_worst_fitness` value")

    def fitness(self) -> List[List[float]]:
        assert self.fully_loaded
        result: List[List[float]] = []
        for iteration in self.iterations:
            iter_result = []
            for particle in iteration.particles:
                iter_result.append(particle.fitness)
            result.append(iter_result)
        return result

    def speed(self) -> List[List[float]]:
        assert self.fully_loaded
        result: List[List[float]] = []
        for iteration in self.iterations:
            iter_result = []
            for particle in iteration.particles:
                iter_result.append(np.sqrt(np.sum(np.array(particle.vel)
                                                  ** 2)))
            result.append(iter_result)
        return result

    def entropy(self) -> List[float]:
        assert self.fully_loaded
        entropies = []
        for iteration in self.iterations:
            positions = []
            velocities = []
            for particle in iteration.particles:
                positions.append(particle.pos)
                velocities.append(particle.vel)
            data = np.hstack([positions, velocities])
            distances = distance_matrix(data, data)
            np.fill_diagonal(distances, np.inf)
            k = 10
            k_nearest_distances = np.sort(distances, axis=1)[:, :k]
            k_distances = k_nearest_distances[:, k-1]
            entropy_estimate = np.mean(np.log(k_distances))
            entropies.append(entropy_estimate)
        return entropies

    def mass(self) -> List[List[float]]:
        assert self.fully_loaded
        assert self.has_mass()
        result: List[List[float]] = []
        for iteration in self.iterations:
            iter_result = []
            for particle in iteration.particles:
                iter_result.append(particle.mass)
            result.append(iter_result)
        return result

    def has_mass(self) -> bool:
        return hasattr(self.iterations[0].particles[0], "mass")

    def update_particles_for_frame(self, frame: int) -> Any:
        """
        Renders a single frame of particles as a plot and returns the figure.
        """
        fig, ax = plt.subplots()
        assert self.fully_loaded
        iteration = self.iterations[frame]
        max_mass = 0.0
        colors = plt.cm.tab10.colors

        if self.has_mass():
            masses = [particle.mass for particle in iteration.particles]
            max_mass = np.max(masses)

        dim = 0
        for i, particle in enumerate(iteration.particles):
            assert len(particle.pos) >= 2
            if self.has_mass() and max_mass != 0.0:
                ax.scatter(particle.pos[dim], particle.pos[dim + 1],
                           s=particle.mass * 10 / max_mass, color=colors[i % len(colors)])
            else:
                ax.scatter(particle.pos[0], particle.pos[1], c='c', s=2)

        ax.grid()
        ax.set_xlim(*self.lim)
        ax.set_ylim(*self.lim)
        ax.set_title(f"Iteration: {frame}" +
                     f" Best: {self.iterations[frame].global_best_fitness:.3e}")
        ax.set_aspect('equal', adjustable='box')

        return fig

    def update_particles_for_histogram_frame(self, frame: int) -> Any:
        """
        Renders a single frame of particles as a plot and returns the figure.
        """
        fig, ax = plt.subplots()
        assert self.fully_loaded
        iteration = self.iterations[frame]

        assert self.has_mass()

        masses = [particle.mass for particle in iteration.particles]

        ax.hist(masses, bins=20, alpha=0.7)

        # ax.grid()
        ax.set_title(f"Iteration: {frame}" +
                     f" Best: {self.iterations[frame].global_best_fitness:.3e}")
        ax.set_xlim(0.0, 0.05)
        ax.set_ylim(0.0, 50)
        # ax.set_aspect('equal', adjustable='box')

        return fig


    def generate_frame_images(self, frames: List[int], output_dir: pathlib.Path) -> None:
        """
        Generate images for the specified frames in parallel.
        """
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.save_frame_image, frame, output_dir): frame
                for frame in frames
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                frame = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error generating frame {frame}: {e}")

    def generate_histogram_frame_images(self, frames: List[int], output_dir: pathlib.Path) -> None:
        """
        Generate images for the specified frames in parallel.
        """
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.save_histogram_frame_image, frame, output_dir): frame
                for frame in frames
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                frame = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error generating frame {frame}: {e}")


    def animate_fitness_histogram(self, destination_path: pathlib.Path,
                          skip_frames: int = 50, start: int = 0,
                          end: int = -1) -> None:
        """
        Generate animation from particle frames using parallel image generation.
        """
        assert self.fully_loaded

        # Define frame range
        if end == -1:
            end = len(self.iterations)
        frames = list(range(start, end, skip_frames))

        # Temporary directory for storing frames
        temp_dir = destination_path.parent / "temp_histogram_frames"
        self.generate_histogram_frame_images(frames, temp_dir)

        # Create animation from generated frames
        fig, ax = plt.subplots()
        img_paths = [temp_dir / f"frame_{frame:04d}.png" for frame in frames]

        def update(frame_idx) -> Any:
            img = plt.imread(img_paths[frame_idx])
            ax.clear()
            ax.imshow(img)
            ax.axis('off')

        ani = FuncAnimation(fig, update, frames=len(img_paths))
        ani.save(destination_path, fps=10)

        # Cleanup temporary frame files
        for img_path in img_paths:
            os.remove(img_path)
        temp_dir.rmdir()

    def save_histogram_frame_image(self, frame: int, output_dir: pathlib.Path) -> None:
        """
        Saves a single frame image.
        """
        fig = self.update_particles_for_histogram_frame(frame)
        frame_path = output_dir / f"frame_{frame:04d}.png"
        fig.savefig(frame_path, dpi=150)
        plt.close(fig)

    def save_frame_image(self, frame: int, output_dir: pathlib.Path) -> None:
        """
        Saves a single frame image.
        """
        fig = self.update_particles_for_frame(frame)
        frame_path = output_dir / f"frame_{frame:04d}.png"
        fig.savefig(frame_path, dpi=150)
        plt.close(fig)

    def animate_particles(self, destination_path: pathlib.Path,
                          skip_frames: int = 50, start: int = 0,
                          end: int = -1) -> None:
        """
        Generate animation from particle frames using parallel image generation.
        """
        assert self.fully_loaded

        # Determine plot limits
        self.lim = [float('inf'), float('-inf')]
        for iteration in self.iterations:
            for particle in iteration.particles:
                self.lim[0] = min(self.lim[0], particle.pos[0], particle.pos[1])
                self.lim[1] = max(self.lim[1], particle.pos[0], particle.pos[1])

        # Define frame range
        if end == -1:
            end = len(self.iterations)
        frames = list(range(start, end, skip_frames))

        # Temporary directory for storing frames
        temp_dir = destination_path.parent / "temp_frames"
        self.generate_frame_images(frames, temp_dir)

        # Create animation from generated frames
        fig, ax = plt.subplots()
        img_paths = [temp_dir / f"frame_{frame:04d}.png" for frame in frames]

        def update(frame_idx) -> Any:
            img = plt.imread(img_paths[frame_idx])
            ax.clear()
            ax.imshow(img)
            ax.axis('off')

        ani = FuncAnimation(fig, update, frames=len(img_paths))
        ani.save(destination_path, fps=10)

        # Cleanup temporary frame files
        for img_path in img_paths:
            os.remove(img_path)
        temp_dir.rmdir()

    # def animate_mass(self, destination_path: pathlib.Path,
    #                  skip_frames: int = 50, start: int = 0,
    #                  end: int = -1) -> None:
    #     assert self.fully_loaded
    #     mass = []
    #     for iteration in self.iterations:
    #         mass_ = []
    #         for particle in iteration.particles:
    #             mass_.append(particle.mass)
    #         mass.append(mass_)
    #     num_bins = 40
    #     flat_mass = np.array(mass).flatten()
    #     self.bins = np.linspace(min(flat_mass), max(flat_mass), num_bins + 1)
    #     max_y = 0
    #     for mass_ in mass:
    #         counts, _ = np.histogram(mass_, self.bins)
    #         max_y = max(max_y, counts.max())
    #     self.max_y = max_y
    #     self.animate(self.update_mass_for_animate, destination_path,
    #                  skip_frames, start, end)

    def plot_global_best_fitness_progress(self,
                                          out_directory: pathlib.Path) -> None:
        if not out_directory.exists():
            os.makedirs(out_directory)
        plt.close()
        plt.cla()
        plt.rcdefaults()
        plt.plot(self.global_best_fitness_progress())
        plt.gca().autoscale(axis='y', tight=False)
        print(f"Saving: {out_directory / 'fitness_over_time.png'}")
        plt.savefig(out_directory / "fitness_over_time.png")
        plt.close()

    def scatter_progress(self, ax: Axes, label: str):
        self.load_full()
        x = []
        y = []
        for i, iteration in enumerate(self.iterations):
            for particle in iteration.particles:
                x.append(i)
                y.append(particle.fitness)

        ax.scatter(x, y, s=0.01, label=label, rasterized=True) 
        ax.set_yscale("log")
        return ax

    def plot_progress(self, ax: Axes, label: str):
        self.load_full()
        y = []
        for i, iteration in enumerate(self.iterations):
            min_fitness = 1e20
            for particle in iteration.particles:
                min_fitness = min(min_fitness, particle.fitness)
            y.append(min_fitness)

        window_size = 100  # Adjust as needed
        # smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        ax.plot(range(len(y)), y, label=label, linewidth=0.8)
        ax.set_yscale("log")
        return ax

    def overview(self, animate: bool, out_directory: pathlib.Path) -> None:
        assert self.fully_loaded
        if not out_directory.exists():
            os.makedirs(out_directory)
        plt.close()

        plt.cla()
        plt.rcdefaults()
        _fig, ax = plt.subplots()
        ax = utils.plot_and_fill(ax, self.fitness())
        plt.gca().autoscale(axis='y', tight=False)
        print(f"Saving: {out_directory / 'fitness_over_time.png'}")
        plt.savefig(out_directory / "fitness_over_time.png")
        plt.close()

        plt.cla()
        plt.rcdefaults()
        _fig, ax = plt.subplots()
        ax = utils.plot_and_fill(ax, self.speed())
        print(f"Saving: {out_directory / 'speed_over_time.png'}")
        plt.savefig(out_directory / "speed_over_time.png")
        plt.close()

        # plt.cla()
        # plt.rcdefaults()
        # fig, ax = plt.subplots()
        # ax.plot(self.entropy())
        # print(f"Saving: {out_directory / 'entropy_over_time.png'}")
        # plt.savefig(out_directory / "entropy_over_time.png")
        # plt.close()

        if self.has_mass():
            plt.cla()
            plt.rcdefaults()
            _fig, ax = plt.subplots()
            ax = utils.plot_and_fill(ax, self.mass(), False)
            print(f"Saving: {out_directory / 'mass_over_time.png'}")
            plt.savefig(out_directory / "mass_over_time.png")
            plt.close()

        if animate:
            print(f"Saving: {out_directory / 'test.gif'}")
            self.animate_particles(out_directory / "test.gif")
            plt.close()

import pathlib
from matplotlib.animation import FuncAnimation  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm  # type: ignore
import utils
import numpy as np
import json
from particle import Particle
from iteration import Iteration
from typing import List, Dict, Any


class PSO:
    def __init__(self, experiment_path: pathlib.Path) -> None:
        self.experiment_path = experiment_path
        self.full_loaded = False

        # Config
        with open(experiment_path / "config.json", 'r') as file:
            config: Dict[str, Any] = json.load(file)
        self.config = config

        # Config
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

    def global_best_fitness_progress(self) -> List[float]:
        result = self.summary["global_best_fitness"]
        if isinstance(result, list) and all(isinstance(i, float)
                                            for i in result):
            return result
        raise ValueError("Incorrect dictionary type.")

    def global_best_fitness(self, idx: int) -> float:
        progress = self.summary["global_best_fitness"]
        if idx != -1 and not (idx >= 0 and idx < len(progress)):
            raise ValueError("Index is out of bounds.")
        result = progress[idx]
        if isinstance(result, float):
            return result
        raise ValueError("Unexpected `global_best_fitness` value")

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

    def update_plot_animate(self, frame: int) -> None:
        assert self.fully_loaded
        plt.cla()
        self.progressbar.update(1)
        iteration = self.iterations[frame]
        for particle in iteration.particles:
            assert len(particle.pos) >= 2
            plt.scatter(particle.pos[0], particle.pos[1])

        plt.grid()
        plt.xlim(self.lim)
        plt.ylim(self.lim)
        plt.title(f"Iteration: {frame}" +
                  f" Best: {self.iterations[frame].global_best_fitness:.3e}")
        plt.gca().set_aspect('equal', adjustable='box')

    def animate(self, destination_path: pathlib.Path,
                skip_frames: int = 50) -> None:
        assert self.fully_loaded
        # Find lim
        self.lim = [float('inf'), float('-inf')]
        for iteration in self.iterations:
            for particle in iteration.particles:
                self.lim[0] = min(self.lim[0],
                                  particle.pos[0],
                                  particle.pos[1])
                self.lim[1] = max(self.lim[1],
                                  particle.pos[0],
                                  particle.pos[1])

        fig, ax = plt.subplots()
        self.progressbar = tqdm(total=int(len(self.iterations) / skip_frames))
        frames = range(0, len(self.iterations), skip_frames)
        ani = FuncAnimation(fig, self.update_plot_animate,
                            frames=frames)
        ani.save(destination_path, writer='imagemagick', fps=10)

    def overview(self, animate: bool, out_directory: pathlib.Path) -> None:
        assert self.fully_loaded
        plt.close()
        plt.rcdefaults()
        utils.plot_and_fill(self.fitness())
        # Set the y-axis limits to automatic
        plt.gca().autoscale(axis='y', tight=False)
        plt.savefig(out_directory / "fitness_over_time.png")

        plt.close()
        plt.cla()
        plt.rcdefaults()
        utils.plot_and_fill(self.speed())
        plt.savefig(out_directory / "speed_over_time.png")

        plt.close()
        if animate:
            self.animate(out_directory / "test.gif")

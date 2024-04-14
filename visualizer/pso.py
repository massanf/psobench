import pathlib
from matplotlib.animation import FuncAnimation  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm  # type: ignore
import utils
import math
import numpy as np
import os
import json
from particle import Particle
from iteration import Iteration
from typing import List, Dict, Any, Callable


class PSO:
    def __init__(self, experiment_path: pathlib.Path) -> None:
        self.experiment_path = experiment_path
        self.full_loaded = False

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

    def global_best_fitness_progress(self) -> List[float]:
        result = self.summary["global_best_fitness"]
        if isinstance(result, list) and all(isinstance(i, float)
                                            for i in result):
            return result
        raise ValueError("Incorrect dictionary type.")

    def evaluation_count(self) -> int:
        result = self.summary["evaluation_count"]
        if isinstance(result, int):
            return result
        raise ValueError("Incorrect dictionary type.")

    def global_best_fitness(self, idx: int=-1) -> float:
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

    def update_particles_for_animate(self, frame: int) -> None:
        assert self.fully_loaded
        plt.cla()
        self.progressbar.update(1)
        iteration = self.iterations[frame]
        if self.has_mass():
            masses = []
            for particle in iteration.particles:
                masses.append(particle.mass)
            max_mass = np.max(masses)
        for particle in iteration.particles:
            assert len(particle.pos) >= 2
            if self.has_mass() and max_mass != 0.0:
                plt.scatter(particle.pos[0], particle.pos[1], s=particle.mass * 50 / max_mass, c='c')
            else:
                plt.scatter(particle.pos[0], particle.pos[1], c='c')

        plt.grid()
        plt.xlim(self.lim)
        plt.ylim(self.lim)
        plt.title(f"Iteration: {frame}" +
                  f" Best: {self.iterations[frame].global_best_fitness:.3e}")
        plt.gca().set_aspect('equal', adjustable='box')

    def update_mass_for_animate(self, frame: int) -> None:
        assert self.fully_loaded
        assert self.has_mass()

        plt.cla()
        self.progressbar.update(1)
        iteration = self.iterations[frame]

        mass = []
        for particle in iteration.particles:
            mass.append(particle.mass)

        plt.hist(mass, bins=self.bins)
        plt.ylim(0, self.max_y)
        plt.title(f"Iteration: {frame}")

    def animate(self, updater: Callable, destination_path: pathlib.Path,
                skip_frames: int = 50, start=0, end=-1) -> None:
        fig, ax = plt.subplots()
        if end == -1:
            end = len(self.iterations)
        frames = range(start, end, skip_frames)
        self.progressbar = tqdm(total=math.ceil((end - start) / skip_frames) + 1)
        ani = FuncAnimation(fig, updater, frames=frames)
        ani.save(destination_path, fps=10)
    
    def animate_particles(self, destination_path: pathlib.Path,
                          skip_frames: int = 50, start=0, end=-1) -> None:
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
        self.animate(self.update_particles_for_animate, destination_path, skip_frames, start, end)

    def animate_mass(self, destination_path: pathlib.Path,
                          skip_frames: int = 50, start=0, end=-1) -> None:
        assert self.fully_loaded
        mass = []
        for iteration in self.iterations:
            mass_ = []
            for particle in iteration.particles:
                mass_.append(particle.mass)
            mass.append(mass_)
        num_bins = 40
        flat_mass = np.array(mass).flatten()
        self.bins = np.linspace(min(flat_mass), max(flat_mass), num_bins + 1)
        max_y = 0
        for mass_ in mass:
            counts, _ = np.histogram(mass_, self.bins)
            max_y = max(max_y, counts.max())
        self.max_y = max_y
        self.animate(self.update_mass_for_animate, destination_path, skip_frames, start, end)

    def plot_global_best_fitness_progress(self, out_directory: pathlib.Path) -> None:
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

    def overview(self, animate: bool, out_directory: pathlib.Path) -> None:
        assert self.fully_loaded
        if not out_directory.exists():
            os.makedirs(out_directory)
        plt.close()

        plt.cla()
        plt.rcdefaults()
        fig, ax = plt.subplots()
        ax = utils.plot_and_fill(ax, self.fitness())
        plt.gca().autoscale(axis='y', tight=False)
        print(f"Saving: {out_directory / 'fitness_over_time.png'}")
        plt.savefig(out_directory / "fitness_over_time.png")
        plt.close()

        plt.cla()
        plt.rcdefaults()
        fig, ax = plt.subplots()
        ax = utils.plot_and_fill(ax, self.speed())
        print(f"Saving: {out_directory / 'speed_over_time.png'}")
        plt.savefig(out_directory / "speed_over_time.png")
        plt.close()

        if self.has_mass():
            plt.cla()
            plt.rcdefaults()
            fig, ax = plt.subplots()
            ax = utils.plot_and_fill(ax, self.mass(), False)
            print(f"Saving: {out_directory / 'mass_over_time.png'}")
            plt.savefig(out_directory / "mass_over_time.png")
            plt.close()

        if animate:
            print(f"Saving: {out_directory / 'test.gif'}")
            self.animate_particles(out_directory / "test.gif")
            plt.close()

from matplotlib.animation import FuncAnimation  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pathlib
import json
from typing import List, Dict, Any, Union
from tqdm import tqdm  # type: ignore

HOME = pathlib.Path(__file__).parent


class Particle:
    def __init__(self, datum: Dict[str, Union[float, List[float]]]) -> None:
        if isinstance(datum["pos"], list) and all(isinstance(i, float)
                                                  for i in datum["pos"]):
            self.pos: List[float] = datum["pos"]

        if isinstance(datum["vel"], list) and all(isinstance(i, float)
                                                  for i in datum["vel"]):
            self.vel: List[float] = datum["vel"]

        if isinstance(datum["fitness"], float):
            self.fitness: float = datum["fitness"]


class Iteration:
    def __init__(self, global_best_fitness: float,
                 particles: List[Particle]) -> None:
        self.particles = particles
        self.global_best_fitness = global_best_fitness

    def visualize(self, title: str, x_lim: List[float],
                  y_lim: List[float], filename: pathlib.Path) -> None:
        plt.cla()
        for particle in self.particles:
            assert len(particle.pos) >= 2
            plt.scatter(particle.pos[0], particle.pos[1])

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(title)

        plt.savefig(filename)


class PSO:
    def __init__(self, file_path: pathlib.Path) -> None:
        with open(file_path, 'r') as file:
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
        result = []
        for iteration in self.iterations:
            result.append(iteration.global_best_fitness)
        return result

    def fitness(self) -> List[List[float]]:
        result: List[List[float]] = []
        for iteration in self.iterations:
            iter_result = []
            for particle in iteration.particles:
                iter_result.append(particle.fitness)
            result.append(iter_result)
        return result

    def speed(self) -> List[List[float]]:
        result: List[List[float]] = []
        for iteration in self.iterations:
            iter_result = []
            for particle in iteration.particles:
                iter_result.append(np.sqrt(np.sum(np.array(particle.vel)
                                                  ** 2)))
            result.append(iter_result)
        return result

    def update_plot_animate(self, frame: int) -> None:
        plt.cla()  # Clear the current axes.
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

    def overview(self, animate: bool) -> None:
        plt.close()
        plt.rcdefaults()
        plot_and_fill(self.fitness())
        # Set the y-axis limits to automatic
        plt.gca().autoscale(axis='y', tight=False)
        plt.savefig(HOME / "graphs" / "fitness_over_time.png")

        plt.close()
        plt.cla()
        plt.rcdefaults()
        plot_and_fill(self.speed())
        plt.savefig(HOME / "graphs" / "speed_over_time.png")

        plt.close()
        if animate:
            self.animate(HOME / "graphs" / "test.gif")


def plot_and_fill(iterations: List[List[float]]) -> None:
    t = np.linspace(0, len(iterations), len(iterations))
    top = []
    bottom = []
    avg = []
    for iteration in iterations:
        top.append(max(iteration))
        bottom.append(min(iteration))
        avg.append(np.average(iteration))
    plt.fill_between(t, top, bottom, color='skyblue', alpha=0.4)
    plt.plot(t, avg)
    plt.yscale("log")


pso = PSO(HOME / "data" / "PSO.json")
pso.animate(HOME / "graphs" / "test.gif")
pso.overview(False)

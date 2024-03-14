# from matplotlib.animation import FuncAnimation  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
# import numpy as np  # type: ignore
import pathlib
import json
import shutil
from typing import List, Dict, Any, Union
import hashlib
from datetime import datetime
import imageio  # type: ignore
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
            assert len(particle.pos) == 2
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

    def animate(self, destination_path: pathlib.Path) -> None:
        # Create a temporary folder for frames output
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        hash_object = hashlib.sha256(current_time.encode())
        hex_dig = hash_object.hexdigest()
        temp_out = HOME / "out" / "frames" / hex_dig
        temp_out.mkdir(parents=True, exist_ok=True)

        # Find lim
        lim = [float('inf'), float('-inf')]
        for iteration in self.iterations:
            for particle in iteration.particles:
                lim[0] = min(lim[0], particle.pos[0])
                lim[1] = max(lim[1], particle.pos[0])
                lim[0] = min(lim[0], particle.pos[1])
                lim[1] = max(lim[1], particle.pos[1])

        # Generate frames
        frames = []
        for idx, iteration in enumerate(tqdm(self.iterations)):
            iteration.visualize(f"{idx}", lim, lim, temp_out / f"{idx}.png")
            frames.append(imageio.imread(temp_out / f"{idx}.png"))

        # Save gif
        imageio.mimsave(destination_path, frames)

        # Cleanup
        shutil.rmtree(temp_out)


pso = PSO(HOME / "data" / "PSO.json")
pso.animate(HOME / "graphs" / "test.gif")

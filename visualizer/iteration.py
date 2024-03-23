import matplotlib.pyplot as plt  # type: ignore
from particle import Particle
from typing import List
import pathlib


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

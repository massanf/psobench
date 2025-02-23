from particle import Particle
from typing import List


class Iteration:
    def __init__(self, global_best_fitness: float,
                 particles: List[Particle]) -> None:
        self.particles = particles
        self.global_best_fitness = global_best_fitness

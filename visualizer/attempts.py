import pathlib
from pso import PSO
import numpy as np

class Attempts:
    def __init__(self, path: pathlib.Path):
        self.data = []
        for attempt in path.glob("*"):
            self.data.append(PSO(attempt))

    def average_global_best_progress(self):
        data = []
        for pso in self.data:
            data.append(pso.global_best_fitness_progress())
        # return data[0]
        return [sum(group) / len(group) for group in zip(*data)]

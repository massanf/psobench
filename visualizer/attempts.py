import pathlib
from typing import List
from pso import PSO
import numpy as np

class Attempts:
    def __init__(self, path: pathlib.Path):
        self.data = []
        for attempt in path.glob("*"):
            self.data.append(PSO(attempt))

    def average_global_best_progress(self) -> List[float]:
        data = []
        for pso in self.data:
            data.append(pso.global_best_fitness_progress())
        return [sum(group) / len(group) for group in zip(*data)]

    def global_best_progress(self) -> List[List[float]]:
        data = []
        for pso in self.data:
            data.append(pso.global_best_fitness_progress())
        data = list(map(list, zip(*data)))
        return data

    def entropy(self) -> List[float]:
        data = []
        for pso in self.data:
            pso.load_full()
            data.append(pso.entropy())
            pso.unload()
        return [sum(group) / len(group) for group in zip(*data)]
    
    def get_all_final_results(self) -> List[float]:
        data = []
        for pso in self.data:
            data.append(pso.global_best_fitness_progress()[-1])
        return data

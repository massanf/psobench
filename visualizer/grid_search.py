import matplotlib.pyplot as plt  # type: ignore
import os
import matplotlib
from matplotlib.animation import FuncAnimation  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, Any
import json
from pso import PSO
from tqdm import tqdm  # type: ignore
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator
import matplotlib.colors as colors  # type: ignore
from typing import List, Tuple
from numpy._typing import _64Bit
import pathlib


class GridSearch:
    def __init__(self, data_path: pathlib.Path) -> None:
        if not data_path.exists():
            raise ValueError(f"Path {data_path} does not exist.")

        if not (data_path / "grid_search_config.json").exists():
            raise ValueError("File grid_search_config.json does"
                             + f"not exist in {data_path}.")

        with open(data_path / "grid_search_config.json", 'r') as file:
            self.search_config: Dict[str, Any] = json.load(file)

        self.name = self.search_config["problem"]["name"]

        keys = list(self.search_config["grid_search"].keys())
        assert len(keys) == 2

        self.arg1 = keys[0]
        self.arg2 = keys[1]

        data = []
        for exp_path in (data_path).glob("*"):
            if not os.path.isdir(exp_path):
                continue
            attempts = len([name for name in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, name))])
            for attempt_path in (exp_path).glob("*"):
                pso = PSO(attempt_path)
                if self.arg1 == "dim":
                    x = pso.config["problem"]["dim"]
                else:
                    x = pso.config["method"]["parameters"][self.arg1]
                if self.arg2 == "dim":
                    y = pso.config["problem"]["dim"]
                y = pso.config["method"]["parameters"][self.arg2]
                data.append((x, y, pso))

        self.x_values = self.search_config["grid_search"][self.arg1]
        self.y_values = self.search_config["grid_search"][self.arg2]

        self.psos = [[[PSO(attempt_path) for _ in range(attempts)] for _ in range(len(self.x_values))]
                     for _ in range(len(self.y_values))]

        self.max_fitness = float("-inf")
        self.min_fitness = float("inf")
        for x, y, z in data:
            ix = np.searchsorted(self.x_values, x)
            iy = np.searchsorted(self.y_values, y)
            self.psos[iy][ix].append(z)
            self.max_fitness = max(self.max_fitness,
                                   np.max(z.global_best_fitness_progress()))
            if self.min_fitness > np.min(z.global_best_fitness_progress()):
                self.best_pso = z
                self.min_fitness = np.min(z.global_best_fitness_progress())

    def best_global_progress(self) -> List[float]:
        best = float("inf")
        for row in self.psos:
            for psos in row:
                final_values = []
                for attempt in psos:
                    final_values.append(attempt.global_best_fitness(-1))
                avg = np.average(final_values)
                if avg < best:
                    best = float(avg)
                    best_psos = psos
        data = []
        best_psos = self.psos[0][0]
        for pso in best_psos:
            data.append(pso.global_best_fitness_progress())
        return [sum(group) / len(group) for group in zip(*data)]
 

    def plot_best_global_progress(self, path: pathlib.Path) -> None:
        plt.close()
        plt.cla()
        plt.yscale('log')
        plt.title(f"{self.name}")
        plt.plot(self.best_global_progress())
        print(f"Saving: {path / 'best_global_progress.png'}")
        plt.savefig(path / "best_global_progress.png", bbox_inches='tight')

    def create_image(self, use_all_range: bool, frame: int = -1) -> Tuple[np.ndarray[Any, np.dtype[np.floating[_64Bit]]], Any]:
        image = np.zeros((len(self.y_values), len(self.x_values)))
        for (i, row) in enumerate(self.psos):
            for (j, z) in enumerate(row):
                fitnesses = []
                for attempt in z:
                    fitnesses.append(attempt.global_best_fitness(frame))
                image[i, j] = np.average(fitnesses)
        if use_all_range:
            norm = colors.LogNorm(vmin=self.min_fitness, vmax=self.max_fitness)
        else:
            norm = colors.LogNorm()
        return (image, norm)

    def draw_heatmap(
            self,
            path: pathlib.Path,
            log_1: bool,
            log_2: bool,
            frame: int = -1,
        ) -> None:
        plt.close()
        image, norm = self.create_image(False, frame)
        
        # Doing finicky stuff because the axes have to be log.
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=100)

        ax.set_title(f"{self.name}")
        ax.axis('off')
        ax.imshow(image, norm=norm,
                  cmap='viridis', origin='lower',
                  extent=[self.x_values[0],
                          self.x_values[-1],
                          self.y_values[0],
                          self.y_values[-1]],
                  aspect='auto')
        newax = fig.add_axes(ax.get_position())
        if log_1:
            newax.set_yscale('log')
        if log_2:
            newax.set_xscale('log')
        newax.set_xlim((self.x_values[0], self.x_values[-1]))
        newax.set_ylim((self.y_values[0], self.y_values[-1]))
        newax.patch.set_facecolor('none')
        newax.set_aspect(ax.get_aspect())
        newax.set_position(ax.get_position())
        newax.set_xlabel(self.arg1)
        newax.set_ylabel(self.arg2)

        plt.savefig(path / "grid_search.png", bbox_inches='tight')
        plt.close()

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
import numpy as np  # type: ignore
from typing import Dict, Any
import json
from pso import PSO
from tqdm import tqdm  # type: ignore
import matplotlib.colors as colors  # type: ignore
from typing import List, Tuple
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

        self.psos = [[PSO(attempt_path) for _ in range(len(self.x_values))]
                     for _ in range(len(self.y_values))]

        self.max_fitness = float("-inf")
        self.min_fitness = float("inf")
        for x, y, z in data:
            ix = np.searchsorted(self.x_values, x)
            iy = np.searchsorted(self.y_values, y)
            self.psos[iy][ix] = z
            self.max_fitness = max(self.max_fitness,
                                   np.max(z.global_best_fitness_progress()))
            if self.min_fitness > np.min(z.global_best_fitness_progress()):
                self.best_pso = z
                self.min_fitness = np.min(z.global_best_fitness_progress())

    def plot_for_animate(
            self,
            frame: int = -1
            ) -> None:
        plt.cla()

        image = np.zeros((len(self.y_values), len(self.x_values)))
        for (i, row) in enumerate(self.psos):
            for (j, z) in enumerate(row):
                image[i, j] = z.global_best_fitness(frame)

        norm = colors.LogNorm(vmin=self.min_fitness, vmax=self.max_fitness)

        plt.imshow(image, cmap='viridis', origin='lower',
                   norm=norm,
                   extent=[self.x_values[0],
                           self.x_values[-1],
                           self.y_values[0],
                           self.y_values[-1]],
                   aspect="auto")
        plt.title(f"{self.name} iter: {frame}")
        plt.xlabel(self.arg1)
        plt.ylabel(self.arg2)
        self.progressbar.update(1)

    def create_image(self, frame: int = -1) -> Tuple[List[List[float]], Any]:
        image = np.zeros((len(self.y_values), len(self.x_values)))
        for (i, row) in enumerate(self.psos):
            for (j, z) in enumerate(row):
                image[i, j] = z.global_best_fitness(frame)

        norm = colors.LogNorm(vmin=self.min_fitness, vmax=self.max_fitness)
        return (image, norm)

    def plot(
            self,
            path: pathlib.Path,
            frame: int = -1,
        ) -> None:
        plt.close()
        image, norm = self.create_image(frame)
        
        font = {'family' : 'DejaVu Sans',
                'size'   : 30}
        plt.rc('font', **font)

        # Doing finicky stuff because the axes have to be log.
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)

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
        newax.set_yscale('log')
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

    def best_pso(self) -> PSO:
        return self.best_pso

    def animate(self, out_path: pathlib.Path) -> None:
        plt.rcParams.update({'figure.max_open_warning': 0})
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        skip_frames = 10
        self.progressbar = tqdm(total=int(
            (len(self.psos[0][0].global_best_fitness_progress())
             / skip_frames) + 2))
        self.progressbar.set_description(self.name)

        self.plot(0)
        plt.colorbar()

        frames = range(0, 1000, skip_frames)
        ani = FuncAnimation(fig, self.plot_for_animate, frames=frames)
        ani.save(out_path /
                 f"grid_progress_{self.name}_{self.arg1}_vs_{self.arg2}.mp4",
                 fps=10)
        plt.close()

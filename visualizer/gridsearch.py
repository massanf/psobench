import matplotlib.pyplot as plt  # type: ignore
import utils
from matplotlib.animation import FuncAnimation  # type: ignore
import numpy as np  # type: ignore
from pso import PSO
from tqdm import tqdm  # type: ignore
import matplotlib.colors as colors  # type: ignore
import pathlib


class GridSearch:
    def __init__(self, data_path: pathlib.Path, arg1: str, arg2: str) -> None:
        self.arg1 = arg1
        self.arg2 = arg2

        if not data_path.exists():
            raise ValueError(f"Path {data_path} does not exist.")

        if arg2 == "dim":
            raise ValueError("`dim` should be arg1.")

        data = []
        for exp_path in (data_path).glob("*"):
            for attempt_path in (exp_path).glob("*"):
                pso = PSO(attempt_path)
                if arg1 == "dim":
                    x = pso.config["problem"]["dim"]
                else:
                    x = pso.config["method"]["parameters"][arg1]
                y = pso.config["method"]["parameters"][arg2]
                data.append((x, y, pso))

        self.x_r_and_s = utils.get_range_and_step([i[0] for i in data])
        self.y_r_and_s = utils.get_range_and_step([i[1] for i in data])

        self.psos = [[PSO(attempt_path) for _ in range(self.x_r_and_s[3])]
                     for _ in range(self.y_r_and_s[3])]

        for x, y, z in data:
            ix = int(round((x - self.x_r_and_s[0]) / self.x_r_and_s[2], 3))
            iy = int(round((y - self.y_r_and_s[0]) / self.y_r_and_s[2], 3))
            self.psos[iy][ix] = z

    def plot(
            self,
            frame: int = -1
            ) -> None:
        plt.cla()

        image = np.zeros((self.y_r_and_s[3], self.x_r_and_s[3]))
        for (i, row) in enumerate(self.psos):
            for (j, z) in enumerate(row):
                image[i, j] = z.global_best_fitness(frame)

        # print(np.min(image), np.max(image))
        norm = colors.LogNorm(vmin=0.00000001, vmax=1000)

        plt.imshow(image, cmap='viridis', origin='lower',
                   norm=norm,
                   extent=[self.x_r_and_s[0] - self.x_r_and_s[2] / 2,
                           self.x_r_and_s[1] + self.x_r_and_s[2] / 2,
                           self.y_r_and_s[0] - self.y_r_and_s[2] / 2,
                           self.y_r_and_s[1] + self.y_r_and_s[2] / 2],
                   aspect="auto")
        plt.title(f"Iteration: {frame}")
        plt.xlabel(self.arg1)
        plt.ylabel(self.arg2)
        self.progressbar.update(1)

    def animate(self, out_path: pathlib.Path) -> None:
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        skip_frames = 10
        self.progressbar = tqdm(total=int(
            (len(self.psos[0][0].global_best_fitness_progress())
             / skip_frames) + 2))

        self.plot(0)
        plt.colorbar()

        frames = range(0, 1000, skip_frames)
        ani = FuncAnimation(fig, self.plot, frames=frames)
        ani.save(out_path, fps=10)

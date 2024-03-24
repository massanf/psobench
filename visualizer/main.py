import matplotlib.pyplot as plt  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
import numpy as np  # type: ignore
import pathlib
import utils
from pso import PSO
from tqdm import tqdm  # type: ignore
import matplotlib.colors as colors  # type: ignore

HOME = pathlib.Path(__file__).parent.parent


class GridSearch:
    def __init__(self, data_path: pathlib.Path, arg1: str, arg2: str) -> None:
        self.arg1 = arg1
        self.arg2 = arg2

        if not data_path.exists():
            raise ValueError(f"Path {data_path} does not exist.")

        data = []
        for exp_path in (data_path).glob("*"):
            for attempt_path in (exp_path).glob("*"):
                pso = PSO(attempt_path)
                x = pso.config["method"]["parameters"][arg1]
                y = pso.config["method"]["parameters"][arg2]
                data.append((x, y, pso))

        self.x_r_and_s = utils.get_range_and_step([i[0] for i in data])
        self.y_r_and_s = utils.get_range_and_step([i[1] for i in data])

        self.psos = [[PSO(attempt_path) for _ in range(self.y_r_and_s[3])]
                     for _ in range(self.x_r_and_s[3])]
        for x, y, z in data:
            ix = int(round((x - self.x_r_and_s[0]) / self.x_r_and_s[2], 3))
            iy = int(round((y - self.y_r_and_s[0]) / self.y_r_and_s[2], 3))
            self.psos[iy][ix] = z

    def plot(
            self,
            frame: int = -1
            ) -> None:
        plt.cla()

        image = np.zeros((self.x_r_and_s[3], self.y_r_and_s[3]))
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
                           self.y_r_and_s[1] + self.y_r_and_s[2] / 2])
        plt.title(f"Iteration: {frame}")
        self.progressbar.update(1)

    def animate(self, out_path: pathlib.Path) -> None:
        skip_frames = 10
        self.progressbar = tqdm(total=int(
            (len(self.psos[0][0].global_best_fitness_progress())
             / skip_frames) + 2))

        self.plot(0)
        plt.colorbar()

        fig, ax = plt.subplots()
        frames = range(0, 1000, skip_frames)
        ani = FuncAnimation(fig, self.plot, frames=frames)
        ani.save(out_path, fps=10)


grid = GridSearch(HOME / "data" / "base_pso_test4" / "Sphere",
                  "phi_g", "phi_p")
grid.animate(HOME / "graphs" / "progress.gif")
# pso = PSO(HOME / "data" / "base_pso_test3" / "g_1_d_2")
# pso.load_full()
# pso.animate(HOME / "graphs" / "animation_g_1.gif", 10)

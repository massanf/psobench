from matplotlib.animation import FuncAnimation  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pathlib
import json
from typing import List, Dict, Any, Union, Tuple
from tqdm import tqdm  # type: ignore
import matplotlib.colors as colors  # type: ignore

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
    def __init__(self, experiment_path: pathlib.Path) -> None:
        self.experiment_path = experiment_path
        self.full_loaded = False

        # Config
        with open(experiment_path / "config.json", 'r') as file:
            config: Dict[str, Any] = json.load(file)
        self.config = config

        # Config
        with open(experiment_path / "summary.json", 'r') as file:
            summary: Dict[str, Any] = json.load(file)
        self.summary = summary

    def load_full(self) -> None:
        # Data
        if not (self.experiment_path / "data.json").exists():
            raise ValueError("Full data not exported.")
        self.fully_loaded = True
        with open(self.experiment_path / "data.json", 'r') as file:
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
        result = self.summary["global_best_fitness"]
        if isinstance(result, list) and all(isinstance(i, float)
                                            for i in result):
            return result
        raise ValueError("Incorrect dictionary type.")

    def global_best_fitness(self, idx: int) -> float:
        progress = self.summary["global_best_fitness"]
        if idx != -1 and not (idx >= 0 and idx < len(progress)):
            raise ValueError("Index is out of bounds.")
        result = progress[idx]
        if isinstance(result, float):
            return result
        raise ValueError("Unexpected `global_best_fitness` value")

    def fitness(self) -> List[List[float]]:
        assert self.fully_loaded
        result: List[List[float]] = []
        for iteration in self.iterations:
            iter_result = []
            for particle in iteration.particles:
                iter_result.append(particle.fitness)
            result.append(iter_result)
        return result

    def speed(self) -> List[List[float]]:
        assert self.fully_loaded
        result: List[List[float]] = []
        for iteration in self.iterations:
            iter_result = []
            for particle in iteration.particles:
                iter_result.append(np.sqrt(np.sum(np.array(particle.vel)
                                                  ** 2)))
            result.append(iter_result)
        return result

    def update_plot_animate(self, frame: int) -> None:
        assert self.fully_loaded
        plt.cla()
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
        assert self.fully_loaded
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
        assert self.fully_loaded
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


def get_range_and_step(array: List[float]) -> Tuple[float, float, float, int]:
    count = len(np.unique(array))
    mx = np.max(np.array(array))
    mn = np.min(np.array(array))
    return (mn, mx, (mx - mn) / (count - 1), count)


def plot_grid_search(
        arg1: str,
        arg2: str,
        data_path: pathlib.Path,
        out_path: pathlib.Path,
        frame: int = -1
        ) -> None:
    print(frame)
    plt.cla()
    data = []
    X = []
    Y = []
    if not data_path.exists():
        raise ValueError(f"Path {data_path} does not exist.")
    for exp_path in (data_path).glob("*"):
        global_bests = []
        for attempt_path in (exp_path).glob("*"):
            pso = PSO(attempt_path)
            global_bests.append(pso.global_best_fitness(frame))
            x = pso.config["method"]["parameters"][arg1]
            y = pso.config["method"]["parameters"][arg2]
        X.append(x)
        Y.append(y)
        data.append((x, y, np.average(global_bests)))

    x_range_and_step = get_range_and_step(X)
    y_range_and_step = get_range_and_step(Y)

    Z = np.zeros(np.array((x_range_and_step[3], y_range_and_step[3])))
    for x, y, z in data:
        ix = int(round((x - x_range_and_step[0]) / x_range_and_step[2], 3))
        iy = int(round((y - y_range_and_step[0]) / y_range_and_step[2], 3))
        Z[iy, ix] = z

    norm = colors.LogNorm(vmin=100000000, vmax=100000000000)

    # plt.figure(figsize=(6.0, 6.0))
    plt.imshow(Z, cmap='viridis', origin='lower',
               norm=norm,
               extent=[x_range_and_step[0] - x_range_and_step[2] / 2,
                       x_range_and_step[1] + x_range_and_step[2] / 2,
                       y_range_and_step[0] - y_range_and_step[2] / 2,
                       y_range_and_step[1] + y_range_and_step[2] / 2])
    plt.xlabel(arg1)
    plt.ylabel(arg2)
    plt.title(f"Iteration: {frame}")
    # plt.savefig(out_path)


fig, ax = plt.subplots()
skip_frames = 20
frames = range(0, 1000, skip_frames)

plot_grid_search("phi_g", "phi_p",
                 HOME / "data" / "base_pso_test" / "CEC2017_F1",
                 HOME / "graphs" / "grid_search.png", 0)
plt.colorbar()

ani = FuncAnimation(fig, lambda x:  plot_grid_search("phi_g", "phi_p",
                    HOME / "data" / "base_pso_test" / "CEC2017_F1",
                    HOME / "graphs" / "grid_search.png", x), frames=frames)
ani.save(HOME / "graphs" / "grid_search.gif", fps=10)

# plot_grid_search(HOME / "data" / "PSO_grid")
# pso = PSO(HOME / "data" / "PSO_grid" / "p0_g0")
# pso.animate(HOME / "graphs" / "test.gif")
# pso.overview(False)

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
import numpy as np  # type: ignore
import pathlib
import utils
from pso import PSO
import matplotlib.colors as colors  # type: ignore

HOME = pathlib.Path(__file__).parent.parent


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

    x_range_and_step = utils.get_range_and_step(X)
    y_range_and_step = utils.get_range_and_step(Y)

    Z = np.zeros(np.array((x_range_and_step[3], y_range_and_step[3])))
    for x, y, z in data:
        ix = int(round((x - x_range_and_step[0]) / x_range_and_step[2], 3))
        iy = int(round((y - y_range_and_step[0]) / y_range_and_step[2], 3))
        Z[iy, ix] = z

    print(np.min(Z), np.max(Z))
    norm = colors.LogNorm(vmin=10000, vmax=10000000)

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
skip_frames = 100
frames = range(0, 1000, skip_frames)

plot_grid_search("phi_g", "phi_p",
                 HOME / "data" / "base_pso_test" / "CEC2017_F3",
                 HOME / "graphs" / "grid_search.png", 0)
plt.colorbar()

ani = FuncAnimation(fig, lambda x:  plot_grid_search("phi_g", "phi_p",
                    HOME / "data" / "base_pso_test" / "CEC2017_F3",
                    HOME / "graphs" / "grid_search.png", x), frames=frames)
ani.save(HOME / "graphs" / "grid_search.gif", fps=10)

# plot_grid_search(HOME / "data" / "PSO_grid")
# pso = PSO(HOME / "data" / "PSO_grid" / "p0_g0")
# pso.animate(HOME / "graphs" / "test.gif")
# pso.overview(False)

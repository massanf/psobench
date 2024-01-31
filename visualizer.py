import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import glob


def plot_progress(ax, path: pathlib.Path):
    df = pd.read_csv(path)
    ax.plot(df["global_best_pos"], label=path.stem)


def plot_all_progresses(path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    for file in glob.glob("data/*.csv"):
        plot_progress(ax, pathlib.Path(file))
    ax.set_xlim(0)
    ax.set_yscale("log")
    ax.legend()
    plt.savefig(path)
    print(f"Graph saved to: {path}")


plot_all_progresses(pathlib.Path("graphs/progresses.png"))

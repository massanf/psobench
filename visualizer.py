import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import glob
import json


def get_global_fitness(data):
    return [x["global_best_fitness"] for x in data["history"]]


def get_average_vel(data):
    avg_vels = []
    for iter in data["history"]:
        avg_vels.append(np.average([x["vel"] for x in iter["particles"]]))
    return avg_vels


def get_average_fitness(data):
    avg_vels = []
    for iter in data["history"]:
        avg_vels.append(np.average([x["fitness"] for x in iter["particles"]]))
    return avg_vels


def plot_all(save_path: pathlib.Path, extractor, title):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    for file_path in glob.glob("data/*.json"):
        with open(file_path, 'r') as file:
            data = json.load(file)
            ax.plot(extractor(data), label=data["setting"]["type"])
    ax.set_xlim(0)
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(title)
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")


plot_all(pathlib.Path("graphs/global_fitness.png"),
         get_global_fitness, "Global Fitness")
plot_all(pathlib.Path("graphs/average_velocity.png"),
         get_average_vel, "Average Velocity")
plot_all(pathlib.Path("graphs/average_fitness.png"),
         get_average_fitness, "Average Fitness")

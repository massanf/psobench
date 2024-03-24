from typing import List
import matplotlib.pyplot as plt  # type: ignore
import numpy as np


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

import pathlib
from pso import PSO
from attempts import Attempts
from tqdm import tqdm
import utils
from matplotlib.ticker import LogLocator, LogFormatterMathtext

class Tests:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.data = {}

        for pso_type in self.path.glob("*"):
            self.data[pso_type.name] = {}
            for function in pso_type.glob("*"):
                self.data[pso_type.name][function.name] = Attempts(function)

    def plot_best_global_progress(self, axs, pso_type: str):
        axs = axs.flatten()
        for i, function in enumerate(tqdm(sorted(self.data[pso_type]))):
            attempts = self.data[pso_type][function]
            axs[i].yaxis.set_major_locator(LogLocator(base=10.0))
            axs[i].yaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=True))
            utils.plot_and_fill(axs[i], attempts.global_best_progress(), label=pso_type)
            axs[i].set_title(function)
            if i == 0:
                axs[i].legend()
        return axs

    def plot_all(self, axs):
        for pso_type in sorted(self.data):
            axs = self.plot_best_global_progress(axs, pso_type)
        return axs

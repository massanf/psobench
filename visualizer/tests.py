import pathlib
import questionary
import numpy as np
from attempts import Attempts
from tqdm import tqdm  # type: ignore
import utils
from typing import List, Dict
from matplotlib.ticker import LogLocator, LogFormatterMathtext


class Tests:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.data: Dict[str, Dict[str, Attempts]] = {}

        for pso_type in self.path.glob("*"):
            self.data[pso_type.name] = {}
            for function in pso_type.glob("*"):
                self.data[pso_type.name][function.name] = Attempts(function)

    def plot_best_global_progress(
        self, axs: np.ndarray, pso_type: str
    ) -> np.ndarray:
        axs = axs.flatten()
        for i, function in enumerate(tqdm(sorted(self.data[pso_type]))):
            attempts = self.data[pso_type][function]
            axs[i].yaxis.set_major_locator(LogLocator(base=10.0))
            axs[i].yaxis.set_major_formatter(
                LogFormatterMathtext(base=10.0, labelOnlyBase=True)
            )
            utils.plot_and_fill(
                axs[i], attempts.global_best_progress(), label=pso_type
            )
            axs[i].set_title(function)
            if i == 0:
                axs[i].legend()
        return axs

    def plot_entropy(self, axs: np.ndarray, pso_type: str) -> np.ndarray:
        axs = axs.flatten()
        for i, function in enumerate(tqdm(sorted(self.data[pso_type]))):
            attempts = self.data[pso_type][function]
            axs[i].plot(attempts.entropy(), label=pso_type)
            axs[i].set_title(function)
            axs[i].set_yscale("linear")
            if i == 0:
                axs[i].legend()
        return axs

    def plot_all(self, axs: np.ndarray) -> np.ndarray:
        psos = questionary.checkbox(
            "Select tests to plot progress:", choices=list(self.data.keys())
        ).ask()
        for pso in psos:
            axs = self.plot_best_global_progress(axs, pso)
        return axs

    def get_final_result(self, pso_type: str) -> Dict[str, List[float]]:
        result = {}
        for i, function in enumerate(sorted(self.data[pso_type])):
            attempts = self.data[pso_type][function]
            result[function] = attempts.get_all_final_results()
        return result

    def get_final_results(self) -> Dict[str, Dict[str, List[float]]]:
        result: Dict[str, Dict[str, List[float]]] = {}

        psos = questionary.checkbox(
            "Select tests to plot entropy:", choices=list(self.data.keys())
        ).ask()
        for pso_type in sorted(psos):
            pso_result = self.get_final_result(pso_type)
            for fn in pso_result:
                if fn not in result:
                    result[fn] = {}
                result[fn][pso_type] = pso_result[fn]
        return result

    def plot_final_results(self, axs: np.ndarray) -> np.ndarray:
        axs = axs.flatten()
        result = self.get_final_results()
        for i, function in enumerate(result):
            axs[i].set_title(function)
            data = result[function].values()
            err_top = []
            avg = []
            err_btm = []
            for datum in data:
                err_top.append(
                    np.quantile(datum, 0.75) - np.quantile(datum, 0.5)
                )
                err_btm.append(
                    np.quantile(datum, 0.5) - np.quantile(datum, 0.25)
                )
                avg.append(np.average(datum) - i * 100)
            axs[i].bar(
                utils.shorter_names(list(result[function].keys())),
                avg,
                yerr=[err_top, err_btm],
                capsize=4,
            )
            axs[i].set_ylim(0)
        return axs

    def plot_all_entropy(self, axs: np.ndarray) -> np.ndarray:
        for pso_type in sorted(self.data):
            if pso_type[0] == "_":
                continue
        psos = questionary.checkbox(
            "Select tests to plot entropy:", choices=list(self.data.keys())
        ).ask()
        for pso in psos:
            axs = self.plot_entropy(axs, pso)
        return axs

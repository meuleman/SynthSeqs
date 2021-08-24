import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from utils.constants import DHS_COLORS


def _performance_csvs(tuning_data_dir, component, usecols=None):
    performance_dir = tuning_data_dir + f"{component}/" + "performance/"
    for filename in os.listdir(performance_dir):
        yield pd.read_csv(performance_dir + filename, usecols=usecols)


def plot_skew_vs_iterations(tuning_data_dir, figure_dir):
    for component in range(1, 17):
        data = pd.concat(
            _performance_csvs(tuning_data_dir, component, usecols=["skew", "iteration"])
        )
        mean_skews = data.groupby("iteration")["skew"].mean()

        plt.plot(mean_skews.index, mean_skews.values, c=DHS_COLORS[component-1])

    plt.savefig(figure_dir + "skew_vs_iteration.pdf")



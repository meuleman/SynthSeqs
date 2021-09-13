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
            _performance_csvs(
                tuning_data_dir,
                component,
                usecols=["iteration", "a_count", "c_count", "g_count", "t_count"],
            )
        )
        at_skews = np.abs(np.log2(
            data.groupby("iteration")["a_count"].sum() / data.groupby("iteration")["t_count"].sum()
        ))
        cg_skews = np.abs(np.log2(
            data.groupby("iteration")["c_count"].sum() / data.groupby("iteration")["g_count"].sum()
        ))
        skews = (at_skews + cg_skews) / 2

        plt.plot(skews.index, skews.values, c=DHS_COLORS[component-1])

    plt.savefig(figure_dir + "skew_vs_iteration.pdf")


plot_skew_vs_iterations(
    "/home/pbromley/synth_seqs_output/tuning/test/",
    "/home/pbromley/synth_seqs_output/figures/",
)

import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from utils.constants import DHS_COLORS


def _performance_csvs(tuning_data_dir, component, usecols=None):
    performance_dir = tuning_data_dir + f"{component}/" + "performance/"
    for filename in os.listdir(performance_dir):
        yield pd.read_csv(performance_dir + filename, usecols=usecols)


def _calculate_skews(df):
    at_skews = np.abs(np.log2(
        df.groupby("iteration")["a_count"].sum() / df.groupby("iteration")["t_count"].sum()
    ))
    cg_skews = np.abs(np.log2(
        df.groupby("iteration")["c_count"].sum() / df.groupby("iteration")["g_count"].sum()
    ))
    return (at_skews + cg_skews) / 2


def plot_skew_vs_iterations(tuning_data_dir, figure_dir):
    for component in range(1, 17):
        data = pd.concat(
            _performance_csvs(
                tuning_data_dir,
                component,
                usecols=["iteration", "a_count", "c_count", "g_count", "t_count"],
            )
        )
        skews = _calculate_skews(data)

        plt.plot(skews.index, skews.values, c=DHS_COLORS[component-1])

    plt.savefig(figure_dir + "skew_vs_iteration.pdf")


def plot_skew_vs_iterations_by_batch_size(tuning_dir, batch_dirs, figure_dir):
    # batch_dirs should be a list of the names of the dirs with completed tuning runs
    for batch_dir in batch_dirs:
        tuning_data_dir = tuning_dir + batch_dir
        for component in [9]: #range(1, 17):
            data = pd.concat(
                _performance_csvs(
                    tuning_data_dir,
                    component,
                    usecols=["iteration", "a_count", "c_count", "g_count", "t_count"],
                )
            )
            skews = _calculate_skews(data)

            plt.plot(skews.index, skews.values, label=batch_dir)

    plt.savefig(figure_dir + "skew_vs_iteration_per_batch_component_9.pdf")


# plot_skew_vs_iterations(
#     "/home/pbromley/synth_seqs_output/tuning/test/",
#     "/home/pbromley/synth_seqs_output/figures/",
# )
plot_skew_vs_iterations_by_batch_size(
    "/home/pbromley/synth_seqs_output/tuning/",
    ["100/", "500/", "1000/", "2000/"],
    "/home/pbromley/synth_seqs_output/figures/",
)

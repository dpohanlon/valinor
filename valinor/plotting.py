import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from tqdm import tqdm

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

import numpy as np

fontsizes = [18,16,14]
colorblindfr = {"main" : ['#56b3e9','#e0d316','#0072b2', '#e69d00','#cc79a7'] , "additional" : ['#EC681E', '#009e74','#000000']} 
colors_palette = colorblindfr["main"]


def plotLossCurve(loss, log=True, name=None):
    plt.plot(np.log(loss))
    plt.ylabel("ELBO")
    plt.xlabel("Steps")
    plt.savefig("valinor_loss.pdf" if name == None else f"valinor_loss_{name}.pdf")
    plt.savefig("valinor_loss.svg" if name == None else f"valinor_loss_{name}.svg")
    plt.clf()


def plotDiagPlots(svi_result, name=None):
    plotLossCurve(svi_result.losses, name=name)

def plot_hist(
    data_dict, xlabel, figsize=(6, 6), nbins=50, figure=None, loc="upper right"
):
    if figure is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = figure[0]
        ax = figure[1]
    data_min = min([i.min() for i in data_dict.values()])
    data_max = max([np.quantile(i, 0.999) for i in data_dict.values()])
    bins = np.linspace(data_min, data_max, nbins)
    for label, data in data_dict.items():
        ax.hist(data, histtype="step", bins=bins, density=True, label=label)
    ax.set_xlabel(xlabel, fontsize=fontsizes[1])
    ax.set_ylabel("Density", fontsize=fontsizes[1])
    ax.tick_params(axis="both", labelsize=fontsizes[2])
    ax.legend(loc=loc)
    fig.tight_layout()

    return (fig, ax)

def produce_modelfit_plot(scoreData_combo, scoreData_single, outputfolder):
    model_val = scoreData_combo["samples"].values
    data_val = scoreData_combo["value"].values
    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig(outputfolder+"modelperformance_combo.svg", bbox_inches="tight")

    model_val = scoreData_single["samples_s"].values
    data_val = scoreData_single["value"].values
    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig(outputfolder+"modelperformance_single.svg", bbox_inches="tight")

    model_val = scoreData_combo["init_count_mean"].values
    data_val = scoreData_combo["plasmid"].values
    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig(
        outputfolder+"modelperformance_combo_plasmid.svg", bbox_inches="tight"
    )

    model_val = scoreData_single["init_count_s_mean"].values
    data_val = scoreData_single["plasmid"].values
    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig(
        outputfolder+"modelperformance_single_plasmid.svg", bbox_inches="tight"
    )
    plt.close(fig)
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

fontsizes = [18, 16, 14]
colorblindfr = {
    "main": ["#56b3e9", "#e0d316", "#0072b2", "#e69d00", "#cc79a7"],
    "additional": ["#EC681E", "#009e74", "#000000"],
}
colors_palette = colorblindfr["main"]

naming_cols = {
    "GuidePair": "Guide pair",
    "gene1": "Singleton gene 1",
    "gene2": "Singleton gene 2",
    "guide1": "Singleton guide 1",
    "guide2": "Singleton guide 2",
    "cell_line": "Cell line",
    "deltaLFC": "dLFC",
    "dev_prior_guide_eff_1_mean": "Deviation from hyper distribution of guide 1",
    "dev_prior_guide_eff_2_mean": "Deviation from hyper distribution of guide 2",
    "gene1": "Gene 1",
    "gene2": "Gene 2",
    "genePair": "Gene pair",
    "guide1": "Guide 1",
    "guide2": "Guide 2",
    "guide_eff_1_mean": "Efficiency guide 1",
    "guide_eff_2_mean": "Efficiency guide 2",
    "guide_eff_mean_1_mean": "Mean of hyper distribution",
    "guide_eff_std_1_mean": "Standard deviation of hyper distribution",
    "gene_ko_growth_1_mean": "Estimated gene 1 effect",
    "gene_ko_growth_12_mean": "Estimated combination effect",
    "gene_ko_growth_12_std": "Uncertainty",
    "gene_ko_growth_1_std": "Uncertainty",
    "gene_ko_growth_2_mean": "Estimated gene 2 effect",
    "gene_ko_growth_2_std": "Uncertainty",
    "ko_growth_s_mean_s_1": "Estimated singleton effect",
    "ko_growth_s_mean_s_2": "Estimated singleton effect",
    "ko_growth_s_std_s_1": "Uncertainty",
    "ko_growth_s_std_s_2": "Uncertainty",
    "lfc": "Combination LFC",
    "lfc_s_1": "Singleton LFC - gene 1",
    "lfc_s_2": "Singleton LFC - gene 2",
    "mv_mean": "Estimated overdispersion",
    "mv_std": "Uncertainty",
    "overdispersion": "Overdispersion",
    "overdispersion_s_1": "Overdispersion",
    "overdispersion_s_2": "Overdispersion",
    "plasmid": "Plasmid counts",
    "rank_valinor_score": "Rank of Valinor Score",
    "rank_valinor_score_s_s_1": "Rank of Valinor Singleton Score",
    "rank_valinor_score_s_s_2": "Rank of Valinor Singleton Score",
    "valinor_score": "Valinor Score",
    "valinor_score_s_s_1": "Valinor Singleton Score - gene 1",
    "valinor_score_s_s_2": "Valinor Singleton Score - gene 2",
    "value": "End of experiment counts",
}


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
    fig.savefig(outputfolder + "modelperformance_combo.svg", bbox_inches="tight")

    model_val = scoreData_single["samples_s"].values
    data_val = scoreData_single["value"].values
    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig(outputfolder + "modelperformance_single.svg", bbox_inches="tight")

    model_val = scoreData_combo["init_count_mean"].values
    data_val = scoreData_combo["plasmid"].values
    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig(
        outputfolder + "modelperformance_combo_plasmid.svg", bbox_inches="tight"
    )

    model_val = scoreData_single["init_count_s_mean"].values
    data_val = scoreData_single["plasmid"].values
    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig(
        outputfolder + "modelperformance_single_plasmid.svg", bbox_inches="tight"
    )
    plt.close(fig)

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


def plotLossCurve(loss, log=True, name=None, outputDir=""):
    plt.plot(np.log(loss))
    plt.ylabel("ELBO")
    plt.xlabel("Steps")

    outputDir = "" if outputDir == "" else outputDir.rstrip("/") + "/"

    plt.savefig(
        f"{outputDir}valinor_loss.pdf"
        if name == None
        else f"{outputDir}valinor_loss_{name}.pdf"
    )
    plt.savefig(
        f"{outputDir}valinor_loss.svg"
        if name == None
        else f"{outputDir}valinor_loss_{name}.png"
    )

    plt.clf()


def plotDiagPlots(svi_result, name=None, outputDir=""):
    plotLossCurve(svi_result.losses, name=name, outputDir=outputDir)


def plot_hist(
    data_dict, xlabel, figsize=(6, 6), nbins=50, figure=None, loc="upper right"
):
    """
    Plot histograms for several different data series.

    Args:
        data_dict (Dict[str, np.ndarray]): A dictionary where keys are labels of the data and values are numpy arrays containing the data points for each entry.
        xlabel (str): The label for the x-axis.
        figsize (Tuple[int, int], optional): The size of the figure in inches. Defaults to (6, 6).
        nbins (int, optional): The number of bins for the histogram. Defaults to 50.
        figure (Tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot], optional):
               A tuple containing a figure and axes object to plot on. If None, a new figure and axes are created.
               Defaults to None.
        loc (str, optional): The location of the legend. Defaults to "upper right".

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot]: A tuple containing the figure and axes
                                                                                objects used for the plot. This can be
                                                                                used for further customization outside
                                                                                the function.
    """
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


def produce_modelfit_plot(df, outputfolder, type="combo"):
    """
    Plot the goodness of fit by comparing input data with Valinor's output.


    Args:
        df (pandas.DataFrame): The dataframe containing the model and actual data values.
        outputfolder (str): The path to the folder where the plots will be saved.
        type (str, optional): The type of plot to produce, either 'combo' or 'single'. Defaults to 'combo'.

    Returns:
        None: The function saves the plots directly to the specified folder and does not return any value.
    """
    sample_name = "samples" if type == "combo" else "samples_s"
    figname = "modelperformance_combo" if type == "combo" else "modelperformance_single"
    model_val = df[sample_name].values
    data_val = df["value"].values
    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig(outputfolder + figname + ".svg", bbox_inches="tight")
    fig.savefig(outputfolder + figname + ".png", bbox_inches="tight", dpi=200)

    plt.close(fig)

    init_count_name = "init_count_mean" if type == "combo" else "init_count_s_mean"
    figname = (
        "modelperformance_combo_plasmid"
        if type == "combo"
        else "modelperformance_single_plasmid"
    )
    model_val = df[init_count_name].values
    data_val = df["plasmid"].values
    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig(outputfolder + figname + ".svg", bbox_inches="tight")
    fig.savefig(outputfolder + figname + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)

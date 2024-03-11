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


def plotLossCurve(loss, log=True, name=None, outputDir=""):
    plt.plot(np.log(loss))
    plt.ylabel("ELBO")
    plt.xlabel("Steps")
    plt.savefig(
        f"{outputDir}valinor_loss.pdf"
        if name == None
        else f"{outputDir}valinor_loss_{name}.pdf"
    )
    plt.clf()


def plotDiagPlots(svi_result, name=None, outputDir=""):
    plotLossCurve(svi_result.losses, name=name, outputDir=outputDir)

import argparse

import os

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

import seaborn as sns

colours = sns.color_palette("tab10")

from pprint import pprint

from tqdm import tqdm

import pandas as pd

from scipy.stats import chisquare

import h5py


class ModelPlotter(object):
    def __init__(
        self,
        data,
        modelOutput,
        location=".",
        singletons=False,
        cutoff=False,
        calibrationFile=None,
        singletonFile=None,
        chronos=False,
    ):

        if not os.path.exists(location):
            os.mkdir(location)

        if location[:-1] != "/":
            location = location + "/"

        # Remove the cases with the same gene targetted twice (there shouldn't be any)
        data = data.query("g1_idx != g2_idx")

        self.data = data
        self.modelOutput = modelOutput
        self.location = location
        self.singletons = singletons
        self.cutoff = cutoff

        self.geneTermName = "g1"
        self.genePairTermName = "gpair"

        # Not true for anchors...
        self.guideTermName = "guide1"
        self.guidePairTermName = "guidepair"

        self.chronos = chronos

        self.valueName = "lfc"

        if self.chronos:
            self.valueName = 'value'

            self.geneTermName = "ko_growth_1"
            self.genePairTermName = "ko_growth_12"

            self.guideTermName = "guide_eff_1"
            self.guidePairTermName = "guide_eff_12"

            self.cell_growth = "cell_growth"
            self.cell_eff = "cell_eff"

        if calibrationFile != None:
            self.calibrationData = pd.read_hdf(calibrationFile, "ace")
            self.calibrationGenValues = pd.read_hdf(calibrationFile, "offsets")[
                "offsets"
            ].values

        if singletonFile:
            self.singletonData = pd.read_hdf(singletonFile, "ace")

    def plotDataModelComparison(self, data, modelOutput, plot=plt):

        # offset = np.array([ 0., 0.29399431, 0.18996913, 0.47189779, -0.21669924, -0.28156997])

        # Discard (3) replicate association
        lfcs = data[self.valueName].values#.reshape(-1, 3)[:, 0]

        # cell_line_idx = np.array(data['cell_line'].values.reshape(-1, 3)[:,:1]).reshape(-1)
        # lfcs = lfcs - offset[cell_line_idx]

        # combinedData = np.concatenate((data[self.valueName].values.reshape(-1, 3)[:, 0], lfcs))
        combinedData = np.concatenate((data[self.valueName].values, lfcs))

        minBinCut = np.quantile(combinedData, 0.01)
        maxBinCut = np.quantile(combinedData, 0.999)

        minBinAll = min(np.min(lfcs), np.min(modelOutput["samples"]))
        maxBinAll = max(np.max(lfcs), np.max(modelOutput["samples"]))

        bins = np.linspace(
            minBinCut if self.cutoff else minBinAll,
            maxBinCut if self.cutoff else maxBinAll,
            100,
        )

        # bins = np.linspace(-5, 2, 75)

        plot.hist(lfcs, bins, histtype="step", label="Data")
        plot.hist(
            modelOutput["samples"],
            bins,
            histtype="step",
            label="Model",
            color=colours[2],
        )

        binsChi2 = np.linspace(minBinAll, maxBinAll, 100)

        y1, _ = np.histogram(lfcs, binsChi2)
        y2, _ = np.histogram(modelOutput["samples"], binsChi2)

        try:
            chi2 = np.round(chisquare((y1 + 1), (y2 + 1)), 2) / len(y1)
        except ValueError as e:
            print(e)
            chi2 = [0.0, 0.0]

        plot.set_xlim(minBinCut if self.cutoff else minBinAll, maxBinCut if self.cutoff else maxBinAll)

        if plot is plt:
            plot.legend(loc=6, fontsize=12)
            plot.xlabel(self.valueName, fontsize=12)

            plot.savefig(f"{self.location}dataModelHist.pdf")
            plot.savefig(f"{self.location}dataModelHist.png", dpi=300)
            plot.clf()
        else:

            plot.annotate(
                r"$\chi^2_r=$%.2f" % (chi2[0]),
                (0.05, 0.05),
                xycoords="axes fraction",
                fontsize=16,
            )

            # plot.set_yscale('log')

            plot.tick_params(axis="both", labelsize=12)
            plot.legend(loc=6, fontsize=16)
            plot.set_xlabel(self.valueName, fontsize=16)

    def plotLFCs(self, lfcs, dataType="", plot=plt):

        plot.hist(lfcs, bins=75, histtype="stepfilled")

        if plot is plt:
            plot.xlabel(self.valueName, fontsize=12)
            plot.savefig(f"{dataType}lfcs.pdf")
            plot.savefig(f"{dataType}lfcs.png", dpi=300)
            plot.clf()
        else:
            plot.tick_params(axis="both", labelsize=12)

            plot.set_xlabel(self.valueName, fontsize=16)

    def plotStrongSynergy(self, data, modelOutput, plot=plt):

        if self.cutoff:
            minStrong = np.quantile(modelOutput["STRONG"], 0.01)
            maxStrong = np.quantile(modelOutput["STRONG"], 0.99)

        plot.plot(
            modelOutput["STRONG"],
            data["syn"].values.reshape(-1, 3)[:, 0],
            ".",
            markersize=1.0,
            alpha=0.5,
        )

        subsample = np.random.randint(0, len(modelOutput), len(modelOutput) // 10)

        sns.kdeplot(
            x=modelOutput["STRONG"].values[subsample],
            y=data["syn"].values.reshape(-1, 3)[:, 0][subsample],
            alpha=0.8,
            ax=plot,
            color=colours[1],
            zorder=10,
        )

        corr = np.round(
            np.corrcoef(modelOutput["STRONG"], data["syn"].values.reshape(-1, 3)[:, 0]),
            2,
        )

        if plot is plt:
            plot.xlabel('GEMINI "strong" score', fontsize=12)
            plot.ylabel("Synergy", fontsize=12)

            if self.cutoff:
                plot.xlim(minStrong, maxStrong)

            plot.savefig(f"{self.location}strongSynergy.pdf")
            plot.savefig(f"{self.location}strongSynergy.png", dpi=300)
            plot.clf()

        else:
            plot.tick_params(axis="both", labelsize=12)

            if self.cutoff:
                plot.set_xlim(minStrong, maxStrong)

            plot.annotate(
                r"$\rho=$%.2f" % (corr[0][1]),
                (0.05, 0.05),
                xycoords="axes fraction",
                fontsize=16,
            )

            plot.set_xlabel('GEMINI "strong" score', fontsize=16)
            plot.set_ylabel("Synergy", fontsize=16)

    def plotGeneTerm(self, data, modelOutput, plot=plt):

        plot.plot(
            modelOutput[self.geneTermName],
            data["ess1"].values,#.reshape(-1, 3)[:, 0],
            ".",
            markersize=1.0,
        )

        corr = np.round(
            # np.corrcoef(modelOutput[self.geneTermName], data["ess1"].values.reshape(-1, 3)[:, 0]), 2
            np.corrcoef(modelOutput[self.geneTermName], data["ess1"].values), 2
        )

        if plot is plt:

            plot.xlabel("Gene term", fontsize=12)
            plot.ylabel("Gene essentiality", fontsize=12)

            plot.savefig(f"{self.location}geneTermEssentiality.pdf")
            plot.savefig(f"{self.location}geneTermEssentiality.png", dpi=300)
            plot.clf()

        else:
            plot.tick_params(axis="both", labelsize=12)

            plot.annotate(
                r"$\rho=$%.2f" % (corr[0][1]),
                (0.05, 0.05),
                xycoords="axes fraction",
                fontsize=16,
            )

            plot.set_xlabel("Gene term", fontsize=16)
            plot.set_ylabel("Gene essentiality", fontsize=16)

    def plotGenePairTerm(self, data, modelOutput, plot=plt):

        # Plot by cell line! Different gene terms!
        # plt.plot(model['gpair'].values[df['cell_line'].values.reshape(-1, 3)[:,0] == 0], df.query('cell_line == 0')['syn'].values.reshape(-1, 3)[:,0], '.')

        cell_lines = np.unique(data["cell_line"])

        if self.cutoff:
            minPair = np.quantile(modelOutput[self.genePairTermName], 0.01)
            maxPair = np.quantile(modelOutput[self.genePairTermName], 0.99)

        plot.plot(
            modelOutput[self.genePairTermName],
            data["syn"].values,#.reshape(-1, 3)[:, 0],
            ".",
            markersize=1.0,
            alpha=0.5,
        )

        subsample = np.random.randint(0, len(modelOutput), len(modelOutput) // 10)

        sns.kdeplot(
            x=modelOutput[self.genePairTermName].values[subsample],
            y=data["syn"].values[subsample],#.reshape(-1, 3)[:, 0][subsample],
            alpha=0.8,
            ax=plot,
            color=colours[1],
            zorder=10,
        )

        # for cell_line in cell_lines:
        #
        #     d = data["syn"].values.reshape(-1, 3)[:, 0]
        #     c = data["cell_line"].values.reshape(-1, 3)[:, 0]
        #
        #     plot.plot(
        #         modelOutput[self.genePairTermName].values[c == cell_line],
        #         d[c == cell_line],
        #         ".",
        #         markersize=1.0,
        #         alpha=0.5,
        #     )

        corr = np.round(
            # np.corrcoef(modelOutput[self.genePairTermName], data["syn"].values.reshape(-1, 3)[:, 0]),
            np.corrcoef(modelOutput[self.genePairTermName], data["syn"].values),
            2,
        )

        if plot is plt:

            plot.xlabel("Gene pair term", fontsize=12)
            plot.ylabel("Synergy", fontsize=12)

            if self.cutoff:
                plot.set_xlim(minPair, maxPair)

            plot.savefig(f"{self.location}genePairSynergy.pdf")
            plot.savefig(f"{self.location}genePairSynergy.png", dpi=300)
            plot.clf()

        else:
            plot.tick_params(axis="both", labelsize=12)

            if self.cutoff:
                plot.set_xlim(minPair, maxPair)

            plot.annotate(
                r"$\rho=$%.2f" % (corr[0][1]),
                (0.05, 0.05),
                xycoords="axes fraction",
                fontsize=16,
            )

            plot.set_xlabel("Gene pair term", fontsize=16)
            plot.set_ylabel("Synergy", fontsize=16)

    def plotGuideTerm(self, data, modelOutput, plot=plt):

        plot.plot(
            modelOutput[self.guideTermName],
            data["ess1"].values,#.reshape(-1, 3)[:, 0],
            ".",
            markersize=1.0,
        )

        corr = np.round(
            np.corrcoef(
                modelOutput[self.guideTermName], data["ess1"].values,#.reshape(-1, 3)[:, 0]
            ),
            2,
        )

        if plot is plt:
            plot.xlabel("Guide term", fontsize=12)
            plot.ylabel("Gene essentiality", fontsize=12)

            plot.savefig(f"{self.location}guideTermEssentiality.pdf")
            plot.savefig(f"{self.location}guideTermEssentiality.png", dpi=300)
            plot.clf()
        else:

            plot.annotate(
                r"$\rho=$%.2f" % (corr[0][1]),
                (0.05, 0.05),
                xycoords="axes fraction",
                fontsize=16,
            )

            plot.tick_params(axis="both", labelsize=12)
            plot.set_xlabel("Guide term", fontsize=16)
            plot.set_ylabel("Gene essentiality", fontsize=16)

    def plotGuidePairTerm(self, data, modelOutput, plot=plt):

        plot.plot(
            modelOutput[self.guidePairTermName],
            data["syn"].values,#.reshape(-1, 3)[:, 0],
            ".",
            markersize=1.0,
            alpha=0.5,
        )

        subsample = np.random.randint(0, len(modelOutput), len(modelOutput) // 10)

        sns.kdeplot(
            x=modelOutput[self.guidePairTermName].values[subsample],
            y=data["syn"].values[subsample],#.reshape(-1, 3)[:, 0][subsample],
            alpha=0.8,
            ax=plot,
            color=colours[1],
            zorder=10,
        )

        corr = np.round(
            np.corrcoef(
                modelOutput[self.guidePairTermName], data["syn"].values,#.reshape(-1, 3)[:, 0]
            ),
            2,
        )

        if plot is plt:

            plot.xlabel("Guide pair term", fontsize=12)
            plot.ylabel("Gene synergy", fontsize=12)

            plot.savefig(f"{self.location}guidePairSynergy.pdf")
            plot.savefig(f"{self.location}guidePairSynergy.png", dpi=300)
            plot.clf()
        else:

            plot.annotate(
                r"$\rho=$%.2f" % (corr[0][1]),
                (0.05, 0.05),
                xycoords="axes fraction",
                fontsize=16,
            )

            plot.tick_params(axis="both", labelsize=12)
            plot.set_xlabel("Guide pair term", fontsize=16)
            plot.set_ylabel("Gene synergy", fontsize=16)

    def plotTermError(self, modelOutputs):

        # maxGeneIdx = np.argmax(modelOutput['g1'])
        # maxGeneVal = modelOutput['g1'].values[maxGeneIdx]
        # maxGeneErr = modelOutput['g1_std'].values[maxGeneIdx]

        for modelOutput in modelOutputs:
            plt.hist(modelOutput["g1_std"], density=True, histtype="step", bins=20)

        plt.savefig(f"{self.location}g1Err.pdf")
        plt.clf()

        for modelOutput in modelOutputs:
            plt.hist(modelOutput[self.geneTermName], density=True, histtype="step", bins=20)

        plt.savefig(f"{self.location}g1.pdf")
        plt.clf()

    def plotCalibDistributions(self, calibData, calibGen):

        cell_lines = np.unique(calibData["cell_line"])
        bins = np.linspace(-1.5, 1.5, 30)

        fig, axs = plt.subplots(len(cell_lines), sharex=True)

        for i, c in enumerate(cell_lines):
            axs[i].hist(
                calibData.query(f"cell_line == {c}")["lfc_neg"],
                histtype="step",
                bins=bins,
                color=colours[i],
            )
            axs[i].axvline(calibGen[i], color=colours[i])

        plt.xlabel(self.valueName, fontsize=12)
        plt.savefig(f"{self.location}calib_neg.pdf")
        plt.clf()

        sns.violinplot(y="cell_line", x="lfc_neg", data=calibData, orient="h")
        plt.savefig(f"{self.location}calib_neg_violins.pdf")
        plt.clf()

    def makePlots(self):

        if not self.singletons:

            dataLFCs = self.data[self.valueName].values,#.reshape(-1, 3)[:, 0]
            modelLFCs = self.modelOutput["samples"]

            self.plotLFCs(dataLFCs, dataType="data")
            self.plotLFCs(modelLFCs, dataType="model")

            self.plotDataModelComparison(self.data, self.modelOutput)

        self.plotStrongSynergy(self.data, self.modelOutput)

        self.plotGeneTerm(self.data, self.modelOutput)
        self.plotGenePairTerm(self.data, self.modelOutput)

        self.plotGuideTerm(self.data, self.modelOutput)
        self.plotGuidePairTerm(self.data, self.modelOutput)

    def makeMultiPlots(self):

        if not self.chronos:
            fig, axs = plt.subplots(2, 3, figsize = (18, 9))
        else:
            fig, axs = plt.subplots(2, 2, figsize = (16, 12))

        if not self.singletons:

            self.plotDataModelComparison(self.data, self.modelOutput, plot=axs[0][0])

        self.plotGeneTerm(self.data, self.modelOutput, plot=axs[0][1])
        self.plotGenePairTerm(self.data, self.modelOutput, plot=axs[1][0])

        self.plotGuideTerm(self.data, self.modelOutput, plot=axs[1][1])

        if self.guidePairTermName in self.data.columns:

            self.plotGuidePairTerm(self.data, self.modelOutput, plot=axs[0][2])

        if not self.chronos:

            self.plotStrongSynergy(self.data, self.modelOutput, plot=axs[1][2])

        axs[0][0].annotate(
            self.location.split("/")[-2],
            (0.10, 0.90),
            xycoords="axes fraction",
            fontsize=14,
        )

        plt.savefig(f"{self.location}multi.png", dpi=300)
        plt.clf()


if __name__ == "__main__":

    # I'd like an argument, please
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-d", type=str, dest="dataFile", help="Pandas HDF5 data input file."
    )
    argParser.add_argument(
        "-sd", type=str, dest="sglFile", help="Pandas HDF5 singleton data input file."
    )
    argParser.add_argument(
        "-cd",
        type=str,
        dest="calibFile",
        help="Pandas HDF5 calibration data input file.",
    )
    argParser.add_argument(
        "-m",
        nargs="+",
        type=str,
        dest="modelOutputFile",
        help="Pandas HDF5 model output file.",
    )
    argParser.add_argument(
        "-l", type=str, dest="location", default=".", help="Plot output location."
    )
    argParser.add_argument(
        "-s",
        action="store_true",
        dest="singletons",
        default=False,
        help="Plot singletons model.",
    )
    argParser.add_argument(
        "-c",
        action="store_true",
        dest="cutoff",
        default=False,
        help="Cutoff plot outliers.",
    )
    argParser.add_argument(
        "--chronos",
        action="store_true",
        dest="chronos",
        default=False,
        help="Assume Chronos likelihood.",
    )

    args = argParser.parse_args()

    data = pd.read_hdf(args.dataFile)
    modelOutput = [pd.read_hdf(f) for f in args.modelOutputFile]
    if len(modelOutput) == 1:
        modelOutput = modelOutput[0]

    plotter = ModelPlotter(
        data,
        modelOutput,
        location=args.location,
        singletons=args.singletons,
        cutoff=args.cutoff,
        singletonFile=args.sglFile,
        calibrationFile=args.calibFile,
        chronos = args.chronos,
    )

    # plotter.makePlots()
    plotter.makeMultiPlots()

    # TODO: make this a a command-line option

    # Only for multiple inputs, for comparison
    # plotter.plotTermError(plotter.modelOutput)

    # plotter.plotCalibDistributions(
    #     plotter.calibrationData, plotter.calibrationGenValues
    # )

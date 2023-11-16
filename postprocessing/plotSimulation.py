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
from sklearn.feature_selection import mutual_info_classif

import h5py

import pickle


class SimPlotter(object):
    def __init__(
        self,
        data,
        modelOutput,
        location=".",
        singletons=False,
        cutoff=False,
        calibrationFile=None,
        singletonFile=None,
    ):

        if not os.path.exists(location):
            os.mkdir(location)

        if location[:-1] != "/":
            location = location + "/"

        self.data = data
        self.modelOutput = modelOutput
        self.location = location
        self.singletons = singletons
        self.cutoff = cutoff

        self.valueName = 'value'

        self.geneTermName = "gene_ko_growth_1_mean"
        self.genePairTermName = "gene_ko_growth_12_mean"

        self.guideTermName = "guide_eff_1_mean"

        self.cell_growth = "cell_line_growth_mean"

        if calibrationFile != None:
            self.calibrationData = pd.read_parquet(calibrationFile)
            self.calibrationGenValues = pd.read_parquet(calibrationFile)[
                "offsets"
            ].values

        if singletonFile:
            self.singletonData = pd.read_parquet(singletonFile)

    def plotDataModelComparison(self, data, modelOutput, plot=plt):

        dataCounts = data[self.valueName]

        minBinCut = np.quantile(dataCounts, 0.001)
        maxBinCut = np.quantile(dataCounts, 0.999)

        minBinAll = min(np.min(dataCounts), np.min(modelOutput["samples"]))
        maxBinAll = max(np.max(dataCounts), np.max(modelOutput["samples"]))

        bins = np.linspace(
            minBinCut if self.cutoff else minBinAll,
            maxBinCut if self.cutoff else maxBinAll,
            100,
        )

        plot.hist(dataCounts, bins, histtype="step", label="Data", lw = 2.0)
        plot.hist(
            modelOutput["samples"],
            bins,
            histtype="step",
            label="Model",
            color=colours[2],
            lw = 2.0
        )

        binsChi2 = np.linspace(minBinAll, maxBinAll, 100)

        y1, _ = np.histogram(dataCounts, binsChi2)
        y2, _ = np.histogram(modelOutput["samples"], binsChi2)

        try:
            chi2 = np.round(chisquare((y1 + 1), (y2 + 1)), 2) / len(y1)
        except ValueError as e:
            print(e)
            chi2 = [0.0, 0.0]

        if plot is plt:

            plot.legend(loc=6, fontsize=18)
            plot.xlabel('Counts', fontsize=18)

            plot.xlim(minBinCut if self.cutoff else minBinAll, maxBinCut if self.cutoff else maxBinAll)

            plot.savefig(f"{self.location}dataModelHist.pdf")
            plot.savefig(f"{self.location}dataModelHist.png", dpi=300)
            plot.clf()

        else:

            plot.annotate(
                r"$\chi^2_r=$%.2f" % (chi2[0]),
                (0.90, 0.05),
                xycoords="axes fraction",
                fontsize=21,
            )

            plot.set_xlim(minBinCut if self.cutoff else minBinAll, maxBinCut if self.cutoff else maxBinAll)

            plot.tick_params(axis="both", labelsize=18)
            plot.legend(loc=6, fontsize=21)
            plot.set_xlabel('Counts', fontsize=21)

    # Separate code to postprocess model output? Merging, LFC, etc

    def calculateLFC(self, data):

        data['lfc'] = np.log2(np.maximum(data['value'], 1) / np.maximum(data['plasmid'], 1))

        return data

    def populateContexts(self, data, contextsFile):

        cell_contexts, context_matrices = pickle.load(open(contextsFile, 'rb'))

        dfs = []

        for cl, contexts in cell_contexts.items():

            gi = np.prod([context_matrices[i] for i in contexts], axis = 0)

            gi_df = pd.DataFrame({
                'gene1': [i for i in range(gi.shape[0]) for _ in range(gi.shape[1])],
                'gene2': [j for _ in range(gi.shape[0]) for j in range(gi.shape[1])],
                'gi': gi.flatten(),
                'cell_line' : cl
            })

            dfs.append(gi_df)

        gi_df = pd.concat(dfs)

        gi_data = data.merge(gi_df, on = ['gene1', 'gene2', 'cell_line'])
        gi_data['context_gi'] = gi_data['gi'] > 1.0

        self.data = gi_data

        return gi_data

    def findContextGI(self, data, threshold = 0.2, n = 1):

        # Get gene pairs that have GI and also only appear in one cell line

        n_cl = len(data['cell_line'].unique())

        data['gene_pair'] = data['gene1'].astype(str) + '_' + data['gene2'].astype(str)
        data['cell_line'] = data['cell_line'].astype(str)

        gi_data_syn = data[np.abs(data['syn']) > threshold]

        gi_data_n = gi_data_syn.groupby('gene_pair')['cell_line'].nunique().reset_index()
        gi_data_n = gi_data_n[gi_data_n['cell_line'] <= n].rename(columns = {'cell_line' : 'n_cl'})

        return gi_data_n.merge(gi_data_syn[['gene_pair', 'cell_line']], on = 'gene_pair').drop_duplicates()

    def plotContextGI(self, data, modelOutput):

        sns.kdeplot(data, x = 'syn')
        plt.savefig('syn.pdf')
        plt.clf()

        sns.kdeplot(data = data[data['context_gi']], x = 'gi')

        plt.savefig('gi.pdf')
        plt.clf()

        clipMin = np.quantile(data['lfc'], 0.0001)
        clipMax = np.quantile(data['lfc'], 0.9999)

        sns.kdeplot(data = data[data['gi'] > 1.7], x = 'lfc', common_norm = False, clip = (clipMin, clipMax), color = 'blue')
        sns.kdeplot(data = data[~data['context_gi']], x = 'lfc', common_norm = False, clip = (clipMin, clipMax), color = 'red')

        plt.xlim(clipMin, clipMax)

        plt.savefig('context_gi.pdf')
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

        fig, axs = plt.subplots(2, 2, figsize = (16, 12))

        if not self.singletons:

            self.plotDataModelComparison(self.data, self.modelOutput, plot=axs[0][0])

        self.plotGeneTerm(self.data, self.modelOutput, plot=axs[0][1])
        self.plotGenePairTerm(self.data, self.modelOutput, plot=axs[1][0])

        self.plotGuideTerm(self.data, self.modelOutput, plot=axs[1][1])

        if self.guidePairTermName in self.data.columns:

            self.plotGuidePairTerm(self.data, self.modelOutput, plot=axs[0][2])

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
        "-d",
        "--combsData",
        type=str, dest="dataFile", help="Pandas Parquet combination data input file."
    )
    argParser.add_argument(
        "-sd",
        "--singlesData",
        type=str, dest="sglFile", help="Pandas Parquet singleton data input file."
    )
    argParser.add_argument(
        "-cd",
        "--controlsData",
        type=str,
        dest="calibFile",
        help="Pandas Parquet control data input file.",
    )
    argParser.add_argument(
        "--simulationGen",
        type=str,
        dest="simulationData",
        help="Pandas Parquet simulation true generated values.",
    )

    # Also add singles model!

    argParser.add_argument(
        "-m",
        nargs="+",
        type=str,
        dest="modelOutputFile",
        help="Pandas Parquet model output file (multiple).",
    )
    argParser.add_argument(
        "-l", type=str, dest="location", default=".", help="Plot output location."
    )
    argParser.add_argument(
        "--contexts", type=str, dest="contextsFile", default=".", help="GI contexts pickle file."
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
        "--simulation",
        action="store_true",
        dest="simulation",
        default=False,
        help="Whether the inputs are simulation, to compare with truth values.",
    )

    args = argParser.parse_args()

    data = pd.read_parquet(args.dataFile)
    modelOutput = [pd.read_parquet(f) for f in args.modelOutputFile]
    if len(modelOutput) == 1:
        modelOutput = modelOutput[0]

    plotter = SimPlotter(
        data,
        modelOutput,
        location=args.location,
        singletons=args.singletons,
        cutoff=args.cutoff,
        singletonFile=args.sglFile,
        calibrationFile=args.calibFile,
    )

    print(plotter.findContextGI(data))

    exit(0)

    data = plotter.populateContexts(data, args.contextsFile)
    data = plotter.calculateLFC(data)

    plotter.plotContextGI(data, modelOutput)

    plotter.plotDataModelComparison(data, modelOutput)

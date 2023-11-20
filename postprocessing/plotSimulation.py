import argparse

import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
from sklearn import metrics

from scipy.stats import norm

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

        self.modelData = pd.concat((data, modelOutput), axis = 1)

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

            gi = np.sum([context_matrices[i] for i in contexts], axis = 0)

            gi_df = pd.DataFrame({
                'gene1': [i for i in range(gi.shape[0]) for _ in range(gi.shape[1])],
                'gene2': [j for _ in range(gi.shape[0]) for j in range(gi.shape[1])],
                'gi': gi.flatten(),
                'cell_line' : cl
            })

            dfs.append(gi_df)

        gi_df = pd.concat(dfs)

        gi_data = data.merge(gi_df, on = ['gene1', 'gene2', 'cell_line'])
        gi_data['context_gi'] = gi_data['gi'] > 1E-4

        self.data = gi_data

        return gi_data

    def findContextGI(self, data, threshold = 0.4, n = 1):

        # Get gene pairs that have GI and also only appear in one cell line

        n_cl = len(data['cell_line'].unique())

        gi_data_syn = data[np.abs(data['syn']) > threshold]

        gi_data_n = gi_data_syn.groupby('gene_pair')['cell_line'].nunique().reset_index()
        gi_data_n = gi_data_n[gi_data_n['cell_line'] <= n].rename(columns = {'cell_line' : 'n_cl'})

        return gi_data_n.merge(gi_data_syn[['gene_pair', 'cell_line']], on = 'gene_pair').drop_duplicates()

    def find_unique_gi_based_on_zscore(self, data, var_name = 'syn', zscore_threshold=3):
        # List to collect rows that meet the criteria
        selected_rows = []

        # Group by gene pair
        for gene_pair, group in tqdm(data.groupby('gene_pair')):
            # For each cell line, calculate the mean and std of other cell lines
            for index, row in group.iterrows():
                other_syn_values = group[group['cell_line'] != row['cell_line']][var_name]

                if len(other_syn_values) > 0:
                    mean_other_syn = other_syn_values.mean()
                    std_other_syn = other_syn_values.std()

                    # Calculate Z-score
                    z_score = abs(row[var_name] - mean_other_syn) / std_other_syn

                    # Check if the Z-score exceeds the threshold
                    if z_score > zscore_threshold:
                        row[f'zscore_{var_name}'] = z_score
                        selected_rows.append(row)

        # Create a DataFrame from the collected rows
        unique_gi_pairs = pd.DataFrame(selected_rows).reset_index(drop=True)
        return unique_gi_pairs

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

    def calculate_uniqueness_probability(self, df):
        # Create an empty DataFrame for the results
        uniqueness_df = pd.DataFrame()

        # Ensure that mean and std columns are numeric
        df['gene_ko_growth_12_mean'] = pd.to_numeric(df['gene_ko_growth_12_mean'], errors='coerce')
        df['gene_ko_growth_12_std'] = pd.to_numeric(df['gene_ko_growth_12_std'], errors='coerce')

        for gene_pair, group in tqdm(df.groupby('gene_pair')):
            cell_lines_data = group.set_index('cell_line')[['gene_ko_growth_12_mean', 'gene_ko_growth_12_std']].T.to_dict()

            # Calculate probabilities
            probabilities = self.calculate_probabilities_for_gene_pair(cell_lines_data)

            # Create a temporary DataFrame to store the results for this gene pair
            temp_df = pd.DataFrame({
                'gene_pair': gene_pair,
                'cell_line': list(probabilities.keys()),
                'uniqueness_probability': list(probabilities.values())
            })

            # Concatenate the temporary DataFrame with the main result DataFrame
            uniqueness_df = pd.concat([uniqueness_df, temp_df], ignore_index=True)

        return uniqueness_df

    def calculate_probabilities_for_gene_pair(self, cell_lines_data):
        probabilities = {}
        for target_cell_line, target_stats in cell_lines_data.items():
            target_mean = target_stats['gene_ko_growth_12_mean']
            target_std = target_stats['gene_ko_growth_12_std']
            sum_other_means = 0
            sum_other_vars = 0

            # Sum up the means and variances of other cell lines
            for other_cell_line, other_stats in cell_lines_data.items():
                if other_cell_line != target_cell_line:
                    other_mean = other_stats['gene_ko_growth_12_mean']
                    other_std = other_stats['gene_ko_growth_12_std']
                    sum_other_means += other_mean
                    sum_other_vars += other_std**2

            # Number of other cell lines
            n_other = len(cell_lines_data) - 1

            # Calculate average mean and variance of other distributions
            if n_other > 0:
                avg_other_mean = sum_other_means / n_other
                avg_other_var = sum_other_vars / n_other

                # Calculate overlap
                overlap = self.calculate_distribution_overlap(target_mean, target_std, avg_other_mean, np.sqrt(avg_other_var))

                # Probability of uniqueness
                probabilities[target_cell_line] = 1 - overlap
            else:
                # If there are no other cell lines, set uniqueness probability to 1
                probabilities[target_cell_line] = 1

        return probabilities

    def calculate_distribution_overlap(self, mean1, std1, mean2, std2):
        # mean_diff = mean1 - mean2
        # var_diff = std1**2 + std2**2
        # diff_dist = norm(mean_diff, np.sqrt(var_diff))
        #
        # # Use the absolute value of the cumulative probability up to zero
        # overlap = abs(diff_dist.cdf(0))
        #
        # # The probability of uniqueness (non-overlap) is 1 - overlap
        # # Ensure it's within [0, 1]
        # uniqueness_probability = min(max(1 - overlap, 0), 1)
        # return uniqueness_probability

        mean_diff = mean1 - mean2
        var_diff = std1**2 + std2**2
        diff_dist = norm(mean_diff, np.sqrt(var_diff))
        overlap = 2 * diff_dist.cdf(0)
        return overlap

    def scoreContextGI(self, data):

        data['gene_pair'] = data['gene1'].astype(str) + '_' + data['gene2'].astype(str)
        data['cell_line'] = data['cell_line'].astype(str)

        data['context_gi'] = np.abs(data['gi']) > 0.01

        print(np.sum(data['context_gi']))

        avg_df = data.groupby(['gene_pair', 'cell_line']).agg({'context_gi' : 'first', 'gi' : 'first', 'context_gi' : 'first', 'lfc' : 'mean', 'gene_ko_growth_12_mean' : 'first', 'gene_ko_growth_12_std' : 'first', 'syn' : 'first'}).reset_index()

        avg_df['score'] = avg_df['gene_ko_growth_12_mean'] / avg_df['gene_ko_growth_12_std']

        sns.kdeplot(data = avg_df, x = 'score', hue = 'context_gi', common_norm = False)
        plt.savefig('g12.pdf')
        plt.clf()
        exit(0)

        unq_score = self.find_unique_gi_based_on_zscore(avg_df, var_name = 'score', zscore_threshold = 0)

        avg_df = avg_df.merge(unq_score[['cell_line', 'gene_pair', 'zscore_score']], on = ['cell_line', 'gene_pair'])

        sns.scatterplot(data = avg_df, x = 'syn', y = 'score', hue = 'context_gi')
        plt.savefig('reg_score.pdf')
        plt.clf()

        sns.regplot(data = avg_df, x = 'gi', y = 'gene_ko_growth_12_mean')
        plt.savefig('reg_gi.pdf')
        plt.clf()

        sns.regplot(data = avg_df, x = 'gi', y = 'zscore_score')
        plt.savefig('reg_zscore.pdf')
        plt.clf()

        fpr, tpr, thresholds = metrics.roc_curve(avg_df['context_gi'], avg_df['zscore_score'])

        plt.plot(fpr, tpr, lw = 3.0)
        plt.savefig(f'gi_roc_context.pdf')
        plt.clf()

        precision, recall, thresholds = metrics.precision_recall_curve(avg_df['context_gi'], avg_df['zscore_score'])

        plt.plot(recall, precision, lw = 3.0)
        plt.savefig(f'gi_pr_context.pdf')
        plt.clf()

        sns.kdeplot(data = avg_df, x = 'zscore_score', hue = 'context_gi', common_norm = False)
        plt.savefig('gi_scatter.pdf')
        plt.clf()

    def scoreContextGI2(self, data):

        data['gene_pair'] = data['gene1'].astype(str) + '_' + data['gene2'].astype(str)
        data['cell_line'] = data['cell_line'].astype(str)

        # context_gi = self.findContextGI(data, 0.6)

        context_gi = self.find_unique_gi_based_on_zscore(data.groupby(['gene_pair', 'cell_line']).agg({'syn' : 'first'}).reset_index(), var_name = 'syn', zscore_threshold = 0)

        context_gi = context_gi[['cell_line', 'gene_pair', 'zscore_syn']].copy()

        merged_df = pd.merge(data, context_gi, on=['gene_pair', 'cell_line'], how='left', indicator=True)
        merged_df['gi'] = merged_df['_merge'].apply(lambda x: 1 if x == 'both' else 0)
        merged_df.drop(columns=['_merge'], inplace=True)

        # sns.kdeplot(data = merged_df, x = 'lfc', hue = 'gi', common_norm = False)
        # plt.savefig('gi.pdf')
        # plt.clf()

        avg_df = merged_df.groupby(['gene_pair', 'cell_line']).agg({'gi' : 'first', 'lfc' : 'mean', 'gene_ko_growth_12_mean' : 'first', 'gene_ko_growth_12_std' : 'first', 'syn' : 'first', 'zscore_syn' : 'first'}).reset_index()

        print(avg_df[avg_df['gene_pair'] == '10_8'])
        print('')
        print(avg_df[avg_df['gene_pair'] == '13_64'])
        print('')
        print(avg_df[avg_df['gene_pair'] == '28_53'])

        sns.regplot(data = avg_df, x = 'syn', y = 'gene_ko_growth_12_mean')
        sns.regplot(data = avg_df[avg_df['gi'] == 1], x = 'syn', y = 'gene_ko_growth_12_mean')
        plt.savefig('syn_reg.pdf')
        plt.clf()

        avg_df['score'] = avg_df['gene_ko_growth_12_mean'] / avg_df['gene_ko_growth_12_std']

        sns.kdeplot(data = avg_df, x = 'score')
        plt.savefig('score.pdf')
        plt.clf()

        # avg_df = avg_df[avg_df['score'] > 5]

        unq_score = self.find_unique_gi_based_on_zscore(avg_df.copy(), var_name = 'gene_ko_growth_12_mean', zscore_threshold = 0)

        print(unq_score[unq_score['zscore_gene_ko_growth_12_mean'] > 30])

        print(len(unq_score))

        avg_df = avg_df.merge(unq_score[['cell_line', 'gene_pair', 'zscore_gene_ko_growth_12_mean']], on = ['cell_line', 'gene_pair'])

        sns.regplot(data = avg_df, x = 'zscore_syn', y = 'zscore_gene_ko_growth_12_mean')
        sns.regplot(data = avg_df[avg_df['gi'] == 1], x = 'zscore_syn', y = 'zscore_gene_ko_growth_12_mean')
        # sns.regplot(data = avg_df[avg_df['gi'] == 1], x = 'syn', y = 'uniqueness_probability')
        plt.savefig('syn_score.pdf')
        plt.clf()

        unq_prob = self.calculate_uniqueness_probability(avg_df)

        avg_df = avg_df.merge(unq_prob, on = ['cell_line', 'gene_pair'])

        sns.regplot(data = avg_df, x = 'syn', y = 'uniqueness_probability')
        sns.regplot(data = avg_df[avg_df['gi'] == 1], x = 'syn', y = 'uniqueness_probability')
        plt.savefig('syn_reg_unq.pdf')
        plt.clf()


        # Split by sign, as there are positive and negative GI in the simulation
        # whereas in reality negative is way more common. Could also just fold the data?

        # Metric for model is not just growth_12, but rather a similar thresholding accounting
        # for posteriors

        for sign in ['pos', 'neg']:

            # param = 'gene_ko_growth_12_mean'
            param = 'uniqueness_probability'

            avg_df_signed = avg_df[avg_df[param] > 0] if sign == 'pos' else avg_df[avg_df[param] < 0]
            # avg_df_signed = avg_df_signed[np.abs(avg_df_signed['syn']) > 0.6]

            print(np.sum(avg_df_signed['gi']), len(avg_df_signed))

            fpr, tpr, thresholds = metrics.roc_curve(avg_df_signed['gi'], avg_df_signed[param] * (1 if sign == 'pos' else -1))

            plt.plot(fpr, tpr, lw = 3.0)
            plt.savefig(f'gi_roc_{sign}.pdf')
            plt.clf()

            precision, recall, thresholds = metrics.precision_recall_curve(avg_df_signed['gi'], avg_df_signed[param] * (1 if sign == 'pos' else -1))

            plt.plot(recall, precision, lw = 3.0)
            plt.savefig(f'gi_pr_{sign}.pdf')
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

    data = plotter.populateContexts(plotter.modelData, args.contextsFile)
    data = plotter.calculateLFC(data)

    plotter.scoreContextGI(data)

    exit(0)

    plotter.plotContextGI(data, modelOutput)

    plotter.plotDataModelComparison(data, modelOutput)

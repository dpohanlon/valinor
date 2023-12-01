#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import argparse
from plotting import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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


def plot_guide_eff_hyper_hist(scoreData_combined):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    tmp = (
        scoreData_combined[["guide1", "guide_eff_mean_1_mean"]]
        .drop_duplicates()["guide_eff_mean_1_mean"]
        .values
    )
    tmp2 = (
        scoreData_combined[["guide2", "guide_eff_mean_2_mean"]]
        .drop_duplicates()["guide_eff_mean_2_mean"]
        .values
    )

    data_dict = {"guide 1": tmp, "guide 2": tmp2}

    xlabel = "Mean of\nguide efficiency hyper distribution"

    fig, ax[0] = plot_hist(
        data_dict, xlabel, figure=[fig, ax[0]], nbins=50, loc="upper right"
    )

    tmp = (
        scoreData_combined[["guide1", "guide_eff_std_1_mean"]]
        .drop_duplicates()["guide_eff_std_1_mean"]
        .values
    )
    tmp2 = (
        scoreData_combined[["guide2", "guide_eff_std_2_mean"]]
        .drop_duplicates()["guide_eff_std_2_mean"]
        .values
    )

    data_dict = {"guide 1": tmp, "guide 2": tmp2}

    xlabel = "Standard deviation of\nguide efficiency hyper distribution"

    fig, ax[1] = plot_hist(
        data_dict, xlabel, figure=[fig, ax[1]], nbins=50, loc="upper right"
    )

    return (fig, ax)


def plot_guide_eff_hist(scoreData_combined):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    tmp = (
        scoreData_combined[["guide1", "cell_line", "guide_eff_1_mean"]]
        .drop_duplicates()["guide_eff_1_mean"]
        .values
    )
    tmp2 = (
        scoreData_combined[["guide2", "cell_line", "guide_eff_2_mean"]]
        .drop_duplicates()["guide_eff_2_mean"]
        .values
    )

    data_dict = {"guide 1": tmp, "guide 2": tmp2}

    xlabel = "Mean of\nguide efficiency"

    fig, ax[0] = plot_hist(
        data_dict, xlabel, figure=[fig, ax[0]], nbins=50, loc="upper right"
    )

    tmp = (
        scoreData_combined[["guide1", "cell_line", "guide_eff_1_std"]]
        .drop_duplicates()["guide_eff_1_std"]
        .values
    )
    tmp2 = (
        scoreData_combined[["guide2", "cell_line", "guide_eff_2_std"]]
        .drop_duplicates()["guide_eff_2_std"]
        .values
    )

    data_dict = {"guide 1": tmp, "guide 2": tmp2}

    xlabel = "Standard deviation of\nguide efficiency"

    fig, ax[1] = plot_hist(
        data_dict, xlabel, figure=[fig, ax[1]], nbins=50, loc="upper right"
    )

    return (fig, ax)


def plot_scatter_genepairlevel(scoreData_combined, xaxis, yaxis):
    tmp = scoreData_combined.groupby(["genePair", "cell_line"]).agg(
        {xaxis["name"]: "mean", yaxis["name"]: "mean"}
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.axhline(0, color="grey")
    ax.axvline(0, color="grey")

    ax.scatter(tmp[xaxis["name"]], tmp[yaxis["name"]], alpha=0.3)

    ax.set_xlabel(xaxis["label"], fontsize=fontsizes[1])
    ax.set_ylabel(yaxis["label"], fontsize=fontsizes[1])
    ax.tick_params(axis="both", labelsize=fontsizes[2])

    return (fig, ax)


def plot_scatter_singlegenelevel(scoreData_combined, gene, xaxis, yaxis):
    tmp = scoreData_combined.groupby([gene, "cell_line"]).agg(
        {xaxis["name"]: "mean", yaxis["name"]: "mean"}
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.axhline(0, color="grey")
    ax.axvline(0, color="grey")

    ax.scatter(tmp[xaxis["name"]], tmp[yaxis["name"]], alpha=0.3)

    ax.set_xlabel(xaxis["label"], fontsize=fontsizes[1])
    ax.set_ylabel(yaxis["label"], fontsize=fontsizes[1])
    ax.tick_params(axis="both", labelsize=fontsizes[2])

    return (fig, ax)


def plot_scatter_singleguidelevel(scoreData_combined, guide, xaxis, yaxis):
    tmp = scoreData_combined.groupby([guide, "cell_line"]).agg(
        {xaxis["name"]: "mean", yaxis["name"]: "mean"}
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.axhline(0, color="grey")
    ax.axvline(0, color="grey")

    ax.scatter(tmp[xaxis["name"]], tmp[yaxis["name"]], alpha=0.3)

    ax.set_xlabel(xaxis["label"], fontsize=fontsizes[1])
    ax.set_ylabel(yaxis["label"], fontsize=fontsizes[1])
    ax.tick_params(axis="both", labelsize=fontsizes[2])

    return (fig, ax)


def plot_genepair_valscore_rank(df, genepair, max_rank, valscore_range):
    nrow = df.cell_line.unique().shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 1 * nrow))
    scatter = ax.scatter(
        df.rank_valinor_score,
        df.cell_line,
        c=df.valinor_score,
        cmap="PuOr_r",
        norm=mcolors.Normalize(vmin=valscore_range[0], vmax=valscore_range[1]),
    )
    ax.set_xlim([-4, max_rank])
    ax.set_title(genepair, fontsize=fontsizes[1])
    ax.set_xlabel(naming_cols["rank_valinor_score"], fontsize=fontsizes[1])
    ax.set_ylabel(naming_cols["cell_line"], fontsize=fontsizes[1])
    ax.tick_params(axis="both", labelsize=fontsizes[2])

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(naming_cols["valinor_score"], fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_facecolor("lightgrey")

    return (fig, ax)


def plot_genepair_valscore_vs_singleton(df, genepair):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    xaxis = ["valinor_score_s_s_1", "valinor_score_s_s_2"]

    for i, axx in enumerate(ax):
        axx.axhline(0, color="grey")
        axx.axvline(0, color="grey")

        axx.scatter(df[xaxis[i]], df.valinor_score, color="blue")
        axx.set_title(genepair, fontsize=fontsizes[1])
        axx.set_xlabel(naming_cols[xaxis[i]], fontsize=fontsizes[1])
        axx.set_ylabel(naming_cols["valinor_score"], fontsize=fontsizes[1])
        axx.tick_params(axis="both", labelsize=fontsizes[2])

    return (fig, ax)


def plot_genepair_singleton_lfc_vs_valscore(df, genepair):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i, axx in enumerate(ax):
        axx.axhline(0, color="grey")
        axx.axvline(0, color="grey")

        axx.scatter(
            df["valinor_score_s_s_" + str(i + 1)],
            df["lfc_s_" + str(i + 1)],
            color="blue",
        )
        axx.set_title(genepair, fontsize=fontsizes[1])
        axx.set_xlabel(
            naming_cols["valinor_score_s_s_" + str(i + 1)], fontsize=fontsizes[1]
        )
        axx.set_ylabel(naming_cols["lfc_s_" + str(i + 1)], fontsize=fontsizes[1])
        axx.tick_params(axis="both", labelsize=fontsizes[2])

    return (fig, ax)


def plot_genepair_combo_lfcs_vs_valscore(df, genepair):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].scatter(df.lfc_s_1, df.valinor_score)
    ax[1].scatter(df.lfc_s_2, df.valinor_score)
    ax[2].scatter(df.lfc, df.valinor_score)

    xaxis = ["lfc_s_1", "lfc_s_2", "lfc"]

    for i, axx in enumerate(ax):
        axx.axhline(0, color="grey")
        axx.axvline(0, color="grey")

        axx.scatter(df[xaxis[i]], df.valinor_score, color="blue")

        axx.set_title(genepair, fontsize=fontsizes[1])
        axx.set_xlabel(naming_cols[xaxis[i]], fontsize=fontsizes[1])
        axx.set_ylabel(naming_cols["valinor_score"], fontsize=fontsizes[1])
        axx.tick_params(axis="both", labelsize=fontsizes[2])

    return (fig, ax)


def plot_globalgenepair_nclns(scoreData_combined, valscore_thr=None):
    if valscore_thr is None:
        valscore_thr = np.quantile(scoreData_combined.valinor_score, 0.05)

    tmp = scoreData_combined.loc[scoreData_combined.valinor_score < valscore_thr].copy()
    tmp["sorted_genePair"] = tmp["genePair"].apply(
        lambda x: "_".join(sorted(x.split("_")))
    )
    tmp = tmp.drop_duplicates(subset=["sorted_genePair", "cell_line", "valinor_score"])
    sl_nclns = tmp["sorted_genePair"].value_counts()

    cln_thr = np.floor(len(scoreData_combined.cell_line.unique()) / 2)

    sl_nclns = sl_nclns[sl_nclns > cln_thr]
    n_pairs = sl_nclns.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(0.2 * n_pairs, 6))
    ax.bar(sl_nclns.index, sl_nclns.values)
    ax.set_xticks(np.arange(0, n_pairs))
    ax.set_xticklabels(sl_nclns.index.values, rotation=90)
    ax.set_title(
        f"Top synthetic lethal gene pairs in more than {cln_thr:.0f} cell lines",
        fontsize=fontsizes[1],
    )
    ax.set_xlabel("Gene pairs", fontsize=fontsizes[1])
    ax.set_ylabel("Number of cell lines", fontsize=fontsizes[1])
    ax.tick_params(axis="both", labelsize=fontsizes[2])

    ax.margins(x=0.01)

    return (fig, ax)


def main():
    parser = argparse.ArgumentParser(description="Load processed valinor output")

    parser.add_argument(
        "--combfile",
        help="Processed table combining data and Valinor output.",
        required=True,
    )

    parser.add_argument(
        "--output_folder",
        help="Folder in which to save the output plots",
        required=True,
    )

    args = parser.parse_args()

    scoreData_combined = pd.read_parquet(args.combfile)

    outputfolder = (
        args.output_folder
        if args.output_folder[-1] == "/"
        else args.output_folder + "/"
    )
    folders = [outputfolder, outputfolder + "topSLgenepairs/"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    fig, ax = plot_guide_eff_hyper_hist(scoreData_combined)
    fig.savefig(
        outputfolder + "guide_efficiencies_hyperdistribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plot_guide_eff_hist(scoreData_combined)
    fig.savefig(
        outputfolder + "guide_efficiencies_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    xaxis = {"name": "valinor_score", "label": naming_cols["valinor_score"]}
    yaxis = {"name": "deltaLFC", "label": naming_cols["deltaLFC"]}
    fig, ax = plot_scatter_genepairlevel(scoreData_combined, xaxis, yaxis)
    fig.savefig(outputfolder + "valscore_vs_dLFC.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    xaxis = {"name": "valinor_score_s_s_1", "label": naming_cols["valinor_score_s_s_1"]}
    yaxis = {"name": "lfc_s_1", "label": naming_cols["lfc_s_1"]}
    fig, ax = plot_scatter_singlegenelevel(scoreData_combined, "gene1", xaxis, yaxis)
    fig.savefig(
        outputfolder + "singleton_1_valscore_vs_lfc.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    xaxis = {"name": "valinor_score_s_s_2", "label": naming_cols["valinor_score_s_s_2"]}
    yaxis = {"name": "lfc_s_2", "label": naming_cols["lfc_s_2"]}
    fig, ax = plot_scatter_singlegenelevel(scoreData_combined, "gene2", xaxis, yaxis)
    fig.savefig(
        outputfolder + "singleton_2_valscore_vs_lfc.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    yaxis = {"name": "guide_eff_1_mean", "label": naming_cols["guide_eff_1_mean"]}
    xaxis = {
        "name": "lfc_s_1",
        "label": naming_cols["lfc_s_1"].replace("gene", "guide"),
    }
    fig, ax = plot_scatter_singleguidelevel(scoreData_combined, "guide1", xaxis, yaxis)
    fig.savefig(
        outputfolder + "singleton_1_lfc_vs_guideeffic.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    xaxis = {"name": "guide_eff_2_mean", "label": naming_cols["guide_eff_2_mean"]}
    xaxis = {
        "name": "lfc_s_2",
        "label": naming_cols["lfc_s_2"].replace("gene", "guide"),
    }
    fig, ax = plot_scatter_singleguidelevel(scoreData_combined, "guide2", xaxis, yaxis)
    fig.savefig(
        outputfolder + "singleton_2_lfc_vs_guideeffic.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # make plots for the synthetic lethal within each cell line
    top_x = 10
    genepairs = scoreData_combined.loc[
        scoreData_combined.rank_valinor_score <= top_x
    ].genePair.unique()
    tmp = scoreData_combined.loc[scoreData_combined.genePair.isin(genepairs)]
    max_rank = scoreData_combined.rank_valinor_score.max()
    min_valscore = scoreData_combined.valinor_score.min()
    max_valscore = scoreData_combined.valinor_score.max()
    for genepair, df in tmp.groupby("genePair"):
        df = df.drop_duplicates(subset=["genePair", "cell_line", "rank_valinor_score"])
        fig, ax = plot_genepair_valscore_rank(
            df, genepair, max_rank, [min_valscore, max_valscore]
        )
        fig.savefig(
            outputfolder + f"topSLgenepairs/{genepair}_ranks.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plot_genepair_valscore_vs_singleton(df, genepair)
        fig.savefig(
            outputfolder
            + f"topSLgenepairs/{genepair}_valscore_vs_singletonvalscore.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        df = df.groupby(["genePair", "gene1", "gene2", "cell_line"]).agg(
            {
                "valinor_score": "mean",
                "valinor_score_s_s_1": "mean",
                "valinor_score_s_s_2": "mean",
                "lfc": "mean",
                "lfc_s_1": "mean",
                "lfc_s_2": "mean",
            }
        )
        fig, ax = plot_genepair_singleton_lfc_vs_valscore(df, genepair)
        fig.savefig(
            outputfolder + f"topSLgenepairs/{genepair}_singleton_lfc_vs_valscore.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plot_genepair_combo_lfcs_vs_valscore(df, genepair)
        fig.savefig(
            outputfolder + f"topSLgenepairs/{genepair}_combo_lfcs_vs_valscore.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # make plots for global SL (among top SL in many cell lines)
    fig, ax = plot_globalgenepair_nclns(scoreData_combined, valscore_thr=None)
    fig.savefig(
        outputfolder + f"topSL_genepairs_number_cell_lines.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

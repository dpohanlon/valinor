import argparse

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

import random
import string

import seaborn as sns

colours = sns.color_palette()

from pprint import pprint

from tqdm import tqdm

import pandas as pd

import time

import pickle

from dko import DoubleKO
from contexts import (
    getContextMatrix,
    generate_context_matrices,
    assign_contexts_to_cell_lines,
)
from utils import negativeBinomial, genePairStr, populateCombinationDF
from singletons import makeSingletonsDF, makeSingletons

# Extend the ACE parameterisation to double KO

# Mutation matrix -> (mutation, essentiality factor)
# or cell type, tissue type, etc

# [[1, 1, ..., 2.0, 1, ..., -0.3, ...],
#  [1, 1, ..., -0.2, 1, ..., 1, ...]]


def genCellLine(
    prototypeDKO,
    resampleFrac=0.2,
    fluctuateStd=0.01,
    context=None,
    gi_contexts=None,
    gi_context_lists=None,
):
    # Simplest: Fluctuate a fraction of essentialities, re-generate the rest to simulate context differences
    # Fluctuate efficiencies

    # More complicated: Re-generate the rest according to 'driver' mutation matrix

    nGenes = prototypeDKO.nGenes
    resampleIdx = np.random.choice(
        range(nGenes), int(resampleFrac * nGenes), replace=False
    )

    newGeneEss = prototypeDKO.geneEssentiality + np.random.normal(
        0, fluctuateStd, len(prototypeDKO.geneEssentiality)
    )
    newGeneEss[resampleIdx] = np.random.normal(0.1, 0.2, size=len(resampleIdx))

    # If we're adding more context, don't include the prototype context GIs!

    if not (gi_contexts == None):

        newSyn = np.random.normal(0, fluctuateStd, size=prototypeDKO.synergies.shape)

    else:

        newSyn = prototypeDKO.synergies + np.random.normal(
            0, fluctuateStd, len(prototypeDKO.synergies)
        )

        newSyn[resampleIdx, :][:, resampleIdx] = np.random.normal(
            0.0, 0.1, size=(len(resampleIdx), len(resampleIdx))
        )

    newSyn = np.tril(newSyn) + np.tril(newSyn, -1).T

    newRNAEfficiencies = np.clip(
        prototypeDKO.sgRNAEfficiencies
        + np.random.normal(0, 0.01, len(prototypeDKO.sgRNAEfficiencies)),
        0,
        1,
    )

    # sgRNAEfficiencies1 = np.clip(
    #     np.random.normal(0.65, 0.02, size=nGenes * 1),
    #     0,
    #     1,
    # )
    #
    # sgRNAEfficiencies2 = np.clip(
    #     np.random.normal(0.95, 0.02, size=nGenes * 1),
    #     0,
    #     1,
    # )
    # newRNAEfficiencies = np.vstack(
    #     (sgRNAEfficiencies1, sgRNAEfficiencies2)
    # ).reshape((-1,), order="F")

    newPairEfficiency = np.clip(
        prototypeDKO.pairEfficiency
        + np.random.normal(0, 0.01, len(prototypeDKO.pairEfficiency)),
        0,
        1,
    )

    newDKO = DoubleKO(
        nGenes=prototypeDKO.nGenes,
        context=context,
        geneEssentiality=newGeneEss,
        synergies=newSyn,
        sgRNAEfficiencies=newRNAEfficiencies,
        pairEfficiency=newPairEfficiency,
        gi_contexts=gi_contexts,
        gi_context_lists=gi_context_lists,
        nGuidesPerGene=prototypeDKO.nGuidesPerGene,
    )

    return newDKO


def addCellLines(
    prototypeDKO,
    dfDKO,
    nCellLines,
    contexts=None,
    gi_contexts=None,
    gi_context_lists=None,
    resampleFrac=0.2,
    fluctuateStd=0.01,
    offsets=None,
    returnCounts=False,
):
    dfDKO["cell_line"] = 0

    dfs = [dfDKO]
    dkos = [prototypeDKO]

    for i in tqdm(range(1, nCellLines)):
        dko = genCellLine(
            prototypeDKO,
            resampleFrac,
            fluctuateStd,
            context=contexts[i] if not contexts is None else None,
            gi_contexts=gi_contexts,
            gi_context_lists=gi_context_lists[i],
        )

        df = populateCombinationDF(dko, returnCounts=returnCounts)

        if not (offsets is None):
            if not returnCounts:
                # Just offset according to the 'true' value as we don't have to
                # correct for essentiality ~0 but not == 0
                df["value"] += offsets[i]
            else:
                # Convert counts to LFC
                countsOffset = (
                    np.power(2, offsets[i]) * dko.nInitialCells - dko.nInitialCells
                )
                df["value"] += countsOffset[i]

        df["g1_idx"] += len(np.unique(df["g1_idx"])) * i
        df["g2_idx"] += len(np.unique(df["g2_idx"])) * i

        if "gene_pair_index" in df.columns:
            df["gene_pair_index"] += len(np.unique(df["gene_pair_index"])) * i

        df["cell_line"] = i

        dfs.append(df)
        dkos.append(dko)

    newDF = pd.concat(dfs)

    return newDF, dkos


def addReplicates(df, nReplicates, od=20, returnCounts=False):
    copies = []
    for i in tqdm(range(nReplicates)):
        replicate = df.copy()

        if not returnCounts:
            std = np.sqrt(1.0 / np.random.gamma(2, 5, size=len(replicate)))

            replicate["value"] = np.random.normal(replicate["value"].values, std)

        else:
            # replicate["value"] = np.random.poisson(replicate["value"])
            replicate["value"] = negativeBinomial(replicate["value"].values, od)

        copies.append(replicate)

    newDF = pd.concat(copies)

    return newDF


def makeDataset(
    nCellLines,
    nGenes,
    nContexts,
    nReplicates,
    nCalib,
    ncGenes,
    nGuidesPerGene,
    od,
    outDir,
):
    returnCounts = True
    nVariantFrac = 0.10

    t = time.time()

    # ms = getGIContextMatrix(num_genes = nGenes, num_contexts = nContexts, total_cell_lines = nCellLines, fraction_gene_pairs_in_context = 0.01)
    #
    # print(ms[0])
    # print(len(ms))
    # print(ms[0].shape)
    #
    # sns.heatmap(ms[0], cmap=sns.color_palette("vlag", as_cmap=True))
    # plt.ylabel("Gene")
    # plt.xlabel("Gene")
    # plt.savefig("contexts0.pdf")
    # plt.clf()
    #
    # sns.heatmap(ms[1], cmap=sns.color_palette("vlag", as_cmap=True))
    # plt.ylabel("Gene")
    # plt.xlabel("Gene")
    # plt.savefig("contexts1.pdf")
    # plt.clf()

    context_matrices = generate_context_matrices(nGenes, nContexts, 0.10, scale=0.1)
    cell_line_to_contexts = assign_contexts_to_cell_lines(
        nCellLines, nContexts, unique_contexts=True
    )

    sns.heatmap(context_matrices[0], cmap=sns.color_palette("vlag", as_cmap=True))
    plt.ylabel("Gene")
    plt.xlabel("Gene")
    plt.savefig("contexts0.pdf")
    plt.clf()

    sns.heatmap(context_matrices[1], cmap=sns.color_palette("vlag", as_cmap=True))
    plt.ylabel("Gene")
    plt.xlabel("Gene")
    plt.savefig("contexts1.pdf")
    plt.clf()

    pickle.dump(
        (cell_line_to_contexts, context_matrices), open("gi_contexts.pkl", "wb")
    )

    contexts = getContextMatrix(
        nCellLines, nGenes, nContexts, nVariantFrac, variance=0.1
    )

    contextsPlot = contexts.copy()
    contextsPlot[contextsPlot == 1.0] = np.nan

    # Maybe I should just always return counts (init, final) and just not use them
    # otherwise?

    # Make a symmetric range about 0
    maxVal = np.max(np.abs(contextsPlot.ravel()))

    # sns.heatmap(contextsPlot, cmap=sns.color_palette("vlag", as_cmap=True), vmin = -maxVal, vmax = maxVal)
    # plt.ylabel("Cell line")
    # plt.xlabel("Gene")
    # plt.savefig("contexts.pdf")
    # plt.clf()

    dko = DoubleKO(
        nGenes=nGenes,
        context=contexts[0],
        gi_contexts=context_matrices,
        gi_context_lists=cell_line_to_contexts[0],
        # sgRNAEfficiencies=sgRNAEfficiencies,
        nGuidesPerGene=nGuidesPerGene,
        od=od,
    )

    # sns.heatmap(dko.synergies[:10, :10], cmap=sns.color_palette("vlag", as_cmap=True), vmin = -0.17, vmax = 0.17)
    # plt.xlabel('Gene 1')
    # plt.ylabel('Gene 2')
    # plt.savefig('syn_mat.png', dpi = 300)
    # exit(0)

    final_counts, sgRNAEssentialities, init_counts, lfcs = dko.getLFCs(
        returnCounts=returnCounts
    )

    dko.plot(
        lfcs, sgRNAEssentialities, init_counts=init_counts, final_counts=final_counts
    )

    dfCombs = populateCombinationDF(dko, returnCounts=returnCounts)

    print("adding cell lines", time.time() - t)
    dfCombs, dkos = addCellLines(
        dko,
        dfCombs,
        nCellLines,
        contexts=contexts,
        gi_contexts=context_matrices,
        gi_context_lists=cell_line_to_contexts,
        # offsets=offsets,
        returnCounts=returnCounts,
    )

    # if not returnCounts:
    print("adding combo replicates", time.time() - t)
    dfCombs = addReplicates(dfCombs, nReplicates, od=od, returnCounts=returnCounts)

    # So the replicates can be projected out
    # Sloooow
    dfCombs = dfCombs.sort_values(
        ["guide_pair_index", "guide1_index", "cell_line"]
    ).reset_index()

    # The same by our new definition
    dfCombs["gene_unq_pair_index"] = dfCombs["gene_pair_index"]
    dfCombs["cell_line_index"] = dfCombs["cell_line"]
    dfCombs["gene1_unq_index"] = dfCombs["g1_idx"]
    dfCombs["gene2_unq_index"] = dfCombs["g2_idx"]

    dfCombs.to_parquet(f"{outDir}/dfCombs_ace.pq")

    print("making singletons", time.time() - t)
    dfSgl = makeSingletonsDF(
        dkos,
        # offsets=offsets,
        returnCounts=returnCounts,
        ncGenes=ncGenes,
    )

    # Offset so they don't clash with the combs
    dfSgl["guide_pair_index"] = (
        dfSgl["guide_pair_index"] + np.max(dfCombs["guide_pair_index"]) + 1
    )

    # if not returnCounts:
    print("adding singleton replicates", time.time() - t)
    dfSgl = addReplicates(dfSgl, nReplicates, returnCounts=returnCounts)

    dfSgl["cell_line_index"] = dfCombs["cell_line"]
    # The same by our new definition
    dfSgl["gene_unq_pair_index"] = dfSgl["gene_pair_index"]
    dfSgl["gene1_unq_index"] = dfSgl["g1_idx"]
    dfSgl["guide1_index"] = dfSgl["guide1_index_s"]

    dfSgl = dfSgl.sort_values(
        ["guide1_index_s", "guide2_index_s", "cell_line"]
    ).reset_index()
    dfSgl.to_parquet(f"{outDir}/dfSgl_ace.pq")

    # Offsets to test calibration

    # The true offset, not one approximated by essentiality close to zero
    offsets = np.concatenate(
        (np.zeros(1), np.random.uniform(-0.5, 0.5, nCellLines - 1))
    )

    calibData = np.array(
        dko.calibrationLFCs(
            offsets, negControlLFCWidth=0.01, n=nCalib, returnCounts=returnCounts
        )
    )

    plt.hist(calibData.ravel(), bins=100)
    plt.savefig("calib.pdf")
    plt.clf()

    # Offset according to singletons (and therefore also combinations)
    guide_pair_index_c = (
        np.array(len(calibData[:, 0, :].ravel()))
        + np.max(dfCombs["guide_pair_index"])
        + 1
    )

    dfCalib = pd.DataFrame(
        {
            "plasmid": calibData[:, 0, :].ravel(),
            "value": calibData[:, 1, :].ravel(),
            "value_pos": calibData[:, 2, :].ravel(),
            "guide_pair_index": guide_pair_index_c,
            "cell_line": np.tile(range(nCellLines), [nCalib, 1]).T.ravel(),
        }
    )

    dfCalib["cell_line_index"] = dfCalib["cell_line"]

    dfCalib.to_parquet(f"{outDir}/dfCalib_ace.pq")

    offsetsCounts = np.zeros(len(offsets))
    for i, dko in enumerate(dkos):
        offsetsCounts[i] = (
            np.power(2, offsets[i]) * dko.nInitialCellsV - dko.nInitialCellsV
        )

    # Save as LFC not as counts - recalculate these!
    dfOffsets = pd.DataFrame(
        {
            "offsets": offsets.ravel(),
            "offsetsCounts": offsetsCounts.ravel(),
        }
    )
    dfOffsets.to_parquet(f"{outDir}/dfOffsets_ace.pq")


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "--out-dir",
        type=str,
        dest="out_dir",
        default=".",
        help="Output directory for the simulated data.",
    )

    argParser.add_argument(
        "--nGenes",
        type=int,
        dest="nGenes",
        default=100,
        help="Number of genes (to form all-to-all pairs).",
    )

    argParser.add_argument(
        "--nCellLines",
        type=int,
        dest="nCellLines",
        default=10,
        help="Total number of cell lines.",
    )

    argParser.add_argument(
        "--nContexts",
        type=int,
        dest="nContexts",
        default=5,
        help="Total number of cell line contexts.",
    )

    argParser.add_argument(
        "--nReplicates",
        type=int,
        dest="nReplicates",
        default=3,
        help="Total number of replicates per guide pair.",
    )

    argParser.add_argument(
        "--nCalib",
        type=int,
        dest="nCalib",
        default=100,
        help="Number of negative control (null calibration) pairs.",
    )

    argParser.add_argument(
        "--ncGenes",
        type=int,
        dest="ncGenes",
        default=3,
        help="Number of negative control (null calibration) genes in singletons.",
    )

    argParser.add_argument(
        "--nGuidesPerGene",
        type=int,
        dest="nGuidesPerGene",
        default=2,
        help="Number of sgRNA guides per gene.",
    )

    argParser.add_argument(
        "--od",
        type=int,
        dest="od",
        default=20,
        help="Final count overdispersion.",
    )

    args = argParser.parse_args()

    makeDataset(
        nGenes=args.nGenes,
        nCellLines=args.nCellLines,
        nContexts=args.nContexts,
        nReplicates=args.nReplicates,
        nCalib=args.nCalib,
        ncGenes=args.ncGenes,
        nGuidesPerGene=args.nGuidesPerGene,
        od=args.od,
        outDir=args.out_dir,
    )

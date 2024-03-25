import numpy as np
import pandas as pd

import re

from pprint import pprint

import argparse

from tqdm import tqdm


def populateIndicesControls(controlsM, indices):

    # Can construct these there as they aren't shared with other data types
    guide_pair_index = {c: i for i, c in enumerate(np.unique(controlsM["GuidePair"]))}

    controlsM["guide_pair_index"] = controlsM["GuidePair"].map(
        lambda x: guide_pair_index[x]
    )

    # But the cell line indices are, so use the global ones

    controlsM["cell_line_index"] = controlsM["cell_line"].map(
        lambda x: indices["cell_line_index"][x]
    )

    return controlsM


def populateIndicesSingletons(singlesM, indices):

    singlesM["guide_pair_index"] = singlesM["GuidePair"].map(
        lambda x: indices["guide_pair_s_index"][x]
    )

    # Add indices to DataFrame

    singlesM["cell_line_index"] = singlesM["cell_line"].map(
        lambda x: indices["cell_line_index"][x]
    )

    singlesM["gene1_index"] = singlesM["SingletonGene"].map(
        lambda x: indices["gene_index"][x]
    )

    for i in range(len(singlesM["SingletonGuide_o"])):
        g = singlesM["SingletonGuide_o"].iloc[i]
        if not (g in indices["guide_index_ori"]):
            print(singlesM["SingletonGene"].iloc[i])

    # _o -> sensitive to order of guides
    # singlesM['guide1_index'] = singlesM['SingletonGuide'].map(lambda x : guide_index[x])
    singlesM["guide1_index"] = singlesM["SingletonGuide_o"].map(
        lambda x: indices["guide_index_ori"][x]
    )

    # With separate combination/singleton efficiencies
    singlesM["guide1_index_s"] = singlesM["SingletonGuide_o_s"].map(
        lambda x: indices["guide_index_ori_s"][x]
    )

    singlesM["gene1_unq_index"] = (
        singlesM["SingletonGene"] + "_" + singlesM["cell_line"]
    ).map(lambda x: indices["gene_unq_index"][x])

    return singlesM


def populateIndicesCombs(dataM, indices):

    dataM["cell_line_index"] = dataM["cell_line"].map(
        lambda x: indices["cell_line_index"][x]
    )

    dataM["gene1_index"] = dataM["gene1"].map(lambda x: indices["gene_index"][x])
    dataM["gene2_index"] = dataM["gene2"].map(lambda x: indices["gene_index"][x])
    dataM["gene_pair_index"] = dataM["genePairUnoriented"].map(
        lambda x: indices["gene_pair_index"][x]
    )

    dataM["guide_pair_index"] = dataM["GuidePair"].map(
        lambda x: indices["guide_pair_index"][x]
    )
    dataM["guide1_index"] = dataM["guide1_o"].map(
        lambda x: indices["guide_index_ori"][x]
    )
    dataM["guide2_index"] = dataM["guide2_o"].map(
        lambda x: indices["guide_index_ori"][x]
    )

    # With separate combination/singleton efficiencies
    dataM["guide1_index_s"] = dataM["guide1_o"].map(
        lambda x: indices["guide_index_ori_s"][x]
    )
    dataM["guide2_index_s"] = dataM["guide2_o"].map(
        lambda x: indices["guide_index_ori_s"][x]
    )

    dataM["gene1_unq_index"] = (dataM["gene1"] + "_" + dataM["cell_line"]).map(
        lambda x: indices["gene_unq_index"][x]
    )
    dataM["gene2_unq_index"] = (dataM["gene2"] + "_" + dataM["cell_line"]).map(
        lambda x: indices["gene_unq_index"][x]
    )
    dataM["gene_unq_pair_index"] = (
        dataM["genePairUnoriented"] + "_" + dataM["cell_line"]
    ).map(lambda x: indices["gene_unq_pair_index"][x])

    dataM["gene1_unq_ori_index"] = (
        dataM["gene1"] + "_" + dataM["cell_line"] + "_1"
    ).map(lambda x: indices["gene_unq_ori_index"][x])
    dataM["gene2_unq_ori_index"] = (
        dataM["gene2"] + "_" + dataM["cell_line"] + "_2"
    ).map(lambda x: indices["gene_unq_ori_index"][x])
    dataM["gene_unq_pair_ori_index"] = (
        dataM["genePairUnoriented"]
        + "_"
        + dataM["cell_line"]
        + "_"
        + dataM["gene1"]
        + "_"
        + dataM["gene2"]
    ).map(lambda x: indices["gene_unq_pair_ori_index"][x])

    return dataM


def calculateIndices(dataM, singlesM):

    # Compute indices for model

    cell_line_index = {c: i for i, c in enumerate(np.unique(dataM["cell_line"]))}

    # Gene pairs unoriented - gene effect should be the same

    gene_index = {
        c: i
        for i, c in enumerate(
            np.unique(np.concatenate((dataM["gene1"], dataM["gene2"])))
        )
    }

    dataM["gene1_o"] = dataM["gene1"] + "_1"
    dataM["gene2_o"] = dataM["gene2"] + "_2"

    gene_index_ori = {
        c: i
        for i, c in enumerate(
            np.unique(np.concatenate((dataM["gene1_o"], dataM["gene2_o"])))
        )
    }

    # Unoriented
    guide_index = {
        c: i
        for i, c in enumerate(
            np.unique(np.concatenate((dataM["guide1"], dataM["guide2"])))
        )
    }

    # Oriented - tag the position to make them unique when calculating the indices

    dataM["guide1_o"] = dataM["guide1"] + "_1"
    dataM["guide2_o"] = dataM["guide2"] + "_2"

    # Sometimes guides appear in singletons but not in combinations, so add these too
    # (e.g., 4 guides vs 2 guides)

    guide_index_ori = {
        c: i
        for i, c in enumerate(
            np.unique(
                np.concatenate(
                    (dataM["guide1_o"], dataM["guide2_o"], singlesM["SingletonGuide_o"])
                )
            )
        )
    }

    # Oriented, with singletons separated

    guide_index_ori_s = {
        c: i
        for i, c in enumerate(
            np.unique(
                np.concatenate(
                    (
                        dataM["guide1_o"],
                        dataM["guide2_o"],
                        singlesM["SingletonGuide_o_s"],
                    )
                )
            )
        )
    }

    gene_pair_index = {
        c: i for i, c in enumerate(np.unique(dataM["genePairUnoriented"]))
    }

    # Unique for each gene (pair), cell line combinations
    gene_unq_index = {
        c: i
        for i, c in enumerate(
            np.unique(
                np.concatenate(
                    (
                        dataM["gene1"] + "_" + dataM["cell_line"],
                        dataM["gene2"] + "_" + dataM["cell_line"],
                    )
                )
            )
        )
    }

    gene_unq_ori_index = {
        c: i
        for i, c in enumerate(
            np.unique(
                np.concatenate(
                    (
                        dataM["gene1"] + "_" + dataM["cell_line"] + "_1",
                        dataM["gene2"] + "_" + dataM["cell_line"] + "_2",
                    )
                )
            )
        )
    }

    gene_unq_pair_index = {
        c: i
        for i, c in enumerate(
            np.unique(dataM["genePairUnoriented"] + "_" + dataM["cell_line"])
        )
    }

    gene_unq_pair_ori_index = {
        c: i
        for i, c in enumerate(
            np.unique(
                dataM["genePairUnoriented"]
                + "_"
                + dataM["cell_line"]
                + "_"
                + dataM["gene1"]
                + "_"
                + dataM["gene2"]
            )
        )
    }

    # Guide pairs oriented - corresponds to data points in the counts array,
    # which are different measurements (and we assume here that guide order
    # matters due to pyogenes and aureus differences)

    # Genes and single guides are shared with the combinations, but the guide pairs
    # (plasmids) of singletons are not (and have different pDNA counts, etc)
    # When this is fitted with the combinations, offset so that the indices don't clash,
    # but when singletons are on their own leave them starting from 0

    guide_pair_s_index = {c: i for i, c in enumerate(np.unique(singlesM["GuidePair"]))}

    guide_pair_index = {c: i for i, c in enumerate(np.unique(dataM["GuidePair"]))}

    print("cell_line_index", len(cell_line_index))
    print("gene_index", len(gene_index))
    print("gene_pair_index", len(gene_pair_index))
    print("guide_index", len(guide_index))
    print("guide_pair_index", len(guide_pair_index))
    print("gene_unq_index", len(gene_unq_index))
    print("gene_unq_pair_index", len(gene_unq_pair_index))

    for c in cell_line_index.keys():
        print(
            c,
            np.unique(dataM[dataM["cell_line"] == c]["replicate"], return_counts=True),
        )

    return {
        "cell_line_index": cell_line_index,
        "gene_index": gene_index,
        "gene_pair_index": gene_pair_index,
        "guide_index": guide_index,
        "guide_pair_index": guide_pair_index,
        "gene_unq_index": gene_unq_index,
        "gene_unq_ori_index": gene_unq_ori_index,
        "gene_unq_pair_index": gene_unq_pair_index,
        "gene_unq_pair_ori_index": gene_unq_pair_ori_index,
        "guide_index_ori": guide_index_ori,
        "guide_index_ori_s": guide_index_ori_s,
        "guide_pair_s_index": guide_pair_s_index,
    }


def pairUnoriented(gset, g1, g2):

    p = g1 + "_" + g2
    rp = g2 + "_" + g1

    if p in gset:
        return p
    elif rp in gset:
        return rp
    else:
        gset.add(p)
        return p


def normReps(data):

    totals = data.groupby("replicate").agg({"value": "sum"}).reset_index()
    totals = totals.rename(columns={"value": "total_value"})

    normedData = data.merge(totals, on="replicate")

    normedData["value_norm"] = (
        normedData["value"] / normedData["total_value"]
    ) * np.mean(normedData["total_value"])

    return normedData


def loadData(countsFile, library, controls):

    # inputs
    # non essentials list controls
    # controls annotation (0 0 and 0 gene)

    data = pd.read_csv(countsFile, low_memory=False)

    # Transform and fill entries

    genePairSet = set()
    guidePairSet = set()

    # Also have a passenger vars here?

    id_vars = [
        "guide1",
        "guide2",
        "plasmid",
        "gene1",
        "gene2",
        "sgRNA",
    ]

    dataM = pd.melt(data, id_vars=id_vars)

    dataM["lfc_scaled"] = calculateScaledLFC(dataM)

    dataM["lfc_norm_scaled"] = calculateScaledLFC(
        dataM, valueVar="value_norm", lfcVar="lfc_norm"
    )

    dataM["genePair"] = dataM["gene1"] + "_" + dataM["gene2"]
    dataM["genePairUnoriented"] = [
        pairUnoriented(genePairSet, a, p) for a, p in dataM[["gene1", "gene2"]].values
    ]

    dataM["GuidePair"] = dataM["guide1"] + "_" + dataM["guide2"]
    dataM["GuidePairUnoriented"] = [
        pairUnoriented(guidePairSet, *p.split("_")) for p in dataM["GuidePair"].values
    ]

    # Remove self combos

    dataM = dataM[dataM["gene1"] != dataM["gene2"]]

    controlsM = dataM[dataM["controls"] == 1]

    singlesM = dataM[dataM["singles"] == 1]

    dataM = dataM[dataM["combinations"] == 1]

    singlesM["SingletonPosition"] = [
        "1" if x[0] not in controls else "2"
        for x in singlesM[["gene1", "gene2"]].values
    ]
    singlesM["SingletonGene"] = [
        x[0] if x[0] not in controls else x[1]
        for x in singlesM[["gene1", "gene2"]].values
    ]
    singlesM["SingletonGuide"] = [
        x[2] if x[0] not in controls else x[3]
        for x in singlesM[["gene1", "gene2", "guide1", "guide2"]].values
    ]
    singlesM["SingletonGuide_o"] = (
        singlesM["SingletonGuide"] + "_" + singlesM["SingletonPosition"]
    )
    singlesM["SingletonGuide_o_s"] = singlesM["SingletonGuide_o"] + "_s"

    return dataM, singlesM, controlsM


def calculateScaling(
    data,
    lfcVar="lfc",
    posControlsVar="PositiveControls",
    negControlsVar="NegativeControls",
    controlsID="Control",
):

    posControls = data[data[controlsID] == posControlsVar].copy()
    negControls = data[data[controlsID] == negControlsVar].copy()

    return np.median(posControls[lfcVar].values), np.median(negControls[lfcVar].values)


def applyScaling(posMedian, negMedian, lfc):

    return (negMedian - lfc) / (posMedian - negMedian)


def calculateScaledLFC(
    data,
    valueVar="value",
    plasmidVar="plasmid",
    lfcVar="lfc",
    posControlsVar="PositiveControls",
    negControlsVar="NegativeControls",
    controlsID="Control",
):

    data[lfcVar] = np.log2((data[valueVar].values + 1) / (data[plasmidVar].values + 1))

    posControls = {}
    negControls = {}

    # Assuming cell_lines are experiments
    for n, g in data.groupby("cell_line"):
        pos, neg = calculateScaling(g)
        posControls[n] = pos
        negControls[n] = neg

    data["negMedian"] = data["cell_line"].map(lambda x: negControls[x])
    data["posMedian"] = data["cell_line"].map(lambda x: posControls[x])

    lfc_scaled = applyScaling(
        data["posMedian"].values, data["negMedian"].values, data[lfcVar].values
    )

    return lfc_scaled


# Normalise so that similar counts across experiments correspond to similar
# effects. No need to do so over cell lines - only one!
def normLibraries(data, singles, controls):

    # Concat these at first to get totals

    allData = pd.concat(
        (
            data[["GuidePair", "value", "library", "plasmid"]],
            singles[["GuidePair", "value", "library", "plasmid"]],
            controls[["GuidePair", "value", "library", "plasmid"]],
        ),
        axis=0,
    )

    # Final counts

    totals = allData.groupby("library").agg({"value": "sum"}).reset_index()
    totals = totals.rename(columns={"value": "total_library_value"})[
        ["library", "total_library_value"]
    ]

    normedData = data.merge(totals, on="library")
    normedSingles = singles.merge(totals, on="library")
    normedControls = controls.merge(totals, on="library")

    normedData["value_norm_library"] = (
        normedData["value"] / normedData["total_library_value"]
    ) * np.median(normedData["total_library_value"])
    normedSingles["value_norm_library"] = (
        normedSingles["value"] / normedSingles["total_library_value"]
    ) * np.median(normedSingles["total_library_value"])
    normedControls["value_norm_library"] = (
        normedControls["value"] / normedControls["total_library_value"]
    ) * np.median(normedControls["total_library_value"])

    # Plasmids - want to drop replicates, etc, first to avoid overcounting
    plasmid_data = allData[["GuidePair", "library", "plasmid"]].drop_duplicates()
    totals_plasmid = (
        plasmid_data.groupby("library").agg({"plasmid": "sum"}).reset_index()
    )
    totals_plasmid = totals_plasmid.rename(
        columns={"plasmid": "plasmid_total_library_value"}
    )[["library", "plasmid_total_library_value"]]

    normedData = normedData.merge(totals_plasmid, on="library")
    normedSingles = normedSingles.merge(totals_plasmid, on="library")
    normedControls = normedControls.merge(totals_plasmid, on="library")

    normedData["plasmid_norm"] = (
        normedData["plasmid"] / normedData["plasmid_total_library_value"]
    ) * np.median(normedData["plasmid_total_library_value"])
    normedSingles["plasmid_norm"] = (
        normedSingles["plasmid"] / normedSingles["plasmid_total_library_value"]
    ) * np.median(normedSingles["plasmid_total_library_value"])
    normedControls["plasmid_norm"] = (
        normedControls["plasmid"] / normedControls["plasmid_total_library_value"]
    ) * np.median(normedControls["plasmid_total_library_value"])

    return normedData, normedSingles, normedControls


def processInputs(countsFiles, outDir, libraries, name):

    dataM_list = []
    singlesM_list = []
    controlsM_list = []

    for library, countsFile in tqdm(
        zip(libraries, countsFiles), total=len(libraries), colour="green"
    ):

        d, s, c = loadData(
            countsFile,
            library,
        )

        if len(library) > 1:

            d.to_parquet(f"{outDir}/encore-{library}-combs.pq")
            s.to_parquet(f"{outDir}/encore-{library}-singles.pq")
            c.to_parquet(f"{outDir}/encore-{library}-controls.pq")

        d["library"] = library
        s["library"] = library
        c["library"] = library

        dataM_list.append(d)
        singlesM_list.append(s)
        controlsM_list.append(c)

    dataM = pd.concat(dataM_list)
    singlesM = pd.concat(singlesM_list)
    controlsM = pd.concat(controlsM_list)

    dataM, singlesM, controlsM = normLibraries(dataM, singlesM, controlsM)

    dataM["value"] = dataM["value_norm_library"]
    singlesM["value"] = singlesM["value_norm_library"]
    controlsM["value"] = controlsM["value_norm_library"]

    dataM["plasmid_raw"] = dataM["plasmid"]
    singlesM["plasmid_raw"] = singlesM["plasmid"]
    controlsM["plasmid_raw"] = controlsM["plasmid"]

    dataM["plasmid"] = dataM["plasmid_norm"]
    singlesM["plasmid"] = singlesM["plasmid_norm"]
    controlsM["plasmid"] = controlsM["plasmid_norm"]

    dataM = calculateScaledLFC(
        dataM, valueVar="value", plasmidVar="plasmid", lfcVar="lfc"
    )
    singlesM = calculateScaledLFC(
        singlesM, valueVar="value", plasmidVar="plasmid", lfcVar="lfc"
    )
    controlsM = calculateScaledLFC(
        controlsM, valueVar="value", plasmidVar="plasmid", lfcVar="lfc"
    )

    # Controls don't have 'genes', as they're assumed that gene effect = 0
    indices = calculateIndices(dataM, singlesM)

    dataM = populateIndicesCombs(dataM, indices)

    singlesM = populateIndicesSingletons(singlesM, indices)

    controlsM = populateIndicesControls(controlsM, indices)

    dataM.to_parquet(f"{outDir}/{name}-combs.pq")
    singlesM.to_parquet(f"{outDir}/{name}-singles.pq")
    controlsM.to_parquet(f"{outDir}/{name}-controls.pq")


if __name__ == "__main__":

    # I'd like an argument, please
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-c",
        type=str,
        dest="countsFiles",
        nargs="+",
        default="",
        help="Counts CSV file(s)",
    )
    argParser.add_argument(
        "-o",
        type=str,
        dest="outDir",
        default=".",
        help="Output directory.",
    )
    argParser.add_argument(
        "-n",
        type=str,
        dest="name",
        default=".",
        help="Output file name.",
    )
    argParser.add_argument(
        "-l",
        "--libraries",
        type=str,
        dest="libraries",
        nargs="+",
        default="COLO1",
        help="Library name.",
    )

    args = argParser.parse_args()

    processInputs(args.countsFiles, args.outDir, args.libraries, args.name)

import jax.numpy as jnp

import pandas as pd

import numpy as np

from functools import partial

import h5py

import yaml

import json

from copy import deepcopy

from numpyro.infer.initialization import init_to_median

from typing import Dict, List, Tuple, Optional

# TODO: Have a better interface to these, especially when first building them
# so that it generalises to more parameters and categories


def loadData(data_files: Dict[str, str]):
    outFiles = {}

    for n, f in data_files.items():
        if f is None:
            d = None
        elif "pq" in f:
            d = pd.read_parquet(f)
        elif "h5" in f:
            d = pd.read_hdf(f)
        else:
            print("File format not recognised:", f)

        outFiles[n] = d

    return outFiles


def loadPriors(priorsFile: str):
    with open(priorsFile) as f:
        priors = yaml.safe_load(f)

    return priors


def getFinalCounts(datasets: Dict[str, str], finalCountVar: str = "value"):
    counts = {}

    for n, d in datasets.items():
        counts[n] = (
            jnp.array(d["value"].values.reshape(-1)) if not (d is None) else None
        )

    return counts


def getInitialCounts(datasets: Dict[str, str], initCountVar: str = "plasmid"):
    counts = {}
    count_indices = {}

    for n, d in datasets.items():

        if not (d is None):

            singletons = "singletons" in n.lower()
            count, indices = getInitialCountsDF(d, initCountVar, singletons)
            counts[n] = count
            count_indices[n] = indices

        else:

            counts[n] = None
            count_indices[n] = None

    return counts, count_indices


# def getInitialCountsDF(df: pd.DataFrame, initCountVar: str, singletons : bool = False) -> pd.Series:
#     """
#     Get initial counts from the DataFrame.

#     Args:
#         df (pd.DataFrame): The input DataFrame.
#         initCountVar (str): The variable for initial count.

#     Returns:
#         pd.Series: Initial counts.
#     """

# # initial_counts_prior = (
# #     data_s.groupby('guide1_index')
# #     .agg({'plasmid': 'mean', 'guide_pair_index': 'first'})
# #     .reset_index()
# #     .sort_values('guide1_index')
# # )

# # t = initial_counts_prior['plasmid'].values[data_s['guide1_index'].values]


#     guideVar = "guide_pair_index" if not singletons else "guide1_index"

#     initial_counts = (
#             df.groupby(guideVar)
#             .agg({"guide_pair_index": "first", initCountVar : "first", guideVar : 'first'})
#             .reset_index(drop=True)
#             .sort_values(guideVar)[[initCountVar, guideVar, 'guide_pair_index']]
#     )

#     return initial_counts[initCountVar].values[df[guideVar].values], initial_counts[guideVar].values.astype(np.int32)

def getInitialCountsDF(df: pd.DataFrame, initCountVar: str, singletons : bool = False) -> pd.Series:
    """
    Get initial counts from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        initCountVar (str): The variable for initial count.

    Returns:
        pd.Series: Initial counts.
    """

    guideVar = "guide_pair_index" if not singletons else "guide1_index"

    initial_counts = (
            df.groupby("guide_pair_index")
            .agg({"guide_pair_index": "first", initCountVar : "first", guideVar : 'first'})
            .reset_index(drop=True)
            .sort_values("guide_pair_index")[[initCountVar, guideVar]]
    )

    return initial_counts[initCountVar].values.astype(np.int32), initial_counts[guideVar].values.astype(np.int32)

def getUniqueGeneGuideIndices(
    indices: Dict[str, np.ndarray],
    singletons: bool = True,
    only_singletons=False,
    shared = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get unique gene and guide indices.

    Args:
        indices (Dict[str, np.ndarray]): Dictionary of indices.
        singletons (bool, optional): If True, include singleton indices. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Unique gene indices and guide indices.
    """

    guide_indices = [
        indices["guide_1_idx"],
    ]
    gene_indices = [
        indices["gene_1_idx"] if not shared else indices["gene_1_common_idx"],
    ]

    if not only_singletons:
        guide_indices += [indices["guide_2_idx"]]
        gene_indices += [indices["gene_2_idx"] if not shared else indices["gene_2_common_idx"]]

    if singletons:
        guide_indices += [indices["guide_s_idx"]]
        gene_indices += [indices["gene_s_idx"] if not shared else indices["gene_s_common_idx"]]

    return np.unique(np.concatenate(gene_indices)), np.unique(
        np.concatenate(guide_indices)
    )

# Only for final counts
def countZeros(df):

    data_sorted = df.sort_values('cell_line_index').reset_index()
    counts = data_sorted.groupby('cell_line_index').size()
    zero_counts = data_sorted[data_sorted['value'] == 0].groupby('cell_line_index').size().reindex(counts.index, fill_value=0)

    zeros_frac = zero_counts.values / counts.values

    return zeros_frac + 1E-6

def reindexVar(df, var):

    oldIndicesUnq = df[var].unique()
    newIndexMap = {oldIndicesUnq[i]: i for i in range(len(oldIndicesUnq))}
    newIndices = np.array([newIndexMap[v] for v in df[var].values])

    return newIndices


def reindexDF(df: pd.DataFrame, singletons=False):

    # Reindexes to avoid cases where guides appear in only the combinations/singles
    # dataset. Not to be used for a model with matched combinations and singles!

    # But there is still a global guide pair index, so be careful with controls....

    df["guide_pair_index"] = reindexVar(df, "guide_pair_index")
    df["guide1_index"] = reindexVar(df, "guide1_index")
    df["gene1_unq_index"] = reindexVar(df, "gene1_unq_index")
    df["cell_line_index"] = reindexVar(df, "cell_line_index")

    if not singletons:

        df["gene_pair_index"] = reindexVar(df, "gene_pair_index")
        df["guide2_index"] = reindexVar(df, "guide2_index")
        df["gene2_unq_index"] = reindexVar(df, "gene2_unq_index")

    return df


def getIndices(
    df: pd.DataFrame,
    dfSingles: Optional[pd.DataFrame] = None,
    dfControls: Optional[pd.DataFrame] = None,
    only_singletons=False,
    singletons=True,
    controls=True,
) -> Dict[str, np.ndarray]:
    """
    Get indices from the dataframes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        dfSingles (pd.DataFrame, optional): DataFrame of singletons. Defaults to None.
        dfControls (pd.DataFrame, optional): DataFrame of controls. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: Dictionary of indices.
    """

    indices = {
        "guide_pair_idx": jnp.array(
            df["guide_pair_index"].values
            if not only_singletons
            else dfSingles["guide_pair_index"].values
        ),

        "guide_pair_unq_idx": jnp.array(
            df["guide_pair_unq_index"].values
            if not only_singletons
            else dfSingles["guide_pair_unq_index"].values
        ),
        "guide_1_idx": jnp.array(
            df["guide1_index"].values
            if not only_singletons
            else dfSingles["guide1_index"].values
        ),
        "guide_1_unq_idx": jnp.array(
            df["guide1_unq_index"].values
            if not only_singletons
            else dfSingles["guide1_unq_index"].values
        ),
        "gene_1_idx": jnp.array(
            df["gene1_unq_index"].values
            if not only_singletons
            else dfSingles["gene1_unq_index"].values
        ),
        # These should be the same value for the same gene across
        # all cell lines, rather than different per cell line
        # like gene_1_idx
        "gene_1_common_idx": jnp.array(
            df["gene1_index"].values
            if not only_singletons
            else dfSingles["gene1_index"].values
        ),
        "cell_line_idx": jnp.array(
            df["cell_line_index"].values
            if not only_singletons
            else dfSingles["cell_line_index"].values
        ),
    }

    if not only_singletons:
        indices["gene_2_idx"] = jnp.array(df["gene2_unq_index"].values)
        indices["gene_2_common_idx"] = jnp.array(df["gene2_index"].values)
        indices["guide_2_idx"] = jnp.array(df["guide2_index"].values)
        indices["guide_2_unq_idx"] = jnp.array(df["guide2_unq_index"].values)
        indices["gene_pair_idx"] = jnp.array(df["gene_unq_pair_index"].values)

        # Rather than indexing for observations, these index for the gene pairs array to specify which gene index corresponds to this pair

        # At the moment these take the first ocurrance to be the gene indices, so if this contains reversed orderings as well this gets ignored

        pair_indices = df.drop_duplicates(subset='gene_unq_pair_index', keep='first')[['gene_unq_pair_index', 'gene1_unq_index', 'gene2_unq_index', 'cell_line_index']].sort_values('gene_unq_pair_index').reset_index(drop=True)

        indices['cell_line_in_pair_idx'] = jnp.array(pair_indices['cell_line_index'].values)
        indices['gene_1_in_pair_idx'] = jnp.array(pair_indices['gene1_unq_index'].values)
        indices['gene_2_in_pair_idx'] = jnp.array(pair_indices['gene2_unq_index'].values)

    if singletons:
        indices["guide_pair_s_idx"] = jnp.array(dfSingles["guide_pair_index"].values)
        indices["guide_pair_unq_s_idx"] = jnp.array(dfSingles["guide_pair_unq_index"].values)

        indices["guide_s_idx"] = jnp.array(dfSingles["guide1_index"].values)
        indices["guide_unq_s_idx"] = jnp.array(dfSingles["guide1_unq_index"].values)

        indices["gene_s_idx"] = jnp.array(dfSingles["gene1_unq_index"].values)

        indices["gene_s_common_idx"] = jnp.array(dfSingles["gene1_index"].values)

        indices["cell_line_s_idx"] = jnp.array(dfSingles["cell_line_index"].values)

    if controls:
        indices["cell_line_c_idx"] = jnp.array(dfControls["cell_line_index"].values)
        indices["guide_pair_c_idx"] = jnp.array(dfControls["guide_pair_index"].values)
        indices["guide_pair_unq_c_idx"] = jnp.array(dfControls["guide_pair_unq_index"].values)

    # Make sure these are all 0D
    for k, v in indices.items():
        indices[k] = v.reshape(-1)

    return indices


def calculateLengths(
    indices: Dict[str, np.ndarray],
    singletons: bool = True,
    neg_controls: bool = True,
    only_singletons=False,
) -> Dict[str, int]:
    """
    Calculate lengths from indices.

    Args:
        indices (Dict[str, np.ndarray]): Dictionary of indices.
        singletons (bool, optional): If True, include singleton lengths. Defaults to True.
        neg_controls (bool, optional): If True, include negative control lengths. Defaults to True.

    Returns:
        Dict[str, int]: Dictionary of lengths.
    """

    # Calculate Numpyro parameter array lengths from indices

    # Without 'unq', guide pairs are the length of initial plasmid counts
    # Otherwise they are the unique pairs over all cell lines
    lengths = {
        "len_cell_lines": len(np.unique(indices["cell_line_idx"])),
        "len_guide_pairs": len(np.unique(indices["guide_pair_idx"])),
        "len_guide_pairs_unq": len(np.unique(indices["guide_pair_unq_idx"])),
    }

    if not only_singletons:

        lengths["len_gene_pairs"] = len(np.unique(indices["gene_pair_idx"]))

    gene_indices, guide_indices = getUniqueGeneGuideIndices(
        indices, singletons=singletons, only_singletons=only_singletons
    )

    if singletons:
        # lengths["len_guide_pairs_s"] = len(np.unique(indices["guide_pair_s_idx"]))

        # Unique guides, not including duplicates with different nulls, as these are not parameterised

        # Ideally I want the same thing for every dataset....

        # FIX ME

        lengths["len_guide_pairs_s"] = len(np.unique(indices["guide_s_idx"]))

    if neg_controls:
        lengths["len_guide_pairs_c"] = len(np.unique(indices["guide_pair_c_idx"]))
        lengths["len_guide_pairs_unq_c"] = len(np.unique(indices["guide_pair_unq_c_idx"]))

    # Unique guides, including each category in case we have unique ones there
    lengths["len_guides"] = len(guide_indices)

    # Unique genes, including each category in case we have unique ones there
    lengths["len_genes"] = len(gene_indices)

    gene_indices_shared, _ = getUniqueGeneGuideIndices(
        indices, singletons=singletons, only_singletons=only_singletons, shared = True
    )

    lengths["len_genes_common"] = len(gene_indices_shared)

    return lengths

def lfc(final, reference):
    fc = final / (reference + 1E-9)

    return np.log2(fc + 1E-9)

def deltaLFC(lfc_combination, lfc_1, lfc_2):
    return lfc_combination - (lfc_1 + lfc_2)

def combinationLFCs(combinations, singles):

    # Calculate gene-averaged dLFCs

    # How am I indexing these into the parameters? -> gene_pair_idx -> gene_unq_pair_index

    # Calculate LFCs

    combinations['lfc'] = lfc(combinations['value'], combinations['plasmid'])
    singles['lfc'] = lfc(singles['value'], singles['plasmid'])

    # Average by gene, cell line

    singles_gene = singles.groupby(['gene1', 'cell_line']).agg({'lfc' : 'mean', 'g1_idx' : 'first', 'plasmid' : 'mean'}).reset_index()
    combs_gene = combinations.groupby(['gene1', 'gene2', 'cell_line', 'plasmid']).agg({'lfc' : 'mean', 'g1_idx' : 'first', 'g2_idx' : 'first'}).reset_index()

    # Merge to associate singles with combinations

    merged_gene = combs_gene.merge(singles_gene, on = ['gene1', 'cell_line'], suffixes = ('', '_1')).merge(singles_gene, left_on = ['gene2', 'cell_line'], right_on = ['gene1', 'cell_line'], suffixes = ('_comb', '_2')).reset_index()
    merged_gene = merged_gene.rename(columns = {'gene1_comb' : 'gene1', 'gene2_comb' : 'gene2'})
    merged_gene = merged_gene.groupby(['gene1', 'gene2', 'cell_line']).agg({'lfc_comb' : 'mean', 'lfc_1' : 'mean', 'lfc_2' : 'mean'}).reset_index()

    # Calculate the dLFC

    merged_gene['dLFC'] = deltaLFC(merged_gene['lfc_comb'], merged_gene['lfc_1'], merged_gene['lfc_2'])

    # Merge back into full combinations dataset shape

    combinations_dLFC = combinations.merge(merged_gene[['gene1', 'gene2', 'lfc_1', 'lfc_2', 'lfc_comb', 'cell_line', 'dLFC']], on = ['gene1', 'gene2', 'cell_line']).reset_index()[["gene_unq_pair_index", 'dLFC']]

    assert len(combinations_dLFC) == len(combinations)

    # Aggregate according to gene_pair_idx, for parameter

    gene_pair_dlfc = combinations_dLFC.sort_values("gene_unq_pair_index").groupby("gene_unq_pair_index").agg({'dLFC' : 'mean'})['dLFC']

    assert len(gene_pair_dlfc) == len(np.unique(combinations["gene_unq_pair_index"]))

    return gene_pair_dlfc.values

def saveModelParams(params: Dict[str, np.ndarray], fileName: str) -> None:
    """
    Save model parameters to a file.

    TODO: Save the transformed and untransformed params

    Args:
        params (Dict[str, np.ndarray]): Dictionary of model parameters.
        fileName (str): Name of the file to save the parameters.

    Returns:
        None
    """

    with h5py.File(fileName, "w") as file:
        for k, v in params.items():
            file.create_dataset(k, data=np.array(v))


def checkBounds(
    indices: Dict[str, np.ndarray],
    lengths: Dict[str, int],
    singletons: bool = True,
    neg_controls: bool = True,
    only_singletons: bool = False,
) -> None:
    """
    Check if the indices are within bounds.

    Args:
        indices (Dict[str, np.ndarray]): Dictionary of indices.
        lengths (Dict[str, int]): Dictionary of lengths.
        singletons (bool, optional): If True, include singleton checks. Defaults to True.
        neg_controls (bool, optional): If True, include negative control checks. Defaults to True.

    Returns:
        None
    """

    gene_indices, guide_indices = getUniqueGeneGuideIndices(
        indices, singletons=singletons, only_singletons=only_singletons
    )

    # Numpyro doesn't check whether we try to index off the end of an array,
    # so check that all of the arrays are the correct size for the indices

    assert np.max(guide_indices) < lengths["len_guides"]
    assert np.max(gene_indices) < lengths["len_genes"]

    assert np.max(indices["cell_line_idx"]) < lengths["len_cell_lines"]

    assert np.max(indices["guide_pair_idx"]) < lengths["len_guide_pairs"]

    if not only_singletons:
        assert np.max(indices["gene_pair_idx"]) < lengths["len_gene_pairs"]

    if singletons:
        assert np.max(indices["cell_line_s_idx"]) < lengths["len_cell_lines"]
        # assert np.max(indices["guide_pair_s_idx"]) < lengths["len_guide_pairs_s"]

    if neg_controls:
        assert np.max(indices["cell_line_c_idx"]) < lengths["len_cell_lines"]
        assert np.max(indices["guide_pair_unq_c_idx"]) < lengths["len_guide_pairs_unq_c"]

    # Also, warn if there are some parameters that remain unused, which is sus

    if np.max(gene_indices) != lengths["len_genes"] - 1:
        print(
            "WARNING: Some model gene parameters are un-referenced (no matching indices)."
        )

    if np.max(guide_indices) != lengths["len_guides"] - 1:
        print(
            "WARNING: Some model guide parameters are un-referenced (no matching indices)."
        )

    if np.max(indices["cell_line_idx"]) != lengths["len_cell_lines"] - 1:
        print(
            "WARNING: Some model cell line parameters are un-referenced (no matching indices)."
        )

    if np.max(indices["guide_pair_idx"]) != lengths["len_guide_pairs"] - 1:
        print(
            "WARNING: Some model guide pair parameters are un-referenced (no matching indices)."
        )

    if not only_singletons:
        if np.max(indices["gene_pair_idx"]) != lengths["len_gene_pairs"] - 1:
            print(
                "WARNING: Some model gene pair parameters are un-referenced (no matching indices)."
            )


def configArgs(args):
    # Take from CLI, read from a config file, or use defaults (in that order)

    config = json.load(open(args.config, "r")) if args.config else {}

    for arg in vars(args):
        if arg != "config":
            config[arg] = getattr(args, arg)

    return config


def getBatchData(data, indices, start_idx, end_idx):

    batch_data = {}
    batch_data["final"] = {
        k: v[start_idx:end_idx] if not v is None else None
        for k, v in data["final"].items()
    }
    batch_data["initial"] = {
        k: v[start_idx:end_idx] if not v is None else None
        for k, v in data["initial"].items()
    }

    batch_indices = {k: v[start_idx:end_idx] for k, v in indices.items()}

    return batch_data, batch_indices

def configure_custom_init(init_dict):

    # Set to values from config dictionary, falling back to median

    def custom_init(site=None):

        if site is None:
            return partial(custom_init)

        if site["name"] in init_dict:
            print(f'Setting {site["name"]}')
            return init_dict[site["name"]]
        else:
            return init_to_median(site)

    return custom_init


def subset_for_mcmc(indices_to_subset, data, lengths, indices):

    print(len(np.unique(indices['guide_pair_idx'])))
    print(np.max(indices['guide_pair_idx']))
    print(len(data['initial']['combinations']))

    indices_to_subset = np.array(indices_to_subset)

    # Everything that touches g_12 needs consistent shapes

    target_data = deepcopy(data)
    target_lengths = deepcopy(lengths)
    target_indices = deepcopy(indices)

    # For the gene pair indices we selected, what corresponding entries are there
    # in the full dataset (potentially many)

    sel_mask = np.isin(indices['gene_pair_idx'], indices_to_subset)
    sel_indices = np.array(list(range(len(indices['gene_pair_idx']))))[sel_mask]

    # Select these from the dataset, via the global index

    for k, v in indices.items():
        if len(v) == len(indices['gene_pair_idx']):
            target_indices[k] = target_indices[k][sel_indices]

    target_data['final']['combinations'] = target_data['final']['combinations'][sel_indices]

    # use guide pair indices that correspond to our gene pairs to index plasmids
    #
    # do I need to do this if its indexed in the model anyway? Do these shapes just mean that everything gets its indexed handled internally?

    # Handled already in call to dkoLH
    # target_data['initial']['combinations'] = target_data['initial']['combinations'][target_indices['guide_pair_idx']]

    # theta_init[indices["guide_pair_idx"]]

    for k, v in target_lengths.items():
        if v == len(indices['gene_pair_idx']):
            target_lengths[k] = len(target_indices['gene_pair_idx'])

    target_gene_indices, target_guide_indices = getUniqueGeneGuideIndices(
        target_indices, singletons=False,
    )

    target_gene_indices_shared, _ = getUniqueGeneGuideIndices(
       target_indices, singletons=False, only_singletons=False, shared = True
    )

    target_lengths["len_genes_common"] = len(target_gene_indices_shared)

    target_lengths['len_gene_pairs'] = len(np.unique(target_indices['gene_pair_idx']))
    target_lengths["len_guide_pairs"] = len(np.unique(target_indices["guide_pair_idx"]))
    target_lengths["len_cell_lines"] = len(np.unique(indices["cell_line_idx"]))
    target_lengths["len_guides"] = len(np.unique(target_guide_indices))
    target_lengths["len_genes"] = len(np.unique(target_gene_indices))

    # These index the cell line and gene by the gene pair index, so should be okay
    # just to index by the target indices

    target_indices["cell_line_in_pair_idx"] = target_indices["cell_line_in_pair_idx"][indices_to_subset]

    target_indices["gene_1_in_pair_idx"] = target_indices["gene_1_in_pair_idx"][indices_to_subset]
    target_indices["gene_2_in_pair_idx"] = target_indices["gene_2_in_pair_idx"][indices_to_subset]

    print(target_indices["guide_pair_idx"])

    # Many to one, rather than the one to many in the model 
    # Does the np.unique... give the correct order when discarding duplicates?
    target_data['initial']['combinations'] = target_data['initial']['combinations'][np.unique(target_indices["guide_pair_idx"])]

    return target_data, target_lengths, target_indices

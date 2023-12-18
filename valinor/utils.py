import jax.numpy as jnp

import pandas as pd

import numpy as np

import h5py

import yaml

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

def loadPriors(priorsFile : str):

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

    for n, d in datasets.items():
        counts[n] = getInitialCountsDF(d, initCountVar) if not (d is None) else None

    return counts


def getInitialCountsDF(df: pd.DataFrame, initCountVar: str) -> pd.Series:
    """
    Get initial counts from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        initCountVar (str): The variable for initial count.

    Returns:
        pd.Series: Initial counts.
    """

    initial_counts = (
        df.groupby("guide_pair_index")
        .agg({"guide_pair_index": "first", f"{initCountVar}": "first"})[
            ["guide_pair_index", f"{initCountVar}"]
        ]
        .reset_index(drop=True)
        .sort_values("guide_pair_index")[f"{initCountVar}"]
    )

    return initial_counts.values


def getUniqueGeneGuideIndices(
    indices: Dict[str, np.ndarray], singletons: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get unique gene and guide indices.

    Args:
        indices (Dict[str, np.ndarray]): Dictionary of indices.
        singletons (bool, optional): If True, include singleton indices. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Unique gene indices and guide indices.
    """

    guide_indices = [indices["guide_1_idx"], indices["guide_2_idx"]]
    gene_indices = [indices["gene_1_idx"], indices["gene_2_idx"]]

    if singletons:
        guide_indices += [indices["guide_s_idx"]]
        gene_indices += [indices["gene_s_idx"]]

    return np.unique(np.concatenate(gene_indices)), np.unique(
        np.concatenate(guide_indices)
    )


def getIndices(
    df: pd.DataFrame,
    dfSingles: Optional[pd.DataFrame] = None,
    dfControls: Optional[pd.DataFrame] = None,
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
        "guide_pair_idx": jnp.array(df["guide_pair_index"].values),
        "gene_pair_idx": jnp.array(df["gene_unq_pair_index"].values),
        "guide_1_idx": jnp.array(df["guide1_index"].values),
        "guide_2_idx": jnp.array(df["guide2_index"].values),
        "gene_1_idx": jnp.array(df["gene1_unq_index"].values),
        "gene_2_idx": jnp.array(df["gene2_unq_index"].values),
        "cell_line_idx": jnp.array(df["cell_line_index"].values),
    }

    if not (dfSingles is None):
        indices["guide_pair_s_idx"] = jnp.array(dfSingles["guide_pair_index"].values)

        indices["guide_s_idx"] = jnp.array(dfSingles["guide1_index"].values)

        indices["gene_s_idx"] = jnp.array(dfSingles["gene1_unq_index"].values)

        indices["cell_line_s_idx"] = jnp.array(dfSingles["cell_line_index"].values)

    if not (dfControls is None):
        indices["cell_line_c_idx"] = jnp.array(dfControls["cell_line_index"].values)
        indices["guide_pair_c_idx"] = jnp.array(dfControls["guide_pair_index"].values)

    # Make sure these are all 0D
    for k, v in indices.items():
        indices[k] = v.reshape(-1)

    return indices


def calculateLengths(
    indices: Dict[str, np.ndarray], singletons: bool = True, neg_controls: bool = True
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

    lengths = {
        "len_cell_lines": len(np.unique(indices["cell_line_idx"])),
        "len_guide_pairs": len(np.unique(indices["guide_pair_idx"])),
        "len_gene_pairs": len(np.unique(indices["gene_pair_idx"])),
    }

    gene_indices, guide_indices = getUniqueGeneGuideIndices(
        indices, singletons=singletons
    )

    if singletons:
        lengths["len_guide_pairs_s"] = len(np.unique(indices["guide_pair_s_idx"]))

    if neg_controls:
        lengths["len_guide_pairs_c"] = len(np.unique(indices["guide_pair_c_idx"]))

    # Unique guides, including each category in case we have unique ones there
    lengths["len_guides"] = len(guide_indices)

    # Unique genes, including each category in case we have unique ones there
    lengths["len_genes"] = len(gene_indices)

    return lengths


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
        indices, singletons=singletons
    )

    # Numpyro doesn't check whether we try to index off the end of an array,
    # so check that all of the arrays are the correct size for the indices

    assert np.max(guide_indices) < lengths["len_guides"]
    assert np.max(gene_indices) < lengths["len_genes"]

    assert np.max(indices["cell_line_idx"]) < lengths["len_cell_lines"]

    assert np.max(indices["guide_pair_idx"]) < lengths["len_guide_pairs"]
    assert np.max(indices["gene_pair_idx"]) < lengths["len_gene_pairs"]

    if singletons:
        assert np.max(indices["cell_line_s_idx"]) < lengths["len_cell_lines"]

        # print(np.max(indices["guide_pair_s_idx"]), lengths["len_guide_pairs_s"])
        # assert np.max(indices["guide_pair_s_idx"]) < lengths["len_guide_pairs_s"]

    if neg_controls:
        assert np.max(indices["cell_line_c_idx"]) < lengths["len_cell_lines"]

        # assert np.max(indices["guide_pair_c_idx"]) < lengths["len_guide_pairs_c"]

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

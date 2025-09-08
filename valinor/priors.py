import pandas as pd

import numpy as np

from typing import Dict, List, Tuple, Any


def calculateOverdispersion(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate overdispersion in the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and standard deviation of overdispersion.
    """

    reps = (
        df.groupby(["GuidePair", "cell_line_index"])
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            cell_line_index=("cell_line_index", "first"),
        )
        .reset_index(drop=True)
    )

    reps["var"] = reps["std"] ** 2
    reps["od"] = reps["var"] / reps["mean"]

    repsC = reps.groupby(["cell_line_index"]).agg(
        mean=("od", "median"), std=("od", "std")
    )
    repsC = repsC.sort_values("cell_line_index")

    repsC["mean"] = repsC["mean"].fillna(repsC["mean"].median())
    repsC["std"] = repsC["std"].fillna(repsC["std"].median())

    return repsC["mean"].values, repsC["std"].values

# def calcInitCountParams(df: pd.DataFrame, initCountVar: str, singletons = True):

#     # Initial values for the count parameters

#     # Params are common to all measurements with the same guide pair (or guide 1) index, incorporating all replicates and null guides, so average over these

#     guideVar = "guide_pair_index" if not singletons else 'guide_index'

#     # Average over plasmid counts per guide pair, if multiple
#     initial_counts = (
#             df.groupby("guide_pair_index")
#             .agg({"guide_pair_index": "first", initCountVar : "median", "guide1_index" : 'first'})
#             .reset_index(drop=True)
#             .sort_values("guide_pair_index")[[initCountVar, guideVar]]
#     )

#     if singletons:
#         # Average over null guides if present
#         initial_counts = initial_counts.groupby('guide1_index').agg({initCountVar : 'mean'}).reset_index()

#     final_counts = (
#             df.groupby("guide_pair_index")
#             .agg({"guide_pair_index": "first", 'value' : "first", "guide1_index" : 'first'})
#             .reset_index(drop=True)
#             .sort_values("guide_pair_index")[['value', guideVar]]
#     )

#     if singletons:
#         # Average over null guides if present
#         final_counts = final_counts.groupby(guideVar).agg({'value' : 'mean'}).reset_index()

#     return initial_counts[initCountVar].values.astype(np.float32), final_counts['value'].values.astype(np.float32)

def calcInitCountParams(df: pd.DataFrame, initCountVar: str, singletons = True):

    # Similar to getInitialCountsDF, but we want to average by the paramVar now, as we're initialising the parameter (which may be associated with many real observations)

    obsVar = "guide_pair_index"
    paramVar = "guide_index" if singletons else obsVar

    # Average over plasmid counts per guide pair, if multiple
    initial_counts = (
            df.groupby(paramVar)
            .agg({initCountVar : "median", obsVar : 'first', paramVar : 'first'})
            .reset_index(drop=True)[[initCountVar, obsVar, paramVar]]
    )

    return initial_counts[initCountVar].values.astype(np.float32)

def calculate_cell_line_stats(df):

    # For when these are controls

    # TODO:if they aren't controls, average over the dataset and make these the difference from the average

    grouped_by_cell = df.groupby(["cell_line_index"])

    df["_computed_log"] = np.log(df["value"] / (df["plasmid"] + 1e-6) + 1e-6)
    means = np.clip(grouped_by_cell["_computed_log"].mean(), -10, 10)
    std_devs = np.clip(grouped_by_cell["_computed_log"].std(), 0.1, 10)
    df.drop(columns=["_computed_log"], inplace=True)

    return means, std_devs


def calculate_gene_stats(df):
    """
    Calculate the mean and standard deviation per gene within each cell line, if 'gene_index' is present.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'cell_line_index', 'plasmid', 'value', and 'gene_index' columns.

    Returns:
    tuple: A tuple containing two 2D numpy arrays:
           - gene_mean_array: Mean log fold change per gene within each cell line.
           - gene_std_dev_array: Standard deviation of log fold change per gene within each cell line.
    """

    # Must match what is used in the rest of the code
    if "gene1_unq_index" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'gene_index' column for gene-level calculations."
        )

    grouped_by_gene_and_cell = df.groupby(["cell_line_index", "gene1_index"])

    # Calculate this once somewhere? Or use precalculated version with normalisation?

    if "lfc_norm_scaled" not in df.columns:
        df["_computed_log"] = np.log(df["value"] / (df["plasmid"] + 1e-6) + 1e-6)
        gene_means = np.clip(grouped_by_gene_and_cell["_computed_log"].mean(), -10, 10)
        gene_std_devs = np.clip(grouped_by_gene_and_cell["_computed_log"].std(), 0.1, 10)
        df.drop(columns=["_computed_log"], inplace=True)
    else:
        gene_means = np.clip(grouped_by_gene_and_cell["lfc_norm_scaled"].mean(), -10, 10)
        gene_std_devs = np.clip(grouped_by_gene_and_cell["lfc_norm_scaled"].std(), 0.1, 10)

    pivot_mean = np.clip(gene_means.unstack(fill_value=np.nan), -10, 10)
    pivot_std_dev = np.clip(gene_std_devs.unstack(fill_value=np.nan), 1E-6, 100)

    gene_mean_array = pivot_mean.to_numpy().astype(np.float32)
    gene_std_dev_array = pivot_std_dev.to_numpy().astype(np.float32)

    cell_line_indices = pivot_mean.index.to_numpy()
    gene_indices = pivot_mean.columns.to_numpy()

    return gene_mean_array, gene_std_dev_array, cell_line_indices, gene_indices


def defaultPriors():
    prior_params = {}

    prior_params["guide_eff_mean"] = (0.9, 0.1)
    prior_params["guide_eff_std"] = (0.1, 0.1)

    prior_params["gene_ko_growth"] = (0.0, 0.1)

    prior_params["cell_line_growth"] = (0.0, 0.1)

    prior_params["mv_mean_scale"] = 1.0
    prior_params["mv_std_scale"] = 1.0
    prior_params["od_pair_scale"] = 0.1

    prior_params["pair_eff_mean"] = (0.9, 0.1)
    prior_params["pair_eff_std"] = (0.1, 0.1)

    prior_params["pair_growth"] = (0.0, 0.1)

    return prior_params

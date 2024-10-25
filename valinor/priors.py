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
            mean=("value", np.mean),
            std=("value", np.std),
            cell_line_index=("cell_line_index", "first"),
        )
        .reset_index(drop=True)
    )

    reps["var"] = reps["std"] ** 2
    reps["od"] = reps["var"] / reps["mean"]

    repsC = reps.groupby(["cell_line_index"]).agg(
        mean=("od", np.median), std=("od", np.std)
    )
    repsC = repsC.sort_values("cell_line_index")

    return repsC["mean"].values, repsC["std"].values


def calculate_cell_line_stats(df):

    grouped_by_cell = df.groupby(["cell_line_index"])

    if "lfc_norm_scaled" not in df:

        means = grouped_by_cell.apply(
            lambda x: np.mean(np.log(x["value"] / (x["plasmid"] + 1e-6) + 1e-6))
        )
        std_devs = grouped_by_cell.apply(
            lambda x: np.std(np.log(x["value"] / (x["plasmid"] + 1e-6) + 1e-6))
        )

    else:

        means = grouped_by_cell.apply(lambda x: np.mean(x["lfc_norm_scaled"]))
        std_devs = grouped_by_cell.apply(lambda x: np.std(x["lfc_norm_scaled"]))

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

    if "lfc_norm_scaled" not in df:

        gene_means = grouped_by_gene_and_cell.apply(
            lambda x: np.mean(np.log(x["value"] / (x["plasmid"] + 1e-6) + 1e-6))
        )
        gene_std_devs = grouped_by_gene_and_cell.apply(
            lambda x: np.std(np.log(x["value"] / (x["plasmid"] + 1e-6) + 1e-6))
        )

    else:

        gene_means = grouped_by_gene_and_cell.apply(
            lambda x: np.mean(x["lfc_norm_scaled"])
        )
        gene_std_devs = grouped_by_gene_and_cell.apply(
            lambda x: np.std(x["lfc_norm_scaled"])
        )

    pivot_mean = gene_means.unstack(fill_value=np.nan)
    pivot_std_dev = gene_std_devs.unstack(fill_value=np.nan)

    gene_mean_array = pivot_mean.to_numpy()
    gene_std_dev_array = pivot_std_dev.to_numpy()

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

    prior_params["pair_eff_mean"] = (0.9, 0.1)
    prior_params["pair_eff_std"] = (0.1, 0.1)

    prior_params["pair_growth"] = (0.0, 0.1)

    return prior_params

from typing import Any, Dict, List, Tuple

import jax
import numpy as np
import pandas as pd


def calculateOverdispersion(
    df: pd.DataFrame, phi: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
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
        mean=("od", "median"),
        std=("od", "std"),
    )
    repsC = repsC.sort_values("cell_line_index")

    repsC["mean"] = repsC["mean"].fillna(repsC["mean"].median())
    repsC["std"] = repsC["std"].fillna(repsC["std"].median())

    if phi:
        mu_ref = reps.groupby("cell_line_index").agg(mu_ref=("mean", "median"))
        mu_ref = mu_ref.reindex(repsC.index)
        mu_ref["mu_ref"] = mu_ref["mu_ref"].fillna(mu_ref["mu_ref"].median())
        mu_ref_v = mu_ref["mu_ref"].to_numpy()

        mv_means = repsC["mean"].to_numpy()
        mv_stds = repsC["std"].to_numpy()

        mu_ref_v = np.clip(mu_ref_v, 1.0, np.inf)
        mv_means = np.clip(mv_means, 1.01, np.inf)
        mv_stds = np.clip(mv_stds, 1e-6, np.inf)

        phi_mean = mu_ref_v / (mv_means - 1.0)
        phi_mean = np.clip(phi_mean, 1e-6, np.inf)

        log_phi_mean = np.log(phi_mean)

        log_phi_std = mv_stds / (mv_means - 1.0)
        log_phi_std = np.clip(log_phi_std, 0.05, 2.0)

        return log_phi_mean.astype(np.float32), log_phi_std.astype(np.float32)

    return repsC["mean"].to_numpy().astype(np.float32), repsC["std"].to_numpy().astype(
        np.float32
    )


def calcInitCountParams(df: pd.DataFrame, initCountVar: str, singletons=True):
    # Similar to getInitialCountsDF, but we want to average by the paramVar now, as we're initialising the parameter (which may be associated with many real observations)

    obsVar = "guide_pair_index"
    paramVar = "guide_index" if singletons else obsVar

    # Average over plasmid counts per guide pair, if multiple
    initial_counts = (
        df.groupby(paramVar)
        .agg({initCountVar: "median", obsVar: "first", paramVar: "first"})
        .reset_index(drop=True)[[initCountVar, obsVar, paramVar]]
    )

    return initial_counts[initCountVar].values.astype(np.float32)


def calculate_cell_line_stats(df, initCountVar="plasmid"):
    # For when these are controls

    # TODO:if they aren't controls, average over the dataset and make these the difference from the average

    grouped_by_cell = df.groupby(["cell_line_index"])

    df["_computed_log"] = np.log(df["value"] / (df[initCountVar] + 1e-6) + 1e-6)
    means = np.clip(grouped_by_cell["_computed_log"].mean(), -10, 10)
    std_devs = np.clip(grouped_by_cell["_computed_log"].std(), 0.1, 10)
    df.drop(columns=["_computed_log"], inplace=True)

    return means, std_devs


def calculate_gene_stats(df, initCountVar="plasmid"):
    # Must match what is used in the rest of the code
    if "gene1_unq_index" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'gene_index' column for gene-level calculations."
        )

    grouped_by_gene_and_cell = df.groupby(["cell_line_index", "gene1_index"])

    # Calculate this once somewhere? Or use precalculated version with normalisation?

    if "lfc_norm_scaled" not in df.columns:
        df["_computed_log"] = np.log(df["value"] / (df[initCountVar] + 1e-6) + 1e-6)
        gene_means = np.clip(grouped_by_gene_and_cell["_computed_log"].mean(), -10, 10)
        gene_std_devs = np.clip(
            grouped_by_gene_and_cell["_computed_log"].std(), 0.1, 10
        )
        df.drop(columns=["_computed_log"], inplace=True)
    else:
        gene_means = np.clip(
            grouped_by_gene_and_cell["lfc_norm_scaled"].mean(), -10, 10
        )
        gene_std_devs = np.clip(
            grouped_by_gene_and_cell["lfc_norm_scaled"].std(), 0.1, 10
        )

    pivot_mean = np.clip(gene_means.unstack(fill_value=np.nan), -10, 10)
    pivot_std_dev = np.clip(gene_std_devs.unstack(fill_value=np.nan), 1e-6, 100)

    gene_mean_array = pivot_mean.to_numpy().astype(np.float32)
    gene_std_dev_array = pivot_std_dev.to_numpy().astype(np.float32)

    cell_line_indices = pivot_mean.index.to_numpy()
    gene_indices = pivot_mean.columns.to_numpy()

    return gene_mean_array, gene_std_dev_array, cell_line_indices, gene_indices


def defaultPriors():
    prior_params = {}

    # prior_params["guide_eff_mean"] = (0.9, 0.1)
    # prior_params["guide_eff_std"] = (0.1, 0.1)

    # prior_params["gene_ko_growth"] = (0.0, 0.1)

    # prior_params["cell_line_growth"] = (0.0, 0.1)

    # prior_params["mv_mean_scale"] = 1.0
    # prior_params["mv_std_scale"] = 1.0
    # prior_params["od_pair_scale"] = 0.1

    # prior_params["pair_eff_mean"] = (0.9, 0.1)
    # prior_params["pair_eff_std"] = (0.1, 0.1)

    # prior_params["pair_growth"] = (0.0, 0.1)

    prior_params["guide_eff_mean"] = (jax.scipy.special.logit(0.8), 0.6)
    prior_params["guide_eff_std"] = (-0.5, 0.5)  # softplus(-0.5) ≈ 0.47 on logit

    prior_params["gene_ko_growth"] = (0.0, 0.5)  # or Laplace in code with b=0.5

    prior_params["cell_line_growth"] = (-1.0, 0.7)  # allow small finals a priori

    prior_params["phi_mean_scale"] = 0.5
    prior_params["phi_pair_scale"] = 0.5

    # only keep if you actually use them in code
    prior_params["pair_eff_mean"] = (jax.scipy.special.logit(0.8), 0.6)
    prior_params["pair_eff_std"] = (-0.5, 0.5)

    prior_params["pair_growth"] = (0.0, 0.5)  # Laplace(0,0.5) in code

    return prior_params

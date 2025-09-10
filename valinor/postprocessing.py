import numpy as np

import pandas as pd

import h5py

import jax
from jax import random
import jax.numpy as jnp

from valinor import models

from typing import Dict, List, Tuple


# Average over samples from the posterior to pack into a Pandas DataFrame
def averageOverSamples(
    samples: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Calculate the mean and standard deviation over the samples in the input dictionary.

    Args:
        samples (Dict[str, np.ndarray]): A dictionary where keys are sample names and values are numpy arrays of samples.

    Returns:
        Tuple[Dict[str, np.ndarray], np.ndarray]: A tuple where the first element is a dictionary of means for each sample,
        and the second element is a numpy array of standard deviations for each sample.
    """

    means = {k: np.mean(s, 0) if s.ndim >= 2 else s for k, s in samples.items()}
    stds = {
        k: np.std(s, 0) if s.ndim >= 2 else np.zeros_like(s) for k, s in samples.items()
    }

    return means, stds


# To be run over params['combs'], etc, so that each category has its own DataFrame
def createDataFrame(paramSamples: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Create a Pandas DataFrame from the provided parameter samples, averaging over samples,
    and appending the samples from the likelihood.

    Args:
        paramSamples (Dict[str, np.ndarray]): A dictionary where keys are sample names and values are either lists or numpy arrays of samples.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a type of sample, and each row represents an observation.
    """

    # Separate out samples from the likelihood from parameter samples

    lhSamples = list(filter(lambda x: "samples" in x, paramSamples))

    # Average over samples from likelihood

    df = pd.DataFrame({n: np.mean(paramSamples[n], 0) for n in lhSamples})

    # Return an empty DF if there are no parameters, e.g., from batching with a size greater than the length

    if list(paramSamples.values())[0].size == 0:

        for k in set(paramSamples.keys()) - set(lhSamples):
            df[f"{k}_mean"] = []
            df[f"{k}_std"] = []

        return df

    # Get the average and std values of the parameter samples

    means, stds = averageOverSamples(paramSamples)

    for k in set(paramSamples.keys()) - set(lhSamples):
        mean_k = means[k]
        std_k = stds[k]

        if np.isscalar(mean_k):
            df[f"{k}_mean"] = float(mean_k)
            df[f"{k}_std"] = 0.0
        else:
            # assume mean_k is a sequence/array
            if len(mean_k) > 1:
                df[f"{k}_mean"] = mean_k
                df[f"{k}_std"] = std_k
            else:
                df[f"{k}_mean"] = mean_k[0]
                df[f"{k}_std"] = std_k[0]
            # ensure floats
            df[f"{k}_mean"] = df[f"{k}_mean"].astype(float)
            df[f"{k}_std"] = df[f"{k}_std"].astype(float)

    return df


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


# Todo: plit up this megafunction


def sampleParams(
    samples: Dict[str, np.ndarray],
    indices: Dict[str, np.ndarray],
    prior_params: Dict[str, np.ndarray],
    alternate: bool = False,
    empirical_gene_priors: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Sample parameters based on the provided samples and indices.

    Args:
        samples (Dict[str, np.ndarray]): A dictionary where keys are sample names and values are numpy arrays of samples.
        indices (Dict[str, np.ndarray]): A dictionary where keys are index names and values are numpy arrays of indices.
        alternate (bool): Whether to use the alternate likelihood.
        empirical_gene_priors (bool): Whether to use empirical gene priors.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: A dictionary of sampled parameters.
    """

    params = {}

    # Determine which datasets are present
    singletons = "guide_init_count_s" in samples
    only_singletons = not "guide_init_count" in samples
    controls = "guide_init_count_c" in samples

    zi = "p_zi" in samples
    zi_s = "p_zi_s" in samples

    raw_mv_cl = samples["raw_mv_cell_line" if not singletons else "raw_mv_cell_line_s"]
    mv_cl = jnp.exp(raw_mv_cl) + 1.0

    base_key = random.PRNGKey(0)
    keys = random.split(base_key, 10)

    # Counter for keys
    key_counter = 0

    if singletons:
        singlesParams = {}

        singlesParams["guide_init_count_s"] = samples["guide_init_count_s"][
            :, indices["guide_pair_s_idx"]
        ]

        singlesParams["init_count_s"] = samples["guide_init_count_s"][
            :, indices["guide_s_idx"]
        ]

        singlesParams["init_count_s"] = jax.nn.softplus(singlesParams["init_count_s"])

        singlesParams["tilde_alpha"] = samples["sko/tilde_alpha"][
            :, indices["guide_s_idx"], indices["cell_line_s_idx"]
        ]

        singlesParams["guide_eff_mean_s"] = samples["sko/guide_eff_mean"][
            :, indices["guide_s_idx"]
        ]
        singlesParams["guide_eff_std_s"] = samples["sko/guide_eff_std"][
            :, indices["guide_s_idx"]
        ]

        singlesParams["guide_eff_s"] = sigmoid(
            singlesParams["guide_eff_mean_s"].squeeze()
            + singlesParams["guide_eff_std_s"].squeeze() * singlesParams["tilde_alpha"]
        )

        # Cell Line Growth
        singlesParams["cell_growth_s"] = samples["cell_line_growth"][
            :, indices["cell_line_s_idx"]
        ]

        # Library Bias
        # singlesParams["library_bias_s"] = samples["library_bias"][:, indices["cell_line_s_idx"]]

        if empirical_gene_priors:
            singlesParams["ko_growth_s"] = samples["gene_ko_growth"][
                :, indices["cell_line_s_idx"], indices["gene_s_common_idx"]
            ]
        else:
            singlesParams["ko_growth_s"] = samples["gene_ko_growth"][
                :, indices["gene_s_idx"]
            ]

        mv_gene = samples["mv_gene" if not singletons else "mv_gene_s"]

        singlesParams["mv_s"] = mv_gene[
            :, indices["cell_line_s_idx"], indices["gene_s_idx"]
        ]  # shape [S, n_singleton_obs]

        if zi_s:
            singlesParams["p_zi_s"] = samples["p_zi_s"][:, indices["cell_line_s_idx"]]

        # Sample from Initial Likelihood
        init_lh, theta_init = models.skoLikelihoodInitial(singlesParams["init_count_s"])
        singlesParams["samples_s_init"] = init_lh.sample(
            random.split(keys[key_counter])[0]
        )[indices["guide_pair_s_idx"]]
        key_counter += 1

        params["singles"] = singlesParams

    ############################
    # Process Controls
    ############################
    if controls:
        controlsParams = {}

        controlsParams["init_count_c"] = samples["guide_init_count_c"][
            :, indices["guide_pair_c_idx"]
        ]
        controlsParams["init_count_c"] = (
            jax.nn.softplus(controlsParams["init_count_c"]) + 1e-6
        )

        controlsParams["cell_growth_c"] = samples["cell_line_growth"][
            :, indices["cell_line_c_idx"]
        ]

        raw_mv_cl = samples["raw_mv_cell_line"]  # [S, n_cell_lines]
        mv_cl = jnp.exp(raw_mv_cl) + 1.0  # [S, n_cell_lines]

        controlsParams["mv_c"] = mv_cl[
            :,
            indices["cell_line_c_idx"],
        ]  # [S, ]

        init_lh_c, theta_init_c = models.skoLikelihoodInitial(
            controlsParams["init_count_c"]
        )
        controlsParams["samples_c_init"] = init_lh_c.sample(
            random.split(keys[key_counter])[0]
        )

        key_counter += 1

        params["controls"] = controlsParams

    if not only_singletons:
        combsParams = {}

        # Initialize Count for Combinations
        combsParams["init_count"] = samples["guide_init_count"][
            :, indices["guide_pair_idx"]
        ]

        # Cell Line Growth for Combinations
        combsParams["cell_line_growth"] = samples["cell_line_growth"][
            :, indices["cell_line_idx"]
        ]

        combsParams["library_bias"] = samples["library_bias"][
            :, indices["cell_line_idx"]
        ]

        # Guide Efficiencies
        if "guide_eff_mean" in samples.keys():
            combsParams["guide_eff_mean_1"] = samples["dko/guide_eff_mean"][
                :, indices["guide_1_idx"]
            ]
            combsParams["guide_eff_mean_2"] = samples["dko/guide_eff_mean"][
                :, indices["guide_2_idx"]
            ]

            combsParams["guide_eff_std_1"] = samples["dko/guide_eff_std"][
                :, indices["guide_1_idx"]
            ]
            combsParams["guide_eff_std_2"] = samples["dko/guide_eff_std"][
                :, indices["guide_2_idx"]
            ]

            combsParams["tilde_alpha_1"] = samples["dko/tilde_alpha"][
                :, indices["guide_1_idx"], indices["cell_line_idx"]
            ]
            combsParams["tilde_alpha_2"] = samples["dko/tilde_alpha"][
                :, indices["guide_2_idx"], indices["cell_line_idx"]
            ]

            # Guide Efficiencies with Sigmoid Transformation
            combsParams["guide_eff_1"] = sigmoid(
                combsParams["guide_eff_mean_1"].squeeze()
                + combsParams["tilde_alpha_1"]
                * combsParams["guide_eff_std_1"].squeeze()
            )
            combsParams["guide_eff_2"] = sigmoid(
                combsParams["guide_eff_mean_2"].squeeze()
                + combsParams["tilde_alpha_2"]
                * combsParams["guide_eff_std_2"].squeeze()
            )
        else:
            # Handle cases where guide_eff_mean is not present
            combsParams["guide_eff_1"] = np.ones_like(
                combsParams["cell_line_growth"]
            )  # or another appropriate default
            combsParams["guide_eff_2"] = np.ones_like(
                combsParams["cell_line_growth"]
            )  # or another appropriate default

        # Gene Knockout Growth
        if empirical_gene_priors:
            combsParams["gene_ko_growth_1"] = samples["gene_ko_growth"][
                :, indices["cell_line_idx"], indices["gene_1_common_idx"]
            ]
            combsParams["gene_ko_growth_2"] = samples["gene_ko_growth"][
                :, indices["cell_line_idx"], indices["gene_2_common_idx"]
            ]
        else:
            combsParams["gene_ko_growth_1"] = samples["gene_ko_growth"][
                :, indices["gene_1_idx"]
            ]
            combsParams["gene_ko_growth_2"] = samples["gene_ko_growth"][
                :, indices["gene_2_idx"]
            ]

        # Gene Pair Knockout Growth
        combsParams["gene_ko_growth_12"] = samples["gene_pair_ko_growth"][
            :, indices["gene_pair_idx"]
        ]

        if "dLFC" in prior_params:
            combsParams["dLFC"] = prior_params["dLFC"][indices["gene_pair_idx"]]

        mv_pair = samples["mv_gene_pair"]
        # index that flat vector by your observation‐level pair indices
        combsParams["mv"] = mv_pair[
            :, indices["gene_pair_idx"]
        ]  # shape [S, n_combo_obs]

        # combsParams["negative_control_bias"] = samples["negative_control_bias"][:, indices["cell_line_idx"]]

        # p_zi if applicable
        if zi:
            combsParams["p_zi"] = samples["p_zi"][:, indices["cell_line_idx"]]
        if zi_s:
            combsParams["p_zi_s"] = samples["p_zi_s"][:, indices["cell_line_idx"]]

        # Sample from Initial Likelihood for Combinations
        init_lh_comb, theta_init_comb = models.dkoLikelihoodInitial(
            combsParams["init_count"]
        )
        combsParams["samples_init"] = init_lh_comb.sample(
            random.split(keys[key_counter])[0]
        )
        key_counter += 1

        params["combs"] = combsParams

    return params


# Even when batching, it's easier just to sample the posterior predictive ('obs') in one go


def samplePosteriorPredictive(samples, indices, annotation=None):

    if "obs_init_c" in samples:

        fileName = (
            "obs_init_c.h5" if annotation == None else f"obs_init_c_{annotation}.h5"
        )

        with h5py.File(fileName, "w") as h5f:
            h5f.create_dataset(
                "obs_init_c", data=samples["obs_init_c"][:, indices["guide_pair_c_idx"]]
            )

    if "obs_c" in samples:

        fileName = "obs_c.h5" if annotation == None else f"obs_c_{annotation}.h5"

        with h5py.File(fileName, "w") as h5f:
            h5f.create_dataset("obs_c", data=samples["obs_c"])

    if "obs_init_s" in samples:

        fileName = (
            "obs_init_s.h5" if annotation == None else f"obs_init_s_{annotation}.h5"
        )

        with h5py.File(fileName, "w") as h5f:

            h5f.create_dataset("obs_init_s", data=samples["obs_init_s"])

    if "obs_s" in samples:

        fileName = "obs_s.h5" if annotation == None else f"obs_s_{annotation}.h5"

        with h5py.File(fileName, "w") as h5f:
            h5f.create_dataset("obs_s", data=samples["obs_s"])

    if "obs_init" in samples:

        fileName = "obs_init.h5" if annotation == None else f"obs_init_{annotation}.h5"

        with h5py.File(fileName, "w") as h5f:
            h5f.create_dataset("obs_init", data=samples["obs_init"])

    if "obs" in samples:

        fileName = "obs.h5" if annotation == None else f"obs_{annotation}.h5"

        with h5py.File(fileName, "w") as h5f:
            h5f.create_dataset("obs", data=samples["obs"])

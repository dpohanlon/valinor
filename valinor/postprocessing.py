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
    gene_effect_means = False,
) -> Dict[str, Dict[str, np.ndarray]]:

    params = {}

    singletons = "guide_init_count_s" in samples
    only_singletons = "guide_init_count" not in samples
    controls = "guide_init_count_c" in samples

    zi = "p_zi" in samples
    zi_s = "p_zi_s" in samples

    base_key = random.PRNGKey(0)
    keys = random.split(base_key, 10)
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

        singlesParams["guide_eff_s"] = samples["sko/guide_eff"][
            :, indices["guide_s_idx"], indices["cell_line_s_idx"]
        ]

        singlesParams["cell_growth_s"] = samples["cell_line_growth"][
            :, indices["cell_line_s_idx"]
        ]

        singlesParams["ko_growth_s"] = samples["gene_ko_growth"][
            :, indices["cell_line_s_idx"], indices["gene_s_idx"]
        ]
        singlesParams["ko_growth_mean_s"] = samples["gene_ko_growth_mean"][
            :, indices["gene_s_idx"]
        ]
        singlesParams["ko_growth_std_s"] = samples["gene_ko_growth_std"][
            :, indices["gene_s_idx"]
        ]
        singlesParams["ko_growth_dev_s"] = (
            singlesParams["ko_growth_s"] - singlesParams["ko_growth_mean_s"]
        )

        mv_gene = samples["mv_gene_s"] if "mv_gene_s" in samples else samples["mv_gene"]
        gene_s_mv_idx = (
            indices["gene_s_common_idx"]
            if "gene_s_common_idx" in indices
            else indices["gene_s_idx"]
        )
        singlesParams["mv_s"] = mv_gene[
            :, indices["cell_line_s_idx"], gene_s_mv_idx
        ]

        if zi_s:
            singlesParams["p_zi_s"] = samples["p_zi_s"][:, indices["cell_line_s_idx"]]

        init_lh, _ = models.skoLikelihoodInitial(singlesParams["init_count_s"])
        singlesParams["samples_s_init"] = init_lh.sample(
            random.split(keys[key_counter])[0]
        )[indices["guide_pair_s_idx"]]
        key_counter += 1

        params["singles"] = singlesParams

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

        raw_mv_cl = samples["raw_mv_cell_line"]
        mv_cl = jnp.exp(raw_mv_cl) + 1.0
        controlsParams["mv_c"] = mv_cl[:, indices["cell_line_c_idx"]]

        init_lh_c, _ = models.skoLikelihoodInitial(controlsParams["init_count_c"])
        controlsParams["samples_c_init"] = init_lh_c.sample(
            random.split(keys[key_counter])[0]
        )
        key_counter += 1

        params["controls"] = controlsParams

    if not only_singletons:
        combsParams = {}

        combsParams["init_count"] = samples["guide_init_count"][:, indices["guide_pair_idx"]]

        combsParams["cell_line_growth"] = samples["cell_line_growth"][
            :, indices["cell_line_idx"]
        ]

        combsParams["library_bias"] = samples["library_bias"][:, indices["cell_line_idx"]]

        combsParams["guide_eff_1"] = samples["dko/guide_eff"][
            :, indices["guide_1_idx"], indices["cell_line_idx"]
        ]
        combsParams["guide_eff_2"] = samples["dko/guide_eff"][
            :, indices["guide_2_idx"], indices["cell_line_idx"]
        ]

        combsParams["gene_ko_growth_1"] = samples["gene_ko_growth"][
            :, indices["cell_line_idx"], indices["gene_1_idx"]
        ]
        combsParams["gene_ko_growth_2"] = samples["gene_ko_growth"][
            :, indices["cell_line_idx"], indices["gene_2_idx"]
        ]

        combsParams["gene_ko_growth_mean_1"] = samples["gene_ko_growth_mean"][
            :, indices["gene_1_idx"]
        ]
        combsParams["gene_ko_growth_mean_2"] = samples["gene_ko_growth_mean"][
            :, indices["gene_2_idx"]
        ]
        combsParams["gene_ko_growth_std_1"] = samples["gene_ko_growth_std"][
            :, indices["gene_1_idx"]
        ]
        combsParams["gene_ko_growth_std_2"] = samples["gene_ko_growth_std"][
            :, indices["gene_2_idx"]
        ]

        combsParams["gene_ko_growth_dev_1"] = (
            combsParams["gene_ko_growth_1"] - combsParams["gene_ko_growth_mean_1"]
        )
        combsParams["gene_ko_growth_dev_2"] = (
            combsParams["gene_ko_growth_2"] - combsParams["gene_ko_growth_mean_2"]
        )

        combsParams["gene_ko_growth_12"] = samples["gene_pair_ko_growth"][
            :, indices["cell_line_idx"], indices["gene_pair_idx"]
        ]
        combsParams["gene_ko_growth_12_mean"] = samples["gene_pair_ko_growth_mean"][
            :, indices["gene_pair_idx"]
        ]
        combsParams["gene_ko_growth_12_std"] = samples["gene_pair_ko_growth_std"][
            :, indices["gene_pair_idx"]
        ]
        combsParams["gene_ko_growth_12_dev"] = (
            combsParams["gene_ko_growth_12"] - combsParams["gene_ko_growth_12_mean"]
        )

        if "dLFC" in prior_params:
            combsParams["dLFC"] = prior_params["dLFC"][indices["gene_pair_idx"]]

        mv_pair = samples["mv_gene_pair"]
        combsParams["mv"] = mv_pair[:, indices["gene_pair_idx"]]

        if zi:
            combsParams["p_zi"] = samples["p_zi"][:, indices["cell_line_idx"]]

        init_lh_comb, _ = models.dkoLikelihoodInitial(combsParams["init_count"])
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

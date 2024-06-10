import numpy as np

import pandas as pd

from jax import random

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

    means = {k: np.mean(s, 0) for k, s in samples.items()}
    stds = {k: np.std(s, 0) for k, s in samples.items()}

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

    # Get the average and std values of the parameter samples

    means, stds = averageOverSamples(paramSamples)

    for k in set(paramSamples.keys()) - set(lhSamples):
        df[f"{k}_mean"] = means[k] if len(means[k]) > 1 else list(means[k])[0]
        df[f"{k}_std"] = stds[k] if len(stds[k]) > 1 else list(stds[k])[0]

        df[f"{k}_mean"] = df[f"{k}_mean"].astype(float)
        df[f"{k}_std"] = df[f"{k}_std"].astype(float)

    return df


# Split up this megafunction


def sampleParams(
    samples: Dict[str, np.ndarray],
    indices: Dict[str, np.ndarray],
    alternate: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Sample parameters based on the provided samples and indices.

    Args:
        samples (Dict[str, np.ndarray]): A dictionary where keys are sample names and values are numpy arrays of samples.
        indices (Dict[str, np.ndarray]): A dictionary where keys are index names and values are numpy arrays of indices.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: A dictionary of sampled parameters.
    """

    params = {}

    # Expand the sampled parameter arrays to the shape of the input dataset, via the indices

    # First axis is the number of samples from the model

    singletons = "guide_init_count_s" in samples
    only_singletons = not "guide_init_count" in samples
    controls = "guide_init_count_c" in samples

    zi = "p_zi" in samples

    if singletons:
        singlesParams = {}

        singlesParams["init_count_s"] = samples["guide_init_count_s"][
            :, indices["guide_pair_s_idx"]
        ]

        # TO DO: Add a switch here

        singlesParams["guide_eff_s"] = samples["guide_eff"][
            :, indices["guide_s_idx"], indices["cell_line_s_idx"]
        ]

        # singlesParams["guide_eff_s"] = samples["guide_eff_s"][:, indices["guide_s_idx"]]

        singlesParams["cell_growth_s"] = samples["cell_line_growth"][
            :, indices["cell_line_s_idx"]
        ]

        singlesParams["library_bias_s"] = samples["library_bias"][
            :, indices["cell_line_s_idx"]
        ]

        singlesParams["ko_growth_s"] = samples["gene_ko_growth"][
            :, indices["gene_s_idx"]
        ]

        singlesParams["mv_s"] = 1.0 / samples["inv_mv_s"]

        if zi:
            singlesParams["p_zi"] = samples["p_zi"][:, indices["cell_line_s_idx"]]

        # Can also add sample_shape if we want to control samples further

        singlesParams["samples_s_init"] = models.skoLikelihoodInitial(
            singlesParams["init_count_s"]
        )[0].sample(random.PRNGKey(42))

        singlesParams["samples_s"] = models.skoLikelihoodFinal(
            singlesParams["init_count_s"],
            singlesParams["guide_eff_s"],
            singlesParams["cell_growth_s"],
            singlesParams["ko_growth_s"],
            singlesParams["mv_s"],
            singlesParams["library_bias_s"],
            alternate,
            singlesParams["p_zi"] if zi else zi,
        )[0].sample(random.PRNGKey(42))

        params["singles"] = singlesParams

    if controls:
        controlsParams = {}

        controlsParams["init_count_c"] = samples["guide_init_count_c"][
            :, indices["guide_pair_c_idx"]
        ]

        controlsParams["cell_growth_c"] = samples["cell_line_growth"][
            :, indices["cell_line_c_idx"]
        ]

        controlsParams["mv_c"] = 1.0 / samples["inv_mv_c"]

        controlsParams["samples_c_init"] = models.skoLikelihoodInitial(
            controlsParams["init_count_c"]
        )[0].sample(random.PRNGKey(42))

        controlsParams["samples_c"] = models.controlLikelihoodFinal(
            controlsParams["init_count_c"],
            controlsParams["cell_growth_c"],
            controlsParams["mv_c"],
        )[0].sample(random.PRNGKey(42))

        params["controls"] = controlsParams

    if not only_singletons:

        combsParams = {}

        combsParams["init_count"] = samples["guide_init_count"][
            :, indices["guide_pair_idx"]
        ]
        combsParams["cell_line_growth"] = samples["cell_line_growth"][
            :, indices["cell_line_idx"]
        ]

        combsParams["guide_eff_1"] = samples[
            "guide_eff"
        ][  # Check whether this should be specified given the hierarchy, ordering, etc
            :, indices["guide_1_idx"], indices["cell_line_idx"]
        ]
        combsParams["guide_eff_2"] = samples[
            "guide_eff"
        ][  # Check whether this should be specified given the hierarchy, ordering, etc
            :, indices["guide_2_idx"], indices["cell_line_idx"]
        ]

        if "guide_eff_mean" in samples.keys():
            combsParams["guide_eff_mean_1"] = samples["guide_eff_mean"][
                :, indices["guide_1_idx"]
            ]
            combsParams["guide_eff_mean_2"] = samples["guide_eff_mean"][
                :, indices["guide_2_idx"]
            ]
        if "guide_eff_std" in samples.keys():
            combsParams["guide_eff_std_1"] = samples["guide_eff_std"][
                :, indices["guide_1_idx"]
            ]
            combsParams["guide_eff_std_2"] = samples["guide_eff_std"][
                :, indices["guide_2_idx"]
            ]

        combsParams["gene_ko_growth_1"] = samples["gene_ko_growth"][
            :, indices["gene_1_idx"]
        ]
        combsParams["gene_ko_growth_2"] = samples["gene_ko_growth"][
            :, indices["gene_2_idx"]
        ]
        combsParams["gene_ko_growth_12"] = samples["gene_pair_ko_growth"][
            :, indices["gene_pair_idx"]
        ]

        combsParams["mv"] = 1.0 / samples["inv_mv"]

        if zi:
            combsParams["p_zi"] = samples["p_zi"][:, indices["cell_line_idx"]]

        combsParams["samples_init"] = models.dkoLikelihoodInitial(
            combsParams["init_count"]
        )[0].sample(random.PRNGKey(42))

        finalLH = (
            models.dkoLikelihoodFinal if alternate else models.dkoLikelihoodFullFinal
        )

        if alternate:
            combsParams["guide_eff_12"] = samples["guide_eff_12"]

            combsParams["samples"] = models.dkoLikelihoodFinal(
                combsParams["init_count"],
                combsParams["guide_eff_1"],
                combsParams["guide_eff_2"],
                combsParams["guide_eff_12"],
                combsParams["cell_line_growth"],
                combsParams["gene_ko_growth_1"],
                combsParams["gene_ko_growth_2"],
                combsParams["gene_ko_growth_12"],
                combsParams["mv"],
                1.0,
                combsParams["p_zi"] if zi else zi,
            )[0].sample(random.PRNGKey(42))

        else:
            combsParams["samples"] = models.dkoLikelihoodFullFinal(
                combsParams["init_count"],
                combsParams["guide_eff_1"],
                combsParams["guide_eff_2"],
                # combsParams["guide_eff_12"],
                combsParams["cell_line_growth"],
                combsParams["gene_ko_growth_1"],
                combsParams["gene_ko_growth_2"],
                combsParams["gene_ko_growth_12"],
                combsParams["mv"],
                1.0,
                combsParams["p_zi"] if zi else zi,
            )[0].sample(random.PRNGKey(42))

        params["combs"] = combsParams

    return params

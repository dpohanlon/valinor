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

    means = {k: np.mean(s, 0) for k, s in samples}
    stds = {k: np.std(s, 0) for k, s in samples}

    return means, std


# To be run over params['combs'], etc, so that each category has its own DataFrame
def createDataFrame(paramSamples: Dict[str, np.ndarray]) -> pd.DataFrame:

    """
    Create a Pandas DataFrame from the provided parameter samples.

    Args:
        paramSamples (Dict[str, np.ndarray]): A dictionary where keys are sample names and values are either lists or numpy arrays of samples.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a type of sample, and each row represents an observation.
    """

    # Separate out samples from the likelihood from parameter samples

    lhSamples = list(filter(lambda x: "samples" in x, paramSamples))

    df = pd.DataFrame({n: paramSamples[n] for x in lhSamples})

    # Get the average and std values of the parameter samples

    means, stds = averageOverSamples(paramSamples)

    for k in set(samples.keys()) - set(lhSamples):
        df[f"{k}_mean"] = means[k]
        df[f"{k}_std"] = stds[k]

    return df


def sampleParams(
    samples: Dict[str, np.ndarray], indices: Dict[str, np.ndarray]
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
    controls = "guide_init_count_c" in samples

    if singletons:

        singlesParams = {}

        singlesParams["init_count_s"] = samples["guide_init_count_s"][
            :, indices["guide_pair_s_idx"]
        ]

        singlesParams["guide_eff_s"] = samples["guide_eff"][
            :, indices["guide_s_idx"], indices["cell_line_s_idx"]
        ]

        singlesParams["cell_growth_s"] = samples["cell_line_growth"][
            :, indices["cell_line_s_idx"]
        ]

        singlesParams["ko_growth_s"] = samples["gene_ko_growth"][
            :, indices["gene_s_idx"]
        ]

        singlesParams["mv_s"] = 1.0 / samples["inv_mv_s"]

        singlesParams["samples_s_init"] = models.skoLikelihoodInitial(
            singlesParams["init_count_s"]
        ).sample(random.PRNGKey(42))
        singlesParams["samples_s"] = models.skoLikelihoodFinal(
            singlesParams["init_count_s"],
            singlesParams["guide_eff_s"],
            singlesParams["cell_growth_s"],
            singlesParams["ko_growth_s"],
            singlesParams["mv_s"],
        ).sample(random.PRNGKey(42))

        params["singles"] = singlesParams

    if controls:

        pass

    combsParams = {}

    combsParams["init_count"] = samples["guide_init_count"][
        :, indices["guide_pair_idx"]
    ]
    combsParams["cell_line_eff"] = samples["cell_line_eff"][:, indices["cell_line_idx"]]
    combsParams["cell_line_growth"] = samples["cell_line_growth"][
        :, indices["cell_line_idx"]
    ]

    combsParams["guide_eff_1"] = samples["guide_eff"][
        :, indices["guide_1_idx"], indices["cell_line_idx"]
    ]
    combsParams["guide_eff_2"] = samples["guide_eff"][
        :, indices["guide_2_idx"], indices["cell_line_idx"]
    ]
    combsParams["guide_eff_12"] = samples["guide_eff_12"]

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

    combsParams["samples_init"] = models.dkoLikelihoodFinal(
        combsParams["init_count"]
    ).sample(random.PRNGKey(42))
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
    ).sample(random.PRNGKey(42))

    params["combs"] = combsParams

    return params

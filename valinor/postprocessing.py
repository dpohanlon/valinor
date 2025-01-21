import numpy as np

import pandas as pd

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

    # Return an empty DF if there are no parameters, e.g., from batching with a size greater than the length

    if list(paramSamples.values())[0].size == 0:

        for k in set(paramSamples.keys()) - set(lhSamples):
            df[f"{k}_mean"] = []
            df[f"{k}_std"] = []

        return df

    # Get the average and std values of the parameter samples

    means, stds = averageOverSamples(paramSamples)

    for k in set(paramSamples.keys()) - set(lhSamples):
        df[f"{k}_mean"] = means[k] if len(means[k]) > 1 else list(means[k])[0]
        df[f"{k}_std"] = stds[k] if len(stds[k]) > 1 else list(stds[k])[0]

        df[f"{k}_mean"] = df[f"{k}_mean"].astype(float)
        df[f"{k}_std"] = df[f"{k}_std"].astype(float)

    return df


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


# Split up this megafunction




def sampleParams(
    samples: Dict[str, np.ndarray],
    indices: Dict[str, np.ndarray],
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

    # Initialize JAX random keys
    # Split the key into multiple unique keys for different sampling operations
    base_key = random.PRNGKey(0)  # You can modify the seed as needed
    keys = random.split(base_key, 10)  # Adjust the number based on needs

    # Counter for keys
    key_counter = 0

    ############################
    # Process Singletons
    ############################
    if singletons:
        singlesParams = {}

        # Initialize Count for Singletons
        singlesParams["init_count_s"] = samples["guide_init_count_s"][:, indices["guide_pair_s_idx"]]

        # Tilde Alpha
        singlesParams["tilde_alpha"] = samples["tilde_alpha"][:, indices["guide_s_idx"], indices["cell_line_s_idx"]]

        # Guide Efficiency Mean and Std
        singlesParams["guide_eff_mean_s"] = samples["guide_eff_mean"][:, indices["guide_s_idx"]]
        singlesParams["guide_eff_std_s"] = samples["guide_eff_std"][:, indices["guide_s_idx"]]

        # Guide Efficiency with Sigmoid Transformation
        singlesParams["guide_eff_s"] = sigmoid(
            singlesParams["guide_eff_mean_s"] + singlesParams["tilde_alpha"] * singlesParams["guide_eff_std_s"]
        )

        # Cell Line Growth
        singlesParams["cell_growth_s"] = samples["cell_line_growth"][:, indices["cell_line_s_idx"]]

        # Library Bias
        singlesParams["library_bias_s"] = samples["library_bias"][:, indices["cell_line_s_idx"]]

        # Gene Knockout Growth
        if empirical_gene_priors:
            singlesParams["ko_growth_s"] = samples["gene_ko_growth"][
                :, indices["cell_line_s_idx"], indices["gene_s_common_idx"]
            ]
        else:
            singlesParams["ko_growth_s"] = samples["gene_ko_growth"][:, indices["gene_s_idx"]]

        # Compute MV with Softplus Transformation
        mvProd = samples["gene_std"][:, :, None] * samples["non_centered_deviation_gene"][:, None, :]
        mv_raw = samples["mv_cell_line"][:, :, None] + mvProd
        mv_transformed = jax.nn.softplus(mv_raw - 1.0) + 1.0 + 1e-6  # Matching model's transformation
        singlesParams["mv_s"] = mv_transformed[:, indices["cell_line_s_idx"], indices["gene_s_idx"]]

        # p_zi if applicable
        if zi:
            singlesParams["p_zi"] = samples["p_zi"][:, indices["cell_line_s_idx"]]

        # Sample from Initial Likelihood
        init_lh, theta_init = models.skoLikelihoodInitial(singlesParams["init_count_s"])
        singlesParams["samples_s_init"] = init_lh.sample(random.split(keys[key_counter])[0])
        key_counter += 1

        # Sample from Final Likelihood
        # Ensure that 'library_bias_s' and 'p_zi' are correctly passed
        lh_s, theta_s = models.skoLikelihoodFinal(
            init_theta_s=theta_init,
            guide_eff_s=singlesParams["guide_eff_s"],
            cell_line_growth_s=singlesParams["cell_growth_s"],
            gene_ko_growth_s=singlesParams["ko_growth_s"],
            mv=singlesParams["mv_s"],
            library_bias=singlesParams["library_bias_s"],  # Use sampled library_bias
            alternate=alternate,
            p_zi=singlesParams["p_zi"] if zi else False,
        )
        singlesParams["samples_s"] = lh_s.sample(random.split(keys[key_counter])[0])
        key_counter += 1

        params["singles"] = singlesParams

    ############################
    # Process Controls
    ############################
    if controls:
        controlsParams = {}

        # Initialize Count for Controls
        controlsParams["init_count_c"] = samples["guide_init_count_c"][:, indices["guide_pair_c_idx"]]

        # Cell Line Growth for Controls
        controlsParams["cell_growth_c"] = samples["cell_line_growth"][:, indices["cell_line_c_idx"]]

        # Compute MV for Controls with Softplus Transformation
        mvProd_c = samples["gene_std"][:, :, None] * samples["non_centered_deviation_gene_c"][:, None, :]
        mv_raw_c = samples["mv_cell_line"][:, :, None] + mvProd_c
        mv_transformed_c = jax.nn.softplus(mv_raw_c - 1.0) + 1.0 + 1e-6  # Matching model's transformation
        controlsParams["mv_c"] = mv_transformed_c[:, indices["cell_line_c_idx"], indices["guide_pair_c_idx"]]

        # Sample from Initial Control Likelihood
        init_lh_c, theta_init_c = models.skoLikelihoodInitial(controlsParams["init_count_c"])
        controlsParams["samples_c_init"] = init_lh_c.sample(random.split(keys[key_counter])[0])
        key_counter += 1

        # Sample from Final Control Likelihood
        lh_c, theta_c = models.controlLikelihoodFinal(
            init_theta=controlsParams["init_count_c"],
            cell_line_growth_c=controlsParams["cell_growth_c"],
            mv=controlsParams["mv_c"],
        )
        controlsParams["samples_c"] = lh_c.sample(random.split(keys[key_counter])[0])
        key_counter += 1

        params["controls"] = controlsParams

    ############################
    # Process Combinations (DKO)
    ############################
    if not only_singletons:
        combsParams = {}

        # Initialize Count for Combinations
        combsParams["init_count"] = samples["guide_init_count"][:, indices["guide_pair_idx"]]

        # Cell Line Growth for Combinations
        combsParams["cell_line_growth"] = samples["cell_line_growth"][:, indices["cell_line_idx"]]

        # Guide Efficiencies
        if "guide_eff_mean" in samples.keys():
            combsParams["guide_eff_mean_1"] = samples["guide_eff_mean"][:, indices["guide_1_idx"]]
            combsParams["guide_eff_mean_2"] = samples["guide_eff_mean"][:, indices["guide_2_idx"]]

            combsParams["guide_eff_std_1"] = samples["guide_eff_std"][:, indices["guide_1_idx"]]
            combsParams["guide_eff_std_2"] = samples["guide_eff_std"][:, indices["guide_2_idx"]]

            combsParams["tilde_alpha_1"] = samples["tilde_alpha"][:, indices["guide_1_idx"], indices["cell_line_idx"]]
            combsParams["tilde_alpha_2"] = samples["tilde_alpha"][:, indices["guide_2_idx"], indices["cell_line_idx"]]

            # Guide Efficiencies with Sigmoid Transformation
            combsParams["guide_eff_1"] = sigmoid(
                combsParams["guide_eff_mean_1"] + combsParams["tilde_alpha_1"] * combsParams["guide_eff_std_1"]
            )
            combsParams["guide_eff_2"] = sigmoid(
                combsParams["guide_eff_mean_2"] + combsParams["tilde_alpha_2"] * combsParams["guide_eff_std_2"]
            )
        else:
            # Handle cases where guide_eff_mean is not present
            combsParams["guide_eff_1"] = 1.0  # or another appropriate default
            combsParams["guide_eff_2"] = 1.0  # or another appropriate default

        # Gene Knockout Growth
        if empirical_gene_priors:
            combsParams["gene_ko_growth_1"] = samples["gene_ko_growth"][
                :, indices["cell_line_idx"], indices["gene_1_common_idx"]
            ]
            combsParams["gene_ko_growth_2"] = samples["gene_ko_growth"][
                :, indices["cell_line_idx"], indices["gene_2_common_idx"]
            ]
        else:
            combsParams["gene_ko_growth_1"] = samples["gene_ko_growth"][:, indices["gene_1_idx"]]
            combsParams["gene_ko_growth_2"] = samples["gene_ko_growth"][:, indices["gene_2_idx"]]

        # Gene Pair Knockout Growth
        combsParams["gene_ko_growth_12"] = samples["gene_pair_ko_growth"][:, indices["gene_pair_idx"]]

        # Compute MV for Combinations with Softplus Transformation
        mvProd_comb = samples["gene_std"][:, :, None] * samples["non_centered_deviation"][:, None, :]
        mv_raw_comb = samples["mv_cell_line"][:, :, None] + mvProd_comb
        mv_transformed_comb = jax.nn.softplus(mv_raw_comb - 1.0) + 1.0 + 1e-6  # Matching model's transformation
        combsParams["mv"] = mv_transformed_comb[:, indices["cell_line_idx"], indices["gene_pair_idx"]]

        # p_zi if applicable
        if zi:
            combsParams["p_zi"] = samples["p_zi"][:, indices["cell_line_idx"]]

        # Sample from Initial Likelihood for Combinations
        init_lh_comb, theta_init_comb = models.dkoLikelihoodInitial(combsParams["init_count"])
        combsParams["samples_init"] = init_lh_comb.sample(random.split(keys[key_counter])[0])
        key_counter += 1

        # Sample from Final Likelihood for Combinations
        if alternate:
            # Guide Efficiencies for Interaction
            combsParams["guide_eff_12"] = samples["guide_eff_12"][:, indices["guide_pair_idx"]]  # Ensure correct indexing

            lh_comb, theta_comb = models.dkoLikelihoodFinal(
                init_theta=theta_init_comb,  # Assuming theta_init_comb is correctly indexed
                guide_eff_1=combsParams["guide_eff_1"],
                guide_eff_2=combsParams["guide_eff_2"],
                guide_eff_12=combsParams["guide_eff_12"],
                cell_line_growth=combsParams["cell_line_growth"],
                gene_ko_growth_1=combsParams["gene_ko_growth_1"],
                gene_ko_growth_2=combsParams["gene_ko_growth_2"],
                gene_ko_growth_12=combsParams["gene_ko_growth_12"],
                mv=combsParams["mv"],
                library_bias=1.0,  # Assuming fixed as per original code
                p_zi=combsParams["p_zi"] if zi else False,
            )
        else:
            lh_comb, theta_comb = models.dkoLikelihoodFullFinal(
                init_theta=theta_init_comb,
                guide_eff_1=combsParams["guide_eff_1"],
                guide_eff_2=combsParams["guide_eff_2"],
                cell_line_growth=combsParams["cell_line_growth"],
                gene_ko_growth_1=combsParams["gene_ko_growth_1"],
                gene_ko_growth_2=combsParams["gene_ko_growth_2"],
                gene_ko_growth_12=combsParams["gene_ko_growth_12"],
                mv=combsParams["mv"],
                library_bias=1.0,  # Assuming fixed as per original code
                p_zi=combsParams["p_zi"] if zi else False,
            )

        combsParams["samples"] = lh_comb.sample(random.split(keys[key_counter])[0])
        key_counter += 1

        params["combs"] = combsParams

    return params

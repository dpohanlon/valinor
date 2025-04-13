import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as transforms


import jax
import jax.numpy as jnp

from numpyro.distributions import Distribution
from typing import Dict, Any

import numpy as np


def negativeBinomial(mean, od, zi=False):
    if zi is False:
        return dist.NegativeBinomial2(mean, od)
    else:
        return dist.ZeroInflatedNegativeBinomial2(mean, od, gate=zi)


def dkoLikelihoodInitial(init_theta: float) -> Distribution:
    """
    Returns a Poisson distribution with the provided parameter.

    Args:
        init_theta (float): The rate parameter (lambda) for the Poisson distribution.

    Returns:
        A Poisson distribution object.
    """

    # return negativeBinomial(init_theta, 1E-6, False), init_theta
    return dist.Poisson(init_theta), init_theta


def dkoLikelihoodFinal(
    init_theta: float,
    guide_eff_1: float,
    guide_eff_2: float,
    guide_eff_12: float,
    cell_line_growth: float,
    gene_ko_growth_1: float,
    gene_ko_growth_2: float,
    gene_ko_growth_12: float,
    mv: float,
    library_bias: float,
    p_zi: float,
) -> Distribution:
    """
    Returns a Negative Binomial distribution calculated from the provided parameters.

    Args:
        init_theta (float): Initial parameter.
        guide_eff_1 (float): Guide efficiency 1.
        guide_eff_2 (float): Guide efficiency 2.
        guide_eff_12 (float): Guide efficiency 12.
        cell_line_growth_v (float): Cell line growth.
        gene_ko_growth_1 (float): Gene knockout growth 1.
        gene_ko_growth_2 (float): Gene knockout growth 2.
        gene_ko_growth_12 (float): Gene knockout growth 12.
        mv (float): MV parameter.

    Returns:
        A Negative Binomial distribution object.
    """

    theta = 1.0 + guide_eff_1 * guide_eff_2 * guide_eff_12 * (
        jnp.exp(
            cell_line_growth
            + (library_bias * gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12)
        )
        - 1.0
    )

    theta *= init_theta

    theta = jax.nn.softplus(theta) + 1E-6
    mv = jax.nn.softplus(mv - 1) + 1.0 + 1e-6

    # dispersion = theta / (mv - 1)

    log_dispersion = jnp.log(theta + 1E-6) - jnp.log(mv - 1 + 1E-6)
    dispersion = jnp.exp(log_dispersion)

    return negativeBinomial(theta, dispersion, p_zi), theta


def dkoLikelihoodFullFinal(
    init_theta: float,
    guide_eff_1: float,
    guide_eff_2: float,
    cell_line_growth: float,
    gene_ko_growth_1: float,
    gene_ko_growth_2: float,
    gene_ko_growth_12: float,
    mv: float,
    library_bias: float,
    p_zi: float,
    singleKO : bool = False
) -> Distribution:
    """
    Returns a Negative Binomial distribution calculated from the provided parameters.

    Args:
        init_theta (float): Initial parameter.
        guide_eff_1 (float): Guide efficiency 1.
        guide_eff_2 (float): Guide efficiency 2.
        cell_line_growth (float): Cell line growth.
        gene_ko_growth_1 (float): Gene knockout growth 1.
        gene_ko_growth_2 (float): Gene knockout growth 2.
        gene_ko_growth_12 (float): Gene knockout growth 12.
        mv (float): MV parameter.

    Returns:
        A Negative Binomial distribution object.
    """

    p_00 = (1.0 - guide_eff_1) * (1.0 - guide_eff_2)
    p_1 = guide_eff_1 * (1.0 - guide_eff_2)
    p_2 = guide_eff_2 * (1.0 - guide_eff_1)

    p_12 = guide_eff_1 * guide_eff_2

    g1 = jnp.clip(library_bias * gene_ko_growth_1, -20, 20)
    g2 = jnp.clip(gene_ko_growth_2, -20, 20)
    g12 = jnp.clip(gene_ko_growth_12, -20, 20)

    # theta = (
    #     init_theta
    #     * jnp.exp(cell_line_growth)
    #     * (p_00 + p_1 * jnp.exp(g1) + p_2 * jnp.exp(g2) + p_12 * jnp.exp(g1 + g2 + g12))
    # )

    # theta = jax.nn.softplus(theta) + 1E-6

    # === Begin Modification: Log-space Reparameterization ===
    epsilon = 1e-6  # Small constant to avoid log(0)
    log_init_theta = jnp.log(init_theta + epsilon)

    # Combine the multiplicative factors:
    # Original multiplier: exp(cell_line_growth) * (p_00 + p_1 * exp(g1) + p_2 * exp(g2) + p_12 * exp(g1 + g2 + g12))
    # Compute the log multiplier:
    mult = p_00 + p_1 * jnp.exp(g1) + p_2 * jnp.exp(g2) + p_12 * jnp.exp(g1 + g2 + g12) if not singleKO else p_00 + p_1 * jnp.exp(g1)

    log_multiplier = cell_line_growth + jnp.log(mult)

    # logsumexp_args = jnp.array([
    #                     jnp.log(p_00),
    #                     jnp.log(p_1) + g1,
    #                     jnp.log(p_2) + g2,
    #                     jnp.log(p_12) + g1 + g2 + g12
    #                 ]) if not singleKO else jnp.array([
    #                    jnp.log(p_00),
    #                    jnp.log(p_1) + g1,
    #                ])

    # log_mult = jax.nn.logsumexp(logsumexp_args)

    # log_multiplier = cell_line_growth + log_mult

    # Add on the log scale and apply softplus for additional stability (optional)
    log_theta = jax.nn.softplus(log_init_theta + log_multiplier)

    # Exponentiate to recover theta on the original scale
    theta = jnp.exp(log_theta)
    # === End Modification ===

    mv = jax.nn.softplus(mv - 1) + 1.0 + 1e-6

    # dispersion = theta / (mv - 1)

    log_dispersion = jnp.log(theta + 1E-6) - jnp.log(mv - 1 + 1E-6)
    dispersion = jnp.exp(log_dispersion)

    return negativeBinomial(theta, dispersion, p_zi), theta


def skoLikelihoodInitial(init_theta: float) -> Distribution:
    """
    Returns a Poisson distribution with the provided parameter.

    Args:
        init_theta (float): The rate parameter (lambda) for the Poisson distribution.

    Returns:
        A Poisson distribution object.
    """

    return dkoLikelihoodInitial(init_theta)


def skoLikelihoodFinal(
    init_theta_s: float,
    guide_eff_s: float,
    cell_line_growth_s: float,
    gene_ko_growth_s: float,
    mv: float,
    library_bias: float,
    alternate: bool = False,
    p_zi=False,
) -> Distribution:
    """
    Returns a Negative Binomial distribution calculated from the provided parameters.

    Args:
        init_theta_s (float): Initial parameter.
        guide_eff_s (float): Guide efficiency.
        cell_line_growth_s (float): Cell line growth.
        gene_ko_growth_s (float): Gene knockout growth.
        mv (float): MV parameter.

    Returns:
        A Negative Binomial distribution object.
    """

    if alternate:
        return dkoLikelihoodFinal(
            init_theta=init_theta_s,
            guide_eff_1=guide_eff_s,
            guide_eff_2=1.0,
            guide_eff_12=1.0,
            cell_line_growth=cell_line_growth_s,
            gene_ko_growth_1=gene_ko_growth_s,
            gene_ko_growth_2=0.0,
            gene_ko_growth_12=0.0,
            mv=mv,
            library_bias=library_bias,
            p_zi=p_zi,
        )

    else:
        return dkoLikelihoodFullFinal(
            init_theta=init_theta_s,
            guide_eff_1=guide_eff_s,
            guide_eff_2=0.0,
            cell_line_growth=cell_line_growth_s,
            gene_ko_growth_1=gene_ko_growth_s,
            gene_ko_growth_2=0.0,
            gene_ko_growth_12=0.0,
            mv=mv,
            library_bias=library_bias,
            p_zi=p_zi,
            singleKO = True,
        )


def controlLikelihoodFinal(
    init_theta_c: float,
    cell_line_growth_c: float,
    mv: float,
) -> Distribution:
    theta = init_theta_c * jnp.exp(cell_line_growth_c)

    theta = jax.nn.softplus(theta) + 1E-6
    mv = jax.nn.softplus(mv - 1) + 1.0 + 1e-6

    # dispersion = theta / (mv - 1)

    log_dispersion = jnp.log(theta + 1E-6) - jnp.log(mv - 1 + 1E-6)
    dispersion = jnp.exp(log_dispersion)

    return negativeBinomial(theta, dispersion), theta


def sample_guide_distributions(
    lengths: Dict[str, int], prior_params: Dict[str, Any], config="partial_pooling"
):
    mean_l, mean_s = prior_params["guide_eff_mean"]
    std_l, std_s = prior_params["guide_eff_std"]

    # guide_eff must still have shape [n_guides, n_cell_lines]

    if config == "partial_pooling":

        with numpyro.plate("guides", lengths["len_guides"]) as g:
            guide_eff_mean = numpyro.sample(
                "guide_eff_mean",
                dist.Normal(loc=mean_l, scale=mean_s),
            )
            guide_eff_std = numpyro.sample(
                "guide_eff_std",
                dist.Normal(loc=std_l, scale=std_s),
            )

        with numpyro.plate("cell_lines", lengths["len_cell_lines"]) as c:
            with numpyro.plate("guides_per_cell", lengths["len_guides"]) as g_c:

                tilde_alpha = numpyro.sample(
                    "tilde_alpha",
                    dist.Normal(0, 1).expand(
                    # dist.Laplace(0, 1).expand(
                        [lengths["len_guides"], lengths["len_cell_lines"]]
                    ),
                )

                # Ensure p_1 has the correct shape
                guide_eff = (
                    guide_eff_mean[:, None] + guide_eff_std[:, None] * tilde_alpha
                )
                sigmoid = transforms.SigmoidTransform()
                guide_eff = numpyro.deterministic("guide_eff", sigmoid(guide_eff))

    elif config == "partial_pooling_low":
        with numpyro.plate("guides", lengths["len_guides"], dim=-2):
            guide_eff_mean = numpyro.sample(
                "guide_eff_mean",
                dist.TruncatedNormal(loc=mean_l, scale=mean_s, low=0.0, high=1.0),
            )
            guide_eff_std = numpyro.sample(
                "guide_eff_std", dist.TruncatedNormal(loc=std_l, scale=std_s, low=0.0)
            )

        with numpyro.plate("cell_lines", lengths["len_cell_lines"], dim=-1):
            guide_eff = numpyro.sample(
                "guide_eff",
                dist.TruncatedNormal(
                    loc=guide_eff_mean.squeeze(),
                    scale=guide_eff_std.squeeze(),
                    low=0.0,
                    high=1.0,
                ),
            )

    elif config == "no_pooling":
        with numpyro.plate("cell_lines", lengths["len_cell_lines"]):
            with numpyro.plate("guides", lengths["len_guides"]):
                guide_eff = numpyro.sample(
                    "guide_eff",
                    dist.TruncatedNormal(loc=mean_l, scale=mean_s, low=0.0, high=1.0),
                )

    elif config == "full_pooling":
        with numpyro.plate("guides", lengths["len_guides"]):
            guide_eff_single = numpyro.sample(
                "guide_eff_single",
                dist.TruncatedNormal(loc=mean_l, scale=mean_s, low=0.0, high=1.0),
            )
            guide_eff = numpyro.deterministic(
                "guide_eff",
                jnp.repeat(
                    guide_eff_single[:, None], lengths["len_cell_lines"], axis=1
                ),
            )

    else:
        raise ValueError(
            "Guide config must be one of ['partial_pooling', 'no_pooling', 'full_pooling']."
        )

    return guide_eff


def sample_gene_distributions(lengths: Dict[str, int], prior_params: Dict[str, Any]):

    if not "gene_effect_means" in prior_params:

        growth_l, growth_s = prior_params["gene_ko_growth"]

        with numpyro.plate("genes", lengths["len_genes"]):
            gene_ko_growth = numpyro.sample(
                "gene_ko_growth", dist.Normal(growth_l, growth_s)
            )

    else:

        # Return a 2D array with [n_cell_lines, n_genes].
        # Later index to a 1D array, where gene indices will be unique for each cell line,
        # as perhaps not all genes will be present for each cell line.

        growth_l = prior_params["gene_effect_means"]
        growth_s = prior_params["gene_effect_stds"]

        # But I can't pass a NaN here, so mask them off

        growth_l_clean = jnp.where(~jnp.isfinite(growth_l), 0.0, growth_l)
        growth_s_clean = jnp.clip(jnp.where(~jnp.isfinite(growth_s), 1.0, growth_s), 1E-6, np.inf)

        mask = jnp.isfinite(growth_l)

        gene_ko_growth = numpyro.sample(
            "gene_ko_growth", dist.Normal(growth_l_clean, growth_s_clean).mask(mask)
        )

    return gene_ko_growth


def sample_mv_cell_line_distributions(
    lengths: Dict[str, int], prior_params: Dict[str, Any]
):

    od_means = np.clip(prior_params["od_means"], 1.0, np.inf) - 1
    od_stds = prior_params["od_stds"]

    mv_mean_s = np.ones(lengths["len_cell_lines"]) * prior_params["mv_mean_scale"]

    mv_cell_line = numpyro.sample(
        "mv_cell_line", dist.Normal(od_means, od_stds)
    )

    mv_cell_line = jax.nn.softplus(mv_cell_line) + 1E-6

    # Define the scale for non-centered deviations (could be learned or set as a prior)
    gene_std = numpyro.sample("gene_std", dist.HalfNormal(mv_mean_s))

    return mv_cell_line, gene_std


def sample_pair_od_distributions(
    mv_cell_line, gene_std, lengths: Dict[str, int], prior_params: Dict[str, Any]
):

    # Sample the non-centered deviations for each gene pair
    non_centered_deviation = numpyro.sample(
        "non_centered_deviation", dist.Normal(0, 1).expand([lengths["len_gene_pairs"]])
        # "non_centered_deviation", dist.Laplace(0, 1).expand([lengths["len_gene_pairs"]])
    )

    # Compute the outer product of gene_std and non_centered_deviation
    outer_product = jnp.outer(
        gene_std, non_centered_deviation
    )  # Shape: [len_cell_lines, len_gene_pairs]

    with numpyro.plate("gene_pairs", lengths["len_gene_pairs"]):

        # Compute the gene pair-specific OD using the non-centered parameterization
        mv_gene_pair_ = numpyro.deterministic(
            "mv_gene_pair_", mv_cell_line[:, None] + outer_product
        )

        mv_gene_pair_ = jax.nn.softplus(mv_gene_pair_) + 1E-6

        mv_gene_pair = numpyro.deterministic("mv_gene_pair", mv_gene_pair_ + 1)

    return mv_gene_pair


def sample_od_distributions(
    mv_cell_line, gene_std, lengths: Dict[str, int], prior_params: Dict[str, Any]
):

    # Sample the non-centered deviations
    non_centered_deviation = numpyro.sample(
        "non_centered_deviation_gene", dist.Normal(0, 1).expand([lengths["len_genes"]])
        # "non_centered_deviation_gene", dist.Laplace(0, 1).expand([lengths["len_genes"]])
    )

    # Compute the outer product of gene_std and non_centered_deviation
    outer_product = jnp.outer(
        gene_std, non_centered_deviation
    )  # Shape: [len_cell_lines, len_genes]

    with numpyro.plate("genes", lengths["len_genes"]):

        # Compute the gene specific OD using the non-centered parameterization
        mv_gene_ = numpyro.deterministic(
            "mv_gene_",
            # mv_cell_line[:, None] * (1 + outer_product) # Double plus ungood (NaN)
            mv_cell_line[:, None] + outer_product,
        )

        mv_gene_ = jax.nn.softplus(mv_gene_) + 1E-6

        # Ensure the final result is in the correct range [1, inf]
        mv_gene = numpyro.deterministic("mv_gene", mv_gene_ + 1)

    return mv_gene


def sample_control_od_distributions(
    mv_cell_line, gene_std, lengths: Dict[str, int], prior_params: Dict[str, Any]
):

    # Sample the non-centered deviations
    non_centered_deviation = numpyro.sample(
        "non_centered_deviation_gene_c",
        dist.Normal(0, 1).expand([lengths["len_guide_pairs_c"]]),
    )

    # Compute the outer product of gene_std and non_centered_deviation
    outer_product = jnp.outer(
        gene_std, non_centered_deviation
    )  # Shape: [len_cell_lines, len_guide_c_pairs]

    with numpyro.plate("guide_pairs", lengths["len_guide_pairs_c"]):

        # Compute the guide pair-specific OD using the non-centered parameterization
        mv_guide_pair_c_ = numpyro.deterministic(
            "mv_guide_pair_c_", mv_cell_line[:, None] + outer_product
        )

        # Ensure the final result is in the correct range [1, inf]
        mv_guide_pair_c = numpyro.deterministic("mv_guide_pair_c", mv_guide_pair_c_ + 1)

    return mv_guide_pair_c


def sample_cell_line_distributions(
    lengths: Dict[str, int],
    prior_params: Dict[str, Any],
):

    if not "control_means" in prior_params:
        growth_cell_l, growth_cell_s = prior_params["cell_line_growth"]
    else:
        # Assume that these are indexed the same as priors (increasing cell line idx)
        growth_cell_l = prior_params["control_means"]
        growth_cell_s = prior_params["control_stds"]

    with numpyro.plate("cell_lines", lengths["len_cell_lines"]) as c:

        growth_cell_l, growth_cell_s = prior_params["cell_line_growth"]

        cell_line_growth = numpyro.sample(
            "cell_line_growth", dist.Normal(loc=growth_cell_l, scale=growth_cell_s)
        )

        cell_line_growth = jnp.clip(cell_line_growth, -20, 20)

        # TODO: Make me configurable
        library_bias = numpyro.sample("library_bias", dist.Normal(loc=1.0, scale=0.0001))

        p_zi = numpyro.sample(
            "p_zi",
            dist.Exponential(10.),
        )

    return cell_line_growth, library_bias, p_zi


def sample_dko_distributions(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    guide_eff,
    gene_ko_growth,
    cell_line_growth,
    mv_gene_pair,
    alternate: bool = False,
    p_zi=False,
    predict=False,
):
    with numpyro.plate("guide_counts", lengths["len_guide_pairs"]):
        init_l, init_s = prior_params["init_count"]

        guide_init_count = numpyro.sample(
            "guide_init_count",
            dist.Normal(loc=init_l, scale=init_s),
        )

    guide_init_count = jax.nn.softplus(guide_init_count) + 1E-6

    with numpyro.plate("gene_pairs", lengths["len_gene_pairs"]):
        pair_growth_l, pair_growth_s = prior_params["pair_growth"]

        gene_pair_ko_growth = numpyro.sample(
            "gene_pair_ko_growth",
            dist.Normal(pair_growth_l, pair_growth_s),
        )

    guide_eff_1 = guide_eff[indices["guide_1_idx"], indices["cell_line_idx"]]
    guide_eff_2 = guide_eff[indices["guide_2_idx"], indices["cell_line_idx"]]

    # Should be THE SAME for all guide pairs of a cell line?
    # Whereas now this is DIFFERENT even for REPLICATES,
    # so should at least be the same for all replicates!

    # Hierarchy reversed wrt above

    mv = mv_gene_pair[indices["cell_line_idx"], indices["gene_pair_idx"]]

    if not "gene_effect_means" in prior_params:

        gene_ko_growth_1 = gene_ko_growth[indices["gene_1_idx"]]
        gene_ko_growth_2 = gene_ko_growth[indices["gene_2_idx"]]

    else:

        # Index 2D array with indices not unique to cell line
        gene_ko_growth_1 = gene_ko_growth[
            indices["cell_line_idx"], indices["gene_1_common_idx"]
        ]
        gene_ko_growth_2 = gene_ko_growth[
            indices["cell_line_idx"], indices["gene_2_common_idx"]
        ]

    gene_ko_growth_12 = gene_pair_ko_growth[indices["gene_pair_idx"]]

    cell_line_growth_v = cell_line_growth[indices["cell_line_idx"]]

    init_lh, theta_init = dkoLikelihoodInitial(guide_init_count)

    if alternate:
        with numpyro.plate("guide_counts", lengths["len_guide_pairs"]):
            pair_eff_mean_l, pair_eff_mean_s = prior_params["pair_eff_mean"]
            pair_eff_std_l, pair_eff_std_s = prior_params["pair_eff_std"]

            guide_pair_eff_mean = numpyro.sample(
                "guide_pair_eff_mean",
                dist.TruncatedNormal(
                    loc=pair_eff_mean_l, scale=pair_eff_mean_s, low=0.0, high=1.0
                ),
            )
            guide_pair_eff_std = numpyro.sample(
                "guide_pair_eff_std",
                dist.TruncatedNormal(loc=pair_eff_std_l, scale=pair_eff_std_s, low=0.0),
            )

        guide_eff_12 = numpyro.sample(
            "guide_eff_12",
            dist.TruncatedNormal(
                loc=guide_pair_eff_mean[indices["guide_pair_idx"]],
                scale=guide_pair_eff_std[indices["guide_pair_idx"]],
                low=0.0,
                high=1.0,
            ),
        )

        lh, theta = dkoLikelihoodFinal(
            theta_init[indices["guide_pair_idx"]],
            guide_eff_1,
            guide_eff_2,
            guide_eff_12,
            cell_line_growth_v,
            gene_ko_growth_1,
            gene_ko_growth_2,
            gene_ko_growth_12,
            mv,
            library_bias=1.0,
            p_zi=p_zi if p_zi is False else p_zi[indices["cell_line_idx"]],
        )

    else:
        lh, theta = dkoLikelihoodFullFinal(
            theta_init[indices["guide_pair_idx"]],
            guide_eff_1,
            guide_eff_2,
            cell_line_growth_v,
            gene_ko_growth_1,
            gene_ko_growth_2,
            gene_ko_growth_12,
            mv,
            library_bias=1.0,
            p_zi=p_zi if p_zi is False else p_zi[indices["cell_line_idx"]],
        )

    # Only evaluate these when performing inference rather than predicting
    # as otherwise there is a problem with the sampling for unbounded
    # discrete distributions

    numpyro.sample("obs_init", init_lh, obs=data["initial"]["combinations"] if not predict else None)
    numpyro.sample("obs", lh, obs=data["final"]["combinations"] if not predict else None)


def sample_sko_distributions(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    guide_eff,
    gene_ko_growth,
    cell_line_growth,
    mv_gene,
    library_bias,
    alternate: bool = False,
    p_zi=False,
    predict=False,
):

    # p = numpyro.param('zip_zoop')

    # Index unique singleton guides, rather than pairs (agg over null guides, etc, as these are not parameterised, unlike in pairs)
    with numpyro.plate("guides_counts_s", lengths["len_guide_pairs_s"]):
        init_s_l, init_s_s = prior_params["init_count_s"]
        guide_init_count_s = numpyro.sample(
            "guide_init_count_s",
            dist.Normal(loc=init_s_l, scale=init_s_s),
        )

    guide_init_count_s = jax.nn.softplus(guide_init_count_s) + 1E-6

    guide_eff_s = guide_eff[indices["guide_s_idx"], indices["cell_line_s_idx"]]

    mv_s = mv_gene[indices["cell_line_s_idx"], indices["gene_s_idx"]]

    if not "gene_effect_means" in prior_params:

        gene_ko_growth_s = gene_ko_growth[indices["gene_s_idx"]]

    else:

        gene_ko_growth_s = gene_ko_growth[
            indices["cell_line_s_idx"], indices["gene_s_common_idx"]
        ]

    cell_line_growth_s = cell_line_growth[indices["cell_line_s_idx"]]

    library_bias_s = (
        library_bias[indices["cell_line_s_idx"]] if library_bias is not None else 1.0
    )

    # Expand this to the init dataset size, with the repeated entries for pairs with different null guides, to match obs (50 -> 300)

    init_lh_s, init_theta_s = skoLikelihoodInitial(guide_init_count_s[indices['guide_initial_s_idx']])

    # Expand initial parameters to the final dataset size from the parameter size, with repeated entries for cell lines (plasmid, no replicates, etc) (50 -> 3600)

    lh_s, theta_s = skoLikelihoodFinal(
        guide_init_count_s[indices['guide_s_idx']],
        guide_eff_s,
        cell_line_growth_s,
        gene_ko_growth_s,
        mv_s,
        library_bias_s,
        alternate,
        p_zi if p_zi is False else p_zi[indices["cell_line_s_idx"]],
    )

    numpyro.sample("obs_init_s", init_lh_s, obs=data["initial"]["singletons"]  if not predict else None)

    numpyro.sample(
        "obs_s",
        lh_s,
        obs=data["final"]["singletons"] if not predict else None,
    )


def sample_control_distributions(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    cell_line_growth,
    mv_guide_pair_c,
    predict=False,
):
    init_c_l, init_c_s = prior_params["init_count_c"]

    with numpyro.plate("guides_counts_c", lengths["len_guide_pairs_c"]):
        guide_init_count_c = numpyro.sample(
            "guide_init_count_c",
            dist.Normal(loc=init_c_l, scale=init_c_s),
        )

    guide_init_count_c = jax.nn.softplus(guide_init_count_c) + 1E-6

    mv_c = mv_guide_pair_c[indices["cell_line_c_idx"], indices["guide_pair_c_idx"]]

    cell_line_growth_c = cell_line_growth[indices["cell_line_c_idx"]]

    init_lh_c, init_theta_c = skoLikelihoodInitial(guide_init_count_c)

    # Could do it with DKO LH, fixing g = 0

    lh_c, theta_c = controlLikelihoodFinal(
        init_theta_c[indices["guide_pair_c_idx"]], cell_line_growth_c, mv_c
    )

    numpyro.sample("obs_init_c", init_lh_c, obs=data["initial"]["controls"] if not predict else None)

    numpyro.sample(
        "obs_c",
        lh_c,
        obs=data["final"]["controls"] if not predict else None,
    )


def valinorControls(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices: Dict[str, jnp.array],
    prior_params: Dict[str, Any],
    no_singletons: bool = False,
    only_singletons: bool = False,
    no_controls: bool = True,
    alternate: bool = False,
    guide_config: str = "partial_pooling",
    zi=False,
    predict=False,
) -> None:

    (
        cell_line_growth,
        library_bias,
        p_zi,
    ) = sample_cell_line_distributions(lengths, prior_params)

    mv_cell_line, gene_std = sample_mv_cell_line_distributions(lengths, prior_params)

    mv_guide_pair_c = sample_control_od_distributions(
        mv_cell_line, gene_std, lengths, prior_params
    )

    sample_control_distributions(
        data, lengths, indices, prior_params, cell_line_growth, mv_guide_pair_c
    )


def valinorSingles(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices: Dict[str, jnp.array],
    prior_params: Dict[str, Any],
    no_singletons: bool = False,
    only_singletons: bool = False,
    no_controls: bool = True,
    alternate: bool = False,
    guide_config: str = "partial_pooling",
    zi=False,
    predict=False,
) -> None:

    guide_eff = sample_guide_distributions(lengths, prior_params, config=guide_config)

    gene_ko_growth = sample_gene_distributions(lengths, prior_params)
    (
        cell_line_growth,
        library_bias,
        p_zi,
    ) = sample_cell_line_distributions(lengths, prior_params)

    # Common to all datasets
    mv_cell_line, gene_std = sample_mv_cell_line_distributions(lengths, prior_params)

    mv_gene = sample_od_distributions(mv_cell_line, gene_std, lengths, prior_params)

    sample_sko_distributions(
        data,
        lengths,
        indices,
        prior_params,
        guide_eff,
        gene_ko_growth,
        cell_line_growth,
        mv_gene,
        None,
        False,
        p_zi if zi else False,
        predict=predict,
    )


def valinorHierarchy(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices: Dict[str, jnp.array],
    prior_params: Dict[str, Any],
    no_singletons: bool = False,
    only_singletons: bool = False,
    no_controls: bool = True,
    alternate: bool = False,
    guide_config: str = "partial_pooling",
    zi=False,
    predict=False,
) -> None:

    guide_eff = sample_guide_distributions(lengths, prior_params, config=guide_config)

    gene_ko_growth = sample_gene_distributions(lengths, prior_params)
    (
        cell_line_growth,
        library_bias,
        p_zi,
    ) = sample_cell_line_distributions(lengths, prior_params)

    # Common to all datasets
    mv_cell_line, gene_std = sample_mv_cell_line_distributions(lengths, prior_params)

    if not only_singletons:

        mv_gene_pair = sample_pair_od_distributions(
            mv_cell_line, gene_std, lengths, prior_params
        )

        sample_dko_distributions(
            data,
            lengths,
            indices,
            prior_params,
            guide_eff,
            gene_ko_growth,
            cell_line_growth,
            mv_gene_pair,
            alternate,
            p_zi if zi else False,
            predict=predict,
        )
    if not no_singletons:

        mv_gene = sample_od_distributions(mv_cell_line, gene_std, lengths, prior_params)

        sample_sko_distributions(
            data,
            lengths,
            indices,
            prior_params,
            guide_eff,
            gene_ko_growth,
            cell_line_growth,
            mv_gene,
            library_bias if not only_singletons else None,
            alternate,
            p_zi if zi else False,
            predict=predict,
        )
    if not no_controls:

        mv_guide_pair_c = sample_control_od_distributions(
            mv_cell_line, gene_std, lengths, prior_params
        )

        sample_control_distributions(
            data,
            lengths,
            indices,
            prior_params,
            cell_line_growth,
            mv_guide_pair_c,
            predict=predict,
        )

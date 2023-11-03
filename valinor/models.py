import numpyro
import numpyro.distributions as dist

import jax
import jax.numpy as jnp

from numpyro.distributions import Distribution
from typing import Dict, Any

import numpy as np


def dkoLikelihoodInitial(init_theta: float) -> Distribution:
    """
    Returns a Poisson distribution with the provided parameter.

    Args:
        init_theta (float): The rate parameter (lambda) for the Poisson distribution.

    Returns:
        A Poisson distribution object.
    """

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
            cell_line_growth + (gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12)
        )
        - 1.0
    )

    theta *= init_theta

    return dist.NegativeBinomial2(theta, theta * mv / (1 - mv)), theta


def dkoLikelihoodFullFinal(
    init_theta: float,
    guide_eff_1: float,
    guide_eff_2: float,
    cell_line_growth: float,
    gene_ko_growth_1: float,
    gene_ko_growth_2: float,
    gene_ko_growth_12: float,
    mv: float,
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

    p_1 = guide_eff_1 * (1.0 - guide_eff_2)
    p_2 = guide_eff_2 * (1.0 - guide_eff_1)
    p_12 = jnp.clip(1.0 - p_1 * p_2, 0.0, 1.0)

    g1 = gene_ko_growth_1 - cell_line_growth
    g2 = gene_ko_growth_2 - cell_line_growth
    g12 = gene_ko_growth_12 - cell_line_growth

    theta = init_theta * (
        p_1 * jnp.exp(g1) + p_2 * jnp.exp(g2) + p_12 * jnp.exp(g1 + g2 + g12)
    )

    theta = jax.nn.softplus(theta)

    return dist.NegativeBinomial2(theta, theta * mv / (1 - mv)), theta


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
    alternate: bool = False,
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
            init_theta = init_theta_s,
            guide_eff_1 = guide_eff_s,
            guide_eff_2 = 0.0,
            guide_eff_12  = 0.0,
            cell_line_growth = cell_line_growth_s,
            gene_ko_growth_1 = gene_ko_growth_s,
            gene_ko_growth_2 = 0.0,
            gene_ko_growth_12 = 0.0,
            mv = mv
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
        )


def sample_guide_distributions(
    lengths: Dict[str, int], prior_params: Dict[str, Any], config="partial_pooling"
):

    mean_l, mean_s = prior_params["guide_eff_mean"]
    std_l, std_s = prior_params["guide_eff_std"]

    # guide_eff must still have shape [n_guides, n_cell_lines]

    if config == "partial_pooling":

        with numpyro.plate("guides", lengths["len_guides"]):

            guide_eff_mean = numpyro.sample(
                "guide_eff_mean",
                dist.TruncatedNormal(loc=mean_l, scale=mean_s, low=0.0, high=1.0),
            )
            guide_eff_std = numpyro.sample(
                "guide_eff_std", dist.TruncatedNormal(loc=std_l, scale=std_s, low=0.0)
            )

        with numpyro.plate("cell_lines", lengths["len_cell_lines"]):

            guide_eff = numpyro.sample(
                "guide_eff",
                dist.TruncatedNormal(
                    loc=jnp.repeat(
                        guide_eff_mean[:, None], lengths["len_cell_lines"], axis=1
                    ),
                    scale=jnp.repeat(
                        guide_eff_std[:, None], lengths["len_cell_lines"], axis=1
                    ),
                    low=0.0,
                    high=1.0,
                ),
            )

    elif config == "partial_pooling_low":

        with numpyro.plate("guides", lengths["len_guides"], dim = -2):

            guide_eff_mean = numpyro.sample(
                "guide_eff_mean",
                dist.TruncatedNormal(loc=mean_l, scale=mean_s, low=0.0, high=1.0),
            )
            guide_eff_std = numpyro.sample(
                "guide_eff_std", dist.TruncatedNormal(loc=std_l, scale=std_s, low=0.0)
            )

        with numpyro.plate("cell_lines", lengths["len_cell_lines"], dim = -1):

            guide_eff = numpyro.sample(
                "guide_eff",
                dist.TruncatedNormal(
                    loc=guide_eff_mean.reshape(-1, 1),
                    scale=guide_eff_std.reshape(-1, 1),
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
    with numpyro.plate("genes", lengths["len_genes"]):
        growth_l, growth_s = prior_params["gene_ko_growth"]
        gene_ko_growth = numpyro.sample(
            "gene_ko_growth", dist.Normal(growth_l, growth_s)
        )

    return gene_ko_growth


def sample_cell_line_distributions(
    lengths: Dict[str, int],
    prior_params: Dict[str, Any],
):
    with numpyro.plate("cell_lines", lengths["len_cell_lines"]):
        growth_cell_l, growth_cell_s = prior_params["cell_line_growth"]

        cell_line_growth = numpyro.sample(
            "cell_line_growth", dist.Normal(loc=growth_cell_l, scale=growth_cell_s)
        )

        od_means = prior_params["od_means"]
        od_stds = prior_params["od_stds"]

        mv_mean_s = prior_params["mv_mean_scale"]
        mv_std_s = prior_params["mv_std_scale"]

        inv_mv_mean = numpyro.sample(
            "inv_mv_mean", dist.TruncatedNormal(loc=od_means, scale=mv_mean_s, low=1.0)
        )
        inv_mv_std = numpyro.sample(
            "inv_mv_std", dist.TruncatedNormal(loc=od_stds, scale=mv_std_s, low=0.0)
        )

    return cell_line_growth, inv_mv_mean, inv_mv_std


def sample_dko_distributions(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    guide_eff,
    gene_ko_growth,
    cell_line_growth,
    inv_mv_mean,
    inv_mv_std,
    alternate: bool = False,
):
    with numpyro.plate("guide_counts", lengths["len_guide_pairs"]):
        init_l, init_s = prior_params["init_count"]

        guide_init_count = numpyro.sample(
            "guide_init_count",
            dist.TruncatedNormal(loc=init_l, scale=init_s, low=0.0),
        )

    with numpyro.plate("gene_pairs", lengths["len_gene_pairs"]):
        pair_growth_l, pair_growth_s = prior_params["pair_growth"]

        gene_pair_ko_growth = numpyro.sample(
            "gene_pair_ko_growth", dist.Normal(pair_growth_l, pair_growth_s)
        )

    guide_eff_1 = guide_eff[indices["guide_1_idx"], indices["cell_line_idx"]]
    guide_eff_2 = guide_eff[indices["guide_2_idx"], indices["cell_line_idx"]]

    inv_mv = numpyro.sample(
        "inv_mv",
        dist.TruncatedNormal(
            loc=inv_mv_mean[indices["cell_line_idx"]],
            scale=inv_mv_std[indices["cell_line_idx"]],
            low=1.0,
        ),
    )

    mv = numpyro.deterministic("mv", 1.0 / inv_mv)

    gene_ko_growth_1 = gene_ko_growth[indices["gene_1_idx"]]
    gene_ko_growth_2 = gene_ko_growth[indices["gene_2_idx"]]
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
        )

    numpyro.sample("obs_init", init_lh, obs=data["initial"]["combinations"])

    numpyro.sample("obs", lh, obs=data["final"]["combinations"])


def sample_sko_distributions(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    guide_eff,
    gene_ko_growth,
    cell_line_growth,
    inv_mv_mean,
    inv_mv_std,
    alternate: bool = False,
):
    with numpyro.plate("guides_counts_s", lengths["len_guide_pairs_s"]):
        init_s_l, init_s_s = prior_params["init_count_s"]

        guide_init_count_s = numpyro.sample(
            "guide_init_count_s",
            dist.TruncatedNormal(loc=init_s_l, scale=init_s_s, low=0.0),
        )

    guide_eff_s = guide_eff[indices["guide_s_idx"], indices["cell_line_s_idx"]]

    inv_mv_s = numpyro.sample(
        "inv_mv_s",
        dist.TruncatedNormal(
            loc=inv_mv_mean[indices["cell_line_s_idx"]],
            scale=inv_mv_std[indices["cell_line_s_idx"]],
            low=1.0,
        ),
    )

    mv_s = numpyro.deterministic("mv_s", 1.0 / inv_mv_s)

    gene_ko_growth_s = gene_ko_growth[indices["gene_s_idx"]]

    cell_line_growth_s = cell_line_growth[indices["cell_line_s_idx"]]

    init_lh_s, init_theta_s = skoLikelihoodInitial(guide_init_count_s)

    lh_s, theta_s = skoLikelihoodFinal(
        init_theta_s[indices["guide_pair_s_idx"]],
        guide_eff_s,
        cell_line_growth_s,
        gene_ko_growth_s,
        mv_s,
        alternate,
    )

    numpyro.sample("obs_init_s", init_lh_s, obs=data["initial"]["singletons"])

    numpyro.sample(
        "obs_s",
        lh_s,
        obs=data["final"]["singletons"],
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
) -> None:
    guide_eff = sample_guide_distributions(lengths, prior_params, config=guide_config)
    gene_ko_growth = sample_gene_distributions(lengths, prior_params)
    cell_line_growth, inv_mv_mean, inv_mv_std = sample_cell_line_distributions(
        lengths, prior_params
    )

    if not only_singletons:
        sample_dko_distributions(
            data,
            lengths,
            indices,
            prior_params,
            guide_eff,
            gene_ko_growth,
            cell_line_growth,
            inv_mv_mean,
            inv_mv_std,
            alternate,
        )
    if not no_singletons:
        sample_sko_distributions(
            data,
            lengths,
            indices,
            prior_params,
            guide_eff,
            gene_ko_growth,
            cell_line_growth,
            inv_mv_mean,
            inv_mv_std,
            alternate,
        )

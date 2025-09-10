import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as transforms

from numpyro import handlers

import jax
import jax.numpy as jnp

from numpyro.distributions import Distribution
from typing import Dict, Any, Tuple

import numpy as np


def observe(name, dist, obs, use=True, weight=1.0):
    with handlers.mask(mask=use), handlers.scale(scale=weight):
        return numpyro.sample(name, dist, obs=obs)


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
    singleKO: bool = False,
    guide_pair_eff=None,
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

    eps = 1e-8  # Small constant to avoid log(0)

    g1 = jnp.clip(gene_ko_growth_1, -20, 20)

    log_p00 = jnp.log1p(-guide_eff_1) + jnp.log1p(-guide_eff_2)
    log_p1 = jnp.log(guide_eff_1) + jnp.log1p(-guide_eff_2)
    log_p2 = jnp.log(guide_eff_2) + jnp.log1p(-guide_eff_1)
    log_p12 = jnp.log(guide_eff_1) + jnp.log(guide_eff_2)

    g2 = jnp.clip(gene_ko_growth_2, -20, 20)
    g12 = jnp.clip(gene_ko_growth_12, -20, 20)

    cat_logits = jnp.stack([log_p00, log_p1, log_p2, log_p12], axis=-1)
    mix_cat = dist.Categorical(logits=cat_logits)

    log_base = jnp.log(init_theta + eps) + cell_line_growth + library_bias  # log

    log_mu00 = log_base
    log_mu1 = log_base + g1
    log_mu2 = log_base + g2
    log_mu12 = log_base + g1 + g2 + g12
    log_mus = jnp.stack([log_mu00, log_mu1, log_mu2, log_mu12], axis=-1)

    mu_comps = jnp.exp(log_mus)
    mu_comps = jnp.clip(mu_comps, 1e-6, 1e6)

    mv_bc = (mv - 1 + 1e-6)[..., None]
    log_disp = jnp.log(mu_comps + eps) - jnp.log(mv_bc)
    raw_phi = jnp.exp(log_disp)
    phi_comps = jnp.clip(raw_phi, 1e-6, 1e6)

    mix_dist = dist.MixtureSameFamily(
        mix_cat, dist.NegativeBinomial2(mu_comps, phi_comps)
    )

    if not (p_zi is False):
        mix_dist = dist.ZeroInflatedDistribution(mix_dist, gate=p_zi)

    return mix_dist, jnp.sum(jax.nn.softmax(cat_logits, axis=-1) * mu_comps, axis=-1)


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

    return dkoLikelihoodFullFinal(
        init_theta=init_theta_s,
        guide_eff_1=guide_eff_s,
        guide_eff_2=0.0,
        cell_line_growth=cell_line_growth_s,
        gene_ko_growth_1=gene_ko_growth_s,
        gene_ko_growth_2=0.0,
        gene_ko_growth_12=0.0,
        mv=mv,
        library_bias=None,
        p_zi=p_zi,
        singleKO=True,
    )


def controlLikelihoodFinal(
    init_theta_c: float,
    lengths,
    indices,
    cell_line_growth_c: float,
    mv: float,
) -> Distribution:

    log_theta = jnp.log(init_theta_c + 1e-6) + cell_line_growth_c
    theta = jnp.exp(log_theta)

    mv = jax.nn.softplus(mv - 1) + 1.0 + 1e-6
    log_dispersion = jnp.log(theta) - jnp.log(mv - 1)
    dispersion = jnp.exp(log_dispersion)

    return negativeBinomial(theta, dispersion), theta


def sample_guide_distributions(
    lengths: Dict[str, int], prior_params: Dict[str, Any], config="partial_pooling"
):
    mean_l, mean_s = prior_params["guide_eff_mean"]
    std_l, std_s = prior_params["guide_eff_std"]

    # guide_eff must still have shape [n_guides, n_cell_lines]

    if config == "partial_pooling":

        with numpyro.plate("guides", lengths["len_guides"], dim=-2):

            guide_eff_mean = numpyro.sample(
                "guide_eff_mean", dist.Normal(loc=mean_l, scale=mean_s)
            )
            guide_eff_std = numpyro.sample(
                "guide_eff_std", dist.Normal(loc=std_l, scale=std_s)
            )

            with numpyro.plate("cell_lines", lengths["len_cell_lines"], dim=-1):

                tilde_alpha = numpyro.sample("tilde_alpha", dist.Normal(0, 1))

                guide_eff_logit = guide_eff_mean + guide_eff_std * tilde_alpha

        guide_eff = numpyro.deterministic("guide_eff", jax.nn.sigmoid(guide_eff_logit))

    elif config == "partial_pooling_low":
        with numpyro.plate("guides", lengths["len_guides"], dim=-2):
            guide_eff_mean = numpyro.sample(
                "guide_eff_mean",
                dist.Normal(loc=mean_l, scale=mean_s),
            )
            guide_eff_std = numpyro.sample(
                "guide_eff_std", dist.Normal(loc=std_l, scale=std_s)
            )

        with numpyro.plate("cell_lines", lengths["len_cell_lines"], dim=-1):
            guide_eff = numpyro.sample(
                "guide_eff_",
                dist.Normal(
                    loc=guide_eff_mean.squeeze(),
                    scale=guide_eff_std.squeeze(),
                ),
            )

            guide_eff = numpyro.deterministic("guide_eff", jax.nn.sigmoid(guide_eff))

    elif config == "no_pooling":
        # sample one guide_eff per guide×cell_line, with shape [n_guides, n_cell_lines]
        with numpyro.plate("guides", lengths["len_guides"], dim=-2):
            with numpyro.plate("cell_lines", lengths["len_cell_lines"], dim=-1):
                guide_eff = numpyro.sample(
                    "guide_eff_",
                    dist.Normal(loc=mean_l, scale=mean_s),
                )
        guide_eff = numpyro.deterministic("guide_eff", jax.nn.sigmoid(guide_eff))

    elif config == "full_pooling":
        # one shared effect per guide, then broadcast over cell lines
        with numpyro.plate("guides", lengths["len_guides"]):
            guide_eff_single = numpyro.sample(
                "guide_eff_single",
                dist.Normal(loc=mean_l, scale=mean_s),
            )

        guide_eff = numpyro.deterministic("guide_eff", jax.nn.sigmoid(guide_eff_single))

    else:
        raise ValueError(
            "Guide config must be one of ['partial_pooling', 'no_pooling', 'full_pooling']."
        )

    return guide_eff


def sample_gene_distributions(lengths: Dict[str, int], prior_params: Dict[str, Any]):

    if not "gene_effect_means" in prior_params:

        growth_l, growth_s = prior_params["gene_ko_growth"]

        with numpyro.plate("genes", lengths["len_genes"]):
            gene_ko_growth_raw = numpyro.sample(
                "gene_ko_growth_raw", dist.Normal(growth_l, growth_s)
            )

    else:

        # Return a 2D array with [n_cell_lines, n_genes].
        # Later index to a 1D array, where gene indices will be unique for each cell line,
        # as perhaps not all genes will be present for each cell line.

        growth_l = prior_params["gene_effect_means"]
        growth_s = prior_params["gene_effect_stds"]

        # But I can't pass a NaN here, so mask them off

        growth_l_clean = jnp.where(~jnp.isfinite(growth_l), 0.0, growth_l)
        growth_s_clean = jnp.clip(
            jnp.where(~jnp.isfinite(growth_s), 1.0, growth_s), 1e-6, np.inf
        )

        mask = jnp.isfinite(growth_l)

        gene_ko_growth_raw = numpyro.sample(
            "gene_ko_growth_raw", dist.Normal(growth_l_clean, growth_s_clean).mask(mask)
        )

    gene_ko_growth = numpyro.deterministic(
        "gene_ko_growth", gene_ko_growth_raw - jnp.mean(gene_ko_growth_raw)
    )

    return gene_ko_growth


def sample_mv_cell_line_distributions_s(
    lengths: Dict[str, int],
    prior_params: Dict[str, Any],
):
    # analogous to sample_mv_cell_line_distributions, but for SKO
    # TODO: Update these initial values to be singleton specific
    od_means_s = np.clip(prior_params["od_means"], 1.0, np.inf) - 1.0
    od_stds_s = np.clip(prior_params["od_stds"], 1e-4, np.inf)
    log_od_means_s = jnp.log(od_means_s + 1e-6)
    log_od_stds_s = od_stds_s / (od_means_s + 1e-6)
    mv_mean_s_scale = prior_params["mv_mean_scale"]

    with numpyro.plate("cell_lines_s", lengths["len_cell_lines"]):
        raw_mv_s = numpyro.sample(
            "raw_mv_cell_line_s",
            dist.Normal(log_od_means_s, log_od_stds_s),
        )
        gene_std_s = numpyro.sample(
            "gene_std_s",
            dist.HalfNormal(mv_mean_s_scale),
        )
    return raw_mv_s, gene_std_s


def sample_od_distributions_s(
    raw_mv_cell_line_s,
    gene_std_s,
    lengths: Dict[str, int],
    prior_params: Dict[str, Any],
):
    # analogous to sample_od_distributions, but namespaced for SKO
    with numpyro.plate("genes_common_s", lengths["len_genes_common"]):
        z_g_s = numpyro.sample("z_gene_s", dist.Normal(0.0, 1.0))
    raw_mv_gene_s = raw_mv_cell_line_s[:, None] + gene_std_s[:, None] * z_g_s

    # Don't clip if you don't exponentiate! Not just softplus.
    # Check whether elsewhere I also assume the logs are exp.

    raw_mv_gene_s = jnp.clip(raw_mv_gene_s, -20.0, 20.0)
    numpyro.deterministic("raw_mv_gene_s", raw_mv_gene_s)

    mv_gene_s = jnp.exp(raw_mv_gene_s) + 1.0

    numpyro.deterministic("mv_gene_s", mv_gene_s)
    return raw_mv_gene_s, mv_gene_s


def sample_mv_cell_line_distributions(
    lengths: Dict[str, int], prior_params: Dict[str, Any]
):

    od_means = jnp.clip(prior_params["od_means"], 1.0, np.inf) - 1.0
    od_stds = jnp.clip(prior_params["od_stds"], 1e-4, np.inf)

    log_od_means = jnp.log(od_means + 1e-6)
    log_od_stds = od_stds / (od_means + 1e-6)

    mv_mean_s = prior_params["mv_mean_scale"]
    with numpyro.plate("cell_lines", lengths["len_cell_lines"]):
        raw_mv = numpyro.sample(
            "raw_mv_cell_line", dist.Normal(log_od_means, log_od_stds)
        )

        gene_std = numpyro.sample("gene_std", dist.HalfNormal(mv_mean_s))
    return raw_mv, gene_std


def sample_pair_od_distributions(
    mv_cell_line_raw,
    raw_mv_gene,
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
):

    # Set scale of overdispersion
    sigma_pair = numpyro.sample(
        "sigma_pair", dist.HalfNormal(prior_params["od_pair_scale"] * 0.1)
    )

    # We may not have all genes pair with others though! -> Flatten

    with numpyro.plate("gene_pairs_common", lengths["len_gene_pairs"]):

        z_p = numpyro.sample("z_pair", dist.Normal(0.0, 1.0))

        c = mv_cell_line_raw[indices["cell_line_in_pair_idx"]]

        g1 = raw_mv_gene[
            indices["cell_line_in_pair_idx"], indices["gene_1_in_pair_idx"]
        ]
        g2 = raw_mv_gene[
            indices["cell_line_in_pair_idx"], indices["gene_2_in_pair_idx"]
        ]

        # There are better ways to combine the OD for pairs with the NB parameterisation

        # raw_mv_pair = g1 + g2 + sigma_pair * z_p
        # raw_mv_pair = g1 + sigma_pair * z_p
        # raw_mv_pair = sigma_pair * z_p
        # raw_mv_pair = g1 + g2 - c
        # raw_mv_pair = c

        raw_mv_pair = c + sigma_pair * z_p

        raw_mv_pair = jnp.clip(raw_mv_pair, -30.0, 30.0)

        mv_gene_pair = jnp.exp(raw_mv_pair) + 1.0
        numpyro.deterministic("mv_gene_pair", mv_gene_pair)

    return mv_gene_pair


def sample_od_distributions(
    raw_mv_cell_line, gene_std, lengths: Dict[str, int], prior_params: Dict[str, Any]
):

    with numpyro.plate("genes_common", lengths["len_genes_common"]):
        z_g = numpyro.sample("z_gene", dist.Normal(0.0, 1.0))
    raw_mv_gene = raw_mv_cell_line[:, None] + gene_std[:, None] * z_g

    numpyro.deterministic("raw_mv_gene", raw_mv_gene)

    mv_gene = jnp.exp(raw_mv_gene) + 1.0

    numpyro.deterministic("mv_gene", mv_gene)

    return raw_mv_gene, mv_gene


def sample_control_od_distributions(
    mv_cell_line, gene_std, lengths: Dict[str, int], prior_params: Dict[str, Any]
):

    non_centered_deviation = numpyro.sample(
        "non_centered_deviation_gene_c",
        dist.Normal(0, 1).expand([lengths["len_guide_pairs_c"]]),
    )

    # Compute the outer product of gene_std and non_centered_deviation
    outer_product = jnp.outer(gene_std, non_centered_deviation)

    with numpyro.plate("guide_pairs", lengths["len_guide_pairs_c"]):

        mv_guide_pair_c_ = numpyro.deterministic(
            "mv_guide_pair_c_", mv_cell_line[:, None] + outer_product
        )

        mv_guide_pair_c = numpyro.deterministic("mv_guide_pair_c", mv_guide_pair_c_ + 1)

    return mv_guide_pair_c


def sample_cell_line_distributions(
    lengths: Dict[str, int],
    prior_params: Dict[str, Any],
):

    with numpyro.plate("cell_lines", lengths["len_cell_lines"]) as c:

        growth_cell_l, growth_cell_s = prior_params["cell_line_growth"]

        cell_line_growth_raw = numpyro.sample(
            "cell_line_growth_raw", dist.Normal(loc=growth_cell_l, scale=growth_cell_s)
        )

        cell_line_growth = numpyro.deterministic(
            "cell_line_growth", cell_line_growth_raw - jnp.mean(cell_line_growth_raw)
        )

        cell_line_growth = jnp.clip(cell_line_growth, -20, 20)

        # TODO: Make me configurable
        library_bias = numpyro.sample("library_bias", dist.Normal(loc=0.0, scale=0.1))

    return cell_line_growth, library_bias


def sample_zero_inflation_s(lengths, prior_params):
    f0 = prior_params["p_zi_s"]
    K = prior_params.get("zi_strength_s", 20.0)
    alpha = f0 * K
    beta = (1.0 - f0) * K

    with numpyro.plate("cell_line_zi_s", lengths["len_cell_lines"]):
        p_zi = numpyro.sample(
            "p_zi_s",
            dist.Beta(concentration1=alpha, concentration0=beta),
        )
    return p_zi


def sample_zero_inflation(lengths, prior_params):
    f0 = prior_params["p_zi"]
    K = prior_params.get("zi_strength", 20.0)
    alpha = f0 * K
    beta = (1.0 - f0) * K

    with numpyro.plate("cell_line_zi", lengths["len_cell_lines"]):
        p_zi = numpyro.sample(
            "p_zi",
            dist.Beta(concentration1=alpha, concentration0=beta),
        )
    return p_zi


def sample_dko_distributions(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    guide_eff,
    gene_ko_growth,
    cell_line_growth,
    mv_gene_pair,
    library_bias,
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

    guide_init_count = jax.nn.softplus(guide_init_count) + 1e-6

    with numpyro.plate("gene_pairs", lengths["len_gene_pairs"]):
        pair_growth_l, pair_growth_s = prior_params["pair_growth"]

        gene_pair_ko_growth_raw = numpyro.sample(
            "gene_pair_ko_growth_raw",
            dist.Normal(pair_growth_l, pair_growth_s),
            # dist.Laplace(pair_growth_l, pair_growth_s),
        )

    gene_pair_ko_growth = numpyro.deterministic(
        "gene_pair_ko_growth",
        gene_pair_ko_growth_raw - jnp.mean(gene_pair_ko_growth_raw),
    )

    guide_eff_1 = guide_eff[indices["guide_1_idx"], indices["cell_line_idx"]]
    guide_eff_2 = guide_eff[indices["guide_2_idx"], indices["cell_line_idx"]]

    mv = mv_gene_pair[indices["gene_pair_idx"]]

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

    library_bias_v = (
        library_bias[indices["cell_line_idx"]] if library_bias is not None else 0.0
    )

    pair_eff_mean_l, pair_eff_mean_s = prior_params["pair_eff_mean"]
    pair_eff_std_l, pair_eff_std_s = prior_params["pair_eff_std"]

    with numpyro.plate("guide_counts", lengths["len_guide_pairs"]):

        guide_pair_eff = numpyro.sample(
            "guide_pair_eff",
            dist.HalfCauchy(0.1),
        )

    lh, theta = dkoLikelihoodFullFinal(
        theta_init[indices["guide_pair_idx"]],
        guide_eff_1,
        guide_eff_2,
        cell_line_growth_v,
        gene_ko_growth_1,
        gene_ko_growth_2,
        gene_ko_growth_12,
        mv,
        library_bias=library_bias_v,
        p_zi=False if p_zi is False else p_zi[indices["cell_line_idx"]],
        guide_pair_eff=guide_pair_eff[indices["guide_pair_idx"]],
    )

    # Only evaluate these when performing inference rather than predicting
    # as otherwise there is a problem with the sampling for unbounded
    # discrete distributions

    return init_lh, lh


def sample_sko_distributions(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    guide_eff,
    gene_ko_growth,
    cell_line_growth,
    mv_gene,
    alternate: bool = False,
    p_zi_s=False,
    predict=False,
):

    # Index unique singleton guides, rather than pairs (agg over null guides, etc, as these are not parameterised, unlike in pairs)
    with numpyro.plate("guides_counts_s", lengths["len_guides"]):
        init_s_l, init_s_s = prior_params["init_count_s"]
        guide_init_count_s = numpyro.sample(
            "guide_init_count_s",
            dist.Normal(loc=init_s_l, scale=init_s_s),
        )

    guide_init_count_s = jax.nn.softplus(guide_init_count_s) + 1e-6

    guide_eff_s = guide_eff[indices["guide_s_idx"], indices["cell_line_s_idx"]]

    mv_s = mv_gene[indices["cell_line_s_idx"], indices["gene_s_common_idx"]]

    if not "gene_effect_means" in prior_params:

        gene_ko_growth_s = gene_ko_growth[indices["gene_s_idx"]]

    else:

        gene_ko_growth_s = gene_ko_growth[
            indices["cell_line_s_idx"], indices["gene_s_common_idx"]
        ]

    cell_line_growth_s = cell_line_growth[indices["cell_line_s_idx"]]

    init_lh_s, init_theta_s = skoLikelihoodInitial(
        guide_init_count_s[indices["guide_initial_s_idx"]]
    )

    lh_s, theta_s = dkoLikelihoodFullFinal(
        init_theta=init_theta_s[indices["guide_pair_s_idx"]],
        guide_eff_1=guide_eff_s,
        guide_eff_2=0.0,
        cell_line_growth=cell_line_growth_s,
        gene_ko_growth_1=gene_ko_growth_s,
        gene_ko_growth_2=0.0,
        gene_ko_growth_12=0.0,
        mv=mv_s,
        library_bias=0.0,
        p_zi=False if p_zi_s is False else p_zi_s[indices["cell_line_s_idx"]],
        singleKO=True,
    )

    return init_lh_s, lh_s


def sample_control_distributions(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    cell_line_growth,
    mv_cell_line_raw,
    predict=False,
):
    init_c_l, init_c_s = prior_params["init_count_c"]

    # Length of init data
    with numpyro.plate("init_counts_c", lengths["len_guide_pairs_c"]):

        guide_init_count_c = numpyro.sample(
            "guide_init_count_c",
            dist.Normal(loc=init_c_l, scale=init_c_s),
        )

    mv_c = numpyro.sample("mv_c", dist.Normal(50, 50))

    guide_init_count_c = jax.nn.softplus(guide_init_count_c) + 1e-6

    cell_line_growth_c = cell_line_growth[indices["cell_line_c_idx"]]

    mv_c = jax.nn.softplus(mv_c - 1) + 1

    init_lh_c, init_theta_c = skoLikelihoodInitial(guide_init_count_c)

    lh_c, theta_c = controlLikelihoodFinal(
        init_theta_c[indices["guide_pair_c_idx"]],
        lengths,
        indices,
        cell_line_growth_c,
        mv_c,
    )

    return init_lh_c, lh_c


def valinorHierarchy(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices: Dict[str, jnp.array],
    prior_params: Dict[str, Any],
    no_singletons: bool = False,
    only_singletons: bool = False,
    no_controls: bool = True,
    alternate: bool = False,
    use_ctrl=True,
    use_sko=True,
    use_dko=True,
    w_ctrl=1.0,
    w_sko=1.0,
    w_dko=1.0,
    guide_config: str = "partial_pooling",
    zi=False,
    zi_s=False,
    predict=False,
) -> None:

    with handlers.scope(prefix="sko"):
        guide_eff_sko = sample_guide_distributions(
            lengths, prior_params, config=guide_config
        )

    with handlers.scope(prefix="dko"):
        guide_eff_dko = sample_guide_distributions(
            lengths, prior_params, config=guide_config
        )

    gene_ko_growth = sample_gene_distributions(lengths, prior_params)
    (
        cell_line_growth,
        library_bias,
    ) = sample_cell_line_distributions(lengths, prior_params)

    if zi:
        p_zi = sample_zero_inflation(lengths, prior_params)
    if zi_s and (not no_singletons):
        p_zi_s = sample_zero_inflation_s(lengths, prior_params)

    mv_cell_line_raw, gene_std = sample_mv_cell_line_distributions(
        lengths, prior_params
    )

    raw_mv_gene, mv_gene = sample_od_distributions(
        mv_cell_line_raw, gene_std, lengths, prior_params
    )

    raw_mv_cell_line_s, gene_std_s = sample_mv_cell_line_distributions_s(
        lengths, prior_params
    )
    raw_mv_gene_s, mv_gene_s = sample_od_distributions_s(
        raw_mv_cell_line_s, gene_std_s, lengths, prior_params
    )

    if not only_singletons:

        mv_gene_pair = sample_pair_od_distributions(
            mv_cell_line_raw, raw_mv_gene, lengths, indices, prior_params
        )

        init_lh, lh = sample_dko_distributions(
            data,
            lengths,
            indices,
            prior_params,
            guide_eff_dko,
            gene_ko_growth,
            cell_line_growth,
            mv_gene_pair,
            library_bias,
            alternate,
            p_zi if zi else False,
            predict=predict,
        )

        observe(
            "obs_init",
            init_lh,
            obs=data["initial"]["combinations"] if not predict else None,
            use=use_dko,
            weight=w_dko,
        )
        observe(
            "obs",
            lh,
            obs=data["final"]["combinations"] if not predict else None,
            use=use_dko,
            weight=w_dko,
        )

    if not no_singletons:

        init_lh_s, lh_s = sample_sko_distributions(
            data,
            lengths,
            indices,
            prior_params,
            guide_eff_sko,
            gene_ko_growth,
            cell_line_growth,
            mv_gene_s,
            alternate,
            p_zi_s if zi_s else False,
            predict=predict,
        )

        observe(
            "obs_init_s",
            init_lh_s,
            obs=data["initial"]["singletons"] if not predict else None,
            use=use_sko,
            weight=w_sko,
        )
        observe(
            "obs_s",
            lh_s,
            obs=data["final"]["singletons"] if not predict else None,
            use=use_sko,
            weight=w_sko,
        )

    if not no_controls:

        init_lh_c, lh_c = sample_control_distributions(
            data,
            lengths,
            indices,
            prior_params,
            cell_line_growth,
            mv_cell_line_raw,
            predict=predict,
        )

        observe(
            "obs_init_c",
            init_lh_c,
            obs=data["initial"]["controls"] if not predict else None,
            use=use_ctrl,
            weight=w_ctrl,
        )
        observe(
            "obs_c",
            lh_c,
            obs=data["final"]["controls"] if not predict else None,
            use=use_ctrl,
            weight=w_ctrl,
        )

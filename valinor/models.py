from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as transforms
from numpyro import handlers
from numpyro.distributions import Distribution


def observe(name, dist, obs, use=True, weight=1.0):
    with handlers.mask(mask=use), handlers.scale(scale=weight):
        return numpyro.sample(name, dist, obs=obs)


def negativeBinomial(mean, od, zi=False):
    if zi is False:
        return dist.NegativeBinomial2(mean, od)
    else:
        return dist.ZeroInflatedNegativeBinomial2(mean, od, gate=zi)


def dkoLikelihoodInitial(init_theta: float, log_exposure: float = 0.0) -> Distribution:
    """
    Returns a Poisson distribution with the provided parameter.

    Args:
        init_theta (float): The rate parameter (lambda) for the Poisson distribution.

    Returns:
        A Poisson distribution object.
    """

    # Don't exposure correct for now - likely same initial counts for all experiments anyway
    return dist.Poisson(init_theta), init_theta
    # return dist.Poisson(init_theta + log_exposure), init_theta


def dkoLikelihoodFullFinal(
    init_theta: float,
    guide_eff_1: float,
    guide_eff_2: float,
    cell_line_growth: float,
    gene_ko_growth_1: float,
    gene_ko_growth_2: float,
    gene_ko_growth_12: float,
    phi: float,
    library_bias: float,
    p_zi: float,
    log_exposure_final: float = 0.0,
    guide_pair_eff=None,
) -> Distribution:
    eps = 1e-8
    library_bias = 0.0 if library_bias is None else library_bias

    p1 = jnp.clip(guide_eff_1, 1e-6, 1.0 - 1e-6)
    p2 = jnp.clip(guide_eff_2, 1e-6, 1.0 - 1e-6)

    log_p00 = jnp.log1p(-p1) + jnp.log1p(-p2)
    log_p1 = jnp.log(p1) + jnp.log1p(-p2)
    log_p2 = jnp.log(p2) + jnp.log1p(-p1)
    log_p12 = jnp.log(p1) + jnp.log(p2)

    if guide_pair_eff is not None:
        log_p12 = log_p12 + jnp.log(jnp.clip(guide_pair_eff, eps, 1.0))

    cat_logits = jnp.stack([log_p00, log_p1, log_p2, log_p12], axis=-1)
    w = jax.nn.softmax(cat_logits, axis=-1)

    g1 = jnp.clip(gene_ko_growth_1, -20.0, 20.0)
    g2 = jnp.clip(gene_ko_growth_2, -20.0, 20.0)
    g12 = jnp.clip(gene_ko_growth_12, -20.0, 20.0)

    init_theta = jnp.clip(init_theta, 1e-6, 1e6)
    log_base = jnp.log(init_theta) + cell_line_growth + log_exposure_final

    log_mu00 = log_base
    log_mu1 = log_base + library_bias + g1
    log_mu2 = log_base + library_bias + g2
    log_mu12 = log_base + g1 + g2 + g12

    log_mus = jnp.stack([log_mu00, log_mu1, log_mu2, log_mu12], axis=-1)
    log_mus = jnp.clip(log_mus, -15.0, 15.0)

    mu_comps = jnp.exp(log_mus)
    mu_comps = jnp.clip(mu_comps, 1e-6, 1e5)

    mu_total = jnp.sum(w * mu_comps, axis=-1)
    mu_total = jnp.clip(mu_total, 1e-6, 1e5)

    phi_total = jnp.clip(jnp.asarray(phi), 1e-5, 1e8)

    lh = dist.NegativeBinomial2(mu_total, phi_total)

    if not (p_zi is False):
        gate = jnp.clip(jnp.asarray(p_zi), 1e-6, 1.0 - 1e-6)
        lh = dist.ZeroInflatedDistribution(lh, gate=gate)

    return lh, mu_total


def skoLikelihoodInitial(init_theta: float, log_exposure=0.0) -> Distribution:
    """
    Returns a Poisson distribution with the provided parameter.

    Args:
        init_theta (float): The rate parameter (lambda) for the Poisson distribution.

    Returns:
        A Poisson distribution object.
    """

    # Dont' use these in initial counts for now
    return dkoLikelihoodInitial(init_theta)
    # return dkoLikelihoodInitial(init_theta, log_exposure)


def skoLikelihoodFinal(
    init_theta_s: float,
    guide_eff_s: float,
    cell_line_growth_s: float,
    gene_ko_growth_s: float,
    phi: float,
    p_zi=False,
    log_exposure_final=0.0,
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
        phi=phi,
        library_bias=None,
        p_zi=p_zi,
        log_exposure_final=log_exposure_final,
    )


def controlLikelihoodFinal(
    init_theta_c: float,
    cell_line_growth_c: float,
    phi_c: float,
) -> Distribution:
    log_theta = jnp.log(init_theta_c + 1e-6) + cell_line_growth_c
    theta = jnp.exp(log_theta)
    phi_c = jnp.clip(phi_c, 1e-5, 1e8)
    return dist.NegativeBinomial2(theta, phi_c), theta


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

            guide_eff_std = jax.nn.softplus(guide_eff_std)

            with numpyro.plate("cell_lines", lengths["len_cell_lines"], dim=-1):
                tilde_alpha = numpyro.sample("tilde_alpha", dist.Normal(0, 1))

                guide_eff_logit = guide_eff_mean + guide_eff_std * tilde_alpha

        guide_eff = numpyro.deterministic(
            "guide_eff", jnp.clip(jax.nn.sigmoid(guide_eff_logit), 1e-6, 1.0 - 1e-6)
        )

    elif config == "partial_pooling_low":
        with numpyro.plate("guides", lengths["len_guides"], dim=-2):
            guide_eff_mean = numpyro.sample(
                "guide_eff_mean",
                dist.Normal(loc=mean_l, scale=mean_s),
            )
            guide_eff_std = numpyro.sample(
                "guide_eff_std", dist.Normal(loc=std_l, scale=std_s)
            )

            guide_eff_std = jax.nn.softplus(guide_eff_std)

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


def sample_phi_cell_line_distributions_s(
    lengths: Dict[str, int],
    prior_params: Dict[str, Any],
):
    log_phi_means_raw = jnp.asarray(
        prior_params.get("log_phi_means_s", prior_params["log_phi_means"])
    )
    log_phi_stds_raw = jnp.asarray(
        prior_params.get("log_phi_stds_s", prior_params["log_phi_stds"])
    )

    log_phi_means_s = jnp.where(jnp.isfinite(log_phi_means_raw), log_phi_means_raw, 3.0)
    log_phi_stds_s = jnp.where(jnp.isfinite(log_phi_stds_raw), log_phi_stds_raw, 1.0)
    log_phi_stds_s = jnp.clip(log_phi_stds_s, 1e-3, 5.0)

    phi_mean_scale_s = jnp.clip(
        jnp.asarray(
            prior_params.get("phi_mean_scale_s", prior_params["phi_mean_scale"])
        ),
        1e-6,
        5.0,
    )

    with numpyro.plate("cell_lines_s", lengths["len_cell_lines"]):
        raw_log_phi_s = numpyro.sample(
            "raw_log_phi_cell_line_s",
            dist.Normal(log_phi_means_s, log_phi_stds_s),
        )
        gene_std_phi_s = numpyro.sample(
            "gene_std_phi_s",
            dist.HalfNormal(phi_mean_scale_s),
        )
    return raw_log_phi_s, gene_std_phi_s


def sample_phi_distributions_s(
    raw_log_phi_cell_line_s,
    gene_std_phi_s,
    lengths: Dict[str, int],
    prior_params: Dict[str, Any],
    record_deterministics: bool = False,
):
    with numpyro.plate("genes_common_s", lengths["len_genes_common"]):
        z_g_s = numpyro.sample("z_gene_phi_s", dist.Normal(0.0, 1.0))

    raw_log_phi_gene_s = (
        raw_log_phi_cell_line_s[:, None] + gene_std_phi_s[:, None] * z_g_s
    )
    raw_log_phi_gene_s = jnp.clip(raw_log_phi_gene_s, -10.0, 20.0)

    if record_deterministics:
        numpyro.deterministic("raw_log_phi_gene_s", raw_log_phi_gene_s)

    phi_gene_s = jnp.exp(raw_log_phi_gene_s)

    if record_deterministics:
        numpyro.deterministic("phi_gene_s", phi_gene_s)

    return raw_log_phi_gene_s, phi_gene_s


def sample_phi_cell_line_distributions(
    lengths: Dict[str, int], prior_params: Dict[str, Any]
):
    log_phi_means_raw = jnp.asarray(prior_params["log_phi_means"])
    log_phi_stds_raw = jnp.asarray(prior_params["log_phi_stds"])

    log_phi_means = jnp.where(jnp.isfinite(log_phi_means_raw), log_phi_means_raw, 3.0)
    log_phi_stds = jnp.where(jnp.isfinite(log_phi_stds_raw), log_phi_stds_raw, 1.0)
    log_phi_stds = jnp.clip(log_phi_stds, 1e-3, 5.0)

    phi_mean_scale = jnp.clip(jnp.asarray(prior_params["phi_mean_scale"]), 1e-6, 5.0)

    with numpyro.plate("cell_lines", lengths["len_cell_lines"]):
        raw_log_phi = numpyro.sample(
            "raw_log_phi_cell_line", dist.Normal(log_phi_means, log_phi_stds)
        )
        gene_std_phi = numpyro.sample("gene_std_phi", dist.HalfNormal(phi_mean_scale))

    return raw_log_phi, gene_std_phi


def sample_pair_phi_distributions(
    raw_log_phi_cell_line,
    raw_log_phi_gene,
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    record_deterministics: bool = False,
):
    sigma_pair_phi = numpyro.sample(
        "sigma_pair_phi", dist.HalfNormal(prior_params["phi_pair_scale"] * 1.0)
    )
    with numpyro.plate("gene_pairs_common", lengths["len_gene_pairs"]):
        z_p = numpyro.sample("z_pair_phi", dist.Normal(0.0, 1.0))
        c = raw_log_phi_cell_line[indices["cell_line_in_pair_idx"]]
        raw_log_phi_pair = c + sigma_pair_phi * z_p
        raw_log_phi_pair = jnp.clip(raw_log_phi_pair, -10.0, 20.0)
        phi_pair = jnp.exp(raw_log_phi_pair)
        if record_deterministics:
            numpyro.deterministic("phi_gene_pair", phi_pair)
    return phi_pair


def sample_phi_distributions(
    raw_log_phi_cell_line,
    gene_std_phi,
    lengths: Dict[str, int],
    prior_params: Dict[str, Any],
    record_deterministics: bool = False,
):
    with numpyro.plate("genes_common", lengths["len_genes_common"]):
        z_g = numpyro.sample("z_gene_phi", dist.Normal(0.0, 1.0))

    raw_log_phi_gene = raw_log_phi_cell_line[:, None] + gene_std_phi[:, None] * z_g
    raw_log_phi_gene = jnp.clip(raw_log_phi_gene, -10.0, 20.0)

    if record_deterministics:
        numpyro.deterministic("raw_log_phi_gene", raw_log_phi_gene)

    phi_gene = jnp.exp(raw_log_phi_gene)
    if record_deterministics:
        numpyro.deterministic("phi_gene", phi_gene)

    return raw_log_phi_gene, phi_gene


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
    f0 = jnp.clip(jnp.asarray(prior_params["p_zi_s"]), 1e-6, 1.0 - 1e-6)
    K = jnp.clip(jnp.asarray(prior_params.get("zi_strength_s", 20.0)), 1e-3, 1e6)
    alpha = f0 * K + 1e-6
    beta = (1.0 - f0) * K + 1e-6
    with numpyro.plate("cell_line_zi_s", lengths["len_cell_lines"]):
        p_zi = numpyro.sample(
            "p_zi_s",
            dist.Beta(concentration1=alpha, concentration0=beta),
        )
    return p_zi


def sample_zero_inflation(lengths, prior_params):
    f0 = jnp.clip(jnp.asarray(prior_params["p_zi"]), 1e-6, 1.0 - 1e-6)
    K = jnp.clip(jnp.asarray(prior_params.get("zi_strength", 20.0)), 1e-3, 1e6)
    alpha = f0 * K + 1e-6
    beta = (1.0 - f0) * K + 1e-6
    with numpyro.plate("cell_line_zi", lengths["len_cell_lines"]):
        p_zi = numpyro.sample(
            "p_zi",
            dist.Beta(concentration1=alpha, concentration0=beta),
        )
    return p_zi


def sample_dko_distributions(
    exposure,
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    guide_eff,
    gene_ko_growth,
    cell_line_growth,
    phi_gene_pair,
    library_bias,
    alternate: bool = False,
    p_zi=False,
    predict=False,
):
    # with numpyro.plate("guide_counts", lengths["len_guide_pairs"]):
    #     init_l, init_s = prior_params["init_count"]

    #     guide_init_count = numpyro.sample(
    #         "guide_init_count",
    #         dist.Normal(loc=init_l, scale=init_s),
    #     )

    # guide_init_count = jax.nn.softplus(guide_init_count) + 1e-6
    # guide_init_count = jnp.clip(guide_init_count, 1e-6, 1e6)

    with numpyro.plate("guide_counts", lengths["len_guide_pairs"]):
        log_init_l, log_init_s = prior_params.get("log_init_count", (0.0, 2.0))
        guide_init_log = numpyro.sample(
            "guide_init_log", dist.Normal(log_init_l, log_init_s)
        )
        guide_init_count = numpyro.deterministic(
            "guide_init_count", jnp.exp(guide_init_log)
        )

    guide_init_count = jnp.exp(guide_init_log)  # no softplus
    guide_init_count = jnp.clip(guide_init_count, 1e-8, 1e7)

    with numpyro.plate("gene_pairs", lengths["len_gene_pairs"]):
        pair_growth_l, pair_growth_s = prior_params["pair_growth"]

        gene_pair_ko_growth_raw = numpyro.sample(
            "gene_pair_ko_growth_raw",
            # dist.Normal(pair_growth_l, pair_growth_s),
            dist.Laplace(pair_growth_l, pair_growth_s),
        )

    gene_pair_ko_growth_raw = jnp.clip(gene_pair_ko_growth_raw, -10.0, 10.0)

    gene_pair_ko_growth = numpyro.deterministic(
        "gene_pair_ko_growth",
        gene_pair_ko_growth_raw - jnp.mean(gene_pair_ko_growth_raw),
    )

    guide_eff_1 = guide_eff[indices["guide_1_idx"], indices["cell_line_idx"]]
    guide_eff_2 = guide_eff[indices["guide_2_idx"], indices["cell_line_idx"]]

    phi = phi_gene_pair[indices["gene_pair_idx"]]

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

    init_lh, theta_init = dkoLikelihoodInitial(
        guide_init_count,
        log_exposure=exposure["initial"]["combinations"] if exposure != None else 0.0,
    )

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
        phi,
        library_bias=library_bias_v,
        p_zi=False if p_zi is False else p_zi[indices["cell_line_idx"]],
        log_exposure_final=exposure["final"]["combinations"]
        if exposure != None
        else 0.0,
        guide_pair_eff=guide_pair_eff[indices["guide_pair_idx"]],
    )

    # Only evaluate these when performing inference rather than predicting
    # as otherwise there is a problem with the sampling for unbounded
    # discrete distributions

    return init_lh, lh


def sample_sko_distributions(
    exposure,
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    guide_eff,
    gene_ko_growth,
    cell_line_growth,
    phi_gene_s,
    library_bias,
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

    phi_s = phi_gene_s[indices["cell_line_s_idx"], indices["gene_s_common_idx"]]

    if not "gene_effect_means" in prior_params:
        gene_ko_growth_s = gene_ko_growth[indices["gene_s_idx"]]

    else:
        gene_ko_growth_s = gene_ko_growth[
            indices["cell_line_s_idx"], indices["gene_s_common_idx"]
        ]

    cell_line_growth_s = cell_line_growth[indices["cell_line_s_idx"]]

    init_lh_s, init_theta_s = skoLikelihoodInitial(
        guide_init_count_s[indices["guide_initial_s_idx"]],
        exposure["initial"]["singletons"] if exposure != None else 0.0,
    )

    lh_s, theta_s = dkoLikelihoodFullFinal(
        init_theta=init_theta_s[indices["guide_pair_s_idx"]],
        guide_eff_1=guide_eff_s,
        guide_eff_2=0.0,
        cell_line_growth=cell_line_growth_s,
        gene_ko_growth_1=gene_ko_growth_s,
        gene_ko_growth_2=0.0,
        gene_ko_growth_12=0.0,
        phi=phi_s,
        library_bias=library_bias[indices["cell_line_s_idx"]]
        if library_bias != None
        else None,
        p_zi=False if p_zi_s is False else p_zi_s[indices["cell_line_s_idx"]],
        log_exposure_final=exposure["final"]["singletons"] if exposure != None else 0.0,
    )

    return init_lh_s, lh_s


def sample_control_phi_distributions(
    raw_log_phi_cell_line,
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
):
    tau_ctrl = numpyro.sample("tau_ctrl_logphi", dist.HalfNormal(0.8))

    with numpyro.plate("cell_lines_ctrl", lengths["len_cell_lines"]):
        delta_ctrl = numpyro.sample("delta_ctrl_logphi", dist.Normal(0.0, tau_ctrl))

    raw_base_ctrl = raw_log_phi_cell_line + delta_ctrl

    phi_pair_scale = prior_params.get(
        "phi_pair_scale", prior_params.get("phi_pair_scale", 0.5)
    )
    sigma_pair_ctrl = numpyro.sample(
        "sigma_ctrl_pair_logphi", dist.HalfNormal(phi_pair_scale * 2.0)
    )

    with numpyro.plate("guide_pairs_c", lengths["len_guide_pairs_c"]):
        zc = numpyro.sample("z_ctrl_pair_logphi", dist.Normal(0.0, 1.0))
        raw_log_phi_pair_c = (
            raw_base_ctrl[indices["cell_line_c_idx"]] + sigma_pair_ctrl * zc
        )
        raw_log_phi_pair_c = jnp.clip(raw_log_phi_pair_c, -10.0, 20.0)
        phi_guide_pair_c = numpyro.deterministic(
            "phi_guide_pair_c", jnp.exp(raw_log_phi_pair_c)
        )

    return phi_guide_pair_c


def sample_control_distributions(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices,
    prior_params: Dict[str, Any],
    cell_line_growth,
    phi_guide_pair_c,
    predict=False,
):
    init_c_l, init_c_s = prior_params["init_count_c"]

    with numpyro.plate("init_counts_c", lengths["len_guide_pairs_c"]):
        guide_init_count_c = numpyro.sample(
            "guide_init_count_c",
            dist.Normal(loc=init_c_l, scale=init_c_s),
        )

    guide_init_count_c = jax.nn.softplus(guide_init_count_c) + 1e-6

    cell_line_growth_c = cell_line_growth[indices["cell_line_c_idx"]]

    init_lh_c, init_theta_c = skoLikelihoodInitial(guide_init_count_c)

    phi_c = phi_guide_pair_c[indices["guide_pair_c_idx"]]

    lh_c, theta_c = controlLikelihoodFinal(
        init_theta_c[indices["guide_pair_c_idx"]],
        cell_line_growth_c,
        phi_c,
    )

    return init_lh_c, lh_c


def valinorHierarchy(
    data: Dict[str, jnp.array],
    exposure: Dict[str, jnp.array],
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
    share_guide_eff: bool = False,
    record_deterministics: bool = True,
) -> None:
    if share_guide_eff:
        with handlers.scope(prefix="guides_shared"):
            shared_guide_eff = sample_guide_distributions(
                lengths, prior_params, config=guide_config
            )
        guide_eff_sko = shared_guide_eff
        guide_eff_dko = shared_guide_eff
    else:
        with handlers.scope(prefix="sko"):
            guide_eff_sko = sample_guide_distributions(
                lengths, prior_params, config=guide_config
            )
        with handlers.scope(prefix="dko"):
            guide_eff_dko = sample_guide_distributions(
                lengths, prior_params, config=guide_config
            )

    gene_ko_growth = sample_gene_distributions(lengths, prior_params)
    cell_line_growth, library_bias = sample_cell_line_distributions(
        lengths, prior_params
    )

    if zi:
        p_zi = sample_zero_inflation(lengths, prior_params)
    if zi_s and (not no_singletons):
        p_zi_s = sample_zero_inflation_s(lengths, prior_params)

    raw_log_phi_cell_line, gene_std_phi = sample_phi_cell_line_distributions(
        lengths, prior_params
    )
    raw_log_phi_gene, phi_gene = sample_phi_distributions(
        raw_log_phi_cell_line,
        gene_std_phi,
        lengths,
        prior_params,
        record_deterministics=record_deterministics,
    )

    raw_log_phi_cell_line_s, gene_std_phi_s = sample_phi_cell_line_distributions_s(
        lengths, prior_params
    )
    raw_log_phi_gene_s, phi_gene_s = sample_phi_distributions_s(
        raw_log_phi_cell_line_s,
        gene_std_phi_s,
        lengths,
        prior_params,
        record_deterministics=record_deterministics,
    )

    if not only_singletons:
        phi_gene_pair = sample_pair_phi_distributions(
            raw_log_phi_cell_line,
            raw_log_phi_gene,
            lengths,
            indices,
            prior_params,
            record_deterministics=record_deterministics,
        )

        init_lh, lh = sample_dko_distributions(
            exposure,
            lengths,
            indices,
            prior_params,
            guide_eff_dko,
            gene_ko_growth,
            cell_line_growth,
            phi_gene_pair,
            None,
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
            exposure,
            lengths,
            indices,
            prior_params,
            guide_eff_sko,
            gene_ko_growth,
            cell_line_growth,
            phi_gene_s,
            None,
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
        phi_guide_pair_c = sample_control_phi_distributions(
            raw_log_phi_cell_line,
            lengths,
            indices,
            prior_params,
        )

        init_lh_c, lh_c = sample_control_distributions(
            data,
            lengths,
            indices,
            prior_params,
            cell_line_growth,
            phi_guide_pair_c,
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

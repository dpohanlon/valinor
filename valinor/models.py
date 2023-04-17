import numpyro
import numpyro.distributions as dist

import jax
import jax.numpy as jnp

import numpy as np


def dkoLikelihoodInitial(init_theta):

    return dist.Poisson(init_theta)


def dkoLikelihoodFinal(
    init_theta,
    guide_eff_1,
    guide_eff_2,
    guide_eff_12,
    cell_line_growth_v,
    gene_ko_growth_1,
    gene_ko_growth_2,
    gene_ko_growth_12,
    mv
):

    theta = 1.0 + guide_eff_1 * guide_eff_2 * guide_eff_12 * (
        jnp.exp(
            cell_line_growth_v
            + (gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12)
        )
        - 1.0
    )

    theta *= init_theta

    return dist.NegativeBinomial2(theta, theta * mv / (1 - mv))


def skoLikelihoodInitial(init_theta):

    return dkoLikelihoodInitial(init_theta)


def skoLikelihoodFinal(
    init_theta_s,
    guide_eff_s,
    cell_line_growth_s,
    gene_ko_growth_s,
    mv,
):

    return dkoLikelihoodFinal(
        init_theta=init_theta_s,
        guide_eff_1=guide_eff_s,
        guide_eff_2=0.0,
        guide_eff_12=0.0,
        cell_line_growth_v=cell_line_growth_s,
        gene_ko_growth_1=gene_ko_growth_s,
        gene_ko_growth_2=0.0,
        gene_ko_growth_12=0.0,
        mv = mv,
    )


def valinorHierarchy(
    data,
    lengths,
    indices,
    prior_params,
    no_singletons=False,
    only_singletons=False,
    no_controls = True,
):

    with numpyro.plate("guides", lengths["len_guides"]):

        mean_l, mean_s = prior_params["guide_eff_mean"]
        std_l, std_s = prior_params["guide_eff_std"]

        guide_eff_mean = numpyro.sample(
            "guide_eff_mean",
            dist.TruncatedNormal(loc=mean_l, scale=mean_s, low=0.0, high=1.0),
        )
        guide_eff_std = numpyro.sample(
            "guide_eff_std", dist.TruncatedNormal(loc=std_l, scale=std_s, low=0.0)
        )

    with numpyro.plate("genes", lengths["len_genes"]):

        growth_l, growth_s = prior_params["gene_ko_growth"]

        gene_ko_growth = numpyro.sample(
            "gene_ko_growth", dist.Normal(growth_l, growth_s)
        )

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

    if not only_singletons:

        with numpyro.plate("guide_counts", lengths["len_guide_pairs"]):

            init_l, init_s = prior_params["init_count"]

            guide_init_count = numpyro.sample(
                "guide_init_count",
                dist.TruncatedNormal(loc=init_l, scale=init_s, low=0.0),
            )

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

        with numpyro.plate("gene_pairs", lengths["len_gene_pairs"]):

            pair_growth_l, pair_growth_s = prior_params["pair_growth"]

            gene_pair_ko_growth = numpyro.sample(
                "gene_pair_ko_growth", dist.Normal(pair_growth_l, pair_growth_s)
            )

        guide_eff_1 = numpyro.sample(
            "guide_eff_1",
            dist.TruncatedNormal(
                loc=guide_eff_mean[indices["guide_1_idx"]],
                scale=guide_eff_std[indices["guide_1_idx"]],
                low=0.0,
                high=1.0,
            ),
        )
        guide_eff_2 = numpyro.sample(
            "guide_eff_2",
            dist.TruncatedNormal(
                loc=guide_eff_mean[indices["guide_2_idx"]],
                scale=guide_eff_std[indices["guide_2_idx"]],
                low=0.0,
                high=1.0,
            ),
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

        init_theta = guide_init_count[indices["guide_pair_idx"]]

        init_lh = dkoLikelihoodInitial(init_theta)

        lh = dkoLikelihoodFinal(
            init_theta,
            guide_eff_1,
            guide_eff_2,
            guide_eff_12,
            cell_line_growth_v,
            gene_ko_growth_1,
            gene_ko_growth_2,
            gene_ko_growth_12,
            mv
        )

        numpyro.sample("obs_init", init_lh, obs=data["initial_counts"])

        numpyro.sample("obs", lh, obs=data["counts"])

    if not no_singletons:

        with numpyro.plate("guides_counts_s", lengths["len_guide_pairs_s"]):

            init_s_l, init_s_s = prior_params["init_count_s"]

            guide_init_count_s = numpyro.sample(
                "guide_init_count_s",
                dist.TruncatedNormal(loc=init_s_l, scale=init_s_s, low=0.0),
            )

        guide_eff_s = numpyro.sample(
            "guide_eff_s",
            dist.TruncatedNormal(
                loc=guide_eff_mean[indices["guide_s_idx"]],
                scale=guide_eff_std[indices["guide_s_idx"]],
                low=0.0,
                high=1.0,
            ),
        )

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

        init_lh_s = skoLikelihoodInitial(guide_init_count_s)

        lh_s = skoLikelihoodFinal(
            init_theta_s, guide_eff_s, cell_line_growth_s, gene_ko_growth_s, mv_s
        )

        numpyro.sample("obs_init_s", init_lh_s, obs=data["initial_counts_s"])

        numpyro.sample(
            "obs_s",
            lh_s,
            obs=data["counts_s"],
        )

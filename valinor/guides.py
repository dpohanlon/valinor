import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from numpyro.distributions import constraints
from numpyro import deterministic

from typing import Dict, List, Tuple, Any

def guide_guide_eff(
    lengths: Dict[str, int],
    prior_params: Dict[str, Any],
    config: str = "partial_pooling",
):
    mean_l, mean_s = prior_params["guide_eff_mean"]
    std_l, std_s   = prior_params["guide_eff_std"]

    if config == "partial_pooling":
        guide_eff_mean_loc = numpyro.param(
            "guide_eff_mean_loc",
            jnp.full((lengths["len_guides"],), mean_l)
        )
        guide_eff_mean_scale = numpyro.param(
            "guide_eff_mean_scale",
            jnp.full((lengths["len_guides"],), jnp.abs(mean_s) + 1e-3),
            constraint=dist.constraints.positive
        )
        guide_eff_std_loc = numpyro.param(
            "guide_eff_std_loc",
            jnp.full((lengths["len_guides"],), std_l)
        )
        guide_eff_std_scale = numpyro.param(
            "guide_eff_std_scale",
            jnp.full((lengths["len_guides"],), jnp.abs(std_s) + 1e-3),
            constraint=dist.constraints.positive
        )
        with numpyro.plate("guides", lengths["len_guides"]):
            guide_eff_mean_ = numpyro.sample(
                "guide_eff_mean",
                dist.TruncatedNormal(guide_eff_mean_loc, guide_eff_mean_scale, low=0.0, high=1.0)
            )
            guide_eff_std_ = numpyro.sample(
                "guide_eff_std",
                dist.TruncatedNormal(guide_eff_std_loc, guide_eff_std_scale, low=0.0, high=1.0)
            )

        with numpyro.plate("cell_lines", lengths["len_cell_lines"]):
            with numpyro.plate("guides_per_cell", lengths["len_guides"]):
                tilde_alpha_loc = numpyro.param(
                    "tilde_alpha_loc",
                    jnp.zeros((lengths["len_guides"], lengths["len_cell_lines"]))
                )
                tilde_alpha_scale = numpyro.param(
                    "tilde_alpha_scale",
                    jnp.ones((lengths["len_guides"], lengths["len_cell_lines"])),
                    constraint=dist.constraints.positive
                )
                tilde_alpha = numpyro.sample(
                    "tilde_alpha",
                    dist.Normal(tilde_alpha_loc, tilde_alpha_scale).expand(
                        [lengths["len_guides"], lengths["len_cell_lines"]]
                    ),
                )
                guide_eff_ = guide_eff_mean_[:, None] + guide_eff_std_[:, None] * tilde_alpha
                numpyro.deterministic("guide_eff", jax.nn.sigmoid(guide_eff_))

    elif config == "no_pooling":
        guide_eff_loc = numpyro.param(
            "guide_eff_loc",
            jnp.full((lengths["len_cell_lines"], lengths["len_guides"]), mean_l)
        )
        guide_eff_scale = numpyro.param(
            "guide_eff_scale",
            jnp.full((lengths["len_cell_lines"], lengths["len_guides"]), jnp.abs(mean_s) + 1e-3),
            constraint=dist.constraints.positive
        )
        with numpyro.plate("cell_lines", lengths["len_cell_lines"]):
            with numpyro.plate("guides", lengths["len_guides"]):
                numpyro.sample(
                    "guide_eff",
                    dist.TruncatedNormal(guide_eff_loc, guide_eff_scale, low=0.0, high=1.0)
                )

    elif config == "full_pooling":
        guide_eff_single_loc = numpyro.param(
            "guide_eff_single_loc",
            jnp.full((lengths["len_guides"],), mean_l)
        )
        guide_eff_single_scale = numpyro.param(
            "guide_eff_single_scale",
            jnp.full((lengths["len_guides"],), jnp.abs(mean_s) + 1e-3),
            constraint=dist.constraints.positive
        )
        with numpyro.plate("guides", lengths["len_guides"]):
            guide_eff_single = numpyro.sample(
                "guide_eff_single",
                dist.TruncatedNormal(
                    guide_eff_single_loc, guide_eff_single_scale, low=0.0, high=1.0
                )
            )
        numpyro.deterministic(
            "guide_eff",
            jnp.repeat(guide_eff_single[:, None], lengths["len_cell_lines"], axis=1),
        )

    else:
        raise ValueError("Invalid guide_config: must be 'partial_pooling', 'no_pooling', or 'full_pooling'.")

# --- Insert the following helper functions anywhere above or below the model ---

def guide_cell_line(lengths, prior_params, zi=False):
    growth_cell_l, growth_cell_s = prior_params["cell_line_growth"]
    cell_line_growth_loc = numpyro.param(
        "cell_line_growth_loc",
        jnp.full((lengths["len_cell_lines"],), growth_cell_l)
    )
    cell_line_growth_scale = numpyro.param(
        "cell_line_growth_scale",
        jnp.full((lengths["len_cell_lines"],), jnp.abs(growth_cell_s) + 1e-3),
        constraint=dist.constraints.positive
    )
    with numpyro.plate("cell_lines", lengths["len_cell_lines"]):
        cell_line_growth = numpyro.sample(
            "cell_line_growth",
            dist.Normal(cell_line_growth_loc, cell_line_growth_scale)
        )
        library_bias_loc = numpyro.param(
            "library_bias_loc",
            jnp.ones(lengths["len_cell_lines"])
        )
        library_bias_scale = numpyro.param(
            "library_bias_scale",
            0.1 * jnp.ones(lengths["len_cell_lines"]),
            constraint=dist.constraints.positive
        )
        numpyro.sample(
            "library_bias",
            dist.Normal(library_bias_loc, library_bias_scale)
        )
        # Turn this on for now, but not used in the model
        # if zi:
        p_zi_loc = numpyro.param(
            "p_zi_loc",
            0.05 * jnp.ones(lengths["len_cell_lines"])
        )
        p_zi_scale = numpyro.param(
            "p_zi_scale",
            0.1 * jnp.ones(lengths["len_cell_lines"]),
            constraint=dist.constraints.positive
        )
        numpyro.sample(
            "p_zi",
            dist.TruncatedNormal(p_zi_loc, p_zi_scale, low=0.0, high=1.0)
        )

def guide_global_overdisp(lengths, prior_params):
    od_means = np.clip(prior_params["od_means"], 1.0, np.inf) - 1
    od_stds  = prior_params["od_stds"]
    mv_cell_line_loc = numpyro.param(
        "mv_cell_line_loc",
        jnp.array(od_means, dtype=jnp.float32)
    )
    mv_cell_line_scale = numpyro.param(
        "mv_cell_line_scale",
        jnp.array(od_stds, dtype=jnp.float32) + 1e-3,
        constraint=dist.constraints.positive
    )
    # with numpyro.plate("cell_lines", lengths["len_cell_lines"]):
    numpyro.sample(
        "mv_cell_line",
        dist.TruncatedNormal(mv_cell_line_loc, mv_cell_line_scale, low=0.0)
    )

def guide_gene_std(lengths, prior_params):
    mv_mean_s = jnp.ones(lengths["len_cell_lines"]) * prior_params["mv_mean_scale"]
    gene_std_scale = numpyro.param(
        "gene_std_scale",
        mv_mean_s,
        constraint=dist.constraints.positive
    )
    # with numpyro.plate("gene_std_plate", lengths["len_cell_lines"]):
    numpyro.sample(
        "gene_std",
        dist.HalfNormal(gene_std_scale)
    )

def guide_gene_ko_growth(lengths, prior_params):
    if "gene_effect_means" not in prior_params:
        growth_l, growth_s = prior_params["gene_ko_growth"]
        gko_loc = numpyro.param(
            "gene_ko_growth_loc",
            jnp.full((lengths["len_genes"],), growth_l)
        )
        gko_scale = numpyro.param(
            "gene_ko_growth_scale",
            jnp.full((lengths["len_genes"],), jnp.abs(growth_s) + 1e-3),
            constraint=dist.constraints.positive
        )
        with numpyro.plate("genes", lengths["len_genes"]):
            numpyro.sample(
                "gene_ko_growth",
                dist.Normal(gko_loc, gko_scale)
            )
    else:
        growth_l = prior_params["gene_effect_means"]
        growth_s = prior_params["gene_effect_stds"]

        mask = ~jnp.isnan(growth_l)
        gko_loc = numpyro.param("gene_ko_growth_masked_loc", growth_l)
        gko_scale = numpyro.param(
            "gene_ko_growth_masked_scale",
            growth_s + 1e-3,
            constraint=dist.constraints.positive
        )

        # Doesn't match model - maybe add to model..?
        # with numpyro.plate("genes", lengths["len_genes"]):
        numpyro.sample(
            "gene_ko_growth",
            dist.Normal(gko_loc, gko_scale).mask(mask)
        )

def guide_single_od(lengths):
    ngene_loc = numpyro.param(
        "non_centered_deviation_gene_loc",
        jnp.zeros(lengths["len_genes"])
    )
    ngene_scale = numpyro.param(
        "non_centered_deviation_gene_scale",
        jnp.ones(lengths["len_genes"]),
        constraint=dist.constraints.positive
    )
    with numpyro.plate("genes", lengths["len_genes"]):
        numpyro.sample(
            "non_centered_deviation_gene",
            dist.Normal(ngene_loc, ngene_scale)
        )

def guide_double_od(lengths, prior_params):
    # OD expansions for gene pairs
    ndp_loc = numpyro.param(
        "non_centered_deviation_loc",
        jnp.zeros(lengths["len_gene_pairs"])
    )
    ndp_scale = numpyro.param(
        "non_centered_deviation_scale",
        jnp.ones(lengths["len_gene_pairs"]),
        constraint=dist.constraints.positive
    )
    with numpyro.plate("gene_pairs", lengths["len_gene_pairs"]):
        numpyro.sample(
            "non_centered_deviation",
            dist.Normal(ndp_loc, ndp_scale)
        )

def guide_pair_growth(lengths, prior_params):
    # Pair-level growth
    pair_growth_l, pair_growth_s = prior_params["pair_growth"]
    pg_loc = numpyro.param("pair_growth_loc", jnp.array(pair_growth_l))
    pg_scale = numpyro.param(
        "pair_growth_scale",
        jnp.array(jnp.abs(pair_growth_s) + 1e-3),
        constraint=dist.constraints.positive
    )
    with numpyro.plate("gene_pairs", lengths["len_gene_pairs"]):
        numpyro.sample(
            "gene_pair_ko_growth",
            dist.Normal(pg_loc, pg_scale)
        )

def guide_control_od(lengths):
    ndc_loc = numpyro.param(
        "non_centered_dev_c_loc",
        jnp.zeros(lengths["len_guide_pairs_c"])
    )
    ndc_scale = numpyro.param(
        "non_centered_dev_c_scale",
        jnp.ones(lengths["len_guide_pairs_c"]),
        constraint=dist.constraints.positive
    )
    with numpyro.plate("guide_pairs", lengths["len_guide_pairs_c"]):
        numpyro.sample(
            "non_centered_deviation_gene_c",
            dist.Normal(ndc_loc, ndc_scale)
        )

def guide_init_counts_double(lengths, prior_params):
    init_l, init_s = prior_params["init_count"]
    ic_loc = numpyro.param(
        "guide_init_count_loc",
        jnp.full((lengths["len_guide_pairs"],), init_l)
    )
    ic_scale = numpyro.param(
        "guide_init_count_scale",
        jnp.full((lengths["len_guide_pairs"],), jnp.abs(init_s) + 1e-3),
        constraint=dist.constraints.positive
    )
    with numpyro.plate("guide_counts", lengths["len_guide_pairs"]):
        numpyro.sample(
            "guide_init_count",
            dist.TruncatedNormal(ic_loc, ic_scale, low=0.0)
        )

def guide_init_counts_single(lengths, prior_params):
    init_s_l, init_s_s = prior_params["init_count_s"]
    ics_loc = numpyro.param(
        "guide_init_count_s_loc",
        jnp.full((lengths["len_guide_pairs_s"],), init_s_l)
    )
    ics_scale = numpyro.param(
        "guide_init_count_s_scale",
        jnp.full((lengths["len_guide_pairs_s"],), jnp.abs(init_s_s) + 1e-3),
        constraint=dist.constraints.positive
    )
    with numpyro.plate("guides_counts_s", lengths["len_guide_pairs_s"]):
        numpyro.sample(
            "guide_init_count_s",
            dist.TruncatedNormal(ics_loc, ics_scale, low=0.0)
        )

def guide_init_counts_control(lengths, prior_params):
    init_c_l, init_c_s = prior_params["init_count_c"]
    icc_loc = numpyro.param(
        "guide_init_count_c_loc",
        jnp.full((lengths["len_guide_pairs_c"],), init_c_l)
    )
    icc_scale = numpyro.param(
        "guide_init_count_c_scale",
        jnp.full((lengths["len_guide_pairs_c"],), jnp.abs(init_c_s) + 1e-3),
        constraint=dist.constraints.positive
    )
    with numpyro.plate("guides_counts_c", lengths["len_guide_pairs_c"]):
        numpyro.sample(
            "guide_init_count_c",
            dist.TruncatedNormal(icc_loc, icc_scale, low=0.0)
        )

def guide_init_counts(lengths, prior_params, no_singletons, only_singletons, no_controls):
    if not only_singletons:
        guide_init_counts_double(lengths, prior_params)

    if not no_singletons:
        guide_init_counts_single(lengths, prior_params)

    if not no_controls:
        guide_init_counts_control(lengths, prior_params)

# --- Replace the previous big guide function with this master function ---

def valinor_controls_guide(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices: Dict[str, jnp.array],
    prior_params: Dict[str, Any],
    zi: bool = False,
    predict: bool = False,
):
    """
    Guide for the controls-only sub-model.
    """

    guide_cell_line(lengths, prior_params, zi=zi)

    guide_global_overdisp(lengths, prior_params)

    guide_gene_std(lengths, prior_params)

    guide_control_od(lengths)

    guide_init_counts_control(lengths, prior_params)

def valinor_singles_guide(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices: Dict[str, jnp.array],
    prior_params: Dict[str, Any],
    guide_config: str = "partial_pooling",
    zi: bool = False,
    predict: bool = False,
):
    """
    Guide for the single knockout sub-model, including controls.
    """


    guide_guide_eff(lengths, prior_params, config=guide_config)

    guide_gene_ko_growth(lengths, prior_params)

    guide_cell_line(lengths, prior_params, zi=zi)

    guide_global_overdisp(lengths, prior_params)
    guide_gene_std(lengths, prior_params)

    guide_single_od(lengths)

    guide_init_counts_single(lengths, prior_params)

def valinor_full_guide(
    data: Dict[str, jnp.array],
    lengths: Dict[str, int],
    indices: Dict[str, jnp.array],
    prior_params: Dict[str, Any],
    no_singletons: bool = False,
    only_singletons: bool = False,
    no_controls: bool = True,
    alternate: bool = False,
    guide_config: str = "partial_pooling",
    zi: bool = False,
    predict: bool = False,
):
    """
    Guide for the full hierarchical model, including double knockouts, single knockouts, and controls.
    Allows for optional exclusion of components.
    """

    if alternate == True:
        print("Alternate hasn't been implemented for this guide!")

    # These have to match the order in the model

    guide_guide_eff(lengths, prior_params, config=guide_config)
    guide_gene_ko_growth(lengths, prior_params)

    guide_cell_line(lengths, prior_params, zi=zi)

    guide_global_overdisp(lengths, prior_params)

    guide_gene_std(lengths, prior_params)
    if not only_singletons:
        guide_double_od(lengths, prior_params)
        guide_init_counts_double(lengths, prior_params)
        guide_pair_growth(lengths, prior_params)

    if not no_singletons:
        guide_single_od(lengths)
        guide_init_counts_single(lengths, prior_params)

    if not no_controls:
        guide_control_od(lengths)
        guide_init_counts_control(lengths, prior_params)


# def valinor_guide(
#     data: Dict[str, jnp.array],
#     lengths: Dict[str, int],
#     indices: Dict[str, jnp.array],
#     prior_params: Dict[str, Any],
#     no_singletons: bool = False,
#     only_singletons: bool = False,
#     no_controls: bool = True,
#     zi: bool = False,
# ) -> None:

#     guide_cell_line(lengths, prior_params, zi=zi)
#     guide_global_overdisp(lengths, prior_params)
#     guide_gene_std(lengths, prior_params)
#     guide_gene_ko_growth(lengths, prior_params)
#     guide_guide_eff(lengths, prior_params)

#     if not no_singletons:
#         guide_single_od(lengths)

#     if not only_singletons:
#         guide_double_od(lengths, prior_params)

#     if not no_controls:
#         guide_control_od(lengths)

#     guide_init_counts(lengths, prior_params, no_singletons, only_singletons, no_controls)

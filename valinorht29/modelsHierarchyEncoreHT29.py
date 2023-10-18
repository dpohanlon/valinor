import numpyro
import numpyro.distributions as dist

import jax
import jax.numpy as jnp

import numpy as np

def chronosModelHierarchy(data, lengths, indices, prior_params, no_singletons = False, only_singletons = False, no_controls = False):

    # No replicates to inform this anyway?

    # inv_mv_init = numpyro.sample('inv_mv_init', dist.TruncatedNormal(loc = 10, scale = 5, low = 1.0))
    # mv_init = numpyro.deterministic('mv_init', 1. / inv_mv_init)

    # library_bias = numpyro.sample('library_bias', dist.Normal(loc = 1.0, scale = 0.1))

    # guide_eff_mean = numpyro.sample('guide_eff_mean', dist.TruncatedNormal(loc = 0.90, scale = 0.1, low = 0.0, high = 1.0))
    # guide_eff_std = numpyro.sample('guide_eff_std', dist.TruncatedNormal(loc = 0.1, scale = 0.1, low = 0.0))

    # Number of guides in each cell line (hyper-distributions)
    with numpyro.plate('guides', lengths['len_guides']):

        guide_eff_mean = numpyro.sample('guide_eff_mean', dist.TruncatedNormal(loc = 0.90, scale = 0.05, low = 0.0, high = 1.0))
        guide_eff_std = numpyro.sample('guide_eff_std', dist.TruncatedNormal(loc = 0.1, scale = 0.05, low = 0.0))

        # with numpyro.plate('cell_lines_g', lengths['len_cell_lines']):

        # eff_tn = dist.TruncatedNormal(loc = guide_eff_mean.reshape(-1, 1), scale = guide_eff_std.reshape(-1, 1), low = 0.0, high = 1.0)
        #
        # print(eff_tn.batch_shape, eff_tn.event_shape)
        #
        # guide_eff = numpyro.sample('guide_eff', eff_tn)

    with numpyro.plate('genes', lengths['len_genes']):

        gene_ko_growth = numpyro.sample('gene_ko_growth', dist.Normal(0.0, 0.1))

    with numpyro.plate('cell_lines', lengths['len_cell_lines']):

        library_bias = numpyro.sample('library_bias', dist.Normal(loc = 1.0, scale = 0.1))

        cell_line_eff = numpyro.sample('cell_line_eff', dist.TruncatedNormal(loc = 0.90, scale = 0.05, low = 0.0, high = 1.0))

        eff_tn = dist.TruncatedNormal(loc = guide_eff_mean.reshape(-1, 1), scale = guide_eff_std.reshape(-1, 1), low = 0.0, high = 1.0)

        guide_eff = numpyro.sample('guide_eff', eff_tn)

        # with numpyro.plate('guides', lengths['len_guides']):

            # Shape [n_guides, n_cell_lines]
            # guide_eff = numpyro.sample('guide_eff', dist.TruncatedNormal(loc = guide_eff_mean, scale = guide_eff_std, low = 0.0, high = 1.0).to_event(1))
        # print('guide_eff shape', guide_eff.shape)
        # 320, 18

        # cell_line_growth = numpyro.sample('cell_line_growth', dist.TruncatedNormal(loc = 0.5, scale = 0.5, low = 0.0)) # Was Normal
        cell_line_growth = numpyro.sample('cell_line_growth', dist.Normal(loc = 0.0, scale = 0.1)) # Was Normal

        # Vectors of overdispersion parameters means and sqrt(var) from data
        # for each cell line

        od_means = prior_params['od_means']
        od_stds = prior_params['od_stds']

        inv_mv_mean = numpyro.sample('inv_mv_mean', dist.TruncatedNormal(loc = od_means, scale = 0.01, low = 1.0))
        inv_mv_std = numpyro.sample('inv_mv_std', dist.TruncatedNormal(loc = od_stds, scale = 0.01, low = 0.0))

    if not only_singletons:

        with numpyro.plate('guide_counts', lengths['len_guide_pairs']):

            guide_init_count = numpyro.sample('guide_init_count', dist.TruncatedNormal(loc = 200., scale = 50., low = 0.0))

            guide_pair_eff_mean = numpyro.sample('guide_pair_eff_mean', dist.TruncatedNormal(loc = 0.90, scale = 0.05, low = 0.0, high = 1.0))
            guide_pair_eff_std = numpyro.sample('guide_pair_eff_std', dist.TruncatedNormal(loc = 0.05, scale = 0.05, low = 0.0))

        with numpyro.plate('gene_pairs', lengths['len_gene_pairs']):

            gene_pair_ko_growth = numpyro.sample('gene_pair_ko_growth', dist.Normal(0.0, 0.05))

        guide_eff_1 = guide_eff[indices['guide_1_idx'], indices['cell_line_idx']]
        guide_eff_2 = guide_eff[indices['guide_2_idx'], indices['cell_line_idx']]

        # These are okay to be sampled for each pair, as each pair is unique (?)
        guide_eff_12 = numpyro.sample('guide_eff_12', dist.TruncatedNormal(loc = guide_pair_eff_mean[indices['guide_pair_idx']], scale = guide_pair_eff_std[indices['guide_pair_idx']], low = 0.0, high = 1.0))

        inv_mv = numpyro.sample('inv_mv', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_idx']], scale = inv_mv_std[indices['cell_line_idx']], low = 1.0))

        mv = numpyro.deterministic('mv', 1. / inv_mv)

        guide_init = guide_init_count

        gene_ko_growth_1 = gene_ko_growth[indices['gene_1_idx']]
        gene_ko_growth_2 = gene_ko_growth[indices['gene_2_idx']]
        gene_ko_growth_12 = gene_pair_ko_growth[indices['gene_pair_idx']]

        cell_line_eff_v = cell_line_eff[indices['cell_line_idx']]
        cell_line_growth_v = cell_line_growth[indices['cell_line_idx']]

        init_theta = guide_init

        # theta = 1. + guide_eff_1 * guide_eff_2 * guide_eff_12 * \
        #           ( jnp.exp(cell_line_growth_v + (gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12) ) - 1. )

        theta = 1. + guide_eff_1 * guide_eff_2 * \
                  ( jnp.exp(cell_line_growth_v + (gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12) ) - 1. )

        theta = init_theta[indices['guide_pair_idx']] * theta

        numpyro.sample('obs_init', dist.Poisson(init_theta), obs=data['initial_counts'])
        # numpyro.sample('obs_init', dist.NegativeBinomial2(init_theta, init_theta * mv_init / (1 - mv_init)), obs=data['initial_counts'])

        numpyro.sample('obs', dist.NegativeBinomial2(theta, theta * mv / (1 - mv)), obs=data['counts'])

    if not no_singletons:

        with numpyro.plate('guides_counts_s', lengths['len_guide_pairs_s']):

            guide_init_count_s = numpyro.sample('guide_init_count_s', dist.TruncatedNormal(loc = 200., scale = 200., low = 0.0))

        guide_eff_s = guide_eff[indices['guide_s_idx'], indices['cell_line_s_idx']]

        inv_mv_s = numpyro.sample('inv_mv_s', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_s_idx']], scale = inv_mv_std[indices['cell_line_s_idx']], low = 1.0))

        mv_s = numpyro.deterministic('mv_s', 1. / inv_mv_s)

        guide_init_s = guide_init_count_s

        gene_ko_growth_s = gene_ko_growth[indices['gene_s_idx']]

        cell_line_eff_s = cell_line_eff[indices['cell_line_s_idx']]
        cell_line_growth_s = cell_line_growth[indices['cell_line_s_idx']]

        # theta_s = guide_init_s[indices['guide_pair_s_idx']] * ( 1. + guide_eff_s * \
        #                        ( jnp.exp( cell_line_growth_s + library_bias * gene_ko_growth_s) - 1. ) )

        theta_s = guide_init_s[indices['guide_pair_s_idx']] * ( 1. + guide_eff_s * \
                               ( jnp.exp( cell_line_growth_s + library_bias[indices['cell_line_s_idx']] * gene_ko_growth_s) - 1. ))

        # theta_s = guide_init_s[indices['guide_pair_s_idx']] * ( 1. + guide_eff_s * \
                               # ( jnp.exp( cell_line_growth_s + gene_ko_growth_s) - 1. ) )

        numpyro.sample('obs_init_s', dist.Poisson(guide_init_s), obs=data['initial_counts_s'])
        # numpyro.sample('obs_init_s', dist.NegativeBinomial2(init_theta_s, init_theta_s * mv_init / (1 - mv_init)), obs=data['initial_counts_s'])

        numpyro.sample('obs_s', dist.NegativeBinomial2(theta_s, theta_s * mv_s / (1 - mv_s)), obs=data['counts_s'])

    if not no_controls:

        with numpyro.plate('guides_counts_c', lengths['len_guide_pairs_c']):

            guide_init_count_c = numpyro.sample('guide_init_count_c', dist.TruncatedNormal(loc = 300., scale = 200., low = 0.0))

        cell_line_growth_c = cell_line_growth[indices['cell_line_c_idx']]

        inv_mv_c = numpyro.sample('inv_mv_c', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_c_idx']], scale = inv_mv_std[indices['cell_line_c_idx']], low = 1.0))

        mv_c = numpyro.deterministic('mv_c', 1. / inv_mv_c)

        # Maybe means this should be + in exp?

        theta_c = guide_init_count_c[indices['guide_pair_c_idx']] * ( 1. + ( jnp.exp(cell_line_growth_c) - 1. ) )

        numpyro.sample('obs_init_c', dist.Poisson(guide_init_count_c), obs=data['initial_counts_c'])
        # numpyro.sample('obs_init_c', dist.NegativeBinomial2(init_theta_c, init_theta_c * mv_init / (1 - mv_init)), obs=data['initial_counts_c'])

        numpyro.sample('obs_c', dist.NegativeBinomial2(theta_c, theta_c * mv_c / (1 - mv_c)), obs=data['counts_c'])

def chronosModelHierarchy2(data, lengths, indices, prior_params, no_singletons = False, only_singletons = False, no_controls = False):

    # Number of guides in each cell line (hyper-distributions)
    with numpyro.plate('guides', lengths['len_guides']):

        guide_eff_mean = numpyro.sample('guide_eff_mean', dist.TruncatedNormal(loc = 0.90, scale = 0.1, low = 0.0, high = 1.0))
        guide_eff_std = numpyro.sample('guide_eff_std', dist.TruncatedNormal(loc = 0.05, scale = 0.1, low = 0.0))

    with numpyro.plate('genes', lengths['len_genes']):

        gene_ko_growth = numpyro.sample('gene_ko_growth', dist.Normal(0.0, 0.1))

    with numpyro.plate('cell_lines', lengths['len_cell_lines']):

        cell_line_eff = numpyro.sample('cell_line_eff', dist.TruncatedNormal(loc = 0.90, scale = 0.1, low = 0.0, high = 1.0))

        eff_tn = dist.TruncatedNormal(loc = guide_eff_mean.reshape(-1, 1), scale = guide_eff_std.reshape(-1, 1), low = 0.0, high = 1.0)

        # print(eff_tn.batch_shape, eff_tn.event_shape)

        guide_eff = numpyro.sample('guide_eff', eff_tn)

        # cell_line_growth = numpyro.sample('cell_line_growth', dist.TruncatedNormal(loc = 0.5, scale = 0.5, low = 0.0)) # Was Normal
        cell_line_growth = numpyro.sample('cell_line_growth', dist.Normal(loc = 0.0, scale = 0.1)) # Was Normal

        # Vectors of overdispersion parameters means and sqrt(var) from data
        # for each cell line

        od_means = prior_params['od_means']
        od_stds = prior_params['od_stds']

        inv_mv_mean = numpyro.sample('inv_mv_mean', dist.TruncatedNormal(loc = od_means, scale = 0.1, low = 1.0))
        inv_mv_std = numpyro.sample('inv_mv_std', dist.TruncatedNormal(loc = od_stds, scale = 0.1, low = 0.0))

    if not only_singletons:

        with numpyro.plate('guide_counts', lengths['len_guide_pairs']):

            guide_init_count = numpyro.sample('guide_init_count', dist.TruncatedNormal(loc = 200., scale = 200., low = 0.0))

            guide_pair_eff_mean = numpyro.sample('guide_pair_eff_mean', dist.TruncatedNormal(loc = 0.20, scale = 0.1, low = 0.0, high = 1.0))
            guide_pair_eff_std = numpyro.sample('guide_pair_eff_std', dist.TruncatedNormal(loc = 0.05, scale = 0.1, low = 0.0))

        with numpyro.plate('gene_pairs', lengths['len_gene_pairs']):

            gene_pair_ko_growth = numpyro.sample('gene_pair_ko_growth', dist.Normal(0.0, 0.1))

        guide_eff_1 = guide_eff[indices['guide_1_idx'], indices['cell_line_idx']]
        guide_eff_2 = guide_eff[indices['guide_2_idx'], indices['cell_line_idx']]

        # These are okay to be sampled for each pair, as each pair is unique (?)
        guide_eff_12 = numpyro.sample('guide_eff_12', dist.TruncatedNormal(loc = guide_pair_eff_mean[indices['guide_pair_idx']], scale = guide_pair_eff_std[indices['guide_pair_idx']], low = 0.0, high = 1.0))

        inv_mv = numpyro.sample('inv_mv', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_idx']], scale = inv_mv_std[indices['cell_line_idx']], low = 1.0))

        mv = numpyro.deterministic('mv', 1. / inv_mv)

        guide_init = guide_init_count

        gene_ko_growth_1 = gene_ko_growth[indices['gene_1_idx']]
        gene_ko_growth_2 = gene_ko_growth[indices['gene_2_idx']]
        gene_ko_growth_12 = gene_pair_ko_growth[indices['gene_pair_idx']]

        cell_line_eff_v = cell_line_eff[indices['cell_line_idx']]
        cell_line_growth_v = cell_line_growth[indices['cell_line_idx']]

        init_theta = guide_init

        # theta = jnp.exp(-cell_line_growth_v) * init_theta[indices['guide_pair_idx']] * ( (1. - (guide_eff_1 + guide_eff_1 + guide_eff_12)) * jnp.exp(cell_line_growth_v) + \
        #                                                                                  guide_eff_1 * jnp.exp(gene_ko_growth_1) + \
        #                                                                                  guide_eff_2 * jnp.exp(gene_ko_growth_2) + \
        #                                                                                  guide_eff_12 * jnp.exp(gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12)
        #                                                                                )

        # theta = jnp.exp(-cell_line_growth_v) * init_theta[indices['guide_pair_idx']] * ( (1. - (guide_eff_1 + guide_eff_1 + guide_eff_12)) * jnp.exp(cell_line_growth_v) + \
        #                                                                                  guide_eff_1 * jnp.exp(gene_ko_growth_1) + \
        #                                                                                  guide_eff_2 * jnp.exp(gene_ko_growth_2) + \
        #                                                                                  guide_eff_12 * guide_eff_1 * guide_eff_2 * jnp.exp(gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12)
        #                                                                                )

        theta = init_theta[indices['guide_pair_idx']] * ( (1. - (guide_eff_1 + guide_eff_1 + guide_eff_12)) + \
                                                                                         guide_eff_1 * jnp.exp(gene_ko_growth_1 - cell_line_growth_v) + \
                                                                                         guide_eff_2 * jnp.exp(gene_ko_growth_2 - cell_line_growth_v) + \
                                                                                         guide_eff_12 * jnp.exp((gene_ko_growth_1 - cell_line_growth_v) + (gene_ko_growth_2 - cell_line_growth_v) + (gene_ko_growth_12 - cell_line_growth_v))
                                                                                       )

        theta = jax.nn.softplus(theta)

        numpyro.sample('obs_init', dist.Poisson(init_theta), obs=data['initial_counts'])
        # numpyro.sample('obs_init', dist.NegativeBinomial2(init_theta, init_theta * mv_init / (1 - mv_init)), obs=data['initial_counts'])

        numpyro.sample('obs', dist.NegativeBinomial2(theta, theta * mv / (1 - mv)), obs=data['counts'])

    if not no_singletons:

        with numpyro.plate('guides_counts_s', lengths['len_guide_pairs_s']):

            guide_init_count_s = numpyro.sample('guide_init_count_s', dist.TruncatedNormal(loc = 200., scale = 200., low = 0.0))

        guide_eff_s = guide_eff[indices['guide_s_idx'], indices['cell_line_s_idx']]

        inv_mv_s = numpyro.sample('inv_mv_s', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_s_idx']], scale = inv_mv_std[indices['cell_line_s_idx']], low = 1.0))

        mv_s = numpyro.deterministic('mv_s', 1. / inv_mv_s)

        guide_init_s = guide_init_count_s

        gene_ko_growth_s = gene_ko_growth[indices['gene_s_idx']]

        cell_line_eff_s = cell_line_eff[indices['cell_line_s_idx']]
        cell_line_growth_s = cell_line_growth[indices['cell_line_s_idx']]

        init_theta_s = guide_init_s

        # Should be the same
        theta_s = jnp.exp(-cell_line_growth_s) * init_theta_s[indices['guide_pair_s_idx']] * ( (1. - guide_eff_s) * jnp.exp(cell_line_growth_s) + \
                                                                                              guide_eff_s * jnp.exp(gene_ko_growth_s) )

        # theta_s = init_theta_s[indices['guide_pair_s_idx']] * ( (1. - guide_eff_s) + \
                                                                                              # guide_eff_s * jnp.exp(gene_ko_growth_s - cell_line_growth_s) )

        theta_s = jax.nn.softplus(theta_s)

        numpyro.sample('obs_init_s', dist.Poisson(init_theta_s), obs=data['initial_counts_s'])
        # numpyro.sample('obs_init_s', dist.NegativeBinomial2(init_theta_s, init_theta_s * mv_init / (1 - mv_init)), obs=data['initial_counts_s'])

        numpyro.sample('obs_s', dist.NegativeBinomial2(theta_s, theta_s * mv_s / (1 - mv_s)), obs=data['counts_s'])

    if not no_controls:

        with numpyro.plate('guides_counts_c', lengths['len_guide_pairs_c']):

            guide_init_count_c = numpyro.sample('guide_init_count_c', dist.TruncatedNormal(loc = 300., scale = 200., low = 0.0))

        cell_line_growth_c = cell_line_growth[indices['cell_line_c_idx']]

        inv_mv_c = numpyro.sample('inv_mv_c', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_c_idx']], scale = inv_mv_std[indices['cell_line_c_idx']], low = 1.0))

        mv_c = numpyro.deterministic('mv_c', 1. / inv_mv_c)

        init_theta_c = guide_init_count_c

        # Maybe means this should be + in exp?

        theta_c = init_theta_c[indices['guide_pair_c_idx']] * ( 1. + ( jnp.exp(cell_line_growth_c) - 1. ) )

        numpyro.sample('obs_init_c', dist.Poisson(init_theta_c), obs=data['initial_counts_c'])
        # numpyro.sample('obs_init_c', dist.NegativeBinomial2(init_theta_c, init_theta_c * mv_init / (1 - mv_init)), obs=data['initial_counts_c'])

        numpyro.sample('obs_c', dist.NegativeBinomial2(theta_c, theta_c * mv_c / (1 - mv_c)), obs=data['counts_c'])


def chronosModelHierarchySeparateEffs(data, lengths, indices, prior_params, no_singletons = False, only_singletons = False, no_controls = False):

    # No replicates to inform this anyway?

    inv_mv_init = numpyro.sample('inv_mv_init', dist.TruncatedNormal(loc = 10, scale = 5, low = 1.0))

    mv_init = numpyro.deterministic('mv_init', 1. / inv_mv_init)

    with numpyro.plate('guides', lengths['len_guides']):

        guide_eff_mean = numpyro.sample('guide_eff_mean', dist.TruncatedNormal(loc = 0.90, scale = 0.1, low = 0.0, high = 1.0))
        guide_eff_std = numpyro.sample('guide_eff_std', dist.TruncatedNormal(loc = 0.05, scale = 0.1, low = 0.0))

    with numpyro.plate('genes', lengths['len_genes']):

        gene_ko_growth = numpyro.sample('gene_ko_growth', dist.Normal(0.0, 0.1))

    with numpyro.plate('cell_lines', lengths['len_cell_lines']):

        cell_line_eff = numpyro.sample('cell_line_eff', dist.TruncatedNormal(loc = 0.90, scale = 0.05, low = 0.0, high = 1.0))

        # cell_line_growth = numpyro.sample('cell_line_growth', dist.TruncatedNormal(loc = 0.5, scale = 0.5, low = 0.0)) # Was Normal
        cell_line_growth = numpyro.sample('cell_line_growth', dist.Normal(loc = 0.0, scale = 0.1)) # Was Normal

        # Vectors of overdispersion parameters means and sqrt(var) from data
        # for each cell line

        od_means = prior_params['od_means']
        od_stds = prior_params['od_stds']

        inv_mv_mean = numpyro.sample('inv_mv_mean', dist.TruncatedNormal(loc = od_means, scale = 0.1, low = 1.0))
        inv_mv_std = numpyro.sample('inv_mv_std', dist.TruncatedNormal(loc = od_stds, scale = 0.1, low = 0.0))

    if not only_singletons:

        with numpyro.plate('guide_counts', lengths['len_guide_pairs']):

            guide_init_count = numpyro.sample('guide_init_count', dist.TruncatedNormal(loc = 200., scale = 200., low = 0.0))

            guide_pair_eff_mean = numpyro.sample('guide_pair_eff_mean', dist.TruncatedNormal(loc = 0.90, scale = 0.1, low = 0.0, high = 1.0))
            guide_pair_eff_std = numpyro.sample('guide_pair_eff_std', dist.TruncatedNormal(loc = 0.05, scale = 0.1, low = 0.0))

        with numpyro.plate('gene_pairs', lengths['len_gene_pairs']):

            gene_pair_ko_growth = numpyro.sample('gene_pair_ko_growth', dist.Normal(0.0, 0.1))

        guide_eff_1 = numpyro.sample('guide_eff_1', dist.TruncatedNormal(loc = guide_eff_mean[indices['guide_1_idx']], scale = guide_eff_std[indices['guide_1_idx']], low = 0.0, high = 1.0))
        guide_eff_2 = numpyro.sample('guide_eff_2', dist.TruncatedNormal(loc = guide_eff_mean[indices['guide_2_idx']], scale = guide_eff_std[indices['guide_2_idx']], low = 0.0, high = 1.0))

        guide_eff_12 = numpyro.sample('guide_eff_12', dist.TruncatedNormal(loc = guide_pair_eff_mean[indices['guide_pair_idx']], scale = guide_pair_eff_std[indices['guide_pair_idx']], low = 0.0, high = 1.0))

        inv_mv = numpyro.sample('inv_mv', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_idx']], scale = inv_mv_std[indices['cell_line_idx']], low = 1.0))

        mv = numpyro.deterministic('mv', 1. / inv_mv)

        guide_init = guide_init_count

        gene_ko_growth_1 = gene_ko_growth[indices['gene_1_idx']]
        gene_ko_growth_2 = gene_ko_growth[indices['gene_2_idx']]
        gene_ko_growth_12 = gene_pair_ko_growth[indices['gene_pair_idx']]

        cell_line_eff_v = cell_line_eff[indices['cell_line_idx']]
        cell_line_growth_v = cell_line_growth[indices['cell_line_idx']]

        init_theta = guide_init

        theta = 1. + guide_eff_1 * guide_eff_2 * guide_eff_12 * \
                  ( jnp.exp(cell_line_growth_v + (gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12) ) - 1. )

        theta = init_theta[indices['guide_pair_idx']] * theta

        numpyro.sample('obs_init', dist.Poisson(init_theta), obs=data['initial_counts'])
        # numpyro.sample('obs_init', dist.NegativeBinomial2(init_theta, init_theta * mv_init / (1 - mv_init)), obs=data['initial_counts'])

        numpyro.sample('obs', dist.NegativeBinomial2(theta, theta * mv / (1 - mv)), obs=data['counts'])

    if not no_singletons:

        with numpyro.plate('guides_counts_s', lengths['len_guide_pairs_s']):

            guide_init_count_s = numpyro.sample('guide_init_count_s', dist.TruncatedNormal(loc = 200., scale = 200., low = 0.0))

        # Don't sample w/ TruncatedNormal but instead just reindex an array of idxs for this cell line? (Like gene effect)
        guide_eff_s = numpyro.sample('guide_eff_s', dist.TruncatedNormal(loc = guide_eff_mean[indices['guide_s_idx']], scale = guide_eff_std[indices['guide_s_idx']], low = 0.0, high = 1.0))

        inv_mv_s = numpyro.sample('inv_mv_s', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_s_idx']], scale = inv_mv_std[indices['cell_line_s_idx']], low = 1.0))

        mv_s = numpyro.deterministic('mv_s', 1. / inv_mv_s)

        guide_init_s = guide_init_count_s

        gene_ko_growth_s = gene_ko_growth[indices['gene_s_idx']]

        cell_line_eff_s = cell_line_eff[indices['cell_line_s_idx']]
        cell_line_growth_s = cell_line_growth[indices['cell_line_s_idx']]

        init_theta_s = guide_init_s

        theta_s = init_theta_s[indices['guide_pair_s_idx']] * ( 1. + guide_eff_s * \
                               ( jnp.exp( cell_line_growth_s + gene_ko_growth_s) - 1. ) )

        numpyro.sample('obs_init_s', dist.Poisson(init_theta_s), obs=data['initial_counts_s'])
        # numpyro.sample('obs_init_s', dist.NegativeBinomial2(init_theta_s, init_theta_s * mv_init / (1 - mv_init)), obs=data['initial_counts_s'])

        numpyro.sample('obs_s', dist.NegativeBinomial2(theta_s, theta_s * mv_s / (1 - mv_s)), obs=data['counts_s'])

    if not no_controls:

        with numpyro.plate('guides_counts_c', lengths['len_guide_pairs_c']):

            guide_init_count_c = numpyro.sample('guide_init_count_c', dist.TruncatedNormal(loc = 300., scale = 200., low = 0.0))

        cell_line_growth_c = cell_line_growth[indices['cell_line_c_idx']]

        inv_mv_c = numpyro.sample('inv_mv_c', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_c_idx']], scale = inv_mv_std[indices['cell_line_c_idx']], low = 1.0))

        mv_c = numpyro.deterministic('mv_c', 1. / inv_mv_c)

        init_theta_c = guide_init_count_c

        # Maybe means this should be + in exp?

        theta_c = init_theta_c[indices['guide_pair_c_idx']] * ( 1. + ( jnp.exp(cell_line_growth_c) - 1. ) )

        numpyro.sample('obs_init_c', dist.Poisson(init_theta_c), obs=data['initial_counts_c'])
        # numpyro.sample('obs_init_c', dist.NegativeBinomial2(init_theta_c, init_theta_c * mv_init / (1 - mv_init)), obs=data['initial_counts_c'])

        numpyro.sample('obs_c', dist.NegativeBinomial2(theta_c, theta_c * mv_c / (1 - mv_c)), obs=data['counts_c'])


def chronosModelHierarchyProd(data, lengths, indices, prior_params, no_singletons = False, only_singletons = False, no_controls = False):

    with numpyro.plate('guides', lengths['len_guides']):

        guide_eff_mean = numpyro.sample('guide_eff_mean', dist.TruncatedNormal(loc = 0.8, scale = 0.05, low = 0.0, high = 1.0))
        guide_eff_std = numpyro.sample('guide_eff_std', dist.TruncatedNormal(loc = 0.05, scale = 0.05, low = 0.0))

        guide_eff_nh = numpyro.sample('guide_eff_nh', dist.TruncatedNormal(loc = 0.8, scale = 0.05, low = 0.0, high = 1.0))

    with numpyro.plate('genes', lengths['len_genes']):

        gene_ko_growth = numpyro.sample('gene_ko_growth', dist.Normal(0.0, 0.1))

    with numpyro.plate('cell_lines', lengths['len_cell_lines']):

        cell_line_eff = numpyro.sample('cell_line_eff', dist.TruncatedNormal(loc = 0.90, scale = 0.1, low = 0.0, high = 1.0))

        # cell_line_growth = numpyro.sample('cell_line_growth', dist.TruncatedNormal(loc = 0.5, scale = 0.5, low = 0.0)) # Was Normal
        cell_line_growth = numpyro.sample('cell_line_growth', dist.Normal(loc = 0.5, scale = 0.5)) # Was Normal

        # Vectors of overdispersion parameters means and sqrt(var) from data
        # for each cell line

        od_means = prior_params['od_means']
        od_stds = prior_params['od_stds'] / 4.

        inv_mv_mean = numpyro.sample('inv_mv_mean', dist.TruncatedNormal(loc = od_means, scale = 1., low = 1.0))
        inv_mv_std = numpyro.sample('inv_mv_std', dist.TruncatedNormal(loc = od_stds, scale = 1., low = 0.0))

    if not only_singletons:

        with numpyro.plate('guide_counts', lengths['len_guide_pairs']):

            guide_init_count = numpyro.sample('guide_init_count', dist.TruncatedNormal(loc = 500., scale = 200., low = 0.0))

            guide_pair_eff_mean = numpyro.sample('guide_pair_eff_mean', dist.TruncatedNormal(loc = 0.9, scale = 0.05, low = 0.0, high = 1.0))
            guide_pair_eff_std = numpyro.sample('guide_pair_eff_std', dist.TruncatedNormal(loc = 0.05, scale = 0.05, low = 0.0))

        with numpyro.plate('gene_pairs', lengths['len_gene_pairs']):

            gene_pair_ko_growth = numpyro.sample('gene_pair_ko_growth', dist.Normal(0.0, 0.25))

        guide_eff_1 = numpyro.sample('guide_eff_1', dist.TruncatedNormal(loc = guide_eff_mean[indices['guide_1_idx']], scale = guide_eff_std[indices['guide_1_idx']], low = 0.0, high = 1.0))
        guide_eff_2 = numpyro.sample('guide_eff_2', dist.TruncatedNormal(loc = guide_eff_mean[indices['guide_2_idx']], scale = guide_eff_std[indices['guide_2_idx']], low = 0.0, high = 1.0))

        guide_eff_12 = numpyro.sample('guide_eff_12', dist.TruncatedNormal(loc = guide_pair_eff_mean[indices['guide_pair_idx']], scale = guide_pair_eff_std[indices['guide_pair_idx']], low = 0.0, high = 1.0))

        inv_mv = numpyro.sample('inv_mv', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_idx']], scale = inv_mv_std[indices['cell_line_idx']], low = 1.0))

        mv = numpyro.deterministic('mv', 1. / inv_mv)

        guide_init = guide_init_count

        gene_ko_growth_1 = gene_ko_growth[indices['gene_1_idx']]
        gene_ko_growth_2 = gene_ko_growth[indices['gene_2_idx']]
        gene_ko_growth_12 = gene_pair_ko_growth[indices['gene_pair_idx']]

        cell_line_eff_v = cell_line_eff[indices['cell_line_idx']]
        cell_line_growth_v = cell_line_growth[indices['cell_line_idx']]

        init_theta = guide_init

        theta = 1. + guide_eff_1 * guide_eff_2 * guide_eff_12 * \
                  ( jnp.exp(cell_line_growth_v * (gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12) ) - 1. )

        theta = init_theta[indices['guide_pair_idx']] * theta

        numpyro.sample('obs_init', dist.Poisson(init_theta), obs=data['initial_counts'])

        numpyro.sample('obs', dist.NegativeBinomial2(theta, theta * mv / (1 - mv)), obs=data['counts'])

    if not no_singletons:

        with numpyro.plate('guides_counts_s', lengths['len_guide_pairs_s']):

            guide_init_count_s = numpyro.sample('guide_init_count_s', dist.TruncatedNormal(loc = 500., scale = 200., low = 0.0))

        guide_eff_s = numpyro.sample('guide_eff_s', dist.TruncatedNormal(loc = guide_eff_mean[indices['guide_s_idx']], scale = guide_eff_std[indices['guide_s_idx']], low = 0.0, high = 1.0))

        inv_mv_s = numpyro.sample('inv_mv_s', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_s_idx']], scale = inv_mv_std[indices['cell_line_s_idx']], low = 1.0))

        mv_s = numpyro.deterministic('mv_s', 1. / inv_mv_s)

        guide_init_s = guide_init_count_s

        gene_ko_growth_s = gene_ko_growth[indices['gene_s_idx']]

        cell_line_eff_s = cell_line_eff[indices['cell_line_s_idx']]
        cell_line_growth_s = cell_line_growth[indices['cell_line_s_idx']]

        init_theta_s = guide_init_s

        theta_s = init_theta_s[indices['guide_pair_s_idx']] * ( 1. + guide_eff_s * \
                               ( jnp.exp( cell_line_growth_s * gene_ko_growth_s) - 1. ) )

        numpyro.sample('obs_init_s', dist.Poisson(init_theta_s), obs=data['initial_counts_s'])

        numpyro.sample('obs_s', dist.NegativeBinomial2(theta_s, theta_s * mv_s / (1 - mv_s)), obs=data['counts_s'])

    if not no_controls:

        with numpyro.plate('guides_counts_c', lengths['len_guide_pairs_c']):

            guide_init_count_c = numpyro.sample('guide_init_count_c', dist.TruncatedNormal(loc = 500., scale = 200., low = 0.0))

        cell_line_growth_c = cell_line_growth[indices['cell_line_c_idx']]

        inv_mv_c = numpyro.sample('inv_mv_c', dist.TruncatedNormal(loc = inv_mv_mean[indices['cell_line_c_idx']], scale = inv_mv_std[indices['cell_line_c_idx']], low = 1.0))

        mv_c = numpyro.deterministic('mv_c', 1. / inv_mv_c)

        init_theta_c = guide_init_count_c

        # Maybe means this should be + in exp?

        theta_c = init_theta_c[indices['guide_pair_c_idx']] * ( 1. + ( jnp.exp(cell_line_growth_c) - 1. ) )

        numpyro.sample('obs_init_c', dist.Poisson(init_theta_c), obs=data['initial_counts_c'])

        numpyro.sample('obs_c', dist.NegativeBinomial2(theta_c, theta_c * mv_c / (1 - mv_c)), obs=data['counts_c'])


def chronosModelHierarchySingleOD(data, lengths, indices, no_singletons = False, only_singletons = False, offset = False):

    with numpyro.plate('guides', lengths['len_guides']):

        guide_eff_mean = numpyro.sample('guide_eff_mean', dist.TruncatedNormal(loc = 0.9, scale = 0.01, low = 0.0, high = 1.0))
        guide_eff_std = numpyro.sample('guide_eff_std', dist.TruncatedNormal(loc = 0.05, scale = 0.01, low = 0.0))

        guide_eff_nh = numpyro.sample('guide_eff_nh', dist.TruncatedNormal(loc = 0.8, scale = 0.05, low = 0.0, high = 1.0))

    with numpyro.plate('genes', lengths['len_genes']):

        gene_ko_growth = numpyro.sample('gene_ko_growth', dist.Normal(0.0, 0.1))

    with numpyro.plate('cell_lines', lengths['len_cell_lines']):

        cell_line_eff = numpyro.sample('cell_line_eff', dist.TruncatedNormal(loc = 0.95, scale = 0.1, low = 0.0, high = 1.0))
        # cell_line_growth = numpyro.sample('cell_line_growth', dist.TruncatedNormal(loc = 0.1, scale = 1.0, low = 0.0)) # Was Normal
        cell_line_growth = numpyro.sample('cell_line_growth', dist.Normal(loc = 0.1, scale = 0.2)) # Was Normal

        inv_mv = numpyro.sample('inv_mv', dist.TruncatedNormal(loc = 10., scale = 10., low = 1.0))

        mv = numpyro.deterministic('mv', 1. / inv_mv)

    inv_mv_init = numpyro.sample('inv_mv_init', dist.TruncatedNormal(loc = 2., scale = 1., low = 1.0))
    mv_init = numpyro.deterministic('mv_init', 1. / inv_mv_init)

    if offset:

        with numpyro.plate('offsets', lengths['len_cell_lines']):

            # Set according to control count distribution
            control_rate = numpyro.sample('control_rate', dist.TruncatedNormal(loc = 3000., scale = 10., low = 0.0))
            control_rate_init = numpyro.sample('control_rate_init', dist.TruncatedNormal(loc = 3000., scale = 10., low = 0.0))

        offset_val = control_rate - control_rate_init

        numpyro.sample('obs_control', dist.Poisson(control_rate[indices["cell_line_c_idx"]]), obs=data['control_counts'])
        numpyro.sample('obs_control_init', dist.Poisson(control_rate_init[indices["cell_line_c_idx"]]), obs=data['control_counts_init'])

    if not only_singletons:

        with numpyro.plate('guide_counts', lengths['len_guide_pairs']):

            guide_init_count = numpyro.sample('guide_init_count', dist.TruncatedNormal(loc = 2000., scale = 100., low = 0.0))

            guide_pair_eff_mean = numpyro.sample('guide_pair_eff_mean', dist.TruncatedNormal(loc = 0.9, scale = 0.05, low = 0.0, high = 1.0))
            guide_pair_eff_std = numpyro.sample('guide_pair_eff_std', dist.TruncatedNormal(loc = 0.05, scale = 0.05, low = 0.0))

        with numpyro.plate('gene_pairs', lengths['len_gene_pairs']):

            gene_pair_ko_growth = numpyro.sample('gene_pair_ko_growth', dist.Normal(0.0, 0.25))

        guide_eff_1 = numpyro.sample('guide_eff_1', dist.TruncatedNormal(loc = guide_eff_mean[indices['guide_1_idx']], scale = guide_eff_std[indices['guide_1_idx']], low = 0.0, high = 1.0))
        guide_eff_2 = numpyro.sample('guide_eff_2', dist.TruncatedNormal(loc = guide_eff_mean[indices['guide_2_idx']], scale = guide_eff_std[indices['guide_2_idx']], low = 0.0, high = 1.0))

        guide_eff_12 = numpyro.sample('guide_eff_12', dist.TruncatedNormal(loc = guide_pair_eff_mean[indices['guide_pair_idx']], scale = guide_pair_eff_std[indices['guide_pair_idx']], low = 0.0, high = 1.0))

        guide_init = guide_init_count

        gene_ko_growth_1 = gene_ko_growth[indices['gene_1_idx']]
        gene_ko_growth_2 = gene_ko_growth[indices['gene_2_idx']]
        gene_ko_growth_12 = gene_pair_ko_growth[indices['gene_pair_idx']]

        cell_line_eff_v = cell_line_eff[indices['cell_line_idx']]
        cell_line_growth_v = cell_line_growth[indices['cell_line_idx']]

        init_theta = guide_init#[indices['guide_pair_idx']]

        theta = 1. + guide_eff_1 * guide_eff_2 * guide_eff_12 * \
                  ( jnp.exp(cell_line_growth_v * (gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12) ) - 1. )

        theta = init_theta[indices['guide_pair_idx']] * theta

        if offset: theta = jax.nn.softplus(theta - offset_val[indices['cell_line_idx']]) + 1E-8

        numpyro.sample('obs_init', dist.Poisson(init_theta), obs=data['initial_counts'])

        numpyro.sample('obs', dist.NegativeBinomial2(theta, theta * mv[indices['cell_line_idx']] / (1 - mv[indices['cell_line_idx']])), obs=data['counts'])

    if not no_singletons:

        # For guides across cell lines
        with numpyro.plate('guides_counts_s', lengths['len_guide_pairs_s']):

            guide_init_count_s = numpyro.sample('guide_init_count_s', dist.TruncatedNormal(loc = 2000., scale = 100., low = 0.0))

        guide_eff_s = numpyro.sample('guide_eff_s', dist.TruncatedNormal(loc = guide_eff_mean[indices['guide_s_idx']], scale = guide_eff_std[indices['guide_s_idx']], low = 0.0, high = 1.0))

        # NO hierarchy
        # guide_eff_s = guide_eff_nh[indices['guide_s_idx']]

        guide_init_s = guide_init_count_s

        gene_ko_growth_s = gene_ko_growth[indices['gene_s_idx']]

        cell_line_eff_s = cell_line_eff[indices['cell_line_s_idx']]
        cell_line_growth_s = cell_line_growth[indices['cell_line_s_idx']]

        init_theta_s = guide_init_s#[indices['guide_pair_s_idx']]

        theta_s = init_theta_s[indices['guide_pair_s_idx']] * ( 1. + guide_eff_s * \
                               ( jnp.exp( cell_line_growth_s * gene_ko_growth_s) - 1. ) )

        # theta_s = init_theta_s[indices['guide_pair_s_idx']] * ( 1. + guide_eff_s * \
        #                        ( jnp.exp( gene_ko_growth_s) - 1. ) )

        if offset: theta_s = jax.nn.softplus(theta_s - offset_val[indices['cell_line_s_idx']]) + 1E-8

        numpyro.sample('obs_init_s', dist.Poisson(init_theta_s), obs=data['initial_counts_s'])
        # numpyro.sample('obs_init_s', dist.NegativeBinomial2(init_theta_s, init_theta_s * mv_init / (1 - mv_init)), obs=data['initial_counts_s'])

        # numpyro.sample('obs_s', dist.Poisson(theta_s), obs=data['counts_s'])
        numpyro.sample('obs_s', dist.NegativeBinomial2(theta_s, theta_s * mv[indices['cell_line_s_idx']] / (1 - mv[indices['cell_line_s_idx']])), obs=data['counts_s'])


def chronosModelHierarchyGuide(data, lengths, indices, no_singletons = False, only_singletons = False, offset = False):

    with numpyro.plate('guides', lengths['len_guides']):

        guide_eff_mean = numpyro.param('guide_eff_mean', 0.9)
        guide_eff_std = numpyro.param('guide_eff_std', 0.1, constraint = dist.constraints.positive)

        guide_eff_nh = numpyro.sample('guide_eff_nh', dist.TruncatedNormal(loc = guide_eff_mean, scale = guide_eff_mean, low = 0.0, high = 1.0))

    with numpyro.plate('genes', lengths['len_genes']):

        gene_ko_growth_mean = numpyro.param('gene_ko_growth_mean', 0.0)
        gene_ko_growth_std = numpyro.param('gene_ko_growth_std', 0.1, constraint = dist.constraints.positive)

        gene_ko_growth = numpyro.sample('gene_ko_growth', dist.Normal(gene_ko_growth_mean, gene_ko_growth_std))

    with numpyro.plate('cell_lines', lengths['len_cell_lines']):

        cell_ko_growth_mean = numpyro.param('cell_ko_growth_mean', 0.1)
        cell_ko_growth_std = numpyro.param('cell_ko_growth_std', 0.1, constraint = dist.constraints.positive)

        cell_line_growth = numpyro.sample('cell_line_growth', dist.TruncatedNormal(loc = cell_ko_growth_mean, scale = cell_ko_growth_std, low = 0.0)) # Was Normal

        inv_mv_mean = numpyro.param('inv_mv_mean', 50.)
        inv_mv_std = numpyro.param('inv_mv_std', 0.1, constraint = dist.constraints.positive)

        inv_mv = numpyro.sample('inv_mv', dist.TruncatedNormal(loc = inv_mv_mean, scale = inv_mv_std, low = 1.0))

    if not no_singletons:

        # For guides across cell lines
        with numpyro.plate('guides_counts_s', lengths['len_guide_pairs_s']):

            guide_init_count_mean = numpyro.param('guide_init_count_mean', 2000.)
            guide_init_count_std = numpyro.param('guide_init_count_std', 0.1, constraint = dist.constraints.positive)

            guide_init_count_s = numpyro.sample('guide_init_count_s', dist.TruncatedNormal(loc = guide_init_count_mean, scale = guide_init_count_std, low = 0.0))

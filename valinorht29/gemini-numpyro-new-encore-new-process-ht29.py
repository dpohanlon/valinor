import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from tqdm import tqdm

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

from pprint import pprint

import re

import numpy as np

import socket

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp

from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive, SVI, Trace_ELBO, TraceMeanField_ELBO
from numpyro.distributions import constraints
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoNormal, AutoDelta, AutoLaplaceApproximation

import pandas as pd

import time

from tqdm import tqdm

import socket

import h5py

import argparse

# from numpyroSimModels import gemini, geminiSingle, geminiSingle_guide
# from numpyroSimModels import chronosModel, chronosModelDKO, chronosModelDKOGuideHierarchy, chronosModelDKOGuides
# from numpyroSimModels import chronosModelOffset

from modelsHierarchyEncoreHT29 import chronosModelHierarchy, chronosModelHierarchy2, chronosModelHierarchySeparateEffs

def calculateOverdispersion(df):

    reps = df.groupby(['GuidePair', 'cell_line_index']).agg(mean = ('value', np.mean), std = ('value', np.std), cell_line_index = ('cell_line_index', 'first')).reset_index(drop = True)

    reps['var'] = reps['std'] ** 2
    reps['od'] = reps['var'] / reps['mean']

    repsC = reps.groupby(['cell_line_index']).agg(mean = ('od', np.median), std = ('od', np.std))
    repsC = repsC.sort_values('cell_line_index')

    print(repsC)

    return repsC['mean'].values, repsC['std'].values

def prepareData(only_singletons = False,
    combsFile = '/nfs/research/petsalaki/users/dohanlon/encore/data/encoreCombsNewProcess.pq',
    singlesFile = '/nfs/research/petsalaki/users/dohanlon/encore/data/encoreSinglesNewProcess.pq',
    controlsFile = '/nfs/research/petsalaki/users/dohanlon/encore/data/encoreControlsNewProcess.pq',
    library = 'COLO1'):

    # if "codon" in socket.gethostname():

    df = pd.read_parquet(combsFile)
    dfSingles = pd.read_parquet(singlesFile)
    dfControls = pd.read_parquet(controlsFile)

    # else:
    #
    #     df = pd.read_hdf('/nfs/research/petsalaki/users/dohanlon/encore/data/dfCombs_ace.h5', key = 'ace')
    #     dfSingles = pd.read_hdf('/nfs/research/petsalaki/users/dohanlon/encore/data/dfSgl_ace.h5', key = 'ace')

    lfcs = jnp.array(df["value"].values.reshape(-1))

    lfcs_s = jnp.array(dfSingles['value'].values).reshape(-1)

    lfcs_c = jnp.array(dfControls['value'].values).reshape(-1)

    guide_pair_idx = jnp.array(
        df["guide_pair_index"].values.reshape(-1)
    ).reshape(-1)

    # With separate singleton and combination efficiencies
    guide_1_idx = jnp.array(
        df["guide1_index"].values.reshape(-1)
    ).reshape(-1)
    guide_2_idx = jnp.array(
        df["guide2_index"].values.reshape(-1)
    ).reshape(-1)

    # Collapse into common guide_pair_index pDNA values, sort by guide_pair_index
    # so that the index of the array is the guide_pair_index (assuming
    # this starts from zero

    initial_counts = df.groupby('guide_pair_index').agg({'guide_pair_index' : 'first', 'plasmid' : 'first'})[['guide_pair_index', 'plasmid']].reset_index(drop = True).sort_values('guide_pair_index')['plasmid']
    initial_counts_s = dfSingles.groupby('guide_pair_index').agg({'guide_pair_index' : 'first', 'plasmid' : 'first'})[['guide_pair_index', 'plasmid']].reset_index(drop = True).sort_values('guide_pair_index')['plasmid']
    initial_counts_c = dfControls.groupby('guide_pair_index').agg({'guide_pair_index' : 'first', 'plasmid' : 'first'})[['guide_pair_index', 'plasmid']].reset_index(drop = True).sort_values('guide_pair_index')['plasmid']

    meanOD, stdOD = calculateOverdispersion(df)

    prior_params = {'od_means' : meanOD, 'od_stds' : stdOD}

    # Guide pairs are indexed separately for combinations and singletons
    # as they have to index their own pDNA array

    guide_pair_s_idx = jnp.array(
        dfSingles["guide_pair_index"].values.reshape(-1)
    ).reshape(-1)

    guide_pair_c_idx = jnp.array(
        dfControls["guide_pair_index"].values.reshape(-1)
    ).reshape(-1)

    gene_unq_pair_index = jnp.array(
        df["gene_pair_index"].values.reshape(-1)
    ).reshape(-1)
    gene_1_idx = jnp.array(df["gene1_index"].values.reshape(-1)).reshape(
        -1
    )
    gene_2_idx = jnp.array(df["gene2_index"].values.reshape(-1)).reshape(
        -1
    )

    cell_line_index = jnp.array(
        df["cell_line_index"].values.reshape(-1)
    ).reshape(-1)

    # With separate combination/singleton efficiencies
    guide_s_idx = np.array(
        dfSingles["guide1_index"].values
    ).reshape(-1)
    gene_s_idx = np.array(
        dfSingles["gene1_index"].values
    ).reshape(-1)

    cell_line_s_idx = jnp.array(
        dfSingles['cell_line_index'].values
    ).reshape(-1)
    cell_line_c_idx = jnp.array(
        dfControls['cell_line_index'].values
    ).reshape(-1)

    len_guide_pairs = len(np.unique(guide_pair_idx))
    len_guides = len(np.unique(np.concatenate((guide_1_idx, guide_2_idx, guide_s_idx))))
    len_guide_pairs_s = len(np.unique(guide_pair_s_idx))
    len_guide_pairs_c = len(np.unique(guide_pair_c_idx))
    len_gene_pairs = len(np.unique(gene_unq_pair_index))
    len_genes = len(np.unique(np.concatenate((gene_1_idx, gene_2_idx, gene_s_idx))))
    len_cell_lines = len(np.unique(cell_line_index))

    lengths = {
        "len_guide_pairs": len_guide_pairs,
        "len_guides": len_guides,
        "len_guide_pairs_s": len_guide_pairs_s,
        "len_guide_pairs_c": len_guide_pairs_c,
        "len_gene_pairs": len_gene_pairs,
        "len_genes": len_genes,
        "len_cell_lines": len_cell_lines,
    }

    indices = {
        "guide_pair_idx": guide_pair_idx,
        "guide_1_idx": guide_1_idx,
        "guide_2_idx": guide_2_idx,
        "gene_pair_idx": gene_unq_pair_index,
        "gene_1_idx": gene_1_idx,
        "gene_2_idx": gene_2_idx,
        "cell_line_idx": cell_line_index,
        "guide_s_idx": guide_s_idx,
        "guide_pair_s_idx" : guide_pair_s_idx,
        "guide_pair_c_idx" : guide_pair_c_idx,
        "gene_s_idx": gene_s_idx,
        "cell_line_s_idx": cell_line_s_idx,
        "cell_line_c_idx": cell_line_c_idx,

    }

    data = {
        "lfcs": lfcs,
        "lfcs_s": lfcs_s,
        "lfcs_c": lfcs_c,
        "counts_s": lfcs_s,
        "counts_c": lfcs_c,
        "counts": lfcs,
        "initial_counts_s" : initial_counts_s.values,
        "initial_counts_c" : initial_counts_c.values,
        "initial_counts" : initial_counts.values,

    }

    pprint(lengths)
    pprint([(k, len(v)) for k, v in data.items()])
    pprint([(k, len(v)) for k, v in indices.items()])

    pprint([(k, np.max(v)) for k, v in indices.items()])

    return lengths, indices, prior_params, data

def runGemini(lengths, indices, prior_params, data, name = '', no_singletons=False, only_singletons=False, no_controls = False):

    model = chronosModelHierarchy

    guide = AutoNormal(model)
    # guide = AutoLaplaceApproximation(model)

    optimizer = numpyro.optim.Adam(step_size=1e-2)

    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO(num_particles = 1))

    svi_result = svi.run(
        random.PRNGKey(42),
        10000,
        data,
        lengths,
        indices,
        prior_params,
        no_singletons=no_singletons,
        only_singletons=only_singletons,
        no_controls = no_controls,
        stable_update=True,
    )

    params = svi_result.params
    for k, v in params.items():
        if ("mean" in k) or ("std" in k):
            print(k, v)
            print(k, v)
        else:
            print(k, v.shape)

    print('guide_eff_auto_loc', params['guide_eff_auto_loc'].shape)
    print('guide_eff_auto_scale', params['guide_eff_auto_scale'].shape)

    with h5py.File(f"score-{name}-params.h5", 'w') as paramFile:
        paramFile.create_dataset('guide_eff_auto_loc', data = np.array(params['guide_eff_auto_loc']))
        paramFile.create_dataset('guide_eff_auto_scale', data = np.array(params['guide_eff_auto_scale']))

    nSamples = 100
    # nSamples = 25

    predictive = Predictive(guide, params=params, num_samples=nSamples)
    p_samples = predictive(
        random.PRNGKey(42),
        data,
        lengths,
        indices,
        prior_params,
        no_singletons=no_singletons,
        only_singletons=only_singletons,
        no_controls = no_controls,
    )

    print(p_samples.keys())

    # print('library bias', np.mean(p_samples['library_bias'], 0), np.std(p_samples['library_bias'], 0))

    samples_s = []
    samples_s_init = []
    samples = []
    offsets = []
#
    cell_line_index = indices["cell_line_idx"]

    if not only_singletons:

        gene_1_idx = indices["gene_1_idx"]
        gene_2_idx = indices["gene_2_idx"]

        guide_1_idx = indices["guide_1_idx"]
        guide_2_idx = indices["guide_2_idx"]

        gene_pair_idx = indices["gene_pair_idx"]
        guide_pair_idx = indices["guide_pair_idx"]

    if not no_singletons:

        gene_s_idx = indices["gene_s_idx"]
        guide_s_idx = indices["guide_s_idx"]

    for i in range(nSamples):

        if not no_singletons:

            # Expand the array of size n_unique_guides to size n_measurements according
            # to the guide pair index
            init_count_s = p_samples['guide_init_count_s'][i][indices['guide_pair_s_idx']]
            # cell_eff_s = p_samples['cell_line_eff'][i][indices['cell_line_s_idx']]

            # guide_eff_s = p_samples['guide_eff'][i][indices['guide_s_idx']]
            guide_eff_s = p_samples['guide_eff'][i][indices['guide_s_idx'], indices['cell_line_s_idx']]
            # guide_eff_s = p_samples['guide_eff_nh'][i][indices['guide_s_idx']]

            # print(guide_eff_s.shape)
            # print(p_samples['guide_eff'].shape)

            cell_growth_s = p_samples['cell_line_growth'][i][indices['cell_line_s_idx']]
            ko_growth_s = p_samples['gene_ko_growth'][i][indices['gene_s_idx']]

            theta_s = init_count_s * (1. + guide_eff_s * (np.exp(cell_growth_s + p_samples['library_bias'][i][indices['cell_line_s_idx']] * ko_growth_s) - 1.))

            # theta_s = np.exp(-cell_growth_s) * init_count_s * ( (1. - guide_eff_s) * np.exp(cell_growth_s) + \
            #                                                                                       guide_eff_s * np.exp(ko_growth_s) )

            # theta_s = init_count_s * ( (1. - guide_eff_s) + guide_eff_s * np.exp(ko_growth_s - cell_growth_s) )

            mv_s = 1./p_samples['inv_mv_s'][i]
            # mv_s_init = 1./p_samples['inv_mv_init'][i]

            # sample = np.random.poisson(theta)
            sample_s = np.array(dist.NegativeBinomial2(theta_s, theta_s * mv_s / (1 - mv_s)).sample(random.PRNGKey(42)))

            sample_s_init = np.array(dist.Poisson(init_count_s).sample(random.PRNGKey(42)))
            # sample_s_init = np.array(dist.NegativeBinomial2(init_count_s, init_count_s * mv_s_init / (1 - mv_s_init)).sample(random.PRNGKey(42)))

            samples_s.append(sample_s)
            samples_s_init.append(sample_s_init)

        if not only_singletons:

            init_count = p_samples['guide_init_count'][i, indices['guide_pair_idx']]
            cell_line_eff = p_samples['cell_line_eff'][i, indices['cell_line_idx']]
            cell_line_growth = p_samples['cell_line_growth'][i, indices['cell_line_idx']]


            # guide_eff_1 = p_samples['guide_eff'][i][indices['guide_1_idx']]
            # guide_eff_2 = p_samples['guide_eff'][i][indices['guide_2_idx']]

            guide_eff_1 = p_samples['guide_eff'][i][indices['guide_1_idx'], indices['cell_line_idx']]
            guide_eff_2 = p_samples['guide_eff'][i][indices['guide_2_idx'], indices['cell_line_idx']]
            guide_eff_12 = p_samples['guide_eff_12'][i]

            # guide_eff_12 = p_samples['guide_pair_eff'][i, indices['guide_pair_idx']]

            gene_ko_growth_1 = p_samples['gene_ko_growth'][i, indices['gene_1_idx']]
            gene_ko_growth_2 = p_samples['gene_ko_growth'][i, indices['gene_2_idx']]
            gene_ko_growth_12 = p_samples['gene_pair_ko_growth'][i, indices['gene_pair_idx']]

            mv = 1./p_samples['inv_mv'][i]

            theta = 1. + guide_eff_1 * guide_eff_2 *\
              ( jnp.exp( cell_line_growth + (gene_ko_growth_1 + gene_ko_growth_2 + gene_ko_growth_12) ) - 1. )

            theta *= init_count

            # theta = np.exp(-cell_line_growth) * init_count * ( (1. - (guide_eff_1 + guide_eff_1 + guide_eff_12)) * np.exp(cell_line_growth) + \
            #                                                                                  guide_eff_1 * np.exp(gene_ko_growth_1) + \
            #                                                                                  guide_eff_2 * np.exp(gene_ko_growth_2) + \
            #                                                                                  guide_eff_12 * np.exp(gene_ko_growth_1 + gene_ko_growth_1 + gene_ko_growth_12)
            #                                                                                )

            # theta = init_count * ( (1. - (guide_eff_1 + guide_eff_1 + guide_eff_12)) + \
            #                                                                                  guide_eff_1 * np.exp(gene_ko_growth_1 - cell_line_growth) + \
            #                                                                                  guide_eff_2 * np.exp(gene_ko_growth_2 - cell_line_growth) + \
            #                                                                                  guide_eff_12 * np.exp((gene_ko_growth_12 - cell_line_growth) + (gene_ko_growth_1 - cell_line_growth) + (gene_ko_growth_2 - cell_line_growth))
            #                                                                                )

            # theta += offsetVal

            sample = np.array(dist.NegativeBinomial2(theta, theta * mv / (1 - mv)).sample(random.PRNGKey(42)))

            samples.append(sample)

    samples = np.mean(np.array(samples), 0)
    samples_s = np.mean(np.array(samples_s), 0)
    samples_s_init = np.mean(np.array(samples_s_init), 0)

    if not no_singletons:

        scoreDF_s = pd.DataFrame(np.array(samples_s).flatten(), columns=["samples_s"])

        init_count_s = p_samples['guide_init_count_s'][:, indices['guide_pair_s_idx']]
        cell_eff_s = p_samples['cell_line_eff'][:, indices['cell_line_s_idx']]

        # guide_eff_s = p_samples['guide_eff_nh'][:, indices['guide_s_idx']]
        # guide_eff_s = p_samples['guide_eff_s']
        guide_s_eff_mean = p_samples['guide_eff_mean'][:, indices['guide_s_idx']]
        guide_s_eff_std = p_samples['guide_eff_std'][:, indices['guide_s_idx']]

        guide_eff_s = p_samples['guide_eff'][:, indices['guide_s_idx'], indices['cell_line_s_idx']]

        cell_growth_s = p_samples['cell_line_growth'][:, indices['cell_line_s_idx']]
        ko_growth_s = p_samples['gene_ko_growth'][:, indices['gene_s_idx']]
        mv_s = p_samples['inv_mv_s']

        theta_s = init_count_s * (1. + guide_eff_s * (np.exp(cell_growth_s + ko_growth_s) - 1.))

        scoreDF_s["init_count_s"] = np.mean(init_count_s, 0)
        # scoreDF_s["cell_eff_s"] = np.mean(cell_eff_s, 0)
        scoreDF_s["guide_s_eff_mean"] = np.mean(guide_s_eff_mean, 0)
        scoreDF_s["guide_s_eff_std"] = np.mean(guide_s_eff_std, 0)
        scoreDF_s["guide_eff_s"] = np.mean(guide_eff_s, 0)
        scoreDF_s["cell_growth_s"] = np.mean(cell_growth_s, 0)
        scoreDF_s["ko_growth_s"] = np.mean(ko_growth_s, 0)
        scoreDF_s["mv_s"] = np.mean(mv_s, 0)
        scoreDF_s["theta_s"] = np.mean(theta_s, 0)

        # scoreDF_s["mv_s_init"] = np.mean(p_samples['inv_mv_init'], 0)

        # print(p_samples['inv_mv_init'])
        # print(type(p_samples['inv_mv_init']))

        scoreDF_s["init_count_s_std"] = np.std(init_count_s, 0)
        # scoreDF_s["cell_eff_s_std"] = np.std(cell_eff_s, 0)
        scoreDF_s["guide_s_eff_mean_std"] = np.std(guide_s_eff_mean, 0)
        scoreDF_s["guide_s_eff_std_std"] = np.std(guide_s_eff_std, 0)
        scoreDF_s["guide_eff_s_std"] = np.std(guide_eff_s, 0)
        # scoreDF_s["cell_growth_s_std"] = np.std(cell_growth_s, 0)
        scoreDF_s["ko_growth_s_std"] = np.std(ko_growth_s, 0)
        scoreDF_s["mv_s_std"] = np.std(mv_s, 0)
        scoreDF_s["theta_s_std"] = np.std(theta_s, 0)
        # scoreDF_s["mv_s_init_std"] = np.std(p_samples['inv_mv'], 0)

        scoreDF_s['samples_s_init'] = samples_s_init

        scoreDF_s.to_hdf(f"score{name}_s.h5", "score", complevel=9, mode = 'w')

    if not only_singletons:

        scoreDF = pd.DataFrame(np.array(samples).flatten(), columns=["samples"])

        init_count = p_samples['guide_init_count'][:, indices['guide_pair_idx']]
        # cell_eff = p_samples['cell_line_eff'][:, indices['cell_line_index']]
        cell_growth = p_samples['cell_line_growth'][:, indices['cell_line_idx']]


        # guide_eff_1 = p_samples['guide_eff'][:, indices['guide_1_idx']]
        # guide_eff_2 = p_samples['guide_eff'][:, indices['guide_2_idx']]

        # guide_eff_1 = p_samples['guide_eff_1']
        # guide_eff_2 = p_samples['guide_eff_2']
        # guide_eff_12 = p_samples['guide_eff_12']

        guide_eff_1 = p_samples['guide_eff'][:, indices['guide_1_idx'], indices['cell_line_idx']]
        guide_eff_2 = p_samples['guide_eff'][:, indices['guide_2_idx'], indices['cell_line_idx']]
        guide_eff_12 = p_samples['guide_eff_12']

        # len(guides)
        # Just guides for the first gene for now (symmetric? maybe!)
        guide_1_eff_mean = p_samples['guide_eff_mean'][:, indices['guide_1_idx']]
        guide_1_eff_std = p_samples['guide_eff_std'][:, indices['guide_1_idx']]
        guide_2_eff_mean = p_samples['guide_eff_mean'][:, indices['guide_2_idx']]
        guide_2_eff_std = p_samples['guide_eff_std'][:, indices['guide_2_idx']]

        # len(guide_pairs)
        guide_pair_eff_mean = p_samples['guide_pair_eff_mean'][:, indices['guide_pair_idx']]
        guide_pair_eff_std = p_samples['guide_pair_eff_std'][:, indices['guide_pair_idx']]

        # guide_eff_12 = p_samples['guide_pair_eff'][:, indices['guide_pair_idx']]

        ko_growth_1 = p_samples['gene_ko_growth'][:, indices['gene_1_idx']]
        ko_growth_2 = p_samples['gene_ko_growth'][:, indices['gene_2_idx']]
        ko_growth_12 = p_samples['gene_pair_ko_growth'][:, indices['gene_pair_idx']]

        # controlRate_s = p_samples['control_rate'][:, indices['cell_line_index']]
        # controlRateInit_s = p_samples['control_rate_init'][:, indices['cell_line_index']]

        # offsetVals = controlRate_s - controlRateInit_s

        mv = p_samples['inv_mv']

        scoreDF["init_count"] = np.mean(init_count, 0)
        # scoreDF["cell_eff"] = np.mean(cell_eff, 0)
        scoreDF["cell_growth"] = np.mean(cell_growth, 0)

        scoreDF["guide_eff_1"] = np.mean(guide_eff_1, 0)
        scoreDF["guide_eff_2"] = np.mean(guide_eff_2, 0)
        scoreDF["guide_eff_12"] = np.mean(guide_eff_12, 0)

        scoreDF["guide_1_eff_mean"] = np.mean(guide_1_eff_mean, 0)
        scoreDF["guide_1_eff_std"] = np.mean(guide_1_eff_std, 0)

        scoreDF["guide_2_eff_mean"] = np.mean(guide_2_eff_mean, 0)
        scoreDF["guide_2_eff_std"] = np.mean(guide_2_eff_std, 0)

        scoreDF["guide_pair_eff_mean"] = np.mean(guide_pair_eff_mean, 0)
        scoreDF["guide_pair_eff_std"] = np.mean(guide_pair_eff_std, 0)

        scoreDF["ko_growth_1"] = np.mean(ko_growth_1, 0)
        scoreDF["ko_growth_2"] = np.mean(ko_growth_2, 0)
        scoreDF["ko_growth_12"] = np.mean(ko_growth_12, 0)

        scoreDF["mv"] = np.mean(mv, 0)

        # scoreDF["offsetVals"] = np.mean(offsetVals, 0)

        scoreDF["init_count_std"] = np.std(init_count, 0)
        # scoreDF["cell_eff_std"] = np.std(cell_eff, 0)
        scoreDF["cell_growth_std"] = np.std(cell_growth, 0)

        scoreDF["guide_eff_1_std"] = np.std(guide_eff_1, 0)
        scoreDF["guide_eff_2_std"] = np.std(guide_eff_2, 0)
        scoreDF["guide_eff_12_std"] = np.std(guide_eff_12, 0)

        scoreDF["guide_1_eff_mean_std"] = np.std(guide_1_eff_mean, 0)
        scoreDF["guide_1_eff_std_std"] = np.std(guide_1_eff_std, 0)
        
        scoreDF["guide_2_eff_mean_std"] = np.std(guide_2_eff_mean, 0)
        scoreDF["guide_2_eff_std_std"] = np.std(guide_2_eff_std, 0)

        scoreDF["guide_pair_eff_mean_std"] = np.std(guide_pair_eff_mean, 0)
        scoreDF["guide_pair_eff_std_std"] = np.std(guide_pair_eff_std, 0)

        scoreDF["ko_growth_1_std"] = np.std(ko_growth_1, 0)
        scoreDF["ko_growth_2_std"] = np.std(ko_growth_2, 0)
        scoreDF["ko_growth_12_std"] = np.std(ko_growth_12, 0)

        scoreDF["mv_std"] = np.std(mv, 0)

        scoreDF.to_hdf(f"score{name}.h5", "score", complevel=9, mode = 'w')

    plt.plot(np.log(svi_result.losses))
    plt.savefig(f"{name}gemini_sim_svi_losses.pdf")
    plt.clf()

    plt.plot(np.log(svi_result.losses)[-1000:])
    plt.savefig(f"{name}gemini_sim_svi_losses_1k.pdf")
    plt.clf()

    plt.plot(np.log(svi_result.losses)[-10000:])
    plt.savefig(f"{name}gemini_sim_svi_losses_10k.pdf")
    plt.clf()

if __name__ == "__main__":

    # I'd like an argument, please
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "--no-singletons",
        action="store_true",
        dest="no_singletons",
        default=False,
        help="No singletons.",
    )

    argParser.add_argument(
        "--only-singletons",
        action="store_true",
        dest="only_singletons",
        default=False,
        help="Only singletons.",
    )

    argParser.add_argument(
        "--no-controls",
        action="store_true",
        dest="no_controls",
        default=False,
        help="No controls.",
    )

    argParser.add_argument("-n", type=str, dest="name", default="", help="Output name.")

    argParser.add_argument(
        "--combsFile",
        type=str,
        dest="combsFile",
        default="",
        help="ENCORE combinations Parquet file",
    )

    argParser.add_argument(
        "--singlesFile",
        type=str,
        dest="singlesFile",
        default="",
        help="ENCORE singles Parquet file",
    )

    argParser.add_argument(
        "--controlsFile",
        type=str,
        dest="controlsFile",
        default="",
        help="ENCORE controls Parquet file",
    )

    argParser.add_argument(
        "--library",
        type=str,
        dest="library",
        default="COLO1",
        help="ENCORE library",
    )

    args = argParser.parse_args()

    lengths, indices, prior_params, data = prepareData(args.only_singletons, args.combsFile, args.singlesFile, args.controlsFile, args.library)

    runGemini(
        lengths, indices, prior_params, data, args.name, args.no_singletons, args.only_singletons, args.no_controls
    )

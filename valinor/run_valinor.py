import argparse

import jax
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

from tqdm import tqdm

from jax import random
import numpyro

from numpyro.infer import Predictive, SVI, TraceMeanField_ELBO
from numpyro.infer.autoguide import AutoNormal, AutoLowRankMultivariateNormal

from numpyro.handlers import seed, trace

from valinor import models
from valinor.guides import valinor_full_guide, valinor_singles_guide, valinor_controls_guide
from valinor.utils import (
    getIndices,
    calculateLengths,
    configArgs,
    saveModelParams,
    loadPriors,
    getBatchData,
    configure_custom_init,
)
from valinor.preprocessing import prepareData, getDeltaLFC
from valinor.postprocessing import sampleParams, createDataFrame
from valinor.plotting import plotDiagPlots

from typing import Dict, List, Tuple, Any

import json

gpu_available = any(device.platform == 'gpu' for device in jax.devices())

if gpu_available:
    numpyro.set_platform('gpu')

def get_model_sites(model, *args):
    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace(*args)
    return list(model_trace.keys())


def initialize_svi(model, guide, config):
    optimizer = numpyro.optim.ClippedAdam(step_size=config["lr"], clip_norm=1.0)
    svi = SVI(
        model,
        guide,
        optimizer,
        loss=TraceMeanField_ELBO(num_particles=config["n_particles"]),
    )
    return svi


def run_svi(
    svi,
    prng_key,
    epochs,
    data,
    lengths,
    indices,
    prior_params,
    init_params=None,
    **kwargs,
):
    return svi.run(
        prng_key,
        epochs,
        init_params=init_params,
        data=data,
        lengths=lengths,
        indices=indices,
        prior_params=prior_params,
        **kwargs,
    )


def save_results(params, config, outputDir):
    saveModelParams(params, f'{outputDir}{config["paramsFileName"]}')
    with open(f"valinorrun_{config['name']}.json", "w") as outfile:
        json.dump(config, outfile)


def sample_posterior(
    predictive, config, data, lengths, indices, prior_params, batch=False
):

    if config["only_singletons"] == False:
        args = {'no_singletons' : config["no_singletons"],
                'only_singletons' : config["only_singletons"],
                'no_controls' : config["no_controls"]}
    else:
        args = {}

    if not batch:
        # Non-batch case: directly sample from the posterior
        samples = predictive(
            random.PRNGKey(42),
            data=data,
            lengths=lengths,
            indices=indices,
            prior_params=prior_params,
            **args,
            zi=config["zi"],
            predict=True,
        )

        sampledParams = sampleParams(
            samples, indices, config["alternateLH"], "gene_effect_means" in prior_params
        )

        combsDF = (
            createDataFrame(sampledParams["combs"])
            if ("combs" in sampledParams)
            else None
        )
        singlesDF = (
            createDataFrame(sampledParams["singles"])
            if ("singles" in sampledParams)
            else None
        )

        return combsDF, singlesDF

    # Batched case
    n_batches = np.ceil(
        (
            len(data["final"]["combinations"])
            if not config["only_singletons"]
            else len(data["final"]["singletons"])
        )
        / config["batch_size"]
    ).astype(int)

    combsDFs, singlesDFs = [], []

    for i in tqdm(range(n_batches)):
        start_idx = i * config["batch_size"]
        end_idx = (i + 1) * config["batch_size"]

        batch_data, batch_indices = getBatchData(data, indices, start_idx, end_idx)

        samples = predictive(
            random.PRNGKey(42),
            data=batch_data,
            lengths=lengths,
            indices=batch_indices,
            prior_params=prior_params,
            no_singletons=config["no_singletons"],
            only_singletons=config["only_singletons"],
            no_controls=config["no_controls"],
            guide_config=config["guide_config"],
            zi=config["zi"],
            predict=True,
        )

        sampledParams = sampleParams(
            samples,
            batch_indices,
            config["alternateLH"],
            "gene_effect_means" in prior_params,
        )

        if config["only_singletons"] == False:
            combsDFs.append(createDataFrame(sampledParams["combs"]))
        if config["no_singletons"] == False:
            singlesDFs.append(createDataFrame(sampledParams["singles"]))

    combsDF = pd.concat(combsDFs) if config["only_singletons"] == False else None
    singlesDF = pd.concat(singlesDFs) if config["no_singletons"] == False else None

    return combsDF, singlesDF


def save_posterior_samples(combsDF, singlesDF, config, outputDir):

    if (config["only_singletons"] == False) and (combsDF is not None):
        combsDF.to_parquet(
            f"{outputDir}combsModel.pq"
            if config["name"] is None
            else f"{outputDir}combsModel_{config['name']}.pq"
        )
    if (config["no_singletons"] == False) and (singlesDF is not None):
        singlesDF.to_parquet(
            f"{outputDir}singlesModel.pq"
            if config["name"] is None
            else f"{outputDir}singlesModel_{config['name']}.pq"
        )


def runValinor(lengths, indices, prior_params, data, config):
    prng_key_controls = random.PRNGKey(42)
    outputDir = (
        f"{config['outputDir'].rstrip('/')}/" if config["outputDir"] != "" else ""
    )

    # Initialize params
    params_controls = None
    params_singles = None

    # Check for controls
    if "controls" in data["final"] and data["final"]["controls"] is not None:

        controls_guide = AutoNormal(models.valinorControls, init_loc_fn=numpyro.infer.init_to_median())
        # controls_guide = valinor_controls_guide

        svi_controls = initialize_svi(
            models.valinorControls, controls_guide, config
        )

        # Shape should be okay, as these are plasmids with single counts, no replicates
        init_params_controls = {
            "guide_init_count_c": data["initial"]["controls"],
        }

        state_controls = run_svi(
            svi_controls,
            prng_key_controls,
            config["epochs"],
            data,
            lengths,
            indices,
            prior_params,
            init_params = init_params_controls
        )
        params_controls = state_controls.params
    else:
        print("No controls data found. Skipping controls step.")

    # Check for singles
    if "singletons" in data["final"] and data["final"]["singletons"] is not None:

        init_params_singles = params_controls if params_controls != None else {}
        init_params_singles["guide_init_count_s"] = prior_params["init_count_s_vals"]

        custom_init = configure_custom_init(init_params_singles)

        singles_guide = AutoNormal(models.valinorSingles, init_loc_fn=custom_init())
        # singles_guide = valinor_singles_guide

        svi_singles = initialize_svi(
            models.valinorSingles, singles_guide, config
        )

        singles_rng_init, _ = random.split(prng_key_controls)
        singles_args = {"guide_config": config["guide_config"], "zi": config["zi"], "stable_update": config["stable_update"]}
        state_singles = run_svi(
            svi_singles,
            singles_rng_init,
            config["epochs"],
            data,
            lengths,
            indices,
            prior_params,
            init_params=init_params_singles,
            **singles_args,
        )
        params_singles = state_singles.params

        plotDiagPlots(state_singles, name=f"{config['name']}_singles", outputDir=outputDir)

        # Sample from the model
        sites_from_model = get_model_sites(
            models.valinorSingles,
            data,
            lengths,
            indices,
            prior_params,
            config["guide_config"],
            config["zi"],
        )

        predictive = Predictive(
            models.valinorSingles,
            guide = singles_guide,
            params=params_singles,
            num_samples=config["nSamples"],
            return_sites=sites_from_model,
        )

        # Sampling posterior
        combsDF, singlesDF = sample_posterior(
            predictive,
            config,
            data,
            lengths,
            indices,
            prior_params,
            batch=config["batch_sample"],
        )

        save_posterior_samples(combsDF, singlesDF, config, outputDir)

    else:
        print("No singles data found. Skipping singles step.")
        params_singles = params_controls

    # Check for combinations
    if "combinations" in data["final"] and data["final"]["combinations"] is not None:

        init_params_combinations = params_singles if params_singles != None else {}
        init_params_combinations["guide_init_count"] = prior_params["init_count_vals"]

        if "dLFC" in prior_params:
            init_params_combinations["gene_pair_ko_growth"] = prior_params["dLFC"]

        custom_init = configure_custom_init(init_params_combinations)

        full_guide = AutoNormal(models.valinorHierarchy, init_loc_fn=custom_init())

        svi_full = initialize_svi(
            models.valinorHierarchy, full_guide, config
        )

        full_rng_init, _ = random.split(prng_key_controls)
        full_args = {
            "no_singletons": config["no_singletons"],
            "only_singletons": config["only_singletons"],
            "no_controls": config["no_controls"],
            "alternate": config["alternateLH"],
            "guide_config": config["guide_config"],
            "zi": config["zi"],
            "stable_update": config["stable_update"],
        }

        # Run the full model with params from singles or controls (if singles are missing)
        state_full = run_svi(
            svi_full,
            full_rng_init,
            config["epochs"],
            data,
            lengths,
            indices,
            prior_params,
            init_params=init_params_combinations,
            **full_args,
        )

        # Plot and save results
        plotDiagPlots(state_full, name=config["name"], outputDir=outputDir)
        params = state_full.params
        save_results(params, config, outputDir)

        # Sample from the model
        sites_from_model = get_model_sites(
            models.valinorHierarchy,
            data,
            lengths,
            indices,
            prior_params,
            config["no_singletons"],
            config["only_singletons"],
            config["no_controls"],
            config["alternateLH"],
            config["guide_config"],
            config["zi"],
        )

        predictive = Predictive(
            models.valinorHierarchy,
            guide = full_guide,
            params=params,
            num_samples=config["nSamples"],
            return_sites=sites_from_model,
        )

        # Sampling posterior
        combsDF, singlesDF = sample_posterior(
            predictive,
            config,
            data,
            lengths,
            indices,
            prior_params,
            batch=config["batch_sample"],
        )
        save_posterior_samples(combsDF, singlesDF, config, outputDir)
    else:
        print("No combinations data found. Skipping combinations step.")


def makeArgs():
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

    argParser.add_argument(
        "--stable-update",
        action="store_true",
        dest="stable_update",
        default=False,
        help="Stable update.",
    )

    argParser.add_argument(
        "--alternateLH",
        action="store_true",
        dest="alternateLH",
        default=False,
        help="Whether to use the alternate approximate likelihood.",
    )

    argParser.add_argument(
        "-n", type=str, dest="name", default=None, help="Output name."
    )

    argParser.add_argument(
        "-o", type=str, dest="outputDir", default="", help="Output directory."
    )

    argParser.add_argument(
        "--paramsFileName",
        type=str,
        dest="paramsFileName",
        default="params.h5",
        help="Parameters file name.",
    )

    argParser.add_argument(
        "--lr", type=float, dest="lr", default=0.01, help="Learning rate."
    )

    argParser.add_argument(
        "--epochs",
        type=int,
        dest="epochs",
        default=50000,
        help="Number of epochs to train for.",
    )

    argParser.add_argument(
        "--nSamples",
        type=int,
        dest="nSamples",
        default=100,
        help="Number of samples to draw from the fitted model.",
    )

    argParser.add_argument(
        "--nParticles",
        type=int,
        dest="n_particles",
        default=1,
        help="Number of particles for the ELBO minimisation",
    )

    argParser.add_argument(
        "--config",
        "-c",
        type=str,
        dest="config",
        default=None,
        help="Config JSON file (overrides all other config)",
    )

    argParser.add_argument(
        "--combinationsFile",
        type=str,
        dest="combinationsFile",
        default=None,
        help="Data combinations file.",
    )

    argParser.add_argument(
        "--singletonsFile",
        type=str,
        dest="singletonsFile",
        default=None,
        help="Data singletons file.",
    )

    argParser.add_argument(
        "--controlsFile",
        type=str,
        dest="controlsFile",
        default=None,
        help="Data controls file.",
    )

    argParser.add_argument(
        "--batch-sample",
        action="store_true",
        dest="batch_sample",
        default=False,
        help="Whether to sample batches of data from the model.",
    )

    argParser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=2**14,
        help="Batch size for batched operations.",
    )

    argParser.add_argument(
        "--guide-config",
        type=str,
        dest="guide_config",
        default="partial_pooling",
        help="Guide pooling type, one of 'no_pooling', 'full pooling', or 'partial_pooling'.",
    )

    argParser.add_argument(
        "--priorsFile",
        type=str,
        dest="priorsFile",
        default=None,
        help="Prior parameters file.",
    )

    argParser.add_argument(
        "--ZINB",
        dest="zi",
        default=False,
        action="store_true",
        help="Set final distributions to be zero inflated.",
    )

    argParser.add_argument(
        "--reindex",
        dest="reindex",
        default=False,
        action="store_true",
        help="Reindex data types that are only of one class (i.e., only combinations or only singles) in the case where guides are not common.",
    )

    return argParser


def run():
    argParser = makeArgs()
    args = argParser.parse_args()

    # Automatically set flags if files are not provided and display warnings
    if args.singletonsFile is None:
        args.no_singletons = True
        print("Warning: No singletons file provided. --no-singletons flag set.")

    if args.controlsFile is None:
        args.no_controls = True
        print("Warning: No controls file provided. --no-controls flag set.")

    if args.combinationsFile is None:
        args.only_singletons = True
        print("Warning: No combinations file provided. --only-singletons flag set.")

    config = configArgs(args)

    data_files = {
        "combinations": config["combinationsFile"],
        "singletons": config["singletonsFile"],
        "controls": config["controlsFile"],
    }

    loaded_priors = loadPriors(args.priorsFile) if args.priorsFile is not None else None

    lengths, indices, prior_params, data = prepareData(
        data_files,
        loaded_priors,
        config["only_singletons"],
        not config["no_singletons"],
        not config["no_controls"],
        config["reindex"],
    )

    runValinor(lengths, indices, prior_params, data, config)

if __name__ == "__main__":
    run()

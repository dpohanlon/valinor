import argparse

import jax.numpy as jnp
import numpy as np
import pandas as pd

from tqdm import tqdm

from jax import random
import numpyro

from numpyro.infer import Predictive, SVI, TraceMeanField_ELBO
from numpyro.infer.autoguide import AutoNormal

from numpyro.handlers import seed, trace

from valinor import models
from valinor.utils import (
    getIndices,
    calculateLengths,
    configArgs,
    saveModelParams,
    loadPriors,
    getBatchData,
)
from valinor.preprocessing import prepareData
from valinor.postprocessing import sampleParams, createDataFrame
from valinor.plotting import plotDiagPlots

from typing import Dict, List, Tuple, Any

import json


def get_model_sites(model, *args):
    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace(*args)
    return list(model_trace.keys())


def runValinor(
    lengths: Dict[str, int],
    indices: Dict[str, Any],
    prior_params: Dict[str, Any],
    data: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Runs the Valinor model on the provided data.

    Args:
        lengths (Dict[str, int]): Dictionary containing length values.
        indices (Dict[str, Any]): Dictionary containing index arrays.
        prior_params (Dict[str, Any]): Dictionary containing prior parameters.
        data (Dict[str, Any]): Dictionary containing data arrays.
        config (Namespace): Configuration options for the run.
    """

    guide = AutoNormal(models.valinorHierarchy)

    optimizer = numpyro.optim.Adam(step_size=config["lr"])

    svi = SVI(
        models.valinorHierarchy,
        guide,
        optimizer,
        loss=TraceMeanField_ELBO(num_particles=config["n_particles"]),
    )

    svi_result = svi.run(
        random.PRNGKey(42),
        config["epochs"],
        data,
        lengths,
        indices,
        prior_params,
        no_singletons=config["no_singletons"],
        only_singletons=config["only_singletons"],
        no_controls=config["no_controls"],
        alternate=config["alternateLH"],
        stable_update=config["stable_update"],
        guide_config=config["guide_config"],
        zi=config["zi"],
    )

    outputDir = ""
    if config["outputDir"] != outputDir:
        outputDir = f"{config['outputDir'].rstrip('/')}/"

    plotDiagPlots(svi_result, name=config["name"], outputDir=outputDir)

    params = svi_result.params

    saveModelParams(params, f'{outputDir}{config["paramsFileName"]}')

    # Run selected post-processing, save samples, make plots, etc

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
        guide,
        params=params,
        num_samples=config["nSamples"],
        return_sites=sites_from_model,
    )

    if not config["batch_sample"]:

        # Sample from posterior
        samples = predictive(
            random.PRNGKey(42),
            data,
            lengths,
            indices,
            prior_params,
            no_singletons=config["no_singletons"],
            only_singletons=config["only_singletons"],
            no_controls=config["no_controls"],
            guide_config=config["guide_config"],
            zi=config["zi"],
        )

        sampledParams = sampleParams(samples, indices, config["alternateLH"])

        if config["only_singletons"] == False:
            combsDF = createDataFrame(sampledParams["combs"])

        if config["no_singletons"] == False:
            singlesDF = createDataFrame(sampledParams["singles"])

    else:

        # Sample from posterior in batches, concatenating to the same dataframe.
        # We can do this as the model is independent of the length of the data.

        n_batches = np.ceil(
            len(data["final"]["combinations"])
            if not config["only_singletons"]
            else len(data["final"]["singletons"]) / config["batch_size"]
        ).astype(int)

        combsDFs = []
        singlesDFs = []

        for i in tqdm(range(n_batches)):

            start_idx = i * config["batch_size"]
            end_idx = (i + 1) * config["batch_size"]

            batch_data, batch_indices = getBatchData(data, indices, start_idx, end_idx)

            samples = predictive(
                random.PRNGKey(42),
                batch_data,
                lengths,
                batch_indices,
                prior_params,
                no_singletons=config["no_singletons"],
                only_singletons=config["only_singletons"],
                no_controls=config["no_controls"],
                guide_config=config["guide_config"],
                zi=config["zi"],
            )

            sampledParams = sampleParams(samples, batch_indices, config["alternateLH"])

            if config["only_singletons"] == False:
                combsDFs.append(createDataFrame(sampledParams["combs"]))

            if config["no_singletons"] == False:
                singlesDFs.append(createDataFrame(sampledParams["singles"]))

        if config["only_singletons"] == False:
            combsDF = pd.concat(combsDFs)

        if config["no_singletons"] == False:
            singlesDF = pd.concat(singlesDFs)

    if config["only_singletons"] == False:

        combsDF.to_parquet(
            f"{outputDir}combsModel.pq"
            if config["name"] is None
            else f"{outputDir}combsModel_{config['name']}.pq"
        )

    if config["no_singletons"] == False:

        singlesDF.to_parquet(
            f"{outputDir}singlesModel.pq"
            if config["name"] is None
            else f"{outputDir}singlesModel_{config['name']}.pq"
        )

    # save the config to a json
    with open(f"valinorrun_{config['name']}.json", "w") as outfile:
        json.dump(config, outfile)


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

    config = configArgs(args)

    # These can be `None`, and the downstream methods will deal with it accordingly
    data_files = {
        "combinations": config["combinationsFile"],
        "singletons": config["singletonsFile"],
        "controls": config["controlsFile"],
    }

    loaded_priors = loadPriors(args.priorsFile) if args.priorsFile != None else None

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

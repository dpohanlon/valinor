import argparse

import jax.numpy as jnp

from jax import random
import numpyro

from numpyro.infer import Predictive, SVI, TraceMeanField_ELBO
from numpyro.infer.autoguide import AutoNormal

from valinor import models
from valinor.utils import getIndices, calculateLengths, configArgs, saveModelParams
from valinor.preprocessing import prepareData
from plotting.plots import plotDiagPlots

from typing import Dict, List, Tuple, Any


def runValinor(
    lengths: Dict[str, int],
    indices: Dict[str, Any],
    prior_params: Dict[str, Any],
    data: Dict[str, Any],
    config: Dict[str, Any],
    name: str = "",
    no_singletons: bool = False,
    only_singletons: bool = False,
    no_controls: bool = False,
) -> None:
    """
    Runs the Valinor model on the provided data.

    Args:
        lengths (Dict[str, int]): Dictionary containing length values.
        indices (Dict[str, Any]): Dictionary containing index arrays.
        prior_params (Dict[str, Any]): Dictionary containing prior parameters.
        data (Dict[str, Any]): Dictionary containing data arrays.
        config (Namespace): Configuration options for the run.
        name (str, optional): Name for the run. Defaults to "".
        no_singletons (bool, optional): If True, singletons are not included. Defaults to False.
        only_singletons (bool, optional): If True, only singletons are included. Defaults to False.
        no_controls (bool, optional): If True, controls are not included. Defaults to False.
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
        stable_update=config["stable_update"],
    )

    params = svi_result.params

    saveModelParams(params, config["paramsFileName"])

    # Run selected post-processing, save samples, make plots, etc

    predictive = Predictive(guide, params=params, num_samples=config["nSamples"])

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
    )

    plotDiagPlots(svi_result)

def makeArgs():
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

    argParser.add_argument(
        "--stable-update",
        action="store_true",
        dest="stable_update",
        default=False,
        help="Stable update.",
    )

    argParser.add_argument("-n", type=str, dest="name", default="", help="Output name.")

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

    lengths, indices, prior_params, data = prepareData(
        data_files,
        config["only_singletons"],
        ~config["no_singletons"],
        ~config["no_controls"],
    )

    runValinor(lengths, indices, prior_params, data, config)


if __name__ == "__main__":
    run()

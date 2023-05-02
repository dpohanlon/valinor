import argparse

from valinor import models
from utils import getIndices, calculateLengths

from typing import Dict, List, Tuple, Any


def prepareData(
    data_files: Dict[str, str],
    only_singletons: bool = False,
    no_singletons: bool = False,
    no_controls: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:

    """
    Prepares data for the Valinor model.

    Args:
        data_files (Dict[str, str]): Path to the data files or dictionary of data files.
        only_singletons (bool, optional): If True, only singleton data is included. Defaults to False.
        no_singletons (bool, optional): If True, singleton data is not included. Defaults to False.
        no_controls (bool, optional): If True, control data is not included. Defaults to True.

    Returns:
        Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]: Tuple containing data, lengths, and indices.
    """

    pass


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

    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO(num_particles=1))

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
        "--config",
        "-c",
        type=str,
        dest="config",
        default=None,
        help="Config JSON file (overrides all other config)",
    )

    args = argParser.parse_args()

    lengths, indices, prior_params, data = prepareData(args.only_singletons)

    runValinor(lengths, indices, prior_params, data, config)

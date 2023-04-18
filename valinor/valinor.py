import argparse

from models import valinorHierarchy
from utils import getIndices, calculateLengths


def prepareData(
    data_files, only_singletons=False, no_singletons=False, no_controls=True
):
    pass


def runValinor(
    lengths,
    indices,
    prior_params,
    data,
    name="",
    no_singletons=False,
    only_singletons=False,
    no_controls=False,
):

    guide = AutoNormal(valinorHierarchy)

    optimizer = numpyro.optim.Adam(step_size=config['lr'])

    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO(num_particles=1))

    svi_result = svi.run(
        random.PRNGKey(42),
        config['epochs'],
        data,
        lengths,
        indices,
        prior_params,
        no_singletons=config['no_singletons'],
        only_singletons=config['only_singletons'],
        no_controls=config['no_controls'],
        stable_update=config['stable_update'],
    )

    params = svi_result.params

    saveModelParams(params, config['paramsFileName'])

    predictive = Predictive(guide, params=params, num_samples=config['nSamples'])

    # Sample from posterior
    samples = predictive(
        random.PRNGKey(42),
        data,
        lengths,
        indices,
        prior_params,
        no_singletons=config['no_singletons'],
        only_singletons=config['only_singletons'],
        no_controls=config['no_controls'],
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

    runValinor(
        lengths,
        indices,
        prior_params,
        data,
        config
    )

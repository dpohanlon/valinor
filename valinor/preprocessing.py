import numpy as np

from valinor.utils import (
    loadData,
    getFinalCounts,
    getInitialCounts,
    getIndices,
    calculateLengths,
    checkBounds,
)

from valinor.priors import calculateOverdispersion, defaultPriors

from typing import Dict, List, Tuple, Any


def prepareData(
    data_files: Dict[str, str],
    only_singletons: bool = False,
    singletons: bool = True,
    controls: bool = True,
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

    datasets = loadData(data_files)

    finalCounts = getFinalCounts(datasets)
    initialCounts = getInitialCounts(datasets)

    meanOD, stdOD = calculateOverdispersion(datasets["combinations"] if not (datasets["combinations"] is None) else datasets["singletons"])

    prior_params = defaultPriors()
    prior_params["od_means"] = meanOD
    prior_params["od_stds"] = stdOD

    prior_params["init_count"] = (np.mean(datasets["combinations"]['plasmid']), np.std(datasets["combinations"]['plasmid']),)

    if not (datasets["singletons"] is None):

        prior_params["init_count_s"] = (np.mean(datasets["singletons"]['plasmid']), np.std(datasets["singletons"]['plasmid']),)

    # These args can be `None`
    indices = getIndices(
        datasets["combinations"], datasets["singletons"], datasets["controls"]
    )

    lengths = calculateLengths(indices, singletons=singletons, neg_controls=controls)

    checkBounds(indices, lengths, singletons = singletons, neg_controls = controls)

    return (
        lengths,
        indices,
        prior_params,
        {"final": finalCounts, "initial": initialCounts},
    )

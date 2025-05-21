import numpy as np

from valinor.utils import (
    loadData,
    getFinalCounts,
    getInitialCounts,
    getIndices,
    calculateLengths,
    checkBounds,
    reindexDF,
    combinationLFCs,
    countZeros
)

from valinor.priors import (
    calculateOverdispersion,
    defaultPriors,
    calculate_cell_line_stats,
    calculate_gene_stats,
    calcInitCountParams,
)

from typing import Dict, List, Tuple, Any


def prepareData(
    data_files: Dict[str, str],
    priors: Dict[str, float],
    only_singletons: bool = False,
    singletons: bool = True,
    controls: bool = True,
    reindex: bool = False,
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
    initialCounts, initialCountIndices = getInitialCounts(datasets)

    meanOD, stdOD = calculateOverdispersion(
        datasets["combinations"]
        if not (datasets["combinations"] is None)
        else datasets["singletons"]
    )

    prior_params = defaultPriors()

    # Update with our input overriding values
    if (priors != None) and (len(priors) > 0):
        for k in prior_params.keys():
            prior_params[k] = priors.get(k, prior_params[k])

    prior_params["od_means"] = meanOD
    prior_params["od_stds"] = stdOD

    if not only_singletons:
        prior_params["init_count"] = (
            np.mean(datasets["combinations"]["plasmid"]),
            np.std(datasets["combinations"]["plasmid"]) * 0.1, # std not realistic
        )

        if "dLFC" in datasets["combinations"]:
            prior_params["dLFC"] = getDeltaLFC(datasets["combinations"])
        elif not (datasets['singletons'] is None):
            prior_params["dLFC"] = combinationLFCs(datasets["combinations"], datasets["singletons"])


        prior_params["init_count_vals"] = calcInitCountParams(datasets["combinations"], initCountVar = 'plasmid', singletons = False)[0]

        prior_params['p_zi'] = countZeros(datasets['combinations'])

    if not (datasets["singletons"] is None):
        prior_params["init_count_s"] = (
            np.mean(datasets["singletons"]["plasmid"]),
            np.std(datasets["singletons"]["plasmid"]),
        )

        prior_params["init_count_s_vals"] = calcInitCountParams(datasets["singletons"], initCountVar = 'plasmid')[0]

        prior_params['p_zi_s'] = countZeros(datasets['singletons'])

    if not (datasets["controls"] is None):
        prior_params["init_count_c"] = (
            np.mean(datasets["controls"]["plasmid"]),
            np.std(datasets["controls"]["plasmid"]),
        )

    # Init params for controls, cell line stats for control DF

    if controls:  # and config == True

        print("Setting control priors using data.")

        control_means, control_stds = calculate_cell_line_stats(datasets["controls"])

        prior_params["control_means"] = control_means
        prior_params["control_stds"] = control_stds

    if singletons:  # and config == True

        print("Setting single gene effect priors using data.")

        # 2D array, genes x cell lines
        gene_means, gene_stds, idx_c, idx_g = calculate_gene_stats(
            datasets["singletons"]
        )

        prior_params["gene_effect_means"] = gene_means
        prior_params["gene_effect_stds"] = gene_stds

    if not singletons:

        # try and guess these

        pass

    if reindex:
        if singletons and not only_singletons:
            print("Only reindex with a single data type!")
        elif only_singletons:
            datasets["singletons"] = reindexDF(datasets["singletons"], singletons = True)
        else:
            datasets["combinations"] = reindexDF(datasets["combinations"])

    # These args can be `None`
    indices = getIndices(
        datasets["combinations"],
        datasets["singletons"],
        datasets["controls"],
        only_singletons,
        singletons,
        controls,
    )

    if singletons:
        # Add the singleton specific indices to map plasmids to their initial values, with duplicates for the null guides that aren't parameterised
        indices['guide_initial_s_idx'] = np.array(initialCountIndices['singletons'])

    lengths = calculateLengths(
        indices,
        singletons=singletons,
        neg_controls=controls,
        only_singletons=only_singletons,
    )

    checkBounds(
        indices,
        lengths,
        singletons=singletons,
        neg_controls=controls,
        only_singletons=only_singletons,
    )

    return (
        lengths,
        indices,
        prior_params,
        {"final": finalCounts, "initial": initialCounts},
    )


def getDeltaLFC(combinations):

    pair_grouped = combinations.groupby("gene_unq_pair_index").agg({"dLFC": "mean"})

    return pair_grouped["dLFC"].values

import numpy as np

import jax.numpy as jnp

from valinor.utils import (
    loadData,
    getFinalCounts,
    getInitialCounts,
    getIndices,
    calculateLengths,
    getInitialCountsDF,
    checkBounds,
    reindexDF,
    combinationLFCs,
    countZeros,
)

from valinor.priors import (
    calculateOverdispersion,
    defaultPriors,
    calculate_cell_line_stats,
    calculate_gene_stats,
    calcInitCountParams,
)

from typing import Dict, List, Tuple, Any


def make_jax(
    lengths,
    indices,
    final_counts,
    initial_counts,
    float_dtype=jnp.float32,
    int_dtype=jnp.int32,
):
    """
    Convert arrays that will participate in JAX tracing to jnp arrays,
    with sensible dtypes:
      - indices -> int_dtype
      - observed count arrays -> int_dtype
    Leaves `lengths` as Python ints on purpose (for numpyro.plate sizes).

    Returns: (lengths, jax_indices, data_dict)
             where data_dict = {"final": jax_final_counts, "initial": jax_initial_counts}
    """

    # -------- helpers --------
    def _asarray(x, dtype=None):
        if x is None:
            return None
        if isinstance(x, jnp.ndarray):
            return x.astype(dtype) if dtype is not None and x.dtype != dtype else x
        if isinstance(x, np.ndarray) or isinstance(x, list):
            return jnp.asarray(x, dtype=dtype)
        # Pandas support (optional)
        try:
            import pandas as pd  # type: ignore

            if isinstance(x, (pd.Series, pd.Index)):
                return jnp.asarray(x.to_numpy(), dtype=dtype)
            if isinstance(x, pd.DataFrame):
                return jnp.asarray(x.values, dtype=dtype)
        except Exception:
            pass
        # scalar or something we don't want to touch
        return x

    def _convert_counts_dict(d):
        out = {}
        for k, v in (d or {}).items():
            out[k] = _asarray(v, dtype=int_dtype) if v is not None else None
        return out

    # -------- indices -> jnp.int --------
    jax_indices = {}
    for k, v in (indices or {}).items():
        jax_indices[k] = _asarray(v, dtype=int_dtype) if v is not None else None

    # -------- data counts -> jnp.int --------
    jax_final = _convert_counts_dict(final_counts or {})
    jax_initial = _convert_counts_dict(initial_counts or {})

    data = {"final": jax_final, "initial": jax_initial}

    # NB: lengths intentionally unchanged (plates want ints)
    return lengths, jax_indices, data


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
            np.std(datasets["combinations"]["plasmid"]),
        )

        if "dLFC" in datasets["combinations"]:
            prior_params["dLFC"] = getDeltaLFC(datasets["combinations"])
        elif not (datasets["singletons"] is None):
            prior_params["dLFC"] = combinationLFCs(
                datasets["combinations"], datasets["singletons"]
            )

        prior_params["init_count_vals"] = calcInitCountParams(
            datasets["combinations"], initCountVar="plasmid", singletons=False
        )

        prior_params["p_zi"] = countZeros(datasets["combinations"])

    if not (datasets["singletons"] is None):
        prior_params["init_count_s"] = (
            np.mean(datasets["singletons"]["plasmid"]),
            np.std(datasets["singletons"]["plasmid"]),
        )

        prior_params["init_count_s_vals"] = calcInitCountParams(
            datasets["singletons"], initCountVar="plasmid"
        )

        prior_params["p_zi_s"] = countZeros(datasets["singletons"])

    if not (datasets["controls"] is None):
        prior_params["init_count_c"] = (
            np.mean(datasets["controls"]["plasmid"]),
            np.std(datasets["controls"]["plasmid"]),
        )

        prior_params["init_count_c_vals"] = calcInitCountParams(
            datasets["controls"], initCountVar="plasmid", singletons=False
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
            datasets["singletons"] = reindexDF(datasets["singletons"], singletons=True)
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
        indices["guide_initial_s_idx"] = np.array(initialCountIndices["singletons"])

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

    lengths, indices, data = make_jax(lengths, indices, finalCounts, initialCounts)

    return (
        lengths,
        indices,
        prior_params,
        data,
    )


def getDeltaLFC(combinations):
    pair_grouped = combinations.groupby("gene_unq_pair_index").agg({"dLFC": "mean"})

    return pair_grouped["dLFC"].values

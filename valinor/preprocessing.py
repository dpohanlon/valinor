from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray
from valinor.priors import (
    calcInitCountParams,
    calculate_cell_line_stats,
    calculate_gene_stats,
    calculateOverdispersion,
    defaultPriors,
)
from valinor.utils import (
    calculateLengths,
    checkBounds,
    combinationLFCs,
    compute_or_load_exposures,
    countZeros,
    getFinalCounts,
    getIndices,
    getInitialCounts,
    getInitialCountsDF,
    loadData,
    reindexDF,
)


def _asarray(x, dtype=None):
    if x is None:
        return None

    if isinstance(x, jnp.ndarray):
        return x.astype(dtype) if dtype is not None and x.dtype != dtype else x

    if isinstance(x, (np.ndarray, list, tuple)):
        return jnp.asarray(x, dtype=dtype) if dtype is not None else jnp.asarray(x)

    try:
        if isinstance(x, (pd.Series, pd.Index)):
            arr = x.to_numpy()
            return (
                jnp.asarray(arr, dtype=dtype) if dtype is not None else jnp.asarray(arr)
            )
        if isinstance(x, pd.DataFrame):
            arr = x.to_numpy()
            return (
                jnp.asarray(arr, dtype=dtype) if dtype is not None else jnp.asarray(arr)
            )
        if isinstance(
            x, ExtensionArray
        ):  # e.g., pandas.arrays.FloatingArray, IntegerArray, BooleanArray
            arr = x.to_numpy()
            return (
                jnp.asarray(arr, dtype=dtype) if dtype is not None else jnp.asarray(arr)
            )
    except Exception:
        pass

    return x


def make_jax(
    lengths,
    indices,
    final_counts,
    initial_counts,
    float_dtype=jnp.float32,
    int_dtype=jnp.int32,
):
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


def jaxify_priors(prior_params, float_dtype=jnp.float32, int_dtype=jnp.int32):
    def _needs_float(v):
        if isinstance(v, jnp.ndarray):
            return v.dtype.kind in ("f", "i", "u", "b")
        if isinstance(v, np.ndarray):
            return v.dtype.kind in ("f", "i", "u", "b")
        return True

    def _convert(v):
        # Tuples are preserved; only elements that are array-like get converted
        if isinstance(v, tuple):
            out = []
            for e in v:
                if isinstance(e, (np.ndarray, jnp.ndarray, list)):
                    out.append(
                        _asarray(e, dtype=float_dtype if _needs_float(e) else int_dtype)
                    )
                else:
                    try:
                        import pandas as pd
                        from pandas.api.extensions import ExtensionArray

                        if isinstance(
                            e, (pd.Series, pd.Index, pd.DataFrame, ExtensionArray)
                        ):
                            out.append(
                                _asarray(
                                    e,
                                    dtype=float_dtype if _needs_float(e) else int_dtype,
                                )
                            )
                        else:
                            out.append(e)
                    except Exception:
                        out.append(e)
            return tuple(out)

        # Lists: keep list type
        if isinstance(v, list):
            out = []
            for e in v:
                out.append(_convert(e))
            return out

        # Dicts: recurse
        if isinstance(v, dict):
            return {k: _convert(val) for k, val in v.items()}

        # Array-like scalars/arrays
        if isinstance(v, (np.ndarray, jnp.ndarray, list)):
            return _asarray(v, dtype=float_dtype if _needs_float(v) else int_dtype)

        try:
            import pandas as pd
            from pandas.api.extensions import ExtensionArray

            if isinstance(v, (pd.Series, pd.Index, pd.DataFrame, ExtensionArray)):
                return _asarray(v, dtype=float_dtype if _needs_float(v) else int_dtype)
        except Exception:
            pass

        return v

    return {k: _convert(v) for k, v in (prior_params or {}).items()}


def prepareData(
    data_files: Dict[str, str],
    priors: Dict[str, float],
    only_singletons: bool = False,
    singletons: bool = True,
    controls: bool = True,
    reindex: bool = False,
    exposure: bool = True,
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

    initCountVar = "initial"

    # Try and guess what the initial count variable is
    if "plasmid" in list(datasets["combinations"]):
        initCountVar = "plasmid"

    finalCounts = getFinalCounts(datasets)
    initialCounts, initialCountIndices = getInitialCounts(
        datasets, initCountVar=initCountVar
    )

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
            np.mean(datasets["combinations"][initCountVar]),
            np.std(datasets["combinations"][initCountVar]),
        )

        if "dLFC" in datasets["combinations"]:
            prior_params["dLFC"] = getDeltaLFC(datasets["combinations"])
        elif not (datasets["singletons"] is None):
            prior_params["dLFC"] = combinationLFCs(
                datasets["combinations"], datasets["singletons"], initCountVar
            )

        prior_params["init_count_vals"] = calcInitCountParams(
            datasets["combinations"], initCountVar=initCountVar, singletons=False
        )

        prior_params["p_zi"] = countZeros(datasets["combinations"])

    if not (datasets["singletons"] is None):
        prior_params["init_count_s"] = (
            np.mean(datasets["singletons"][initCountVar]),
            np.std(datasets["singletons"][initCountVar]),
        )

        prior_params["init_count_s_vals"] = calcInitCountParams(
            datasets["singletons"], initCountVar=initCountVar
        )

        prior_params["p_zi_s"] = countZeros(datasets["singletons"])

    if not (datasets["controls"] is None):
        prior_params["init_count_c"] = (
            np.mean(datasets["controls"][initCountVar]),
            np.std(datasets["controls"][initCountVar]),
        )

        prior_params["init_count_c_vals"] = calcInitCountParams(
            datasets["controls"], initCountVar=initCountVar, singletons=False
        )

    # Init params for controls, cell line stats for control DF

    if controls:  # and config == True
        print("Setting control priors using data.")

        control_means, control_stds = calculate_cell_line_stats(
            datasets["controls"], initCountVar
        )

        prior_params["control_means"] = control_means
        prior_params["control_stds"] = control_stds

    if singletons:  # and config == True
        print("Setting single gene effect priors using data.")

        # 2D array, genes x cell lines
        gene_means, gene_stds, idx_c, idx_g = calculate_gene_stats(
            datasets["singletons"], initCountVar
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
    prior_params = jaxify_priors(
        prior_params, float_dtype=jnp.float32, int_dtype=jnp.int32
    )

    if exposure:
        exp_map, exp_arrays = compute_or_load_exposures(
            datasets,
            cell_col="cell_line_index",
            rep_col="replicate_index",
            init_col=initCountVar,
            final_col="value",
        )

        return (lengths, indices, prior_params, data, exp_arrays)

    return (
        lengths,
        indices,
        prior_params,
        data,
    )


def getDeltaLFC(combinations):
    pair_grouped = combinations.groupby("gene_unq_pair_index").agg({"dLFC": "mean"})

    return pair_grouped["dLFC"].values

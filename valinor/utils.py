import jax.numpy as jnp

# TODO: Have a better interface to these, especially when first building them
# so that it generalises to more parameters and categories


def calculateOverdispersion(df):

    reps = (
        df.groupby(["GuidePair", "cell_line_index"])
        .agg(
            mean=("value", np.mean),
            std=("value", np.std),
            cell_line_index=("cell_line_index", "first"),
        )
        .reset_index(drop=True)
    )

    reps["var"] = reps["std"] ** 2
    reps["od"] = reps["var"] / reps["mean"]

    repsC = reps.groupby(["cell_line_index"]).agg(
        mean=("od", np.median), std=("od", np.std)
    )
    repsC = repsC.sort_values("cell_line_index")

    return repsC["mean"].values, repsC["std"].values


def getInitialCounts(df, initCountVar):

    initial_counts = (
        df.groupby("guide_pair_index")
        .agg({"guide_pair_index": "first", "initCountVar": "first"})[
            ["guide_pair_index", "initCountVar"]
        ]
        .reset_index(drop=True)
        .sort_values("guide_pair_index")["initCountVar"]
    )

    return initial_counts


def getUniqueGeneGuideIndices(indices, singletons=True):

    guide_indices = [indices["guide_1_idx"], indices["guide_2_idx"]]
    gene_indices = [indices["gene_1_idx"], indices["gene_2_idx"]]

    if singletons:

        guide_indices += [indices["guide_s_idx"]]
        gene_indices += [indices["gene_s_idx"]]

    return np.unique(np.concatenate(gene_indices)), np.unique(
        np.concatenate(guide_indices)
    )


def getIndices(df, dfSingles, dfControls):

    indices = {
        "guide_pair_idx": jnp.array(df["guide_pair_index"].values),
        "gene_pair_idx": jnp.array(df["gene_unq_pair_index"].values),
        "guide_1_idx": jnp.array(df["guide1_index_s"].values),
        "guide_2_idx": jnp.array(df["guide2_index_s"].values),
        "gene_1_idx": jnp.array(df["gene1_unq_index"].values),
        "gene_2_idx": jnp.array(df["gene2_unq_index"].values),
        "cell_line_idx": jnp.array(df["cell_line_index"].values),
    }

    if dfSingles != None:

        indices["guide_pair_s_idx"] = jnp.array(dfSingles["guide_pair_index"].values)

        indices["guide_s_idx"] = jnp.array(dfSingles["guide1_index_s"].values)

        indices["gene_s_idx"] = jnp.array(dfSingles["guide1_unq_index"].values)

        indices["cell_line_s_idx"] = jnp.array(dfSingles["cell_line_index"].values)

    if dfControls != None:

        indices["cell_line_c_idx"] = jnp.array(dfControls["cell_line_index"].values)

    # Make sure these are all 0D
    for k, v in indices.items():
        indices[k] = v.reshape(-1)

    return indices


def calculateLengths(indices, singletons=True, neg_controls=True):

    # Calculate Numpyro parameter array lengths from indices

    lengths = {
        "len_cell_lines": len(np.unique(indices["cell_line_index"])),
        "len_guide_pairs": len(np.unique(indices["guide_pair_idx"])),
        "len_gene_pairs": len(np.unique(indices["gene_unq_pair_index"])),
    }

    gene_indices, guide_indices = getUniqueGeneGuideIndices(
        indices, singletons=singletons
    )

    if singletons:

        lengths["len_guide_pairs_s"] = len(np.unique(indices["guide_pair_s_idx"]))

    if neg_controls:

        lengths["len_guide_pairs_c"] = len(np.unique(indices["guide_pair_c_idx"]))

    # Unique guides, including each category in case we have unique ones there
    lengths["len_guides"] = len(guide_indices)

    # Unique genes, including each category in case we have unique ones there
    lengths["len_genes"] = len(gene_indices)

    return lengths


def checkBounds(indices, lengths, singletons=True, neg_controls=True):

    gene_indices, guide_indices = getUniqueGeneGuideIndices(
        indices, singletons=singletons
    )

    # Numpyro doesn't check whether we try to index off the end of an array,
    # so check that all of the arrays are the correct size for the indices

    assert np.max(guide_indices) < lengths["len_guides"]
    assert np.max(gene_indices) < lengths["len_genes"]

    assert np.max(indices["cell_line_index"]) < lengths["len_cell_lines"]

    assert np.max(indices["guide_pair_idx"]) < lengths["len_guide_pairs"]
    assert np.max(indices["gene_pair_idx"]) < lengths["len_gene_pairs"]

    if singletons:

        assert np.max(indices["cell_line_s_index"]) < lengths["len_cell_lines"]

        assert np.max(indices["guide_pair_s_idx"]) < lengths["len_guide_pairs_s"]

    if neg_controls:

        assert np.max(indices["cell_line_c_index"]) < lengths["len_cell_lines"]

        assert np.max(indices["guide_pair_c_idx"]) < lengths["len_guide_pairs_c"]

    # Also, warn if there are some parameters that remain unused, which is sus

    if np.max(gene_indices) != lengths["len_genes"] - 1:
        print(
            "WARNING: Some model gene parameters are un-referenced (no matching indices)."
        )

    if np.max(guide_indices) != lengths["len_guides"] - 1:
        print(
            "WARNING: Some model guide parameters are un-referenced (no matching indices)."
        )

    if np.max(indices["cell_line_index"]) != lengths["len_cell_lines"] - 1:
        print(
            "WARNING: Some model cell line parameters are un-referenced (no matching indices)."
        )

    if np.max(indices["guide_pair_idx"]) != lengths["len_guide_pairs"] - 1:
        print(
            "WARNING: Some model guide pair parameters are un-referenced (no matching indices)."
        )

    if np.max(indices["gene_pair_idx"]) != lengths["len_gene_pairs"] - 1:
        print(
            "WARNING: Some model gene pair parameters are un-referenced (no matching indices)."
        )

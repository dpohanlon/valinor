#!/usr/bin/env python3

import argparse
import re
from collections import OrderedDict

import numpy as np
import pandas as pd


def make_dense_index(*series):
    seen = OrderedDict()
    k = 0
    for s in series:
        if s is None:
            continue
        s = pd.Index(s).dropna()
        for v in s:
            if v not in seen:
                seen[v] = k
                k += 1
    return seen


def populateIndicesControls(controlsM, indices):
    controlsM["GuidePair"] = controlsM["GuidePair"].astype("string")
    controlsM["cell_line"] = controlsM["cell_line"].astype("string")

    gp_keys = set(indices["guide_pair_index"].keys())
    gpu_keys = set(indices["guide_pair_unq_index"].keys())

    key_unq = controlsM["GuidePair"] + "_" + controlsM["cell_line"]
    mask = controlsM["GuidePair"].isin(gp_keys) & key_unq.isin(gpu_keys)
    if (~mask).any():
        n_drop = int((~mask).sum())
        print(f"[populateIndicesControls] Dropping {n_drop} control rows with unseen guide pairs.")

    controlsM = controlsM.loc[mask].copy()

    controlsM["guide_pair_index"] = controlsM["GuidePair"].map(indices["guide_pair_index"])
    controlsM["guide_pair_unq_index"] = key_unq.loc[mask].map(indices["guide_pair_unq_index"])
    controlsM["cell_line_index"] = controlsM["cell_line"].map(indices["cell_line_index"])

    return controlsM


def populateIndicesSingletons(singlesM, indices):
    singlesM["GuidePair"] = singlesM["GuidePair"].astype("string")
    singlesM["SingletonGuide"] = singlesM["SingletonGuide"].astype("string")
    singlesM["SingletonGuide_o"] = singlesM["SingletonGuide_o"].astype("string")
    singlesM["SingletonGuide_o_s"] = singlesM["SingletonGuide_o_s"].astype("string")
    singlesM["SingletonGene"] = singlesM["SingletonGene"].astype("string")
    singlesM["cell_line"] = singlesM["cell_line"].astype("string")

    singlesM["guide_pair_index"] = singlesM["GuidePair"].map(
        lambda x: indices["guide_pair_s_index"][x]
    )

    singlesM["guide_pair_unq_index"] = (
        singlesM["GuidePair"] + "_" + singlesM["cell_line"]
    ).map(lambda x: indices["guide_pair_unq_s_index"][x])

    singlesM["cell_line_index"] = singlesM["cell_line"].map(
        lambda x: indices["cell_line_index"][x]
    )

    singlesM["gene1_index"] = singlesM["SingletonGene"].map(
        lambda x: indices["gene_index"][x]
    )

    singlesM["guide_index"] = singlesM["SingletonGuide"].map(
        lambda x: indices["guide_index"][x]
    )

    singlesM["guide1_index"] = singlesM["SingletonGuide_o"].map(
        lambda x: indices["guide_index_ori"][x]
    )

    singlesM["guide1_index_s"] = singlesM["SingletonGuide_o_s"].map(
        lambda x: indices["guide_index_ori_s"][x]
    )

    singlesM["guide1_unq_index"] = (
        singlesM["SingletonGuide_o"] + "_" + singlesM["cell_line"]
    ).map(lambda x: indices["guide_unq_index_ori"][x])

    singlesM["gene1_unq_index"] = (
        singlesM["SingletonGene"] + "_" + singlesM["cell_line"]
    ).map(lambda x: indices["gene_unq_index"][x])

    return singlesM


def populateIndicesCombs(dataM, indices):
    dataM["cell_line"] = dataM["cell_line"].astype("string")
    dataM["gene1"] = dataM["gene1"].astype("string")
    dataM["gene2"] = dataM["gene2"].astype("string")
    dataM["guide1"] = dataM["guide1"].astype("string")
    dataM["guide2"] = dataM["guide2"].astype("string")
    dataM["guide1_o"] = dataM["guide1_o"].astype("string")
    dataM["guide2_o"] = dataM["guide2_o"].astype("string")
    dataM["GuidePair"] = dataM["GuidePair"].astype("string")
    dataM["GenePairUnoriented"] = dataM["GenePairUnoriented"].astype("string")

    dataM["cell_line_index"] = dataM["cell_line"].map(
        lambda x: indices["cell_line_index"][x]
    )

    dataM["gene1_index"] = dataM["gene1"].map(lambda x: indices["gene_index"][x])
    dataM["gene2_index"] = dataM["gene2"].map(lambda x: indices["gene_index"][x])

    dataM["gene_pair_index"] = dataM["GenePairUnoriented"].map(
        lambda x: indices["gene_pair_index"][x]
    )

    dataM["guide_pair_index"] = dataM["GuidePair"].map(
        lambda x: indices["guide_pair_index"][x]
    )

    dataM["guide1_index"] = dataM["guide1"].map(lambda x: indices["guide_index"][x])
    dataM["guide2_index"] = dataM["guide2"].map(lambda x: indices["guide_index"][x])

    dataM["guide1_index_o"] = dataM["guide1_o"].map(
        lambda x: indices["guide_index_ori"][x]
    )
    dataM["guide2_index_o"] = dataM["guide2_o"].map(
        lambda x: indices["guide_index_ori"][x]
    )

    dataM["guide_pair_unq_index"] = (dataM["GuidePair"] + "_" + dataM["cell_line"]).map(
        lambda x: indices["guide_pair_unq_index"][x]
    )

    dataM["guide1_unq_index"] = (dataM["guide1_o"] + "_" + dataM["cell_line"]).map(
        lambda x: indices["guide_unq_index_ori"][x]
    )
    dataM["guide2_unq_index"] = (dataM["guide2_o"] + "_" + dataM["cell_line"]).map(
        lambda x: indices["guide_unq_index_ori"][x]
    )

    dataM["guide1_index_s"] = dataM["guide1_o"].map(
        lambda x: indices["guide_index_ori_s"][x]
    )
    dataM["guide2_index_s"] = dataM["guide2_o"].map(
        lambda x: indices["guide_index_ori_s"][x]
    )

    dataM["gene1_unq_index"] = (dataM["gene1"] + "_" + dataM["cell_line"]).map(
        lambda x: indices["gene_unq_index"][x]
    )
    dataM["gene2_unq_index"] = (dataM["gene2"] + "_" + dataM["cell_line"]).map(
        lambda x: indices["gene_unq_index"][x]
    )

    dataM["gene_unq_pair_index"] = (
        dataM["GenePairUnoriented"] + "_" + dataM["cell_line"]
    ).map(lambda x: indices["gene_unq_pair_index"][x])

    dataM["gene1_unq_ori_index"] = (
        dataM["gene1"] + "_" + dataM["cell_line"] + "_1"
    ).map(lambda x: indices["gene_unq_ori_index"][x])
    dataM["gene2_unq_ori_index"] = (
        dataM["gene2"] + "_" + dataM["cell_line"] + "_2"
    ).map(lambda x: indices["gene_unq_ori_index"][x])

    dataM["gene_unq_pair_ori_index"] = (
        dataM["GenePairUnoriented"]
        + "_"
        + dataM["cell_line"]
        + "_"
        + dataM["gene1"]
        + "_"
        + dataM["gene2"]
    ).map(lambda x: indices["gene_unq_pair_ori_index"][x])

    return dataM


def calculateIndices(dataM, singlesM):
    for c in [
        "gene1", "gene2", "guide1", "guide2", "cell_line",
        "GenePairUnoriented", "GuidePair", "replicate",
    ]:
        dataM[c] = dataM[c].astype("string")

    for c in ["SingletonGuide", "SingletonGuide_o", "SingletonGuide_o_s", "cell_line"]:
        singlesM[c] = singlesM[c].astype("string")

    dataM["gene1_o"] = dataM["gene1"] + "_1"
    dataM["gene2_o"] = dataM["gene2"] + "_2"
    dataM["guide1_o"] = dataM["guide1"] + "_1"
    dataM["guide2_o"] = dataM["guide2"] + "_2"

    gene_index = make_dense_index(dataM["gene1"], dataM["gene2"])
    guide_index = make_dense_index(dataM["guide1"], dataM["guide2"])

    gene_pair_index = make_dense_index(dataM["GenePairUnoriented"])
    guide_pair_index = make_dense_index(dataM["GuidePair"])

    cell_line_index = make_dense_index(dataM["cell_line"])

    gene_index_ori = make_dense_index(dataM["gene1_o"], dataM["gene2_o"])
    guide_index_ori = make_dense_index(
        dataM["guide1_o"], dataM["guide2_o"], singlesM["SingletonGuide_o"]
    )
    guide_index_ori_s = make_dense_index(
        dataM["guide1_o"], dataM["guide2_o"], singlesM["SingletonGuide_o_s"]
    )

    gene_unq_index = make_dense_index(
        dataM["gene1"].astype("string") + "_" + dataM["cell_line"].astype("string"),
        dataM["gene2"].astype("string") + "_" + dataM["cell_line"].astype("string"),
    )
    gene_unq_ori_index = make_dense_index(
        dataM["gene1"].astype("string") + "_" + dataM["cell_line"].astype("string") + "_1",
        dataM["gene2"].astype("string") + "_" + dataM["cell_line"].astype("string") + "_2",
    )
    gene_unq_pair_index = make_dense_index(
        dataM["GenePairUnoriented"].astype("string") + "_" + dataM["cell_line"].astype("string")
    )
    gene_unq_pair_ori_index = make_dense_index(
        dataM["GenePairUnoriented"].astype("string") + "_" + dataM["cell_line"].astype("string")
        + "_" + dataM["gene1"].astype("string") + "_" + dataM["gene2"].astype("string")
    )

    guide_unq_index_ori = make_dense_index(
        dataM["guide1_o"].astype("string") + "_" + dataM["cell_line"].astype("string"),
        dataM["guide2_o"].astype("string") + "_" + dataM["cell_line"].astype("string"),
        singlesM["SingletonGuide_o"].astype("string") + "_" + singlesM["cell_line"].astype("string"),
    )

    guide_pair_s_index = make_dense_index(singlesM["GuidePair"])
    guide_pair_unq_s_index = make_dense_index(
        singlesM["GuidePair"].astype("string") + "_" + singlesM["cell_line"].astype("string")
    )
    guide_pair_unq_index = make_dense_index(
        dataM["GuidePair"].astype("string") + "_" + dataM["cell_line"].astype("string")
    )

    print("cell_line_index", len(cell_line_index))
    print("gene_index", len(gene_index))
    print("gene_pair_index", len(gene_pair_index))
    print("guide_index", len(guide_index))
    print("guide_pair_index", len(guide_pair_index))
    print("gene_unq_index", len(gene_unq_index))
    print("gene_unq_pair_index", len(gene_unq_pair_index))

    for c in cell_line_index.keys():
        print(
            c,
            np.unique(dataM[dataM["cell_line"] == c]["replicate"], return_counts=True),
        )

    return {
        "cell_line_index": cell_line_index,
        "gene_index": gene_index,
        "gene_pair_index": gene_pair_index,
        "guide_index": guide_index,
        "guide_pair_index": guide_pair_index,
        "guide_pair_unq_index": guide_pair_unq_index,
        "gene_unq_index": gene_unq_index,
        "gene_unq_ori_index": gene_unq_ori_index,
        "gene_unq_pair_index": gene_unq_pair_index,
        "gene_unq_pair_ori_index": gene_unq_pair_ori_index,
        "guide_index_ori": guide_index_ori,
        "guide_unq_index_ori": guide_unq_index_ori,
        "guide_index_ori_s": guide_index_ori_s,
        "guide_pair_s_index": guide_pair_s_index,
        "guide_pair_unq_s_index": guide_pair_unq_s_index,
    }


def normCounts(data):
    rep_depth = (
        data.groupby(["cell_line", "replicate"], observed=True)["value"]
        .sum()
        .reset_index(name="rep_total")
    )
    cell_median = (
        rep_depth.groupby("cell_line", observed=True)["rep_total"]
        .median()
        .reset_index(name="cell_med")
    )
    rep_depth = rep_depth.merge(cell_median, on="cell_line", how="left")
    rep_depth["rep_scale"] = rep_depth["cell_med"] / rep_depth["rep_total"].replace(0, np.nan)

    out = data.merge(
        rep_depth[["cell_line", "replicate", "rep_scale"]],
        on=["cell_line", "replicate"],
        how="left",
    )
    s = out["rep_scale"].fillna(1.0)
    out["value_rep"] = (out["value"] * s).astype(float)
    out["initial_rep"] = (out["initial"] * s).astype(float)

    cell_totals = (
        out.groupby("cell_line", observed=True)["value_rep"]
        .sum()
        .reset_index(name="cell_total")
    )
    global_med = cell_totals["cell_total"].median()
    cell_totals["cell_scale"] = global_med / cell_totals["cell_total"].replace(0, np.nan)

    out = out.merge(cell_totals[["cell_line", "cell_scale"]], on="cell_line", how="left")
    cs = out["cell_scale"].fillna(1.0)
    out["value_norm"] = (out["value_rep"] * cs).astype(float)
    out["initial_norm"] = (out["initial_rep"] * cs).astype(float)

    return out


def removeSoloSingles(singlesM, dataM):
    dataM["guide1"] = dataM["guide1"].astype("string")
    dataM["guide2"] = dataM["guide2"].astype("string")
    dataM["gene1"] = dataM["gene1"].astype("string")
    dataM["gene2"] = dataM["gene2"].astype("string")

    singlesM["SingletonGuide"] = singlesM["SingletonGuide"].astype("string")
    singlesM["SingletonGene"] = singlesM["SingletonGene"].astype("string")

    guidesInData = pd.Index(dataM["guide1"]).union(pd.Index(dataM["guide2"]))
    genesInData = pd.Index(dataM["gene1"]).union(pd.Index(dataM["gene2"]))

    mask = singlesM["SingletonGuide"].isin(guidesInData) & singlesM["SingletonGene"].isin(genesInData)
    singlesM = singlesM[mask]
    return singlesM


def removeUnusedSingles(singlesM, dataM):
    dataM["guide1"] = dataM["guide1"].astype("string")
    dataM["guide2"] = dataM["guide2"].astype("string")
    singlesM["SingletonGuide_o"] = singlesM["SingletonGuide_o"].astype("string")

    g1 = pd.Index(dataM["guide1"] + "_1")
    g2 = pd.Index(dataM["guide2"] + "_2")
    combsGuideOrientations = g1.union(g2)

    singlesM = singlesM[singlesM["SingletonGuide_o"].isin(combsGuideOrientations)]
    return singlesM


def split_datasets(data):
    isnaA = data["gene1"].isna()
    isnaB = data["gene2"].isna()
    mask_controls = isnaA & isnaB
    mask_singles = (~mask_controls) & (isnaA ^ isnaB)
    mask_combs = (~mask_controls) & (~mask_singles)

    controlsM = data.loc[mask_controls].copy()
    singlesM = data.loc[mask_singles].copy()
    dataM = data.loc[mask_combs].copy()

    g = singlesM["singles_target_gene"].astype("string")
    g1 = singlesM["gene1"].astype("string")
    g2 = singlesM["gene2"].astype("string")

    eq1 = g1.fillna("") == g.fillna("")
    eq2 = g2.fillna("") == g.fillna("")
    only1 = g1.notna() & g2.isna()
    only2 = g2.notna() & g1.isna()

    singlesM["SingletonPosition"] = np.where(
        eq1, "1", np.where(eq2, "2", np.where(only1, "1", "2"))
    )

    singlesM["SingletonGene"] = np.where(
        singlesM["SingletonPosition"] == "1", singlesM["gene1"], singlesM["gene2"]
    )
    singlesM["SingletonGuide"] = np.where(
        singlesM["SingletonPosition"] == "1", singlesM["guide1"], singlesM["guide2"]
    )
    singlesM["SingletonGuide_o"] = singlesM["SingletonGuide"].astype("string") + "_" + singlesM["SingletonPosition"]
    singlesM["SingletonGuide_o_s"] = singlesM["SingletonGuide_o"].astype("string") + "_s"

    singlesM["SingletonGuide"] = singlesM["SingletonGuide"].astype("string")
    dataM["guide1"] = dataM["guide1"].astype("string")
    dataM["guide2"] = dataM["guide2"].astype("string")

    singlesM = removeSoloSingles(singlesM, dataM)
    singlesM = removeUnusedSingles(singlesM, dataM)

    s_avg = singlesM.groupby(["variable", "SingletonGuide"])["lfc"].mean()
    s_avg.index = s_avg.index.set_names(["variable", "guide"])

    idx1 = pd.MultiIndex.from_frame(
        dataM[["variable", "guide1"]].rename(columns={"guide1": "guide"})
    )
    idx2 = pd.MultiIndex.from_frame(
        dataM[["variable", "guide2"]].rename(columns={"guide2": "guide"})
    )

    dataM["lfc_1"] = s_avg.reindex(idx1).to_numpy().astype("float32")
    dataM["lfc_2"] = s_avg.reindex(idx2).to_numpy().astype("float32")

    dataM["dLFC"] = dataM["lfc"] - (dataM["lfc_1"].fillna(0.0) + dataM["lfc_2"].fillna(0.0))

    return dataM, singlesM, controlsM


def assert_combination_index_contiguity(indices, dataM):
    def _assert_map(name):
        vals = np.array(list(indices[name].values()), dtype=int)
        if len(vals) == 0:
            return
        n = vals.max() + 1
        assert n == len(vals) and set(vals) == set(range(n)), f"{name} must be contiguous 0..{len(vals)-1}"

    for name in ("gene_index", "guide_index", "gene_pair_index", "guide_pair_index"):
        _assert_map(name)

    checks = [
        ("gene1_index", "gene_index"),
        ("gene2_index", "gene_index"),
        ("guide1_index", "guide_index"),
        ("guide2_index", "guide_index"),
        ("gene_pair_index", "gene_pair_index"),
        ("guide_pair_index", "guide_pair_index"),
    ]
    for col, m in checks:
        arr = dataM[col].to_numpy()
        if arr.size == 0:
            continue
        assert np.issubdtype(arr.dtype, np.integer), f"{col} must be integer dtype"
        maxv = int(arr.max(initial=-1))
        minv = int(arr.min(initial=0))
        assert 0 <= minv, f"{col} has negative index"
        assert maxv < len(indices[m]), f"{col} has value {maxv} >= len({m})={len(indices[m])}"


def _pair_unoriented(gset, a, b):
    a = str(a)
    b = str(b)
    p = a + "~" + b
    rp = b + "~" + a
    if p in gset:
        return p
    if rp in gset:
        return rp
    gset.add(p)
    return p


def loadData_generic(
    input_file: str,
    initial_col: str,
    final_col: str,
    gene1_col: str,
    gene2_col: str,
    guide1_col: str,
    guide2_col: str,
    experiment_col: str,
    replicate_col: str | None,
):
    raw = pd.read_csv(input_file, low_memory=False)

    def _norm_series(s):
        na_tokens = {"", "NA", "Na", "na", "N/A", "None", "nan", "NaN"}
        s = s.astype("string").str.strip()
        return s.where(~s.isin(na_tokens), pd.NA)

    raw["gene1"] = _norm_series(raw[gene1_col]) if gene1_col in raw else pd.Series(pd.NA, index=raw.index, dtype="string")
    raw["gene2"] = _norm_series(raw[gene2_col]) if gene2_col in raw else pd.Series(pd.NA, index=raw.index, dtype="string")

    raw["guide1"] = _norm_series(raw[guide1_col])
    raw["guide2"] = _norm_series(raw[guide2_col])

    raw["cell_line"] = _norm_series(raw[experiment_col])
    if replicate_col and replicate_col in raw:
        raw["replicate"] = _norm_series(raw[replicate_col])
    else:
        raw["replicate"] = "R1"

    raw["GuidePair"] = raw["guide1"].astype("string") + "~" + raw["guide2"].astype("string")

    s_guides = set()
    raw["GuidePairUnoriented"] = [
        _pair_unoriented(s_guides, a, b)
        for a, b in raw[["guide1", "guide2"]].astype("string").itertuples(index=False, name=None)
    ]

    s_genes = set()
    raw["GenePairUnoriented"] = [
        _pair_unoriented(s_genes, a, b)
        for a, b in raw[["gene1", "gene2"]].astype("string").itertuples(index=False, name=None)
    ]

    # counts
    raw["initial"] = pd.to_numeric(raw[initial_col], errors="coerce").fillna(0.0)
    raw["value"] = pd.to_numeric(raw[final_col], errors="coerce").fillna(0.0)
    raw["lfc"] = np.log2((raw["value"] + 1.0) / (raw["initial"] + 1.0))

    # singleton target gene: if exactly one gene present, use that; else NA
    isnaA = raw["gene1"].isna()
    isnaB = raw["gene2"].isna()
    stg = pd.Series(pd.NA, index=raw.index, dtype="string")
    stg = stg.where(stg.notna(), raw["gene1"].where(~isnaA & isnaB))
    stg = stg.where(stg.notna(), raw["gene2"].where(isnaA & ~isnaB))
    raw["singles_target_gene"] = stg.astype("string")

    # match older pipeline shape
    raw["variable"] = raw["cell_line"].astype("string") + " " + raw["replicate"].astype("string")

    id_vars = [
        "guide1", "guide2",
        "GuidePair", "GuidePairUnoriented", "GenePairUnoriented",
        "gene1", "gene2",
        "singles_target_gene",
        "initial", "value", "lfc",
        "cell_line", "replicate", "variable",
    ]
    data = raw[id_vars].copy()
    return data


def processInputs(
    input_file: str,
    initial_counts_column: str,
    final_counts_column: str,
    gene_1_column: str,
    gene_2_column: str,
    guide_1_column: str,
    guide_2_column: str,
    experiment_column: str,
    output_file: str,
    replicate_column: str | None = None,
):
    data = loadData_generic(
        input_file=input_file,
        initial_col=initial_counts_column,
        final_col=final_counts_column,
        gene1_col=gene_1_column,
        gene2_col=gene_2_column,
        guide1_col=guide_1_column,
        guide2_col=guide_2_column,
        experiment_col=experiment_column,
        replicate_col=replicate_column,
    )

    data = normCounts(data)

    dataM, singlesM, controlsM = split_datasets(data)

    dataM["value"] = dataM["value_norm"]
    singlesM["value"] = singlesM["value_norm"]
    controlsM["value"] = controlsM["value_norm"]

    dataM["plasmid_raw"] = dataM["initial"]
    singlesM["plasmid_raw"] = singlesM["initial"]
    controlsM["plasmid_raw"] = controlsM["initial"]

    dataM["plasmid"] = dataM["initial_norm"]
    singlesM["plasmid"] = singlesM["initial_norm"]
    controlsM["plasmid"] = controlsM["initial_norm"]

    indices = calculateIndices(dataM, singlesM)
    dataM = populateIndicesCombs(dataM, indices)
    assert_combination_index_contiguity(indices, dataM)

    singlesM = populateIndicesSingletons(singlesM, indices)
    controlsM = populateIndicesControls(controlsM, indices)

    dataM.to_parquet(f"{output_file}-combs.pq")
    singlesM.to_parquet(f"{output_file}-singles.pq")
    controlsM.to_parquet(f"{output_file}-controls.pq")


def main():
    ap = argparse.ArgumentParser(description="Prepare double knockout CRISPR data for downstream modeling.")
    ap.add_argument("--input_file", required=True, type=str, help="Path to input CSV.")
    ap.add_argument("--initial_counts_column", required=True, type=str, help="Column with initial/plasmid counts.")
    ap.add_argument("--final_counts_column", required=True, type=str, help="Column with final/observed counts.")
    ap.add_argument("--gene_1_column", required=True, type=str, help="Column with gene 1 identifier (can be NA).")
    ap.add_argument("--gene_2_column", required=True, type=str, help="Column with gene 2 identifier (can be NA).")
    ap.add_argument("--guide_1_column", required=True, type=str, help="Column with guide 1 identifier.")
    ap.add_argument("--guide_2_column", required=True, type=str, help="Column with guide 2 identifier.")
    ap.add_argument("--experiment_column", required=True, type=str, help="Column indicating experiment / cell line.")
    ap.add_argument(
        "--replicate_column",
        required=False,
        default=None,
        type=str,
        help="Optional column indicating replicate label. If omitted, all rows are assumed to be one replicate.",
    )
    ap.add_argument("--output_file", required=True, type=str, help="Output file prefix (no extension).")

    args = ap.parse_args()

    processInputs(
        input_file=args.input_file,
        initial_counts_column=args.initial_counts_column,
        final_counts_column=args.final_counts_column,
        gene_1_column=args.gene_1_column,
        gene_2_column=args.gene_2_column,
        guide_1_column=args.guide_1_column,
        guide_2_column=args.guide_2_column,
        experiment_column=args.experiment_column,
        output_file=args.output_file,
        replicate_column=args.replicate_column,
    )


if __name__ == "__main__":
    main()

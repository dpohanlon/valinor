#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import argparse
import json
from plotting import *


def find_1_to_1_mappings(df, base_column):
    one_to_one_columns = []

    for column in df.columns:
        if column != base_column:
            # Create a mapping from base_column to the current column
            mapping = df.groupby(base_column)[column].nunique()

            # Check if the mapping is 1-to-1
            if mapping.max() == 1 and mapping.min() == 1:
                # Check if the reverse mapping is also 1-to-1
                reverse_mapping = df.groupby(column)[base_column].nunique()
                if reverse_mapping.max() == 1:
                    one_to_one_columns.append(column)

    return one_to_one_columns


def rename_gene_columns(df):
    if "gene1" not in df.columns:
        gene1_col = find_1_to_1_mappings(df, "gene1_index")
        if len(gene1_col) != 1:
            raise ValueError(
                "Can't determine the gene1 column. Please modify your data input files."
            )
        else:
            gene1_col = gene1_col[0]
    else:
        gene1_col = "gene1"

    # Indexed singleton files don't need to have a gene2_index column
    if "gene2" not in df.columns and "gene2_index" in df.columns:
        gene2_col = find_1_to_1_mappings(df, "gene2_index")
        if len(gene2_col) > 1:
            raise ValueError(
                "Can't determine the gene2 column. Please modify your data input files."
            )
        else:
            gene2_col = gene2_col[0]
    else:
        gene2_col = "gene2"

    cols_rename = {
        gene1_col: "gene1",
        gene2_col: "gene2",
    }

    return cols_rename


def load_datasets(data_combo, data_single, val_combo, val_single):
    score = pd.read_parquet(val_combo)

    score_s = pd.read_parquet(val_single)

    dataset_name = "combs"
    data = (
        pd.read_hdf(data_combo, dataset_name)
        if "h5" in data_combo
        else pd.read_parquet(data_combo)
    )

    dataset_name = "singles"
    data_s = (
        pd.read_hdf(data_single, dataset_name)
        if "h5" in data_single
        else pd.read_parquet(data_single)
    )

    # sometimes there are no gene1 or gene2 columns but there are named after e.g. the promoter U6/H1 or Cas9 enzyme Pyogenes/Aureus
    # find the columns that correspond to gene1/gene2 and rename them
    # in the singleton file there will be a SingletonGene and SingletonGuide column, so this conversion is not necessary
    cols_rename = rename_gene_columns(data)
    data = data.rename(columns=cols_rename)

    return (data, data_s, score, score_s)


def merge_data_scores(data, score):
    df_combined = pd.concat(
        (score.reset_index(drop=True), data.reset_index(drop=True)), axis=1
    )
    return df_combined


def calc_valinor_score(df_combs, df_singles):
    df_combs["valinor_score"] = (
        df_combs["gene_ko_growth_12_mean"] / df_combs["gene_ko_growth_12_std"]
    )
    df_singles["valinor_score_s"] = (
        df_singles["ko_growth_s_mean"] / df_singles["ko_growth_s_std"]
    )

    df_combs["rank_" + "valinor_score"] = df_combs.groupby("cell_line")[
        "valinor_score"
    ].rank("dense")
    df_singles["rank_" + "valinor_score_s"] = df_singles.groupby("cell_line")[
        "valinor_score_s"
    ].rank("dense")

    return (df_combs, df_singles)


def calc_overdisp(df_combs, df_singles):
    # calculate the overdispersion based on the variance in counts across replicates for a given guide pair in a given cell line
    numeric_columns = df_combs.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = numeric_columns + ["GuidePair"]
    reps = (
        df_combs[numeric_columns]
        .groupby(["GuidePair", "cell_line_index"])
        .agg(mean=("value", np.mean), std=("value", np.std))
        .reset_index()
    )
    reps["var"] = reps["std"] ** 2
    reps["overdispersion"] = reps["var"] / reps["mean"]
    df_combs = df_combs.merge(
        reps[["GuidePair", "cell_line_index", "overdispersion"]],
        on=["GuidePair", "cell_line_index"],
    )

    numeric_columns = df_singles.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = numeric_columns + ["GuidePair", "SingletonGuide"]
    reps = (
        df_singles.groupby(["GuidePair", "SingletonGuide", "cell_line_index"])
        .agg(mean=("value", np.mean), std=("value", np.std))
        .reset_index()
    )
    reps["var"] = reps["std"] ** 2
    reps["overdispersion"] = reps["var"] / reps["mean"]
    df_singles = df_singles.merge(
        reps[["GuidePair", "SingletonGuide", "cell_line_index", "overdispersion"]],
        on=["GuidePair", "SingletonGuide", "cell_line_index"],
    )

    return (df_combs, df_singles)


def average_NE_singletons(df_singles):
    # if a gene has multiple singleton measurements, i.e. was paired with multiple non-essential or non-targeting guides
    # then we just average over the measurements
    df_singles_NEav = (
        df_singles.groupby(["SingletonGuide", "cell_line", "replicate"])
        .agg(
            {
                "lfc": "mean",
                "SingletonGene": "first",
                "ko_growth_s_mean": "mean",
                "ko_growth_s_std": "mean",
                "valinor_score_s": "mean",
                "rank_valinor_score_s": "mean",
                "overdispersion": "mean",
                "init_count_s_mean": "mean",
                "samples_s": "mean",
            }
        )
        .reset_index()
    )

    return df_singles_NEav


def combine_single_combo(df_combs, df_singles_NEav):
    colstorename = {i: i + "_s" for i in df_singles_NEav.columns}

    scoreData_combined = (
        df_combs.merge(
            df_singles_NEav.rename(columns=colstorename),
            left_on=["guide1", "cell_line", "replicate"],
            right_on=["SingletonGuide_s", "cell_line_s", "replicate_s"],
            how="left",
        )
        .drop(
            columns=[
                "SingletonGuide_s",
                "cell_line_s",
                "replicate_s",
                "SingletonGene_s",
            ]
        )
        .rename(columns={i: i + "_1" for i in colstorename.values()})
        .merge(
            df_singles_NEav.rename(columns=colstorename),
            left_on=["guide2", "cell_line", "replicate"],
            right_on=["SingletonGuide_s", "cell_line_s", "replicate_s"],
            how="left",
        )
        .drop(
            columns=[
                "SingletonGuide_s",
                "cell_line_s",
                "replicate_s",
                "SingletonGene_s",
            ]
        )
        .rename(columns={i: i + "_2" for i in colstorename.values()})
    )

    return scoreData_combined


def calc_deltaLFC(scoreData_combined):
    scoreData_combined["deltaLFC"] = scoreData_combined["lfc"] - (
        scoreData_combined["lfc_s_1"] + scoreData_combined["lfc_s_2"]
    )
    return scoreData_combined


def average_replicates(scoreData_combined):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    drop_cols = ["variable", "replicate", "guide1_o", "guide2_o"]
    drop_cols = [col for col in drop_cols if col in scoreData_combined.columns]
    groupby_cols = [
        "GuidePair",
        "gene2",
        "gene1",
        "cell_line",
        "guide1",
        "guide2",
        "genePair",
        "genePairUnoriented",
        "GuidePairUnoriented",
    ]
    scoreData_combined = scoreData_combined.drop(columns=drop_cols)
    numeric_cols = scoreData_combined.select_dtypes(include=numerics).columns

    scoreData_combined = (
        scoreData_combined.groupby(groupby_cols)[list(numeric_cols)]
        .mean()
        .reset_index()
    )
    return scoreData_combined


def create_overview_stats(data, data_s, data_combo, data_single, val_combo, val_single):
    av_replic_percln = "{:.2f}".format(
        data.groupby("cell_line")["replicate"].nunique().mean()
    )
    av_replic_percln_s = "{:.2f}".format(
        data_s.groupby("cell_line")["replicate"].nunique().mean()
    )

    # TO DO: creating date and introduce the settings from valinor
    combs_attr = {
        "n_cell_lines": data["cell_line"].unique().shape[0],
        "cell_lines": ", ".join(data["cell_line"].unique()),
        "n_replicates": av_replic_percln,
        "n_gene_pairs": len(data["genePair"].unique()),
        "n_genes": len(set(data["gene1"]).union(data["gene2"])),
        "n_guide_pairs": len(data["GuidePair"].unique()),
        "n_guides": len(set(data["guide1"]).union(data["guide2"])),
        "file_name": val_combo,
        #         "creation_date": "20.10.2022",  # <- this should be stored as meta data with the valinor output
        "input_path": data_combo,  # <- this points to the data file that was used as valinor input
        #         "normalised_replicates": True,  # <- this is a Valinor setting, check how to set this, needs to be exported as meta
        #         "normalised_total": True,  # <- this is a Valinor setting, check how to set this, needs to be exported as meta
    }

    combs_attr_s = {
        "n_cell_lines": data_s["cell_line"].unique().shape[0],
        "cell_lines": ", ".join(data_s["cell_line"].unique()),
        "n_replicates": av_replic_percln_s,
        "n_genes": len(set(data_s["SingletonGene"])),
        "n_guides": len(set(data_s["SingletonGuide"])),
        "file_name": val_single,
        #         "creation_date": "20.10.2022",  # <- this should be stored as meta data with the valinor output
        "input_path": data_single,  # <- this points to the data file that was used as valinor input
        #         "normalised_replicates": True,  # <- this is a Valinor setting, check how to set this, needs to be exported as meta
        #         "normalised_total": True,  # <- this is a Valinor setting, check how to set this, needs to be exported as meta
    }

    return (combs_attr, combs_attr_s)


def calc_LFC(df_combs, df_singles):
    # Calculate the LFC from the counts that were used as input into Valinor
    # don't rely on already existing 'lfc' columns
    tmp = df_combs["value"] / df_combs["plasmid"]
    tmp[np.isinf(tmp)] = np.nan
    # also set 0 to nan since log2 doesn't exist for 0
    tmp[tmp == 0] = np.nan
    df_combs["lfc"] = np.log2(tmp)

    tmp = df_singles["value"] / df_singles["plasmid"]
    tmp[np.isinf(tmp)] = np.nan
    tmp[tmp == 0] = np.nan
    df_singles["lfc"] = np.log2(tmp)

    return (df_combs, df_singles)


def process_valinor_pred(data_combo, data_single, val_combo, val_single, report_folder):
    data, data_s, score, score_s = load_datasets(
        data_combo, data_single, val_combo, val_single
    )

    combs_attr, combs_attr_s = create_overview_stats(
        data, data_s, data_combo, data_single, val_combo, val_single
    )

    df_combs = merge_data_scores(data, score)
    df_singles = merge_data_scores(data_s, score_s)

    df_combs, df_singles = calc_LFC(df_combs, df_singles)

    df_combs, df_singles = calc_valinor_score(df_combs, df_singles)

    df_combs, df_singles = calc_overdisp(df_combs, df_singles)

    if report_folder is not None:
        folders = [f"{report_folder}/input", f"{report_folder}/plots"]

        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

        with open(f"{report_folder}/input/combs_attr.json", "w") as outfile:
            json.dump(combs_attr, outfile)
        with open(f"{report_folder}/input/combs_attr_s.json", "w") as outfile:
            json.dump(combs_attr_s, outfile)

        produce_modelfit_plot(df_combs, df_singles, f"{report_folder}/plots/")

    df_singles_NEav = average_NE_singletons(df_singles)

    scoreData_combined = combine_single_combo(df_combs, df_singles_NEav)

    scoreData_combined = calc_deltaLFC(scoreData_combined)

    # we don't really need the replicate counts/LFCs -> just average across them
    scoreData_combined = average_replicates(scoreData_combined)

    return scoreData_combined


def main():
    parser = argparse.ArgumentParser(
        description="Script to process model and data files."
    )

    parser.add_argument("--val_combo", help="Path to the val_combo file.")
    parser.add_argument("--val_single", help="Path to the val_single file.")
    parser.add_argument("--data_combo", help="Path to the data_combo file.")
    parser.add_argument("--data_single", help="Path to the data_single file.")
    parser.add_argument(
        "--output_file",
        help="Processed output file. Default is valinoroutput_processed.pq",
        default="../valinoroutput_processed.pq",
    )
    parser.add_argument(
        "--report_folder",
        help="Optional: if given this will produce files necessary for the report in the given location",
    )

    args = parser.parse_args()

    scoreData_combined = process_valinor_pred(
        args.data_combo,
        args.data_single,
        args.val_combo,
        args.val_single,
        args.report_folder,
    )

    scoreData_combined.to_parquet(args.output_file)


if __name__ == "__main__":
    main()

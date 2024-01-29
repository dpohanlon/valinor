#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import argparse
import json
from priors import defaultPriors
from plotting import *


def find_1_to_1_mappings(df, base_column):
    """
    Identifies columns in the DataFrame that have a 1-to-1 mapping with the entries in the specified base column.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        base_column (str): The column name in the DataFrame to be used as a base for finding 1-to-1 mappings.

    Returns:
        List[str]: A list of column names that have a 1-to-1 mapping with the entries in the base column.
    """

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
    """
    Finds columns that contain gene1 and gene2 annotations and returns a dictionary to rename the columns accordingly.

    Args:
        df (pandas.DataFrame): The DataFrame where gene columns need to be renamed.

    Returns:
        Dict[str, str]: A dictionary mapping original column names to their new names.

    Raises:
        ValueError: If unable to determine the correct gene column names from the data.
    """

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
    """
    Loads and processes the datasets for the Valinor model.

    Args:
        data_combo (str): Path to the data file for combinations.
        data_single (str): Path to the data file for singletons.
        val_combo (str): Path to the Valinor output file for combinations.
        val_single (str): Path to the Valinor output file for singletons.

    Returns:
        tuple: A tuple containing four elements:
            - data (pandas.DataFrame): Data file for combinations.
            - data_s (pandas.DataFrame): Data file for singletons.
            - score (pandas.DataFrame): Valinor output file for combinations.
            - score_s (pandas.DataFrame): Valinor output file for singletons.
    """

    score = pd.read_parquet(val_combo)

    score_s = pd.read_parquet(val_single) if val_single else None

    dataset_name = "combs"
    data = (
        pd.read_hdf(data_combo, dataset_name)
        if "h5" in data_combo
        else pd.read_parquet(data_combo)
    )

    dataset_name = "singles"
    if data_single:
        data_s = (
            pd.read_hdf(data_single, dataset_name)
            if "h5" in data_single
            else pd.read_parquet(data_single)
        )
    else:
        data_s = None

    # sometimes there are no gene1 or gene2 columns but there are named after e.g. the promoter U6/H1 or Cas9 enzyme Pyogenes/Aureus
    # find the columns that correspond to gene1/gene2 and rename them
    # in the singleton file there will be a SingletonGene and SingletonGuide column, so this conversion is not necessary
    cols_rename = rename_gene_columns(data)
    data = data.rename(columns=cols_rename)

    return (data, data_s, score, score_s)


def merge_data_scores(data, score):
    """
    Merges two pandas DataFrames, 'data' and 'score', into a single DataFrame.

    This function resets the index of both DataFrames before concatenating them along the columns.

    Args:
        data (pandas.DataFrame): Data file containing counts, gene and guide annotations.
        score (pandas.DataFrame): Valinor output containing the synthetic lethality scores, guide efficiencies etc.

    Returns:
        pandas.DataFrame: A DataFrame resulting from the concatenation of 'data' and 'score'.
    """
    df_combined = pd.concat(
        (score.reset_index(drop=True), data.reset_index(drop=True)), axis=1
    )
    return df_combined


def calc_valinor_score(df, type="combo"):
    """
    Calculates the Valinor scores for combination and singletons based on the specified type ('combo' or 'singles').

    Args:
        df (pandas.DataFrame): The DataFrame containing the merged Valinor output an input data.
        type (str, optional): The type of data to process. Can be either 'combo' for combination data or 'singles'
                              for singleton data. Defaults to 'combo'.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns for Valinor scores and their ranks.

    Raises:
        ValueError: If the specified type is neither 'combo' nor 'singles'.
    """
    if type == "combo":
        df["valinor_score"] = df["gene_ko_growth_12_mean"] / df["gene_ko_growth_12_std"]
        df["rank_" + "valinor_score"] = df.groupby("cell_line")["valinor_score"].rank(
            "dense"
        )

        # Singleton valinor score can be determined from the combs file
        df["valinor_score_s_1"] = (
            df["gene_ko_growth_1_mean"] / df["gene_ko_growth_1_std"]
        )
        df["rank_" + "valinor_score_s_1"] = df.groupby("cell_line")[
            "valinor_score_s_1"
        ].rank("dense")

        df["valinor_score_s_2"] = (
            df["gene_ko_growth_2_mean"] / df["gene_ko_growth_2_std"]
        )
        df["rank_" + "valinor_score_s_2"] = df.groupby("cell_line")[
            "valinor_score_s_2"
        ].rank("dense")
    elif type == "singles":
        df["valinor_score_s"] = df["ko_growth_s_mean"] / df["ko_growth_s_std"]
        df["rank_" + "valinor_score_s"] = df.groupby("cell_line")[
            "valinor_score_s"
        ].rank("dense")
    else:
        raise ValueError(f"{type} is not known. Choose either combo or singles.")

    return df


def calc_overdisp(df, type="combo"):
    """
    Calculates overdispersion for each guide pair in a given cell line based on the variance in counts across replicates.

    Args:
        df (pandas.DataFrame): The DataFrame containing guide pair and cell line data.
        type (str, optional): The type of data to process. Can be either 'combo' or 'singles'. Defaults to 'combo'.

    Returns:
        pandas.DataFrame: The DataFrame with an added 'overdispersion' column.

    Raises:
        ValueError: If the specified type is neither 'combo' nor 'singles'.
    """

    # calculate the overdispersion based on the variance in counts across replicates for a given guide pair in a given cell line
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if type == "combo":
        numeric_columns = numeric_columns + ["GuidePair"]
        reps = (
            df[numeric_columns]
            .groupby(["GuidePair", "cell_line_index"])
            .agg(mean=("value", np.mean), std=("value", np.std))
            .reset_index()
        )
        reps["var"] = reps["std"] ** 2
        reps["overdispersion"] = reps["var"] / reps["mean"]
        df = df.merge(
            reps[["GuidePair", "cell_line_index", "overdispersion"]],
            on=["GuidePair", "cell_line_index"],
        )
    elif type == "singles":
        numeric_columns = numeric_columns + ["GuidePair", "SingletonGuide"]
        reps = (
            df.groupby(["GuidePair", "SingletonGuide", "cell_line_index"])
            .agg(mean=("value", np.mean), std=("value", np.std))
            .reset_index()
        )
        reps["var"] = reps["std"] ** 2
        reps["overdispersion"] = reps["var"] / reps["mean"]
        df = df.merge(
            reps[["GuidePair", "SingletonGuide", "cell_line_index", "overdispersion"]],
            on=["GuidePair", "SingletonGuide", "cell_line_index"],
        )
    else:
        raise ValueError(f"{type} is not known. Choose either combo or singles.")

    return df


def average_NE_singletons(df_singles):
    """
    Averages multiple singleton measurements for each guide across its non-essential gene partners in a DataFrame.

    Args:
        df_singles (pandas.DataFrame): The DataFrame containing singleton measurements.

    Returns:
        pandas.DataFrame: A DataFrame with averaged values for each gene across different metrics.
    """

    # if a gene has multiple singleton measurements, i.e. was paired with multiple non-essential or non-targeting guides
    # then we just average over the measurements
    df_singles_NEav = (
        df_singles.groupby(["SingletonGuide", "cell_line", "replicate"])
        .agg(
            {
                "lfc": "mean",
                "SingletonGene": "first",
                "guide_eff_s_mean": "mean",
                "guide_eff_s_std": "mean",
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
    """
    Combines singleton and combination data into a single DataFrame.

    This function merges the singleton data (averaged over non-essential guides) and combination data into one DataFrame. It handles
    renaming and aligning columns appropriately to ensure a consistent and unified structure in the resulting DataFrame.

    Args:
        df_combs (pandas.DataFrame): The DataFrame containing combination data.
        df_singles_NEav (pandas.DataFrame): The DataFrame containing averaged singleton data.

    Returns:
        pandas.DataFrame: The combined DataFrame with both singleton and combination data.
    """

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
    """
    Calculates the delta Log Fold Change (deltaLFC) for the combined score data.

    This function adds a new column 'deltaLFC' to the DataFrame, which is calculated as the difference
    between the combination LFC and the sum of singleton lfc values.

    Args:
        scoreData_combined (pandas.DataFrame): DataFrame that contains combined singleton and combination data and Valinor outputs.

    Returns:
        pandas.DataFrame: The updated DataFrame including the 'deltaLFC' column.
    """

    scoreData_combined["deltaLFC"] = scoreData_combined["lfc"] - (
        scoreData_combined["lfc_s_1"] + scoreData_combined["lfc_s_2"]
    )
    return scoreData_combined


def average_replicates(scoreData_combined):
    """
    Averages values (e.g. LFCs) across replicates.

    This function averages the numerical data across replicates for each guide pair and cell line.

    Args:
        scoreData_combined (pandas.DataFrame): DataFrame that contains combined singleton and combination data and Valinor outputs.

    Returns:
        pandas.DataFrame: The DataFrame with averaged values across replicates.
    """

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
    """
    Creates overview statistics for the given datasets.

    This function computes and compiles key statistics for both the combination and singleton datasets. Statistics include
    the number of cell lines, replicates, gene pairs, genes, guide pairs, guides, and file information.

    Args:
        data (pandas.DataFrame): The DataFrame containing combination data.
        data_s (pandas.DataFrame): The DataFrame containing singleton data.
        data_combo (str): File path of the combination data.
        data_single (str): File path of the singleton data.
        val_combo (str): Validation file path for the combination data.
        val_single (str): Validation file path for the singleton data.

    Returns:
        tuple: A tuple containing two dictionaries, one for combination data stats (combs_attr) and one for singleton data stats (combs_attr_s).
    """

    av_replic_percln = "{:.2f}".format(
        data.groupby("cell_line")["replicate"].nunique().mean()
    )
    combs_attr = {
        "n_cell_lines": data["cell_line"].unique().shape[0],
        "cell_lines": ", ".join(data["cell_line"].unique()),
        "n_replicates": av_replic_percln,
        "n_gene_pairs": len(data["genePair"].unique()),
        "n_genes": len(set(data["gene1"]).union(data["gene2"])),
        "n_guide_pairs": len(data["GuidePair"].unique()),
        "n_guides": len(set(data["guide1"]).union(data["guide2"])),
        "file_name": val_combo,
        "input_path": data_combo,
    }

    if data_s is None:
        combs_attr_s = {
            "n_cell_lines": "NA",
            "cell_lines": "NA",
            "n_replicates": "NA",
            "n_genes": "NA",
            "n_guides": "NA",
            "file_name": "NA",
            "input_path": "NA",
        }
    else:
        av_replic_percln_s = "{:.2f}".format(
            data_s.groupby("cell_line")["replicate"].nunique().mean()
        )

        combs_attr_s = {
            "n_cell_lines": data_s["cell_line"].unique().shape[0],
            "cell_lines": ", ".join(data_s["cell_line"].unique()),
            "n_replicates": av_replic_percln_s,
            "n_genes": len(set(data_s["SingletonGene"])),
            "n_guides": len(set(data_s["SingletonGuide"])),
            "file_name": val_single,
            "input_path": data_single,
        }

    return (combs_attr, combs_attr_s)


def calc_LFC(df):
    """
    Calculates the Log Fold Change (LFC) for the given DataFrame.

    This function computes the LFC as the log2 of the ratio of 'value' to 'plasmid' columns in the DataFrame.
    It handles infinite and zero values by setting them to NaN.

    Args:
        df (pandas.DataFrame): The DataFrame containing 'value' and 'plasmid' columns.

    Returns:
        pandas.DataFrame: The DataFrame with an added 'lfc' column representing the Log Fold Change.
    """

    tmp = df["value"] / df["plasmid"]
    tmp[np.isinf(tmp)] = np.nan
    # also set 0 to nan since log2 doesn't exist for 0
    tmp[tmp == 0] = np.nan
    df["lfc"] = np.log2(tmp)
    return df


def calc_guide_eff_deviation(scoreData_combined, priors):
    """
    Calculates deviations from prior parameters for guide efficiency in the combined data.

    This function computes the deviations of guide efficiency mean and standard deviation values from their prior
    parameters. It adds multiple columns to the DataFrame indicating these deviations.

    Args:
        scoreData_combined (pandas.DataFrame): DataFrame that contains combined singleton and combination data and Valinor outputs.
        priors (dict): A dictionary containing prior parameter values for guide efficiency.

    Returns:
        pandas.DataFrame: The updated DataFrame including new columns for deviations from prior parameters.
    """

    # calculate the deviations from priors for hierarchical parameters - partial_pooling scenario
    if "guide_eff_mean_1_mean" in scoreData_combined.columns:
        scoreData_combined["dev_prior_guide_eff_mean_1"] = (
            scoreData_combined["guide_eff_mean_1_mean"] - priors["guide_eff_mean"][0]
        ) / priors["guide_eff_mean"][1]
        scoreData_combined["dev_prior_guide_eff_mean_2"] = (
            scoreData_combined["guide_eff_mean_2_mean"] - priors["guide_eff_mean"][0]
        ) / priors["guide_eff_mean"][1]

        scoreData_combined["dev_prior_guide_eff_std_1"] = (
            scoreData_combined["guide_eff_std_1_mean"] - priors["guide_eff_std"][0]
        ) / priors["guide_eff_std"][1]
        scoreData_combined["dev_prior_guide_eff_std_2"] = (
            scoreData_combined["guide_eff_std_2_mean"] - priors["guide_eff_std"][0]
        ) / priors["guide_eff_std"][1]

        scoreData_combined["dev_prior_guide_eff_1_mean"] = (
            scoreData_combined["guide_eff_1_mean"]
            - scoreData_combined["guide_eff_mean_1_mean"]
        ) / scoreData_combined["guide_eff_std_1_mean"]
        scoreData_combined["dev_prior_guide_eff_2_mean"] = (
            scoreData_combined["guide_eff_2_mean"]
            - scoreData_combined["guide_eff_mean_2_mean"]
        ) / scoreData_combined["guide_eff_std_2_mean"]

    elif "guide_eff_1_mean" in scoreData_combined.columns:
        # full_pooling or no_pooling -> no hierarchy
        scoreData_combined["dev_prior_guide_eff_1_mean"] = (
            scoreData_combined["guide_eff_1_mean"] - priors["guide_eff_mean"][0]
        ) / priors["guide_eff_mean"][1]
        scoreData_combined["dev_prior_guide_eff_2_mean"] = (
            scoreData_combined["guide_eff_2_mean"] - priors["guide_eff_mean"][0]
        ) / priors["guide_eff_mean"][1]

    return scoreData_combined


def process_valinor_pred(
    data_combo, data_single, val_combo, val_single, report_folder, prior_params
):
    """
    Processes Valinor predictions by combining, scoring, and analyzing data from multiple sources.

    This function orchestrates the entire process of loading datasets, merging them, calculating scores and
    overdispersion, and optionally generating reports. It handles both combination and singleton datasets and merges them
    into a comprehensive dataset for analysis.

    Args:
        data_combo (str): Path to the combination data file.
        data_single (str): Path to the singleton data file, if available.
        val_combo (str): Path to the validation file for combination data.
        val_single (str): Path to the validation file for singleton data, if available.
        report_folder (str): Directory path to save the reports.
        prior_params (dict): Dictionary of prior parameters for calculations.

    Returns:
        pandas.DataFrame: DataFrame that contains combined singleton and combination data and Valinor outputs.
    """

    data, data_s, score, score_s = load_datasets(
        data_combo, data_single, val_combo, val_single
    )

    combs_attr, combs_attr_s = create_overview_stats(
        data, data_s, data_combo, data_single, val_combo, val_single
    )

    df_combs = merge_data_scores(data, score)
    df_combs = calc_LFC(df_combs)
    df_combs = calc_valinor_score(df_combs, type="combo")
    df_combs = calc_overdisp(df_combs, type="combo")

    if data_s is None:
        df_singles = pd.DataFrame()
        scoreData_combined = df_combs.copy()

        # use the Singleton Valinor Scores obtained from the combination measurements
        scoreData_combined = scoreData_combined.rename(
            columns={
                "valinor_score_s_1": "valinor_score_s_s_1",
                "valinor_score_s_2": "valinor_score_s_s_2",
            }
        )
    else:
        df_singles = merge_data_scores(data_s, score_s)
        df_singles = calc_LFC(df_singles)
        df_singles = calc_valinor_score(df_singles, type="singles")
        df_singles = calc_overdisp(df_singles, type="singles")
        df_singles_NEav = average_NE_singletons(df_singles)
        scoreData_combined = combine_single_combo(df_combs, df_singles_NEav)
        scoreData_combined = calc_deltaLFC(scoreData_combined)

    if report_folder is not None:
        folders = [f"{report_folder}/input", f"{report_folder}/plots"]

        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

        with open(f"{report_folder}/input/combs_attr.json", "w") as outfile:
            json.dump(combs_attr, outfile)
        with open(f"{report_folder}/input/combs_attr_s.json", "w") as outfile:
            json.dump(combs_attr_s, outfile)

        produce_modelfit_plot(df_combs, f"{report_folder}/plots/", type="combo")
        if data_s is not None:
            produce_modelfit_plot(df_singles, f"{report_folder}/plots/", type="singles")

        # produce_modelfit_plot(df_combs, df_singles, f"{report_folder}/plots/")

    # we don't really need the replicate counts/LFCs -> just average across them
    scoreData_combined = average_replicates(scoreData_combined)

    # calculate the deviation of guide efficiencies from the prior
    scoreData_combined = calc_guide_eff_deviation(scoreData_combined, prior_params)

    return scoreData_combined


def run_processvalinor(
    valinorPriorFile,
    data_combo,
    data_single,
    val_combo,
    val_single,
    report_folder,
    output_file,
):
    if valinorPriorFile:
        with open(valinorPriorFile, "r") as file:
            prior_params = json.load(file)
    else:
        prior_params = defaultPriors()

    scoreData_combined = process_valinor_pred(
        data_combo,
        data_single,
        val_combo,
        val_single,
        report_folder,
        prior_params,
    )

    scoreData_combined.to_parquet(output_file)


def main():
    """
    Main function to process model and data files for the Valinor model.

    This function parses command line arguments for data file paths and saves the processed data.
    """

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
        "--valinorPriorFile",
        help="Optional: JSON that contains the priors used to run valinor",
    )

    parser.add_argument(
        "--report_folder",
        help="Optional: if given this will produce files necessary for the report in the given location",
    )

    args = parser.parse_args()

    run_processvalinor(
        args.valinorPriorFile,
        args.data_combo,
        args.data_single,
        args.val_combo,
        args.val_single,
        args.report_folder,
        args.output_file,
    )


if __name__ == "__main__":
    main()

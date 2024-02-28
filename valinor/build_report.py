#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import jinja2
from datetime import date
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil
import os
import re
import argparse
import json
import importlib.util
import importlib.resources as pkg_resources
from valinor.processvalinoroutput import *

alt.data_transformers.disable_max_rows()

fontsizes = [18, 16, 14]

naming_cols = {
    "GuidePair": "Guide pair",
    "gene1": "Singleton gene 1",
    "gene2": "Singleton gene 2",
    "guide1": "Singleton guide 1",
    "guide2": "Singleton guide 2",
    "cell_line": "Cell line",
    "deltaLFC": "dLFC",
    "dev_prior_guide_eff_1_mean": "Deviation from hyper distribution of guide 1",
    "dev_prior_guide_eff_2_mean": "Deviation from hyper distribution of guide 2",
    "gene1": "Gene 1",
    "gene2": "Gene 2",
    "genePair": "Gene pair",
    "guide1": "Guide 1",
    "guide2": "Guide 2",
    "guide_eff_1_mean": "Efficiency guide 1",
    "guide_eff_2_mean": "Efficiency guide 2",
    "guide_eff_s_mean_s_1": "Efficiency guide 1",
    "guide_eff_s_mean_s_2": "Efficiency guide 2",
    "guide_eff_mean_1_mean": "Mean of hyper distribution",
    "guide_eff_std_1_mean": "Standard deviation of hyper distribution",
    "gene_ko_growth_1_mean": "Estimated gene 1 effect",
    "gene_ko_growth_12_mean": "Estimated combination effect",
    "gene_ko_growth_12_std": "Uncertainty",
    "gene_ko_growth_1_std": "Uncertainty",
    "gene_ko_growth_2_mean": "Estimated gene 2 effect",
    "gene_ko_growth_2_std": "Uncertainty",
    "ko_growth_s_mean_s_1": "Estimated singleton effect",
    "ko_growth_s_mean_s_2": "Estimated singleton effect",
    "ko_growth_s_std_s_1": "Uncertainty",
    "ko_growth_s_std_s_2": "Uncertainty",
    "lfc": "Combination LFC",
    "lfc_s_1": "Singleton LFC - gene 1",
    "lfc_s_2": "Singleton LFC - gene 2",
    "mv_mean": "Estimated overdispersion",
    "mv_std": "Uncertainty",
    "overdispersion": "Overdispersion",
    "overdispersion_s_1": "Overdispersion",
    "overdispersion_s_2": "Overdispersion",
    "plasmid": "Plasmid counts",
    "rank_valinor_score": "Rank of Valinor Score",
    "rank_valinor_score_s_s_1": "Rank of Valinor Singleton Score",
    "rank_valinor_score_s_s_2": "Rank of Valinor Singleton Score",
    "valinor_score": "Valinor Score",
    "valinor_score_s_s_1": "Valinor Singleton Score - gene 1",
    "valinor_score_s_s_2": "Valinor Singleton Score - gene 2",
    "value": "End of experiment counts",
}


def plot_hist(
    data_dict, xlabel, figsize=(6, 6), nbins=50, figure=None, loc="upper right"
):
    """
    Plot histograms for several different data series.

    Args:
        data_dict (Dict[str, np.ndarray]): A dictionary where keys are labels of the data and values are numpy arrays containing the data points for each entry.
        xlabel (str): The label for the x-axis.
        figsize (Tuple[int, int], optional): The size of the figure in inches. Defaults to (6, 6).
        nbins (int, optional): The number of bins for the histogram. Defaults to 50.
        figure (Tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot], optional):
               A tuple containing a figure and axes object to plot on. If None, a new figure and axes are created.
               Defaults to None.
        loc (str, optional): The location of the legend. Defaults to "upper right".

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot]: A tuple containing the figure and axes
                                                                                objects used for the plot. This can be
                                                                                used for further customization outside
                                                                                the function.
    """
    if figure is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = figure[0]
        ax = figure[1]
    data_min = min([i.min() for i in data_dict.values()])
    data_max = max([np.quantile(i, 0.999) for i in data_dict.values()])
    bins = np.linspace(data_min, data_max, nbins)
    for label, data in data_dict.items():
        ax.hist(data, histtype="step", bins=bins, density=True, label=label)
    ax.set_xlabel(xlabel, fontsize=fontsizes[1])
    ax.set_ylabel("Density", fontsize=fontsizes[1])
    ax.tick_params(axis="both", labelsize=fontsizes[2])
    ax.legend(loc=loc)
    fig.tight_layout()

    return (fig, ax)


def sample_genepairs(scoreData_combined):
    """
    Selects a sample of gene pairs from the combined score data.

    This function first selects the top 100 synthetic lethality (SL) pairs based on the Valinor score and then
    randomly samples an additional 300 gene pairs. The resulting set combines these two selections.
    The final selection might be less than 400 gene pairs because one gene pair might be present several times in the Top 100 SL ordering.

    Args:
        scoreData_combined (pandas.DataFrame): DataFrame that contains combined singleton and combination data and Valinor outputs.

    Returns:
        tuple: A tuple containing a set of selected gene pairs (combining top SL pairs and random samples) and a list containing the number of top SL pairs and randomly selected pairs.
    """

    # pick the top SL pairs based on rank
    # the number of top SL depends on the number of cell lines to not lead to huge data needing to be loaded into the report page
    n_clns = scoreData_combined["cell_line"].unique().shape[0]
    ntop = int(400 / n_clns)
    nrandom = int(1200 / n_clns)
    topsl = (
        scoreData_combined[["genePair", "valinor_score"]]
        .drop_duplicates()
        .sort_values("valinor_score")
        .genePair[:ntop]
        .unique()
    )
    # then randomly select pairs
    selectpairs = scoreData_combined.genePair.sample(nrandom, random_state=42).unique()
    selectpairs = set(np.concatenate((selectpairs, topsl)))
    nrandom = len(selectpairs.difference(topsl))
    return (selectpairs, [ntop, nrandom])


def create_valinorrun_settings(valinorsettings):
    """
    Creates a string of Valinor run settings, excluding file paths.

    This function formats the settings for a Valinor run, excluding any settings that contain 'File' in their keys.
    The resulting string is HTML-formatted with each setting on a new line.

    Args:
        valinorsettings (dict): Dictionary of Valinor settings.

    Returns:
        str: A HTML-formatted string of Valinor run settings.
    """

    valinorrun_settings = [
        f"{i}: {j}" for i, j in valinorsettings.items() if "File" not in i
    ]
    valinorrun_settings = "<br>".join(valinorrun_settings)

    return valinorrun_settings


def create_overview_stats(combs_attr, combs_attr_s, topslgene_str):
    """
    Creates HTML-formatted overview statistics for combination and singleton datasets.

    This function generates HTML-formatted strings summarizing key statistics of both combination and singleton datasets.
    It includes information such as the number of cell lines, replicates, genes, gene pairs, and guides.

    Args:
        combs_attr (dict): Dictionary containing statistics for the combination dataset.
        combs_attr_s (dict): Dictionary containing statistics for the singleton dataset.
        topslgene_str (str): A string informing about the number of selected gene pairs if the --subsetSLpairs was set.

    Returns:
        tuple: A tuple of three strings, each containing HTML-formatted statistics for different aspects of the datasets.
    """

    overview_stats = [
        "{} cell lines: {}".format(
            combs_attr["n_cell_lines"], combs_attr["cell_lines"]
        ),
        "{} replicates per cell line (on average)".format(combs_attr["n_replicates"]),
    ]
    overview_stats = "<br>".join(overview_stats)

    overview_stats_combs = [
        "{} gene pairs (orientation aware) out of {} genes".format(
            combs_attr["n_gene_pairs"], combs_attr["n_genes"]
        ),
        "{} guide pairs (orientation aware) out of {} guides".format(
            combs_attr["n_guide_pairs"], combs_attr["n_guides"]
        ),
        topslgene_str,
    ]
    overview_stats_combs = "<br>".join(overview_stats_combs)

    overview_stats_s = [
        "{} genes".format(combs_attr_s["n_genes"]),
        "{} guides".format(combs_attr_s["n_guides"]),
    ]

    overview_stats_s = "<br>".join(overview_stats_s)

    return (overview_stats, overview_stats_combs, overview_stats_s)


def create_file_settings(mergedfile, combs_attr, combs_attr_s, valinorsettings):
    """
    Generates HTML-formatted settings for input and output files used in the Valinor analysis.

    This function creates two lists: one for input file settings, including paths from the Valinor settings, and
    another for output files, detailing files related to combinations, singletons, and the merged dataset.

    Args:
        mergedfile (str): File path of the combined singleton and combination data and Valinor outputs.
        combs_attr (dict): Dictionary containing attributes for the combinations dataset.
        combs_attr_s (dict): Dictionary containing attributes for the singletons dataset.
        valinorsettings (dict): Dictionary of Valinor settings including file paths.

    Returns:
        tuple: A tuple containing two strings, one for input file settings and another for output file settings, both HTML-formatted.
    """

    input_files = [
        f"{i}: {j}"
        for i, j in valinorsettings.items()
        if "File" in i
        if i != "paramsFileName"
    ]
    input_files = "<br>".join(input_files)

    output_files = [
        "Combinations: {}".format(combs_attr["file_name"]),
        "Singletons: {}".format(combs_attr_s["file_name"]),
        "Merged, replicates averaged: {}".format(mergedfile),
    ]
    output_files = "<br>".join(output_files)

    return (input_files, output_files)


def create_startpage_stats(
    mergedfile, combs_attr_f, combs_attr_s_f, valinorrun_json, topslgene_str
):
    """
    Generates a dictionary of HTML-formatted statistics for the start page of the Valinor analysis.

    This function reads the Valinor settings and attributes from provided files, then uses these to create html code that details
    overview statistics, file settings, and Valinor run settings. It compiles these into a dictionary suitable
    for display on a start page.

    Args:
        mergedfile (str): File path of the combined singleton and combination data and Valinor outputs.
        combs_attr_f (str): File path to the JSON file with combination dataset attributes.
        combs_attr_s_f (str): File path to the JSON file with singleton dataset attributes.
        valinorrun_json (str): File path to the JSON file with Valinor run settings.
        topslgene_str (str): A string informing about the number of selected gene pairs if the --subsetSLpairs was set.
    Returns:
        tuple: A dictionary containing various HTML-formatted statistics for the start page and a dict containing the valinor settins such as pooling etc.
    """

    with open(valinorrun_json, "r") as file:
        valinorsettings = json.load(file)

    with open(combs_attr_f, "r") as file:
        combs_attr_df = json.load(file)

    with open(combs_attr_s_f, "r") as file:
        combs_attr_s_df = json.load(file)

    overview_stats, overview_stats_combs, overview_stats_s = create_overview_stats(
        combs_attr_df, combs_attr_s_df, topslgene_str
    )

    valinorrun_settings = create_valinorrun_settings(valinorsettings)

    input_files, output_files = create_file_settings(
        mergedfile, combs_attr_df, combs_attr_s_df, valinorsettings
    )

    startpage_stats = {
        "overviewstats": overview_stats,
        "overviewstatscomb": overview_stats_combs,
        "overviewstatssingle": overview_stats_s,
        "valinorrun": valinorrun_settings,
        "valinorruninput": input_files,
        "valinorrunoutput": output_files,
    }

    return startpage_stats, valinorsettings


def produce_lossfunc_examples():
    """
    Produces visual examples of good and bad loss functions.

    This function generates two plots: one representing a 'good' loss function with an appropriate decay rate
    and another representing a 'bad' loss function with a slower decay rate. The plots are saved as SVG files.

    Args:
        None

    Returns:
        None
    """

    x = np.arange(0, 1000)
    y = np.exp(-1 * x * 0.05)

    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 3))
    ax1.plot(x, y)
    ax1.set_xlabel("Steps", fontsize=fontsizes[1])
    ax1.set_ylabel("Loss", fontsize=fontsizes[1])
    ax1.tick_params(axis="both", labelsize=fontsizes[2])
    fig1.tight_layout()

    x = np.arange(0, 1000)
    y = np.exp(-1 * x * 0.005)

    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 3))
    ax2.plot(x, y)
    ax2.set_xlabel("Steps", fontsize=fontsizes[1])
    ax2.set_ylabel("Loss", fontsize=fontsizes[1])
    ax2.tick_params(axis="both", labelsize=fontsizes[2])
    fig2.tight_layout()

    return (fig1, fig2)


def copy_loss_plot(target_file, report_folder):
    new_path = os.path.join(f"{report_folder}/plots/", "valinor_loss.svg")
    shutil.copy(target_file, new_path)


def produce_modelfit_examples():
    """
    Generates visual examples of good and bad model fits using histograms.

    This function creates two sets of histograms: one demonstrating a good model fit where the model-generated
    values closely match the data, and another showing a bad model fit where there is a mismatch. The histograms
    are saved as SVG files.

    Args:
        None

    Returns:
        None
    """

    # create images for good and bad model performance
    k = 10  # Number of successes
    p = 0.01  # Probability of success on each trial

    rng = np.random.default_rng()
    data_val = rng.negative_binomial(k, p, 10000)

    rng = np.random.default_rng(31)
    model_val = rng.negative_binomial(k, p, 10000)

    fig1, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")

    rng = np.random.default_rng(31)
    model_val = rng.negative_binomial(k, 0.02, 10000)

    fig2, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")

    return (fig1, fig2)


def produce_paramfit_guideff_hyper(source, valinorsettings):
    if (valinorsettings["guide_config"] not in ["no_pooling", "full_pooling"]) and (
        "guide_eff_mean_1_mean" in source.columns
    ):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        dev = source.dev_prior_guide_eff_mean_1.unique()
        dev2 = source.dev_prior_guide_eff_mean_2.unique()
        dev = np.unique(np.concatenate([dev, dev2]))

        # Add shaded region
        ax[0].axvspan(-1, 1, facecolor="grey", alpha=0.5)
        ax[0].axvline(0, color="darkgrey")
        ax[0].hist(dev, bins=len(dev) // 10, alpha=0.8)
        ax[0].set_ylabel("Number of guides", fontsize=fontsizes[1])
        ax[0].set_xlabel("Deviation from prior", fontsize=fontsizes[1])
        ax[0].tick_params(axis="both", which="major", labelsize=fontsizes[2])
        ax[0].set_title(naming_cols["guide_eff_mean_1_mean"], fontsize=fontsizes[0])
        # Create custom legend
        grey_patch = mpatches.Patch(
            color="grey", alpha=0.5, label="Within 68% of the prior\ndistribution"
        )

        dev = source.dev_prior_guide_eff_std_1.unique()
        dev2 = source.dev_prior_guide_eff_std_2.unique()
        dev = np.unique(np.concatenate([dev, dev2]))

        # Add shaded region
        ax[1].axvspan(-1, 1, facecolor="grey", alpha=0.5)
        ax[1].axvline(0, color="darkgrey")
        ax[1].hist(dev, bins=len(dev) // 10, alpha=0.8)
        ax[1].set_ylabel("Number of guides", fontsize=fontsizes[1])
        ax[1].set_xlabel("Deviation from prior", fontsize=fontsizes[1])
        ax[1].tick_params(axis="both", which="major", labelsize=fontsizes[2])
        ax[1].set_title(naming_cols["guide_eff_std_1_mean"], fontsize=fontsizes[0])
        # Create custom legend
        grey_patch = mpatches.Patch(
            color="grey", alpha=0.5, label="Within 68% of the prior\ndistribution"
        )
        ax[1].legend(handles=[grey_patch], fontsize=12)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add the text in the middle
        text = "Not applicable\nbecause pooling strategy set to {}".format(
            valinorsettings["guide_config"]
        )
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)

        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def produce_paramfit_guideff(
    source_csv,
):
    selection = alt.selection_single(fields=["cell_line"], bind="legend", empty="none")

    # Base chart with only the outline (using mark_line)
    base_line = (
        alt.Chart(source_csv)
        .transform_aggregate(  # this is just so unique values for guides are used and they are not counted multiple times based on how often a guide occurrs in a combination
            unique_dev_prior_guide_eff_1_mean="mean(dev_prior_guide_eff_1_mean)",
            groupby=["guide1", "cell_line"],
        )
        .transform_density(
            density="unique_dev_prior_guide_eff_1_mean",
            groupby=["cell_line"],
            counts=True,
            as_=["unique_dev_prior_guide_eff_1_mean", "density"],
        )
        .mark_line()
        .encode(
            x=alt.X(
                "unique_dev_prior_guide_eff_1_mean:Q",
                title=naming_cols["dev_prior_guide_eff_1_mean"],
            ),
            y="density:Q",
            color=alt.Color("cell_line:O", scale=alt.Scale(scheme="viridis")),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.5)),
        )
        .add_selection(selection)
    )

    # Additional chart for the filled area when selected
    base_area = (
        alt.Chart(source_csv)
        .transform_aggregate(  # this is just so unique values for guides are used and they are not counted multiple times based on how often a guide occurrs in a combination
            unique_dev_prior_guide_eff_1_mean="mean(dev_prior_guide_eff_1_mean)",
            groupby=["guide1", "cell_line"],
        )
        .transform_density(
            density="unique_dev_prior_guide_eff_1_mean",
            groupby=["cell_line"],
            counts=True,
            as_=["unique_dev_prior_guide_eff_1_mean", "density"],
        )
        .mark_area()
        .encode(
            x="unique_dev_prior_guide_eff_1_mean:Q",
            y="density:Q",
            color=alt.Color(
                "cell_line:O",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            opacity=alt.condition(selection, alt.value(0.8), alt.value(0)),
        )
        .properties(title="Guide 1 effiency")
    )

    # Base chart with only the outline (using mark_line)
    base_line2 = (
        alt.Chart(source_csv)
        .transform_aggregate(  # this is just so unique values for guides are used and they are not counted multiple times based on how often a guide occurrs in a combination
            unique_dev_prior_guide_eff_2_mean="mean(dev_prior_guide_eff_2_mean)",
            groupby=["guide1", "cell_line"],
        )
        .transform_density(
            density="unique_dev_prior_guide_eff_2_mean",
            groupby=["cell_line"],
            counts=True,
            as_=["unique_dev_prior_guide_eff_2_mean", "density"],
        )
        .mark_line()
        .encode(
            x=alt.X(
                "unique_dev_prior_guide_eff_2_mean:Q",
                title=naming_cols["dev_prior_guide_eff_2_mean"],
            ),
            y="density:Q",
            color=alt.Color(
                "cell_line:O",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.5)),
        )
        .add_selection(selection)
    )

    # Additional chart for the filled area when selected
    base_area2 = (
        alt.Chart(source_csv)
        .transform_aggregate(  # this is just so unique values for guides are used and they are not counted multiple times based on how often a guide occurrs in a combination
            unique_dev_prior_guide_eff_2_mean="mean(dev_prior_guide_eff_2_mean)",
            groupby=["guide1", "cell_line"],
        )
        .transform_density(
            density="unique_dev_prior_guide_eff_2_mean",
            groupby=["cell_line"],
            counts=True,
            as_=["unique_dev_prior_guide_eff_2_mean", "density"],
        )
        .mark_area()
        .encode(
            x="unique_dev_prior_guide_eff_2_mean:Q",
            y="density:Q",
            color=alt.Color(
                "cell_line:O",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            opacity=alt.condition(selection, alt.value(0.8), alt.value(0)),
        )
        .properties(title="Guide 2 effiency")
    )

    grey_rect = (
        alt.Chart(pd.DataFrame({"x_min": [-1], "x_max": [1]}))
        .mark_rect(color="grey", opacity=0.2)
        .encode(x="x_min:Q", x2="x_max:Q")
    )

    # Vertical line
    vline = (
        alt.Chart().mark_rule(color="grey").encode(x="a:Q").transform_calculate(a="0")
    )

    # Combine the line and area charts
    final_chart = (grey_rect + base_line + base_area + vline) | (
        grey_rect + base_line2 + base_area2 + vline
    )
    return final_chart


def produce_datastats_countdistr(
    source_csv,
):
    # plot distribution of counts
    # color by cell lines
    # very slow for the whole dataset, but renders!
    selection = alt.selection_single(fields=["cell_line"], bind="legend")

    base = (
        alt.Chart(source_csv)
        .mark_bar(opacity=0.3, binSpacing=0)
        .encode(
            alt.Color(
                "cell_line:O",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.01)),
        )
        .add_selection(selection)
    )

    base = base.encode(
        alt.X("value:Q", bin=alt.Bin(maxbins=100), title=naming_cols["value"]),
        alt.Y("count()", stack=None),
    ) | base.encode(
        alt.X("plasmid:Q", bin=alt.Bin(maxbins=100), title=naming_cols["plasmid"]),
        alt.Y("count()", stack=None),
    )
    return base


def produce_datastats_overdisp(
    source,
    source_csv,
):
    # use the overdispersion calculated from the replicate data rather than the mv parameter fitted by Valinor
    selection = alt.selection_single(fields=["cell_line"], bind="legend")
    min_od = np.min([source.overdispersion.min()])
    max_od = np.max(
        [
            (
                np.quantile(source.overdispersion, 0.99)
                if source.overdispersion.max()
                - np.quantile(source.overdispersion, 0.99)
                > 100
                else source.overdispersion.max()
            )
        ]
    )

    base_line = (
        alt.Chart(source_csv)
        .transform_density(
            density="overdispersion",
            groupby=["cell_line"],
            counts=True,
            extent=[min_od, max_od],
            as_=["overdispersion", "Density"],
        )
        .mark_line()
        .encode(
            x=alt.X("overdispersion:Q", title=naming_cols["overdispersion"]),
            y="Density:Q",
            color=alt.Color(
                "cell_line:O",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.5)),
        )
        .add_selection(selection)
    )

    base_area = (
        alt.Chart(source_csv)
        .transform_density(
            density="overdispersion",
            groupby=["cell_line"],
            counts=True,
            extent=[min_od, max_od],
            as_=["overdispersion", "Density"],
        )
        .mark_area()
        .encode(
            x=alt.X("overdispersion:Q", title="Overdispersion"),
            y="Density:Q",
            color=alt.Color(
                "cell_line:O",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            opacity=alt.condition(selection, alt.value(0.8), alt.value(0)),
        )
    )

    chart = base_line + base_area
    return chart


def produce_diagnplots_dlfc_valscore(
    source_csv,
):
    cell_line_selection = alt.selection_single(fields=["cell_line"], bind="legend")

    tooltip = [
        "genePair:O",
        "cell_line:O",
        "valinor_score:Q",
        "rank_valinor_score:Q",
        "deltaLFC:Q",
        "lfc_s_1:Q",
        "lfc_s_2:Q",
        "lfc:Q",
    ]
    # Create a dictionary that maps original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    chart = (
        alt.Chart(source_csv)
        .mark_circle(size=100)
        .encode(
            x=alt.X("valinor_score:Q", title=naming_cols["valinor_score"]),
            y=alt.Y("deltaLFC:Q", title=naming_cols["deltaLFC"]),
            color=alt.condition(
                cell_line_selection,
                "cell_line:O",
                alt.value("lightgray"),
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            opacity=alt.condition(
                cell_line_selection,
                alt.value(0.7),  # opacity for selected points
                alt.value(0.2),  # opacity for non-selected points
            ),
            tooltip=tooltip_display,
            # tooltip=[naming_cols[field.split(':')[0].replace('mean_','')]+':'+field.split(':')[1] for field in tooltip]
        )
        .transform_aggregate(
            valinor_score="mean(valinor_score)",
            deltaLFC="mean(deltaLFC)",
            rank_valinor_score="mean(rank_valinor_score)",
            lfc_s_1="mean(lfc_s_1)",
            lfc_s_2="mean(lfc_s_2)",
            lfc="mean(lfc)",
            groupby=["genePair", "cell_line"],
        )
        .add_selection(cell_line_selection)
    )

    # Horizontal line for Gene 2 Chart
    hline = (
        alt.Chart().mark_rule(color="grey").encode(y="a:Q").transform_calculate(a="0")
    )

    # Vertical line for Gene 2 Chart
    vline = (
        alt.Chart().mark_rule(color="grey").encode(x="a:Q").transform_calculate(a="0")
    )

    # Combine the charts for Gene 2 and make them interactive
    combined_chart = (chart + hline + vline).interactive()
    return combined_chart


def produce_diagnplots_kogrowths(
    source_csv,
):
    cell_line_selection = alt.selection_single(fields=["cell_line"], bind="legend")

    tooltip = [
        "genePair:O",
        "cell_line:O",
        "gene_ko_growth_12_mean:Q",
        "gene_ko_growth_12_std:Q",
        "valinor_score:Q",
        "rank_valinor_score:Q",
        "deltaLFC:Q",
        "lfc:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    chart = (
        alt.Chart(source_csv)
        .mark_circle(size=100)
        .transform_aggregate(
            gene_ko_growth_12_mean="mean(gene_ko_growth_12_mean)",
            gene_ko_growth_12_std="mean(gene_ko_growth_12_std)",
            valinor_score="mean(valinor_score)",
            deltaLFC="mean(deltaLFC)",
            rank_valinor_score="mean(rank_valinor_score)",
            lfc_s_1="mean(lfc_s_1)",
            lfc_s_2="mean(lfc_s_2)",
            lfc="mean(lfc)",
            groupby=["genePair", "cell_line"],
        )
        .encode(
            x=alt.X(
                "gene_ko_growth_12_mean:Q", title=naming_cols["gene_ko_growth_12_mean"]
            ),
            y=alt.Y(
                "gene_ko_growth_12_std:Q", title=naming_cols["gene_ko_growth_12_std"]
            ),
            color=alt.Color(
                "cell_line:O",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            opacity=alt.condition(
                cell_line_selection,
                alt.value(1),  # opacity for selected points
                alt.value(0.2),  # opacity for non-selected points
            ),
            tooltip=tooltip_display,
        )
        .add_selection(cell_line_selection)
    )

    # Vertical line at x=0
    vline = (
        alt.Chart().mark_rule(color="grey").encode(x="a:Q").transform_calculate(a="0")
    )
    chart = (chart + vline).interactive()
    return chart


def produce_diagnplots_singletons(
    source_csv,
):
    tooltip = [
        "gene1:O",
        "cell_line:O",
        "valinor_score_s_s_1:Q",
        "lfc_s_1:Q",
        "ko_growth_s_mean_s_1:Q",
        "ko_growth_s_std_s_1:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    # Main Chart
    main_chart = (
        alt.Chart(source_csv)
        .mark_circle(size=100)
        .encode(
            x=alt.X("valinor_score_s_s_1:Q", title=naming_cols["valinor_score_s_s_1"]),
            y=alt.Y("lfc_s_1:Q", title=naming_cols["lfc_s_1"]),
            color=alt.Color(
                "ko_growth_s_std_s_1:Q",
                scale=alt.Scale(scheme="blues", reverse=True),
                legend=alt.Legend(title=naming_cols["ko_growth_s_std_s_1"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            valinor_score_s_s_1="mean(valinor_score_s_s_1)",
            lfc_s_1="mean(lfc_s_1)",
            ko_growth_s_std_s_1="mean(ko_growth_s_std_s_1)",
            ko_growth_s_mean_s_1="mean(ko_growth_s_mean_s_1)",
            groupby=["gene1", "cell_line"],
        )
    )

    tooltip = [
        "gene2:O",
        "cell_line:O",
        "valinor_score_s_s_2:Q",
        "lfc_s_2:Q",
        "ko_growth_s_mean_s_2:Q",
        "ko_growth_s_std_s_2:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    # Main Chart for Gene 2
    main_chart_gene2 = (
        alt.Chart(source_csv)
        .mark_circle(size=100)
        .encode(
            x=alt.X("valinor_score_s_s_2:Q", title=naming_cols["valinor_score_s_s_2"]),
            y=alt.Y("lfc_s_2:Q", title=naming_cols["lfc_s_2"]),
            color=alt.Color(
                "ko_growth_s_std_s_2:Q",
                scale=alt.Scale(scheme="blues", reverse=True),
                legend=alt.Legend(title=naming_cols["ko_growth_s_std_s_2"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            valinor_score_s_s_2="mean(valinor_score_s_s_2)",
            lfc_s_2="mean(lfc_s_2)",
            ko_growth_s_std_s_2="mean(ko_growth_s_std_s_2)",
            ko_growth_s_mean_s_2="mean(ko_growth_s_mean_s_2)",
            groupby=["gene2", "cell_line"],
        )
    )

    # Horizontal line at y=0
    hline = (
        alt.Chart().mark_rule(color="grey").encode(y="a:Q").transform_calculate(a="0")
    )

    # Vertical line at x=0
    vline = (
        alt.Chart().mark_rule(color="grey").encode(x="a:Q").transform_calculate(a="0")
    )

    # Combine the charts and make them interactive
    chart = (main_chart + hline + vline).interactive()
    # Combine the charts for Gene 2 and make them interactive
    chart_gene2 = (main_chart_gene2 + hline + vline).interactive()

    # Horizontally concatenate the Gene 1 and Gene 2 charts
    combined_chart = alt.hconcat(chart, chart_gene2)
    return combined_chart


def produce_diagnplots_guideeff_vslfc(
    source_csv,
):
    tooltip = [
        "guide1:O",
        "gene1:O",
        "cell_line:O",
        "lfc_s_1:Q",
        "guide_eff_s_mean_s_1:Q",
        "valinor_score_s_s_1:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]
    chart1 = (
        alt.Chart(source_csv)
        .mark_circle(size=100)
        .encode(
            x=alt.X("lfc_s_1:Q", title=naming_cols["lfc_s_1"]),
            y=alt.Y(
                "guide_eff_s_mean_s_1:Q", title=naming_cols["guide_eff_s_mean_s_1"]
            ),
            color=alt.Color(
                "guide_eff_s_mean_s_1:Q",
                scale=alt.Scale(scheme="blues", reverse=False),
                legend=alt.Legend(title=naming_cols["guide_eff_s_mean_s_1"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            lfc_s_1="mean(lfc_s_1)",
            guide_eff_s_mean_s_1="mean(guide_eff_s_mean_s_1)",
            valinor_score_s_s_1="mean(valinor_score_s_s_1)",
            groupby=["guide1", "gene1", "cell_line"],
        )
    )

    tooltip = [
        "guide2:O",
        "gene2:O",
        "cell_line:O",
        "lfc_s_2:Q",
        "guide_eff_s_mean_s_2:Q",
        "valinor_score_s_s_2:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    chart2 = (
        alt.Chart(source_csv)
        .mark_circle(size=100)
        .encode(
            x=alt.X("lfc_s_2:Q", title=naming_cols["lfc_s_2"]),
            y=alt.Y(
                "guide_eff_s_mean_s_2:Q", title=naming_cols["guide_eff_s_mean_s_2"]
            ),
            color=alt.Color(
                "guide_eff_s_mean_s_2:Q",
                scale=alt.Scale(scheme="blues", reverse=False),
                legend=alt.Legend(title=naming_cols["guide_eff_s_mean_s_2"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            lfc_s_2="mean(lfc_s_2)",
            guide_eff_s_mean_s_2="mean(guide_eff_s_mean_s_2)",
            valinor_score_s_s_2="mean(valinor_score_s_s_2)",
            groupby=["guide2", "gene2", "cell_line"],
        )
    )

    vline = (
        alt.Chart().mark_rule(color="grey").encode(x="a:Q").transform_calculate(a="0")
    )
    hline = (
        alt.Chart().mark_rule(color="grey").encode(y="a:Q").transform_calculate(a="1")
    )

    chart1 = (chart1 + vline + hline).interactive()
    chart2 = (chart2 + vline + hline).interactive()
    combined_chart = alt.hconcat(chart1, chart2)
    return combined_chart


def produce_hitprior_lfc_vs_valscore_avg(
    source,
    source_csv,
):
    tmp = source.groupby(["genePair"])[["valinor_score", "lfc"]].mean()
    x_diff = (tmp["valinor_score"].max() - tmp["valinor_score"].min()) / 10
    xrange = (
        tmp["valinor_score"].values.min() - x_diff,
        tmp["valinor_score"].values.max() + x_diff,
    )
    y_diff = (tmp["lfc"].max() - tmp["lfc"].min()) / 10
    yrange = (tmp["lfc"].min() - y_diff, tmp["lfc"].max() + y_diff)

    tooltip = [
        "genePair:O",
        "rank_valinor_score:Q",
        "valinor_score:Q",
        "lfc:Q",
        "gene_ko_growth_12_mean:Q",
        "gene_ko_growth_12_std:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    base = (
        alt.Chart(source_csv)
        .mark_circle(size=60, color="black")
        .encode(
            x=alt.X(
                "valinor_score:Q",
                scale=alt.Scale(domain=[xrange[0], xrange[1]], nice=False),
                title=naming_cols["valinor_score"],
            ),
            y=alt.Y(
                "lfc:Q",
                scale=alt.Scale(domain=[yrange[0], yrange[1]], nice=False),
                title=naming_cols["lfc"],
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            valinor_score="mean(valinor_score)",
            lfc="mean(lfc)",
            gene_ko_growth_12_mean="mean(gene_ko_growth_12_mean)",
            gene_ko_growth_12_std="mean(gene_ko_growth_12_std)",
            rank_valinor_score="mean(rank_valinor_score)",
            groupby=["genePair"],
        )
    )

    rect = pd.DataFrame(
        {
            "x1": [xrange[0]],
            "x2": [xrange[1]],
            "y1": [yrange[0]],
            "y2": [yrange[1]],
            "zero": [0],
        }
    )

    span1 = (
        alt.Chart(rect)
        .mark_rect(opacity=0.10, color="green")
        .encode(
            x="x1",
            x2="zero",
            y="y1",  # 0 pixels from top
            y2="zero",  # 0 pixels from top
        )
    )

    span2 = (
        alt.Chart(rect)
        .mark_rect(opacity=0.10, color="red")
        .encode(
            x="zero",
            x2="x2",
            y="y1",  # 0 pixels from top
            y2="zero",  # 0 pixels from top
        )
    )

    span3 = (
        alt.Chart(rect)
        .mark_rect(opacity=0.10, color="red")
        .encode(
            x="x1",
            x2="zero",
            y="zero",  # 0 pixels from top
            y2="y2",  # 0 pixels from top
        )
    )

    span4 = (
        alt.Chart(rect)
        .mark_rect(opacity=0.10, color="blue")
        .encode(
            x="zero",
            x2="x2",
            y="zero",  # 0 pixels from top
            y2="y2",  # 0 pixels from top
        )
    )
    # add dummy selection
    plot = (base + span1 + span2 + span3 + span4).add_selection(alt.selection_single())
    return plot


def produce_hitprior_lfc_vs_valscore_hist(
    source,
    source_csv,
):
    tmp = source.groupby(["genePair", "cell_line"])[["valinor_score", "lfc"]].mean()
    x_diff = (tmp["valinor_score"].max() - tmp["valinor_score"].min()) / 10
    xrange = (
        tmp["valinor_score"].values.min() - x_diff,
        tmp["valinor_score"].values.max() + x_diff,
    )
    y_diff = (tmp["lfc"].max() - tmp["lfc"].min()) / 10
    yrange = (tmp["lfc"].min() - y_diff, tmp["lfc"].max() + y_diff)

    # Create a selection_interval for selecting rectangular area in scatterplot
    brush = alt.selection_interval(
        empty="none",
        init={
            "x": [xrange[0], tmp["valinor_score"].min() + 4 * x_diff],
            "y": [yrange[0], tmp["lfc"].min() + 4 * y_diff],
        },
    )

    tooltip = [
        "genePair:O",
        "cell_line:O",
        "valinor_score:Q",
        "lfc:Q",
        "gene_ko_growth_12_mean:Q",
        "gene_ko_growth_12_std:Q",
        "rank_valinor_score:Q",
    ]

    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    # Create scatterplot
    scatter = (
        alt.Chart(source_csv)
        .mark_circle(size=60, color="black")
        .encode(
            x=alt.X(
                "valinor_score:Q",
                scale=alt.Scale(domain=[xrange[0], xrange[1]], nice=False),
                title=naming_cols["valinor_score"],
            ),
            y=alt.Y(
                "lfc:Q",
                scale=alt.Scale(domain=[yrange[0], yrange[1]], nice=False),
                title=naming_cols["lfc"],
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            valinor_score="mean(valinor_score)",
            lfc="mean(lfc)",
            gene_ko_growth_12_mean="mean(gene_ko_growth_12_mean)",
            gene_ko_growth_12_std="mean(gene_ko_growth_12_std)",
            rank_valinor_score="mean(rank_valinor_score)",
            groupby=["genePair", "cell_line"],
        )
        .add_selection(brush)
    )
    # Create histogram
    histogram = (
        alt.Chart(source_csv)
        .mark_bar()
        .encode(
            x=alt.X("count(cell_line):Q", title="Count of cell lines"),
            y=alt.Y("genePair:O", sort="-x", title=naming_cols["genePair"]),
        )
        .transform_aggregate(
            valinor_score="mean(valinor_score)", groupby=["genePair", "cell_line"]
        )
        .transform_filter(brush)
        .transform_window(
            rank="rank(count(cell_line))",
            sort=[alt.SortField("count(cell_line):Q", order="descending")],
        )
        .transform_filter((alt.datum.rank <= 10))  # Take only the top 10 genePairs
    )

    rect = pd.DataFrame(
        {
            "x1": [xrange[0]],
            "x2": [xrange[1]],
            "y1": [yrange[0]],
            "y2": [yrange[1]],
            "zero": [0],
        }
    )

    span1 = (
        alt.Chart(rect)
        .mark_rect(opacity=0.10, color="green")
        .encode(
            x="x1",
            x2="zero",
            y="y1",  # 0 pixels from top
            y2="zero",  # 0 pixels from top
        )
    )

    span2 = (
        alt.Chart(rect)
        .mark_rect(opacity=0.10, color="red")
        .encode(
            x="zero",
            x2="x2",
            y="y1",  # 0 pixels from top
            y2="zero",  # 0 pixels from top
        )
    )

    span3 = (
        alt.Chart(rect)
        .mark_rect(opacity=0.10, color="red")
        .encode(
            x="x1",
            x2="zero",
            y="zero",  # 0 pixels from top
            y2="y2",  # 0 pixels from top
        )
    )

    span4 = (
        alt.Chart(rect)
        .mark_rect(opacity=0.10, color="blue")
        .encode(
            x="zero",
            x2="x2",
            y="zero",  # 0 pixels from top
            y2="y2",  # 0 pixels from top
        )
    )
    chart = (scatter + span1 + span2 + span3 + span4) & histogram  # & histogram
    return chart


def produce_hitprior_kogrowths(
    source_csv,
):
    # Create a selection_interval for selecting rectangular area in scatterplot
    brush = alt.selection_interval()  # empty='none'

    tooltip = [
        "genePair:O",
        "cell_line:O",
        "valinor_score:Q",
        "lfc:Q",
        "gene_ko_growth_12_mean:Q",
        "gene_ko_growth_1_mean:Q",
        "gene_ko_growth_2_mean:Q",
        "rank_valinor_score:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    # Create scatterplot
    scatter1 = (
        alt.Chart(source_csv)
        .mark_circle(size=60, color="black")
        .encode(
            x=alt.X(
                "gene_ko_growth_12_mean:Q", title=naming_cols["gene_ko_growth_12_mean"]
            ),
            y=alt.Y(
                "gene_ko_growth_1_mean:Q", title=naming_cols["gene_ko_growth_1_mean"]
            ),
            color=alt.condition(
                brush,
                alt.Color("genePair:O", scale=alt.Scale(scheme="sinebow"), legend=None),
                alt.value("grey"),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            valinor_score="mean(valinor_score)",
            lfc="mean(lfc)",
            gene_ko_growth_1_mean="mean(gene_ko_growth_1_mean)",
            gene_ko_growth_2_mean="mean(gene_ko_growth_2_mean)",
            gene_ko_growth_12_mean="mean(gene_ko_growth_12_mean)",
            rank_valinor_score="mean(rank_valinor_score)",
            groupby=["genePair", "cell_line"],
        )
        .add_selection(brush)
    )

    tooltip = [
        "genePair:O",
        "cell_line:O",
        "valinor_score:Q",
        "lfc:Q",
        "gene_ko_growth_12_mean:Q",
        "gene_ko_growth_1_mean:Q",
        "gene_ko_growth_2_mean:Q",
        "rank_valinor_score:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    scatter2 = (
        alt.Chart(source_csv)
        .mark_circle(size=60, color="black")
        .encode(
            x=alt.X(
                "gene_ko_growth_12_mean:Q", title=naming_cols["gene_ko_growth_12_mean"]
            ),
            y=alt.Y(
                "gene_ko_growth_2_mean:Q", title=naming_cols["gene_ko_growth_2_mean"]
            ),
            color=alt.condition(
                brush,
                alt.Color("genePair:O", scale=alt.Scale(scheme="sinebow"), legend=None),
                alt.value("grey"),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            valinor_score="mean(valinor_score)",
            lfc="mean(lfc)",
            gene_ko_growth_1_mean="mean(gene_ko_growth_1_mean)",
            gene_ko_growth_2_mean="mean(gene_ko_growth_2_mean)",
            gene_ko_growth_12_mean="mean(gene_ko_growth_12_mean)",
            rank_valinor_score="mean(rank_valinor_score)",
            groupby=["genePair", "cell_line"],
        )
        .add_selection(brush)
    )

    # Horizontal line at y=0
    hline = (
        alt.Chart().mark_rule(color="grey").encode(y="a:Q").transform_calculate(a="0")
    )

    # Vertical line at x=0
    vline = (
        alt.Chart().mark_rule(color="grey").encode(x="a:Q").transform_calculate(a="0")
    )

    chart = (scatter1 + hline + vline) | (scatter2 + hline + vline)
    return chart


def produce_geneview_valscore_rank(
    source,
    source_csv,
):
    min_valinor_score = source.valinor_score.min()
    max_valinor_score = source.valinor_score.max()

    max_rank = source.rank_valinor_score.max()

    tooltip = [
        "genePair:O",
        "cell_line:O",
        "valinor_score:Q",
        "rank_valinor_score:Q",
        "gene_ko_growth_12_mean:Q",
        "gene_ko_growth_12_std:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    sym_valinor_score = np.max([-1 * min_valinor_score, max_valinor_score])

    base = (
        alt.Chart(source_csv)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                "rank_valinor_score:Q",
                title=naming_cols["rank_valinor_score"],
                scale=alt.Scale(domain=[-1, max_rank]),
            ),
            y=alt.Y("cell_line:O", title=naming_cols["cell_line"]),
            color=alt.Color(
                "valinor_score:Q",
                scale=alt.Scale(
                    scheme="blueorange",
                    domain=[-1 * sym_valinor_score, sym_valinor_score],
                ),
                legend=alt.Legend(title=naming_cols["valinor_score"]),
            ),
            tooltip=tooltip_display,
        )
        .interactive()
        .configure_view(  # Removes the border around the plot
            fill="grey"  # Replace 'your_color' with your desired color, e.g., '#f0f0f0'
        )
    )
    return base


def produce_geneview_valscore_lfc(
    source_csv,
):
    single = alt.selection_single()

    tooltip = [
        "genePair:O",
        "cell_line:O",
        "lfc_s_1:Q",
        "lfc_s_2:Q",
        "lfc:Q",
        "deltaLFC:Q",
        "valinor_score:Q",
        "rank_valinor_score:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    base = (
        alt.Chart(source_csv)
        .mark_circle(size=100)
        .encode(
            x=alt.X("valinor_score:Q", title=naming_cols["valinor_score"]),
            y=alt.Y("lfc_s_1:Q", title=naming_cols["lfc_s_1"]),
            color=alt.condition(
                single,
                "cell_line:O",
                alt.value("lightgray"),
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            lfc_s_1="mean(lfc_s_1)",
            lfc_s_2="mean(lfc_s_2)",
            lfc="mean(lfc)",
            rank_valinor_score="mean(rank_valinor_score)",
            valinor_score="mean(valinor_score)",
            deltaLFC="mean(deltaLFC)",
            groupby=["cell_line", "genePair"],
        )
        .properties(width=250, height=250)
        .add_selection(single)
    )
    line_x = (
        alt.Chart(pd.DataFrame({"x": [0], "genePair": np.nan}))
        .mark_rule()
        .encode(x="x")
    )
    line_y = (
        alt.Chart(pd.DataFrame({"y": [0], "genePair": np.nan}))
        .mark_rule()
        .encode(y="y")
    )

    base = (
        base + line_x + line_y
        | base.encode(y=alt.Y("lfc_s_2:Q", title=naming_cols["lfc_s_2"]))
        + line_x
        + line_y
        | base.encode(y=alt.Y("lfc:Q", title=naming_cols["lfc"])) + line_x + line_y
    )
    base = base.resolve_scale(y="shared")
    return base


def produce_geneview_growth_lfc(
    source_csv,
):
    single = alt.selection_single()

    tooltip = [
        "genePair:O",
        "cell_line:O",
        "valinor_score:Q",
        "rank_valinor_score:Q",
        "deltaLFC:Q",
        "valinor_score_s_s_1:Q",
        "valinor_score_s_s_2:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    base = (
        alt.Chart(source_csv)
        .mark_circle(size=100)
        .encode(
            x=alt.X("valinor_score:Q", title=naming_cols["valinor_score"]),
            y=alt.Y("valinor_score_s_s_1:Q", title=naming_cols["valinor_score_s_s_1"]),
            color=alt.condition(
                single,
                "cell_line:O",
                alt.value("lightgray"),
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            valinor_score_s_s_1="mean(valinor_score_s_s_1)",
            valinor_score_s_s_2="mean(valinor_score_s_s_2)",
            valinor_score="mean(valinor_score)",
            rank_valinor_score="mean(rank_valinor_score)",
            deltaLFC="mean(deltaLFC)",
            groupby=["cell_line", "genePair"],
        )
        .properties(width=250, height=250)
        .add_selection(single)
    )
    line_x = (
        alt.Chart(pd.DataFrame({"x": [0], "genePair": np.nan}))
        .mark_rule()
        .encode(x="x")
    )
    line_y = (
        alt.Chart(pd.DataFrame({"y": [0], "genePair": np.nan}))
        .mark_rule()
        .encode(y="y")
    )

    base = (
        base + line_x + line_y
        | base.encode(
            y=alt.Y("valinor_score_s_s_2:Q", title=naming_cols["valinor_score_s_s_2"])
        )
        + line_x
        + line_y
    )
    # base = base | base.encode(y=alt.Y('valinor_score_s_s_2:Q', title='Valinor Score Gene 2'))

    base = base.resolve_scale(y="shared")
    return base


def produce_geneview_valscore_lfc_single(
    source_csv,
):
    single = alt.selection_single()

    tooltip = [
        "genePair:O",
        "cell_line:O",
        "valinor_score:Q",
        "rank_valinor_score:Q",
        "deltaLFC:Q",
        "valinor_score_s_s_1:Q",
        "valinor_score_s_s_2:Q",
        "lfc_s_1:Q",
        "lfc_s_2:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    base = (
        alt.Chart(source_csv)
        .mark_circle(size=100)
        .encode(
            x=alt.X("valinor_score_s_s_1:Q", title=naming_cols["valinor_score_s_s_1"]),
            y=alt.Y("lfc_s_1:Q", title=naming_cols["lfc_s_1"]),
            color=alt.condition(
                single,
                "cell_line:O",
                alt.value("lightgray"),
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=naming_cols["cell_line"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            valinor_score_s_s_1="mean(valinor_score_s_s_1)",
            valinor_score_s_s_2="mean(valinor_score_s_s_2)",
            lfc_s_1="mean(lfc_s_1)",
            lfc_s_2="mean(lfc_s_2)",
            valinor_score="mean(valinor_score)",
            rank_valinor_score="mean(rank_valinor_score)",
            deltaLFC="mean(deltaLFC)",
            groupby=["cell_line", "genePair"],
        )
        .properties(width=250, height=250)
        .add_selection(single)
    )
    line_x = (
        alt.Chart(pd.DataFrame({"x": [0], "genePair": np.nan}))
        .mark_rule()
        .encode(x="x")
    )
    line_y = (
        alt.Chart(pd.DataFrame({"y": [0], "genePair": np.nan}))
        .mark_rule()
        .encode(y="y")
    )

    base = (
        base + line_x + line_y
        | base.encode(
            x=alt.X("valinor_score_s_s_2:Q", title=naming_cols["valinor_score_s_s_2"]),
            y=alt.Y("lfc_s_2:Q", title=naming_cols["lfc_s_2"]),
        )
        + line_x
        + line_y
    )
    # base = base | base.encode(y=alt.Y('valinor_score_s_s_2:Q', title='Valinor Score Gene 2'))

    base = base.resolve_scale(y="shared", x="shared")
    return base


def postprocess_altairhtml(htmlfile, datafile, id_handle, select_on=None):
    with open(htmlfile, "r") as file:
        altair_plot = file.read()

    filejson_replace = '{"url": "' + datafile + '"}'

    if "datasets" in altair_plot:
        front = altair_plot.split('"datasets"')[0]
        back = altair_plot.split('"datasets"')[1]
        datasets = back.split("};")[0]
        back = "};" + back.split("};")[1]
        tmp = datasets[:-1] + ", 'datadatadata' : csv" + "}"
        datasets_replace = [datasets, tmp]
    else:
        datasets_replace = [
            '"$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json"',
            '"$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets" : {\'datadatadata\' : csv}',
        ]

    altairplot_instr = (
        altair_plot.split("unction(vegaEmbed) {\n")[1]
        .split("})(vegaEmbed);")[0]
        .strip(" ")
        .strip("\n")
        .replace(
            filejson_replace, '{"name": "datadatadata", "format": {"type": "csv",}}'
        )
        .replace(datasets_replace[0], datasets_replace[1])
        .replace("vis", id_handle)
    )

    if not "datasets" in altairplot_instr:
        # If there are no embedded datasets, add ours

        altairplot_instr = altairplot_instr.replace(
            '"$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json"',
            '"$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets" : {\'datadatadata\' : csv}',
        )
    else:
        # If there are, add ours to the list

        altairplot_instr = altairplot_instr.replace(
            '"datasets": {', "\"datasets\": {'datadatadata' : csv,"
        )

    if select_on != None:
        # Add JS to insert gene pair from selection menu into Vega plot filter
        if "datasets" in altair_plot:
            if "transform" in altair_plot:
                insertions = [
                    m.end() for m in re.finditer("transform", altairplot_instr)
                ]
                for i in insertions[::-1]:
                    altairplot_instr = (
                        altairplot_instr[: i + 5]
                        + '"filter": { "field": \''
                        + select_on
                        + '\', "equal": selectedValue }},{'
                        + altairplot_instr[i + 5 :]
                    )

                slice_from = altairplot_instr.index("var spec")
                slice_to = altairplot_instr.index("var embedOpt")
                exciseplots = altairplot_instr[slice_from:slice_to].replace(
                    "spec", "newSpec"
                )
                altairplot_instr = (
                    altairplot_instr[:slice_from] + altairplot_instr[slice_to:]
                )

                slice_to = altairplot_instr.index("vegaEmbed")
                altairplot_instr = (
                    altairplot_instr[:slice_to]
                    + "const input = document.getElementById('gene-pair-selection');"
                    + "\n"
                    "input.addEventListener('change', (event) => {" + "\n"
                    "    const selectedValue = event.target.value;"
                    + "\n"
                    + exciseplots
                    + "\n"
                    "vegaEmbed('#{}', newSpec, embedOpt);".format(id_handle) + "\n"
                    "});" + "\n"
                )

        else:
            slice_to = altairplot_instr.index("vegaEmbed")

            altairplot_instr = (
                altairplot_instr[:slice_to]
                + "const input = document.getElementById('gene-pair-selection');"
                + "\n"
                "input.addEventListener('change', (event) => {" + "\n"
                "    const selectedValue = event.target.value;" + "\n"
                "    const newSpec = Object.assign({}, spec, {" + "\n"
                "    transform: [{ filter: { field: '"
                + select_on
                + "', equal: selectedValue } }]"
                + "\n"
                "    });" + "\n"
                "vegaEmbed('#{}', newSpec, embedOpt);".format(id_handle) + "\n"
                "});" + "\n"
            )
    with open(htmlfile.split(".")[0] + "_mod.html", "w") as file:
        file.write(altairplot_instr)

    return altairplot_instr


def populate_genelist(source):
    # Populate gene pair list
    html_pairs_list = []

    for p in sorted(source["genePair"].unique()):
        html_pairs_list.append(f'<option value="{p}" label = "{p}">')

    html_pairs = "\n".join(html_pairs_list)
    return html_pairs


def create_report(
    source_csv,
    startpage_stats,
    # guide_eff_priors_str,
    html_pairs,
    report_folder,
):
    # postprocess the saved html dso it works in the final report
    # not really elegant constantly writing and loading files
    # but not sure how to do it in place
    hit_prioritisation_cell_avg_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/hit_prioritisation_cell_avg.html",
        source_csv,
        "hit_prioritisation_cell_avg_plot",
    )
    hit_prioritisation_lfcvsval_histo = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/hit_prioritisation_lfcvsval_histo.html",
        source_csv,
        "hit_prioritisation_lfcvsval_histo",
    )
    hit_prioritisation_growthdefecscatter = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/hit_prioritisation_growthdefecscatter.html",
        source_csv,
        "hit_prioritisation_growthdefecscatter",
    )

    parameterfits_guideeffs_devprior_combo = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/parameterfits_guideeffs_devprior_combo.html",
        source_csv,
        "parameterfits_guideeffs_devprior_combo",
    )

    datastats_count_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/data_stats_counts.html",
        source_csv,
        "datastats_count_plot",
    )
    datastats_disp_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/data_stats_dispersion.html",
        source_csv,
        "datastats_disp_plot",
    )

    diagnplots_dlfcval_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/diagnostic_plots_dLFCvsValinor.html",
        source_csv,
        "diagnplots_dlfcval_plot",
    )
    diagnplots_kogrowthstd_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/diagnostic_plots_kogrowhstd.html",
        source_csv,
        "diagnplots_kogrowthstd_plot",
    )
    diagnplots_singlelfcval_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/diagnostic_plots_singleLFCvsvalscore.html",
        source_csv,
        "diagnplots_singlelfcval_plot",
    )
    diagnplots_singleLFCeffic_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/diagnostic_plots_singleLFCvsefficiency.html",
        source_csv,
        "diagnplots_singleLFCeffic_plot",
    )

    genepair_valinorscore_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/gene_pair_valscore_percln.html",
        source_csv,
        "genepair_valinorscore_plot",
        select_on="genePair",
    )
    genepair_valinorscorevslfc_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/gene_pair_valscorevslfc.html",
        source_csv,
        "genepair_valinorscorevslfc_plot",
        select_on="genePair",
    )
    genepair_valscorescombovssingle_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/gene_pair_valscorescombovssingle.html",
        source_csv,
        "genepair_valscorescombovssingle_plot",
        select_on="genePair",
    )
    genepair_valscoresinglevslfc_plot = postprocess_altairhtml(
        f"{report_folder}/altair_snippets/gene_pair_valscoresinglevslfc.html",
        source_csv,
        "genepair_valscoresinglevslfc_plot",
        select_on="genePair",
    )

    # Template handling
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=""))
    # template = env.get_template('Bootstrap, from Twitter.html')
    # html = template.render()
    template = env.get_template(f"{report_folder}/template_tabs.html")
    html = template.render(
        date=date.today(),
        overviewstats=startpage_stats["overviewstats"],
        overviewstatscomb=startpage_stats["overviewstatscomb"],
        overviewstatssingle=startpage_stats["overviewstatssingle"],
        valinorrun=startpage_stats["valinorrun"],
        valinorruninput=startpage_stats["valinorruninput"],
        valinorrunoutput=startpage_stats["valinorrunoutput"],
        # guide_eff_priors_str=guide_eff_priors_str,
        parameterfits_guideeffs_devprior_combo=parameterfits_guideeffs_devprior_combo,
        datastats_count_plot=datastats_count_plot,
        datastats_disp_plot=datastats_disp_plot,
        diagnplots_dlfcval_plot=diagnplots_dlfcval_plot,
        diagnplots_kogrowthstd_plot=diagnplots_kogrowthstd_plot,
        diagnplots_singlelfcval_plot=diagnplots_singlelfcval_plot,
        diagnplots_singleLFCeffic_plot=diagnplots_singleLFCeffic_plot,
        hit_prioritisation_cell_avg_plot=hit_prioritisation_cell_avg_plot,
        hit_prioritisation_lfcvsval_histo=hit_prioritisation_lfcvsval_histo,
        hit_prioritisation_growthdefecscatter=hit_prioritisation_growthdefecscatter,
        genepair_valinorscore_plot=genepair_valinorscore_plot,
        genepair_valinorscorevslfc_plot=genepair_valinorscorevslfc_plot,
        genepair_valscorescombovssingle_plot=genepair_valscorescombovssingle_plot,
        genepair_valscoresinglevslfc_plot=genepair_valscoresinglevslfc_plot,
        gene_pair_list=html_pairs,
    )

    # Write the HTML file
    with open(f"{report_folder}/report.html", "w") as f:
        f.write(html)


def create_necessary_folders(folders):
    # Create folders if they don't exist
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def get_resource_path(package, resource):
    # Try to find the package's spec
    package_spec = importlib.util.find_spec(package)

    if package_spec and package_spec.origin:
        # Non-editable mode: Use importlib.resources
        try:
            with pkg_resources.path(package, resource) as res_path:
                return str(res_path)
        except FileNotFoundError:
            pass

    # Editable mode: Construct the path manually
    base_dir = os.path.dirname(os.path.dirname(package_spec.origin))
    return os.path.join(base_dir, package.replace(".", "/"), resource)


def run():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument(
        "--combfile",
        default=None,
        help="processed table combining data and Valinor output, if not supplied this will be created here",
    )

    parser.add_argument("--valinorLossFile", help="SVG file that shows the loss curve")

    parser.add_argument(
        "--valinorConfigFile", help="JSON that contains the config for the valinor run"
    )

    # parser.add_argument("--valinorPriorFile", help="JSON that contains the priors")

    parser.add_argument(
        "--subsetSLpairs",
        action="store_true",
        help="Don't export all the data but only a subset containing the most synthetic lethal pairs and several more randomly sampled ones.",
    )

    parser.add_argument("--val_combo", default=None, help="Path to the val_combo file.")
    parser.add_argument(
        "--val_single", default=None, help="Path to the val_single file."
    )
    parser.add_argument(
        "--data_combo", default=None, help="Path to the data_combo file."
    )
    parser.add_argument(
        "--data_single", default=None, help="Path to the data_single file."
    )
    parser.add_argument(
        "--output_file", default=None, help="Name of processed output file."
    )
    parser.add_argument(
        "--valinorPriorFile",
        help="Optional: JSON that contains the priors used to run valinor",
    )

    parser.add_argument(
        "--report_folder",
        help="This is the folder in which the necessary report files will be saved",
    )

    args = parser.parse_args()

    if args.combfile is None:
        if (
            (args.data_combo is None)
            | (args.data_single is None)
            | (args.val_combo is None)
            | (args.val_single is None)
            | (args.report_folder is None)
            | (args.output_file is None)
        ):
            raise ValueError(
                "To create the report, the valinor output needs to be processed first. Please provide: --data_combo, --data_single, --val_combo, --val_single, --report_folder, --output_file"
            )
        else:
            scoreData_combined = run_processvalinor(
                args.valinorPriorFile,
                args.data_combo,
                args.data_single,
                args.val_combo,
                args.val_single,
                args.report_folder,
            )

            scoreData_combined.to_parquet(args.output_file)
    else:
        scoreData_combined = pd.read_parquet(args.combfile)

    folders = [
        f"{args.report_folder}/input",
        f"{args.report_folder}/plots",
        f"{args.report_folder}/altair_snippets",
    ]
    create_necessary_folders(folders)

    resource_path = get_resource_path("valinor.valinorreport", "template_tabs.html")
    shutil.copy(resource_path, os.path.join(args.report_folder, "template_tabs.html"))
    resource_path = get_resource_path("valinor.valinorreport", "favicon.ico")
    shutil.copy(resource_path, os.path.join(args.report_folder, "favicon.ico"))
    resource_path = get_resource_path("valinor.valinorreport", "valinor.png")
    shutil.copy(resource_path, os.path.join(args.report_folder, "valinor.png"))
    resource_path = get_resource_path("valinor.valinorreport", "valinor_white.png")
    shutil.copy(resource_path, os.path.join(args.report_folder, "valinor_white.png"))

    if args.subsetSLpairs:
        selectpairs, npairs = sample_genepairs(scoreData_combined)

        source = scoreData_combined.loc[
            scoreData_combined.genePair.isin(selectpairs)
        ].copy()

        topslgene_str = "Report only displays data of subset of gene pairs (--subsetSLpairs flag set): {} gene pairs are displayed in next tabs, {} are top synthetic lethal, {} were randomly selected".format(
            npairs[0] + npairs[1], npairs[0], npairs[1]
        )

    else:
        source = scoreData_combined.copy()
        topslgene_str = ""

    source.to_csv(f"{args.report_folder}/input/report_input.csv", index=False)
    source_csv = f"{args.report_folder}/input/report_input.csv"

    ## HOME
    # Overview stats
    combs_attr_f = f"{args.report_folder}/input/combs_attr.json"
    combs_attr_s_f = f"{args.report_folder}/input/combs_attr_s.json"

    startpage_stats, valinorsettings = create_startpage_stats(
        args.combfile,
        combs_attr_f,
        combs_attr_s_f,
        args.valinorConfigFile,
        topslgene_str,
    )

    # Loss Function
    fig1, fig2 = produce_lossfunc_examples()
    fig1.savefig(
        f"{args.report_folder}/plots/good_lossfunction.svg", bbox_inches="tight"
    )
    plt.close(fig1)
    fig2.savefig(
        f"{args.report_folder}/plots/bad_lossfunction.svg", bbox_inches="tight"
    )
    plt.close(fig2)

    copy_loss_plot(args.valinorLossFile, args.report_folder)

    # Model Fit
    fig1, fig2 = produce_modelfit_examples()
    fig1.savefig(f"{args.report_folder}/plots/good_modelfit.svg", bbox_inches="tight")
    plt.close(fig1)
    fig2.savefig(f"{args.report_folder}/plots/bad_modelfit.svg", bbox_inches="tight")
    plt.close(fig2)

    fig = produce_paramfit_guideff_hyper(source, valinorsettings)
    fig.savefig(
        f"{args.report_folder}/plots/parameterfits_guideeffs_metahier_combo.svg",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)

    chart = produce_paramfit_guideff(source_csv)
    chart.save(
        f"{args.report_folder}/altair_snippets/parameterfits_guideeffs_devprior_combo.html"
    )

    ## DATA STATS
    # Count distributions
    chart = produce_datastats_countdistr(source_csv)
    chart.save(f"{args.report_folder}/altair_snippets/data_stats_counts.html")

    # Count overdispersion
    chart = produce_datastats_overdisp(source, source_csv)
    chart.save(f"{args.report_folder}/altair_snippets/data_stats_dispersion.html")

    ## DIAGNOSTIC PLOTS
    # dLFC vs. Valinor Score
    chart = produce_diagnplots_dlfc_valscore(source_csv)
    chart.save(
        f"{args.report_folder}/altair_snippets/diagnostic_plots_dLFCvsValinor.html"
    )
    # ko_growth_12 vs. gene_ko_growth_12_std
    chart = produce_diagnplots_kogrowths(source_csv)
    chart.save(f"{args.report_folder}/altair_snippets/diagnostic_plots_kogrowhstd.html")
    # singleton LFC vs. singleton gene effect
    chart = produce_diagnplots_singletons(source_csv)
    chart.save(
        f"{args.report_folder}/altair_snippets/diagnostic_plots_singleLFCvsvalscore.html"
    )
    # guide eff vs. guide lfc Singletons
    chart = produce_diagnplots_guideeff_vslfc(source_csv)
    chart.save(
        f"{args.report_folder}/altair_snippets/diagnostic_plots_singleLFCvsefficiency.html"
    )

    ## HIT PRIORITISATION
    # LFC vs. Valinor Score - genePair averaged
    chart = produce_hitprior_lfc_vs_valscore_avg(source, source_csv)
    chart.save(f"{args.report_folder}/altair_snippets/hit_prioritisation_cell_avg.html")

    # LFC vs. Valinor score - Number of cell lines
    chart = produce_hitprior_lfc_vs_valscore_hist(source, source_csv)
    chart.save(
        f"{args.report_folder}/altair_snippets/hit_prioritisation_lfcvsval_histo.html"
    )
    # growth 1 vs. growth 12 vs. growth 2
    chart = produce_hitprior_kogrowths(source_csv)
    chart.save(
        f"{args.report_folder}/altair_snippets/hit_prioritisation_growthdefecscatter.html"
    )

    ## GENE PAIR VIEW
    # Valinor score per cell line
    chart = produce_geneview_valscore_rank(source, source_csv)
    chart.save(f"{args.report_folder}/altair_snippets/gene_pair_valscore_percln.html")
    # Combination Valinor score vs. LFCs
    chart = produce_geneview_valscore_lfc(source_csv)
    chart.save(f"{args.report_folder}/altair_snippets/gene_pair_valscorevslfc.html")
    # growth 12 vs singleton growth
    chart = produce_geneview_growth_lfc(source_csv)
    chart.save(
        f"{args.report_folder}/altair_snippets/gene_pair_valscoresinglevslfc.html"
    )
    # Singleton Valinor Scores vs. LFCs
    chart = produce_geneview_valscore_lfc_single(source_csv)
    chart.save(
        f"{args.report_folder}/altair_snippets/gene_pair_valscorescombovssingle.html"
    )

    ## MAKE REPORT
    html_pairs = populate_genelist(source)

    create_report(
        source_csv,
        startpage_stats,
        # guide_eff_priors_str,
        html_pairs,
        args.report_folder,
    )


if __name__ == "__main__":
    run()

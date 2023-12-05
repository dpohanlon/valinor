#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import jinja2
from datetime import date
import h5py
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil
import os
import re
import argparse
import json

alt.data_transformers.disable_max_rows()

fontsizes = [18, 16, 14]
colorblindfr = {
    "main": ["#56b3e9", "#e0d316", "#0072b2", "#e69d00", "#cc79a7"],
    "additional": ["#EC681E", "#009e74", "#000000"],
}
colors_palette = colorblindfr["main"]

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


def calc_overdisp(scoreData_combo, scoreData_single):
    # calculate the overdispersion based on the variance in counts across replicates for a given guide pair in a given cell line
    numeric_columns = scoreData_combo.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    numeric_columns = numeric_columns + ["GuidePair"]
    reps = (
        scoreData_combo[numeric_columns]
        .groupby(["GuidePair", "cell_line_index"])
        .agg(mean=("value", np.mean), std=("value", np.std))
        .reset_index()
    )
    reps["var"] = reps["std"] ** 2
    reps["overdispersion"] = reps["var"] / reps["mean"]
    scoreData_combo = scoreData_combo.merge(
        reps[["GuidePair", "cell_line_index", "overdispersion"]],
        on=["GuidePair", "cell_line_index"],
    )

    numeric_columns = scoreData_single.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    numeric_columns = numeric_columns + ["GuidePair", "SingletonGuide"]
    reps = (
        scoreData_single.groupby(["GuidePair", "SingletonGuide", "cell_line_index"])
        .agg(mean=("value", np.mean), std=("value", np.std))
        .reset_index()
    )
    reps["var"] = reps["std"] ** 2
    reps["overdispersion"] = reps["var"] / reps["mean"]
    scoreData_single = scoreData_single.merge(
        reps[["GuidePair", "SingletonGuide", "cell_line_index", "overdispersion"]],
        on=["GuidePair", "SingletonGuide", "cell_line_index"],
    )

    return (scoreData_combo, scoreData_single)


def combine_single_combo(scoreData_combo, scoreData_single_NEav, combo_cols):
    colstorename = {i: i + "_s" for i in scoreData_single_NEav.columns}

    scoreData_combined = (
        scoreData_combo[combo_cols]
        .merge(
            scoreData_single_NEav.rename(columns=colstorename),
            left_on=["guide1", "cell_line", "replicate"],
            right_on=["SingletonGuide_s", "cell_line_s", "replicate_s"],
            how="left",
        )
        .rename(columns={i: i + "_1" for i in colstorename.values()})
        .merge(
            scoreData_single_NEav.rename(columns=colstorename),
            left_on=["guide2", "cell_line", "replicate"],
            right_on=["SingletonGuide_s", "cell_line_s", "replicate_s"],
            how="left",
        )
        .rename(columns={i: i + "_2" for i in colstorename.values()})
    )

    return scoreData_combined


def calc_deltaLFC(scoreData_combined):
    scoreData_combined["deltaLFC"] = scoreData_combined["lfc"] - (
        scoreData_combined["lfc_s_1"] + scoreData_combined["lfc_s_2"]
    )
    return scoreData_combined


def sample_genepairs(scoreData_combined):
    # pick the top SL pairs based on rank
    topsl = (
        scoreData_combined[["genePair", "valinor_score"]]
        .drop_duplicates()
        .sort_values("valinor_score")
        .genePair[:100]
        .unique()
    )
    # then randomly select pairs
    selectpairs = scoreData_combined.genePair.sample(300).unique()
    selectpairs = set(np.concatenate((selectpairs, topsl)))
    return selectpairs


def create_valinorrun_settings(valinorsettings):
    valinorrun_settings = [
        f"{i}: {j}" for i, j in valinorsettings.items() if "File" not in i
    ]
    valinorrun_settings = "<br>".join(valinorrun_settings)

    return valinorrun_settings


def create_overview_stats(combs_attr, combs_attr_s):
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
    ]
    overview_stats_combs = "<br>".join(overview_stats_combs)

    overview_stats_s = [
        "{} genes".format(combs_attr_s["n_genes"]),
        "{} guides".format(combs_attr_s["n_guides"]),
    ]

    overview_stats_s = "<br>".join(overview_stats_s)

    return (overview_stats, overview_stats_combs, overview_stats_s)


def create_file_settings(mergedfile, combs_attr, combs_attr_s, valinorsettings):
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


def create_startpage_stats(mergedfile, combs_attr_f, combs_attr_s_f, valinorrun_json):
    with open(valinorrun_json, "r") as file:
        valinorsettings = json.load(file)

    with open(combs_attr_f, "r") as file:
        combs_attr_df = json.load(file)

    with open(combs_attr_s_f, "r") as file:
        combs_attr_s_df = json.load(file)

    overview_stats, overview_stats_combs, overview_stats_s = create_overview_stats(
        combs_attr_df, combs_attr_s_df
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

    return startpage_stats


def produce_lossfunc_examples():
    # produce example of a good loss function

    x = np.arange(0, 1000)
    y = np.exp(-1 * x * 0.05)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(x, y)
    ax.set_xlabel("Steps", fontsize=fontsizes[1])
    ax.set_ylabel("Loss", fontsize=fontsizes[1])
    ax.tick_params(axis="both", labelsize=fontsizes[2])
    fig.tight_layout()
    fig.savefig("plots/good_lossfunction.svg", bbox_inches="tight")
    plt.close(fig)

    x = np.arange(0, 1000)
    y = np.exp(-1 * x * 0.005)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(x, y)
    ax.set_xlabel("Steps", fontsize=fontsizes[1])
    ax.set_ylabel("Loss", fontsize=fontsizes[1])
    ax.tick_params(axis="both", labelsize=fontsizes[2])
    fig.tight_layout()
    fig.savefig("plots/bad_lossfunction.svg", bbox_inches="tight")
    plt.close(fig)


def copy_loss_plot(target_file):
    new_path = os.path.join("plots/", "valinor_loss.svg")
    shutil.copy(target_file, new_path)


def produce_modelfit_examples():
    # create images for good and bad model performance
    k = 10  # Number of successes
    p = 0.01  # Probability of success on each trial

    rng = np.random.default_rng()
    data_val = rng.negative_binomial(k, p, 10000)

    rng = np.random.default_rng(31)
    model_val = rng.negative_binomial(k, p, 10000)

    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig("plots/good_modelfit.svg", bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(31)
    model_val = rng.negative_binomial(k, 0.02, 10000)

    fig, ax = plot_hist({"model": model_val, "data": data_val}, xlabel="Counts")
    fig.savefig("plots/bad_modelfit.svg", bbox_inches="tight")
    plt.close(fig)


def get_prior_val(variable, param="loc"):
    match = re.search(r"{}\s*=\s*([\d.]+)".format(param), variable)
    if match:
        value = float(match.group(1))
        return value
    else:
        raise Exception("regular expression no found")


def produce_paramfit_guideff_hyper(
    source,
    # priors,
):
    if "guide_eff_mean_1_mean" in source.columns:
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
        text = "Could not be displayed because\ncolumns such as 'guide_eff_mean_1_mean' were missing"
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)

        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(
        "plots/parameterfits_guideeffs_metahier_combo.svg",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)


def produce_paramfit_guideff(
    source_json,
):
    selection = alt.selection_single(fields=["cell_line"], bind="legend", empty="none")

    # Base chart with only the outline (using mark_line)
    base_line = (
        alt.Chart(source_json)
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
        alt.Chart(source_json)
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
        alt.Chart(source_json)
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
        alt.Chart(source_json)
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
    final_chart.save("altair_snippets/parameterfits_guideeffs_devprior_combo.html")


def produce_datastats_countdistr(
    source_json,
):
    # plot distribution of counts
    # color by cell lines
    # very slow for the whole dataset, but renders!
    selection = alt.selection_single(fields=["cell_line"], bind="legend")

    base = (
        alt.Chart(source_json)
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

    base.save("altair_snippets/data_stats_counts.html")


def produce_datastats_overdisp(
    source,
    source_json,
):
    # use the overdispersion calculated from the replicate data rather than the mv parameter fitted by Valinor
    selection = alt.selection_single(fields=["cell_line"], bind="legend")
    min_od = np.min([source.overdispersion.min()])
    max_od = np.max(
        [
            np.quantile(source.overdispersion, 0.99)
            if source.overdispersion.max() - np.quantile(source.overdispersion, 0.99)
            > 100
            else source.overdispersion.max()
        ]
    )

    base_line = (
        alt.Chart(source_json)
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
        alt.Chart(source_json)
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

    chart.save("altair_snippets/data_stats_dispersion.html")


def produce_diagnplots_dlfc_valscore(
    source_json,
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
        alt.Chart(source_json)
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
            tooltip=tooltip_display
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

    combined_chart.save("altair_snippets/diagnostic_plots_dLFCvsValinor.html")


def produce_diagnplots_kogrowths(
    source_json,
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
        alt.Chart(source_json)
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
    chart.save("altair_snippets/diagnostic_plots_kogrowhstd.html")


def produce_diagnplots_singletons(
    source_json,
):
    tooltip = [
        "gene1:O",
        "cell_line:O",
        "valinor_score_s_s_1:Q",
        "lfc_s_1:Q",
        "gene_ko_growth_1_mean:Q",
        "gene_ko_growth_1_std:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    # Main Chart
    main_chart = (
        alt.Chart(source_json)
        .mark_circle(size=100)
        .encode(
            x=alt.X("valinor_score_s_s_1:Q", title=naming_cols["valinor_score_s_s_1"]),
            y=alt.Y("lfc_s_1:Q", title=naming_cols["lfc_s_1"]),
            color=alt.Color(
                "gene_ko_growth_1_std:Q",
                scale=alt.Scale(scheme="blues", reverse=True),
                legend=alt.Legend(title=naming_cols["gene_ko_growth_1_std"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            valinor_score_s_s_1="mean(valinor_score_s_s_1)",
            lfc_s_1="mean(lfc_s_1)",
            gene_ko_growth_1_std="mean(gene_ko_growth_1_std)",
            gene_ko_growth_1_mean="mean(gene_ko_growth_1_mean)",
            groupby=["gene1", "cell_line"],
        )
    )

    tooltip = [
        "gene2:O",
        "cell_line:O",
        "valinor_score_s_s_2:Q",
        "lfc_s_2:Q",
        "gene_ko_growth_2_mean:Q",
        "gene_ko_growth_2_std:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    # Main Chart for Gene 2
    main_chart_gene2 = (
        alt.Chart(source_json)
        .mark_circle(size=100)
        .encode(
            x=alt.X("valinor_score_s_s_2:Q", title=naming_cols["valinor_score_s_s_2"]),
            y=alt.Y("lfc_s_2:Q", title=naming_cols["lfc_s_2"]),
            color=alt.Color(
                "gene_ko_growth_2_std:Q",
                scale=alt.Scale(scheme="blues", reverse=True),
                legend=alt.Legend(title=naming_cols["gene_ko_growth_2_std"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            valinor_score_s_s_2="mean(valinor_score_s_s_2)",
            lfc_s_2="mean(lfc_s_2)",
            gene_ko_growth_2_std="mean(gene_ko_growth_2_std)",
            gene_ko_growth_2_mean="mean(gene_ko_growth_2_mean)",
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

    combined_chart.save("altair_snippets/diagnostic_plots_singleLFCvsvalscore.html")


def produce_diagnplots_guideeff_vslfc(
    source_json,
):
    tooltip = [
        "guide1:O",
        "gene1:O",
        "cell_line:O",
        "lfc_s_1:Q",
        "guide_eff_1_mean:Q",
        "valinor_score_s_s_1:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]
    chart1 = (
        alt.Chart(source_json)
        .mark_circle(size=100)
        .encode(
            x=alt.X("lfc_s_1:Q", title=naming_cols["lfc_s_1"]),
            y=alt.Y("guide_eff_1_mean:Q", title=naming_cols["guide_eff_1_mean"]),
            color=alt.Color(
                "guide_eff_1_mean:Q",
                scale=alt.Scale(scheme="blues", reverse=False),
                legend=alt.Legend(title=naming_cols["guide_eff_1_mean"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            lfc_s_1="mean(lfc_s_1)",
            guide_eff_1_mean="mean(guide_eff_1_mean)",
            valinor_score_s_s_1="mean(valinor_score_s_s_1)",
            groupby=["guide1", "gene1", "cell_line"],
        )
    )

    tooltip = [
        "guide2:O",
        "gene2:O",
        "cell_line:O",
        "lfc_s_2:Q",
        "guide_eff_2_mean:Q",
        "valinor_score_s_s_2:Q",
    ]
    # Map original field names to custom labels for display
    tooltip_display = [
        alt.Tooltip(field, title=naming_cols.get(field.split(":")[0]))
        for field in tooltip
    ]

    chart2 = (
        alt.Chart(source_json)
        .mark_circle(size=100)
        .encode(
            x=alt.X("lfc_s_2:Q", title=naming_cols["lfc_s_2"]),
            y=alt.Y("guide_eff_2_mean:Q", title=naming_cols["guide_eff_2_mean"]),
            color=alt.Color(
                "guide_eff_2_mean:Q",
                scale=alt.Scale(scheme="blues", reverse=False),
                legend=alt.Legend(title=naming_cols["guide_eff_2_mean"]),
            ),
            tooltip=tooltip_display,
        )
        .transform_aggregate(
            lfc_s_2="mean(lfc_s_2)",
            guide_eff_2_mean="mean(guide_eff_2_mean)",
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
    combined_chart.save("altair_snippets/diagnostic_plots_singleLFCvsefficiency.html")


def produce_hitprior_lfc_vs_valscore_avg(
    source,
    source_json,
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
        alt.Chart(source_json)
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

    plot.save("altair_snippets/hit_prioritisation_cell_avg.html")


def produce_hitprior_lfc_vs_valscore_hist(
    source,
    source_json,
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
            "x": [xrange[0], tmp["valinor_score"].min() + 3 * x_diff],
            "y": [yrange[0], tmp["lfc"].min() + 3 * y_diff],
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
        alt.Chart(source_json)
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
        alt.Chart(source_json)
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

    chart.save("altair_snippets/hit_prioritisation_lfcvsval_histo.html")


def produce_hitprior_kogrowths(
    source_json,
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
        alt.Chart(source_json)
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
        alt.Chart(source_json)
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
    chart.save("altair_snippets/hit_prioritisation_growthdefecscatter.html")


def produce_geneview_valscore_rank(
    source,
    source_json,
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
        alt.Chart(source_json)
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

    base.save("altair_snippets/gene_pair_valscore_percln.html")


def produce_geneview_valscore_lfc(
    source_json,
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
        alt.Chart(source_json)
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

    base.save("altair_snippets/gene_pair_valscorevslfc.html")


def produce_geneview_growth_lfc(
    source_json,
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
        alt.Chart(source_json)
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

    base.save("altair_snippets/gene_pair_valscorescombovssingle.html")


def produce_geneview_valscore_lfc_single(
    source_json,
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
        alt.Chart(source_json)
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
    base.save("altair_snippets/gene_pair_valscoresinglevslfc.html")


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

    for p in source.sort_values("genePair")["genePair"].unique():
        html_pairs_list.append(f'<option value="{p}" label = "{p}">')

    html_pairs = "\n".join(html_pairs_list)
    return html_pairs


def create_report(
    source_json,
    startpage_stats,
    # guide_eff_priors_str,
    html_pairs,
):
    # postprocess the saved html to it works in the final report
    # not really elegant constantly writing and loading files
    # but not sure how to do it in place
    hit_prioritisation_cell_avg_plot = postprocess_altairhtml(
        "altair_snippets/hit_prioritisation_cell_avg.html",
        source_json,
        "hit_prioritisation_cell_avg_plot",
    )
    hit_prioritisation_lfcvsval_histo = postprocess_altairhtml(
        "altair_snippets/hit_prioritisation_lfcvsval_histo.html",
        source_json,
        "hit_prioritisation_lfcvsval_histo",
    )
    hit_prioritisation_growthdefecscatter = postprocess_altairhtml(
        "altair_snippets/hit_prioritisation_growthdefecscatter.html",
        source_json,
        "hit_prioritisation_growthdefecscatter",
    )

    parameterfits_guideeffs_devprior_combo = postprocess_altairhtml(
        "altair_snippets/parameterfits_guideeffs_devprior_combo.html",
        source_json,
        "parameterfits_guideeffs_devprior_combo",
    )

    datastats_count_plot = postprocess_altairhtml(
        "altair_snippets/data_stats_counts.html",
        source_json,
        "datastats_count_plot",
    )
    datastats_disp_plot = postprocess_altairhtml(
        "altair_snippets/data_stats_dispersion.html",
        source_json,
        "datastats_disp_plot",
    )

    diagnplots_dlfcval_plot = postprocess_altairhtml(
        "altair_snippets/diagnostic_plots_dLFCvsValinor.html",
        source_json,
        "diagnplots_dlfcval_plot",
    )
    diagnplots_kogrowthstd_plot = postprocess_altairhtml(
        "altair_snippets/diagnostic_plots_kogrowhstd.html",
        source_json,
        "diagnplots_kogrowthstd_plot",
    )
    diagnplots_singlelfcval_plot = postprocess_altairhtml(
        "altair_snippets/diagnostic_plots_singleLFCvsvalscore.html",
        source_json,
        "diagnplots_singlelfcval_plot",
    )
    diagnplots_singleLFCeffic_plot = postprocess_altairhtml(
        "altair_snippets/diagnostic_plots_singleLFCvsefficiency.html",
        source_json,
        "diagnplots_singleLFCeffic_plot",
    )

    genepair_valinorscore_plot = postprocess_altairhtml(
        "altair_snippets/gene_pair_valscore_percln.html",
        source_json,
        "genepair_valinorscore_plot",
        select_on="genePair",
    )
    genepair_valinorscorevslfc_plot = postprocess_altairhtml(
        "altair_snippets/gene_pair_valscorevslfc.html",
        source_json,
        "genepair_valinorscorevslfc_plot",
        select_on="genePair",
    )
    genepair_valscorescombovssingle_plot = postprocess_altairhtml(
        "altair_snippets/gene_pair_valscorescombovssingle.html",
        source_json,
        "genepair_valscorescombovssingle_plot",
        select_on="genePair",
    )
    genepair_valscoresinglevslfc_plot = postprocess_altairhtml(
        "altair_snippets/gene_pair_valscoresinglevslfc.html",
        source_json,
        "genepair_valscoresinglevslfc_plot",
        select_on="genePair",
    )

    # Template handling
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=""))
    # template = env.get_template('Bootstrap, from Twitter.html')
    # html = template.render()
    template = env.get_template("template_tabs.html")
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
    with open("report.html", "w") as f:
        f.write(html)


def create_necessary_folders(folders):
    # Create folders if they don't exist
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument(
        "--combfile", help="processed table combining data and Valinor output"
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

    args = parser.parse_args()

    folders = [
        "input",
        "plots",
        "altair_snippets",
    ]
    create_necessary_folders(folders)

    scoreData_combined = pd.read_parquet(args.combfile)

    if args.subsetSLpairs:
        selectpairs = sample_genepairs(scoreData_combined)

        source = scoreData_combined.loc[
            scoreData_combined.genePair.isin(selectpairs)
        ].copy()
    else:
        source = scoreData_combined.copy()

    source.to_csv("input/report_input.csv", index=False)
    source_json = "input/report_input.csv"

    ## HOME
    # Overview stats
    combs_attr_f = "input/combs_attr.json"
    combs_attr_s_f = "input/combs_attr_s.json"

    startpage_stats = create_startpage_stats(
        args.combfile, combs_attr_f, combs_attr_s_f, args.valinorConfigFile
    )

    # Loss Function
    produce_lossfunc_examples()
    copy_loss_plot(args.valinorLossFile)

    # Model Fit
    produce_modelfit_examples()

    # guide_eff_priors_str = "<br>".join(guide_eff_priors_str)
    produce_paramfit_guideff_hyper(source)
    produce_paramfit_guideff(source_json)

    ## DATA STATS
    # Count distributions
    produce_datastats_countdistr(source_json)
    # Count overdispersion
    produce_datastats_overdisp(source, source_json)

    ## DIAGNOSTIC PLOTS
    # dLFC vs. Valinor Score
    produce_diagnplots_dlfc_valscore(source_json)
    # ko_growth_12 vs. gene_ko_growth_12_std
    produce_diagnplots_kogrowths(source_json)
    # singleton LFC vs. singleton gene effect
    produce_diagnplots_singletons(source_json)
    # guide eff vs. guide lfc Singletons
    produce_diagnplots_guideeff_vslfc(source_json)

    ## HIT PRIORITISATION
    # LFC vs. Valinor Score - genePair averaged
    produce_hitprior_lfc_vs_valscore_avg(source, source_json)
    # LFC vs. Valinor score - Number of cell lines
    produce_hitprior_lfc_vs_valscore_hist(source, source_json)
    # growth 1 vs. growth 12 vs. growth 2
    produce_hitprior_kogrowths(source_json)

    ## GENE PAIR VIEW
    # Valinor score per cell line
    produce_geneview_valscore_rank(source, source_json)
    # Combination Valinor score vs. LFCs
    produce_geneview_valscore_lfc(source_json)
    # growth 12 vs singleton growth
    produce_geneview_growth_lfc(source_json)
    # Singleton Valinor Scores vs. LFCs
    produce_geneview_valscore_lfc_single(source_json)

    ## MAKE REPORT
    html_pairs = populate_genelist(source)

    create_report(
        source_json,
        startpage_stats,
        # guide_eff_priors_str,
        html_pairs,
    )


if __name__ == "__main__":
    main()

import argparse

import jax

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np
import pandas as pd

from tqdm import tqdm

from jax import random
import numpyro

from numpyro.infer import Predictive, SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal, AutoLowRankMultivariateNormal

from numpyro.handlers import seed, trace, substitute

import optax

from valinor import models
from valinor.guides import (
    valinor_full_guide,
    valinor_singles_guide,
    valinor_controls_guide,
)
from valinor.utils import (
    getIndices,
    calculateLengths,
    configArgs,
    saveModelParams,
    loadPriors,
    getBatchData,
    configure_custom_init,
    subset_for_mcmc,
)
from valinor.preprocessing import prepareData, getDeltaLFC
from valinor.postprocessing import (
    sampleParams,
    createDataFrame,
    samplePosteriorPredictive,
)
from valinor.plotting import plotDiagPlots

from typing import Dict, List, Tuple, Any

import json

from pprint import pprint

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib import rcParams

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

gpu_available = any(device.platform == "gpu" for device in jax.devices())

if gpu_available:
    numpyro.set_platform("gpu")


def make_minibatches(N, B, key):
    idx = jax.random.permutation(key, N)
    for start in range(0, N, B):
        yield idx[start : start + B]


def get_model_sites(model, *args, **kwargs):
    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace(*args, **kwargs)
    sites = list(model_trace.keys())

    # Don't sample obs as we don't need them for this part
    sites = filter(lambda x: not ("obs" in x), sites)

    return list(sites)


def obs_config(mode, data, config):
    has_ctrl = (
        "controls" in data["final"] and data["final"]["controls"] is not None
    ) and not config.get("no_controls", True)
    has_sko = (
        "singletons" in data["final"] and data["final"]["singletons"] is not None
    ) and not config.get("no_singletons", False)
    has_dko = (
        "combinations" in data["final"] and data["final"]["combinations"] is not None
    ) and not config.get("only_singletons", False)

    n_ctrl = int(has_ctrl) and len(data["final"]["controls"]) or 0
    n_sko = int(has_sko) and len(data["final"]["singletons"]) or 0
    n_dko = int(has_dko) and len(data["final"]["combinations"]) or 0

    base = n_dko if n_dko > 0 else n_sko
    w_ctrl = base / max(n_ctrl, 1)
    w_sko = base / max(n_sko, 1)
    w_dko = base / max(n_dko, 1)

    print(w_ctrl, w_sko, w_dko)

    if mode == "controls":
        return dict(
            use_ctrl=has_ctrl,
            use_sko=False,
            use_dko=False,
            w_ctrl=w_ctrl,
            w_sko=1.0,
            w_dko=1.0,
        )
    if mode == "singles":
        return dict(
            use_ctrl=False,
            use_sko=has_sko,
            use_dko=False,
            w_ctrl=1.0,
            w_sko=w_sko,
            w_dko=1.0,
        )
    if mode == "dko":
        return dict(
            use_ctrl=False,
            use_sko=False,
            use_dko=has_dko,
            w_ctrl=1.0,
            w_sko=1.0,
            w_dko=w_dko,
        )
    if "singles" in mode and "dko" in mode:
        return dict(
            use_ctrl=False,
            use_sko=has_sko,
            use_dko=has_dko,
            w_ctrl=1.0,
            w_sko=w_sko,
            w_dko=w_dko,
        )
    if mode == "full":
        return dict(
            use_ctrl=has_ctrl,
            use_sko=has_sko,
            use_dko=has_dko,
            w_ctrl=w_ctrl,
            w_sko=w_sko,
            w_dko=w_dko,
        )
    raise ValueError(f"Unknown mode {mode}")


def initialize_svi(model, guide, lr, nParticles, nEpochs):
    warmup = int(0.25 * nEpochs)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=nEpochs - warmup,
        end_value=lr * 0.05,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )

    optimizer = numpyro.optim.optax_to_numpyro(tx)

    svi = SVI(
        model,
        guide,
        optimizer,
        loss=Trace_ELBO(num_particles=nParticles),  # maybe mean field is okay?
    )

    return svi


def run_svi(
    svi,
    prng_key,
    epochs,
    data,
    lengths,
    indices,
    prior_params,
    init_params=None,
    **kwargs,
):
    return svi.run(
        prng_key,
        epochs,
        init_params=init_params,
        data=data,
        lengths=lengths,
        indices=indices,
        prior_params=prior_params,
        **kwargs,
    )


def save_results(params, config, outputDir):
    saveModelParams(params, f"{outputDir}{config['paramsFileName']}")
    with open(f"valinorrun_{config['name']}.json", "w") as outfile:
        json.dump(config, outfile)


# Save the posterior predictive 'obs' independent of batches, etc, in one go.
# Sample only obs sites, rather than the rest, to avoid running out of memory. Can also batch this separately over num_samples.
def save_posterior_predictive(
    model,
    guide,
    params,
    num_samples,
    sites,
    data,
    lengths,
    indices,
    prior_params,
    config,
    annotation,
    **kwargs,
):
    sites = list(filter(lambda x: "obs" in x, sites))

    predictive = Predictive(
        model,
        guide=guide,
        params=params,
        num_samples=num_samples,
        return_sites=sites,
        parallel=False,
    )

    if config["only_singletons"] == False:
        args = {
            "no_singletons": config["no_singletons"],
            "only_singletons": config["only_singletons"],
            "no_controls": config["no_controls"],
            "guide_config": config["guide_config"],
        }
    else:
        args = {"guide_config": config["guide_config"]}

    with jax.default_device(jax.devices("cpu")[0]):
        samples = predictive(
            random.PRNGKey(42),
            data=data,
            lengths=lengths,
            indices=indices,
            prior_params=prior_params,
            # **args,
            **kwargs,
            predict=True,
        )

    samplePosteriorPredictive(samples, indices, annotation=annotation)


def sample_posterior(
    predictive, config, data, lengths, indices, prior_params, batch=False, **kwargs
):
    if config["only_singletons"] == False:
        args = {
            "no_singletons": config["no_singletons"],
            "only_singletons": config["only_singletons"],
            "no_controls": config["no_controls"],
            "guide_config": config["guide_config"],
        }
    else:
        args = {"guide_config": config["guide_config"]}

    if not batch:
        with jax.default_device(jax.devices("cpu")[0]):
            # Non-batch case: directly sample from the posterior
            samples = predictive(
                random.PRNGKey(42),
                data=data,
                lengths=lengths,
                indices=indices,
                prior_params=prior_params,
                # **args,
                **kwargs,
                predict=True,
            )

            sampledParams = sampleParams(
                samples,
                indices,
                prior_params,
                config["alternateLH"],
                "gene_effect_means" in prior_params,
            )

        combsDF = (
            createDataFrame(sampledParams["combs"])
            if ("combs" in sampledParams)
            else None
        )
        singlesDF = (
            createDataFrame(sampledParams["singles"])
            if ("singles" in sampledParams)
            else None
        )

        return combsDF, singlesDF

    # Batched case
    if kwargs.get("use_dko", False):
        total = len(data["final"]["combinations"])
        head = "dko"
    elif kwargs.get("use_sko", False):
        total = len(data["final"]["singletons"])
        head = "sko"
    else:
        total = len(data["final"]["controls"])
        head = "ctrl"

    n_batches = int(np.ceil(total / config["batch_size"]))

    combsDFs, singlesDFs = [], []

    for i in tqdm(range(n_batches)):
        start_idx = i * config["batch_size"]
        end_idx = (i + 1) * config["batch_size"]

        batch_data, batch_indices = getBatchData(
            data, indices, start_idx, end_idx, head=head
        )

        with jax.default_device(jax.devices("cpu")[0]):
            samples = predictive(
                random.PRNGKey(42),
                data=batch_data,
                lengths=lengths,
                indices=batch_indices,
                prior_params=prior_params,
                **kwargs,
                predict=True,
            )

            sampledParams = sampleParams(
                samples,
                batch_indices,
                prior_params,
                config["alternateLH"],
                "gene_effect_means" in prior_params,
            )

        if "combs" in sampledParams:
            combsDFs.append(createDataFrame(sampledParams["combs"]))
        if "singles" in sampledParams:
            singlesDFs.append(createDataFrame(sampledParams["singles"]))

    combsDF = pd.concat(combsDFs) if len(combsDFs) > 0 else None
    singlesDF = pd.concat(singlesDFs) if len(singlesDFs) > 0 else None

    return combsDF, singlesDF


def save_posterior_samples(combsDF, singlesDF, config, outputDir, singlesStage="Only"):
    if (config["only_singletons"] == False) and (combsDF is not None):
        combsName = "OnlyCombsModel" if config["no_singletons"] else "CombsModel"
        combsDF.to_parquet(
            f"{outputDir}{combsName}.pq"
            if config["name"] is None
            else f"{outputDir}{combsName}_{config['name']}.pq"
        )
    if (config["no_singletons"] == False) and (singlesDF is not None):
        singlesDF.to_parquet(
            f"{outputDir}{singlesStage}SinglesModel.pq"
            if config["name"] is None
            else f"{outputDir}{singlesStage}SinglesModel_{config['name']}.pq"
        )


def runValinor(lengths, indices, prior_params, data, config):
    if config["zi"] == False:
        config["zi"] = None

    indices["guide_pair_init_idx"] = indices["guide_pair_idx"]

    prng_key = random.PRNGKey(42)

    outputDir = (
        f"{config['outputDir'].rstrip('/')}/" if config["outputDir"] != "" else ""
    )

    fit_mode = config.get("fit_mode", "full")

    print("Fit mode:", fit_mode)

    use_ctrl_d = (
        "controls" in data["final"] and data["final"]["controls"] is not None
    ) and not config.get("no_controls", True)
    use_sko_d = (
        "singletons" in data["final"] and data["final"]["singletons"] is not None
    ) and not config.get("no_singletons", False)
    use_dko_d = (
        "combinations" in data["final"] and data["final"]["combinations"] is not None
    ) and not config.get("only_singletons", False)

    # Some combination of these is screwing up the likelihood?

    init_params_common = {}
    if config.get("zi", False):
        init_params_common["p_zi"] = prior_params.get("p_zi", None)
    if config.get("zi", False) and not config.get("zi_ns", False) and use_sko_d:
        init_params_common["p_zi_s"] = prior_params.get("p_zi_s", None)

    if "dLFC" in prior_params:
        # init_params_common["gene_pair_ko_growth_raw"] = prior_params["dLFC"]
        init_params_common["gene_pair_ko_growth_raw"] = prior_params["dLFC"] - np.mean(
            prior_params["dLFC"]
        )
    if "init_count_vals" in prior_params and use_dko_d:
        init_params_common["guide_init_count"] = prior_params["init_count_vals"]
    if "init_count_s_vals" in prior_params and use_sko_d:
        init_params_common["guide_init_count_s"] = prior_params["init_count_s_vals"]
    if "initial" in data and "controls" in data["initial"] and use_ctrl_d:
        init_params_common["guide_init_count_c"] = prior_params["init_count_c_vals"]

    custom_init = configure_custom_init(init_params_common)

    valinor_model = models.valinorHierarchy
    valinor_guide = AutoLowRankMultivariateNormal(
        models.valinorHierarchy, init_loc_fn=custom_init(), rank=64
    )

    common_config = {
        "guide_config": config["guide_config"],
        "zi": config["zi"],
        "zi_s": config["zi"] and not config["zi_ns"],
        "no_singletons": config["no_singletons"],
        "only_singletons": config["only_singletons"],
        "no_controls": config["no_controls"],
        "alternate": config["alternateLH"],
    }

    common_svi_config = {
        "stable_update": config["stable_update"],
    }

    print("Fitting")

    if fit_mode == "controls" or fit_mode == "full":
        print("Fitting controls")

        svi_controls = initialize_svi(
            valinor_model,
            valinor_guide,
            config["lr"],
            config["n_particles"],
            config["epochs"],
        )

        controls_args = {
            **common_config,
            **obs_config("controls", data, config),
        }

        state_c = run_svi(
            svi_controls,
            prng_key,
            config["epochs"],
            data,
            lengths,
            indices,
            prior_params,
            **controls_args,
            **common_svi_config,
            predict=False,
        )

        params_c = state_c.params

        plotDiagPlots(state_c, name=f"{config['name']}_controls", outputDir=outputDir)

        sites_from_model = get_model_sites(
            valinor_model, data, lengths, indices, prior_params, **controls_args
        )

        predictive = Predictive(
            valinor_model,
            guide=valinor_guide,
            params=params_c,
            num_samples=config["nSamples"],
            return_sites=sites_from_model,
            parallel=False,
        )

        samples = predictive(
            prng_key,
            data=data,
            lengths=lengths,
            indices=indices,
            prior_params=prior_params,
            **controls_args,
            predict=True,
        )

        sampleVars = [
            "cell_line_growth",
            "cell_lines",
            "raw_mv_cell_line",
            "gene_std",
        ]  # , 'library_bias']#, 'negative_control_bias']

        controlsDF = pd.DataFrame({n: np.mean(samples[n], 0) for n in sampleVars})

        controlsDF.to_parquet(
            f"{outputDir}controlsModel.pq"
            if config["name"] is None
            else f"{outputDir}controlsModel_{config['name']}.pq"
        )

        annotation = "only" if fit_mode == "controls" else "first-stage"

        save_posterior_predictive(
            valinor_model,
            valinor_guide,
            params_c,
            config["nSamples"],
            sites_from_model,
            data,
            lengths,
            indices,
            prior_params,
            config,
            annotation,
            **controls_args,
        )

    if "singles" in fit_mode or fit_mode == "full":
        print("Fitting singles")

        prng_key, _ = random.split(prng_key)

        svi_singles = initialize_svi(
            valinor_model,
            valinor_guide,
            config["lr"],
            config["n_particles"],
            config["epochs"],
        )

        singles_args = {
            **common_config,
            **obs_config("singles", data, config),
        }

        state_s = run_svi(
            svi_singles,
            prng_key,
            config["epochs"],
            data,
            lengths,
            indices,
            prior_params,
            init_params=params_c if (fit_mode == "full" and use_ctrl_d) else None,
            **singles_args,
            **common_svi_config,
            predict=False,
        )

        params_s = state_s.params

        plotDiagPlots(state_s, name=f"{config['name']}_singles", outputDir=outputDir)

        sites_from_model = get_model_sites(
            valinor_model, data, lengths, indices, prior_params, **singles_args
        )

        predictive = Predictive(
            valinor_model,
            guide=valinor_guide,
            params=params_s,
            num_samples=config["nSamples"],
            return_sites=sites_from_model,
            parallel=False,
        )

        combsDF, singlesDF = sample_posterior(
            predictive,
            config,
            data,
            lengths,
            indices,
            prior_params,
            batch=config["batch_sample"],
            **singles_args,
        )

        stage = "FirstStage" if fit_mode in ("full", "singles-dko") else "Only"

        save_posterior_samples(
            combsDF, singlesDF, config, outputDir, singlesStage=stage
        )

        save_posterior_predictive(
            valinor_model,
            valinor_guide,
            params_s,
            config["nSamples"],
            sites_from_model,
            data,
            lengths,
            indices,
            prior_params,
            config,
            stage,
            **singles_args,
        )

        singles_vals = valinor_guide.median(params_s)

    if "dko" in fit_mode or fit_mode == "full":
        print("Fitting dko")

        custom_init = configure_custom_init(init_params_common)
        dko_guide = AutoLowRankMultivariateNormal(
            models.valinorHierarchy, init_loc_fn=custom_init(), rank=64
        )

        prng_key, _ = random.split(prng_key)

        svi_dko = initialize_svi(
            valinor_model,
            dko_guide,
            config["lr"],
            config["n_particles"],
            config["epochs"],
        )

        dko_args = {
            **common_config,
            **obs_config(fit_mode, data, config),
        }

        # ---- minibatched DKO training ----
        N = int(lengths["len_guide_pairs"])
        B = int(config["batch_size"])  # reuse CLI --batch-size

        prng_key, key_init = random.split(prng_key)
        init_batch_iter = make_minibatches(N, B, key_init)
        init_batch = next(init_batch_iter)

        init_params = params_s if (fit_mode == "full" and use_sko_d) else None

        state_d = svi_dko.init(
            prng_key,
            data=data,
            lengths=lengths,
            indices=indices,
            prior_params=prior_params,
            init_params=init_params,
            predict=False,
            dko_subsample_size=B,
            dko_batch_idx=init_batch,
            **dko_args,
        )

        # one full pass over all DKO pairs per epoch
        for _ in tqdm(range(config["epochs"]), desc="DKO epochs"):
            prng_key, key_epoch = random.split(prng_key)
            for batch in make_minibatches(N, B, key_epoch):
                if config.get("stable_update", False):
                    state_d, _ = svi_dko.stable_update(
                        state_d,
                        data=data,
                        lengths=lengths,
                        indices=indices,
                        prior_params=prior_params,
                        predict=False,
                        dko_subsample_size=B,
                        dko_batch_idx=batch,
                        **dko_args,
                    )
                else:
                    state_d, _ = svi_dko.update(
                        state_d,
                        data=data,
                        lengths=lengths,
                        indices=indices,
                        prior_params=prior_params,
                        predict=False,
                        dko_subsample_size=B,
                        dko_batch_idx=batch,
                        **dko_args,
                    )
        # -----------------------------------

        params_d = state_d.params

        plotDiagPlots(state_d, name=f"{config['name']}_combs", outputDir=outputDir)

        sites_from_model = get_model_sites(
            valinor_model, data, lengths, indices, prior_params, **dko_args
        )

        predictive = Predictive(
            valinor_model,
            guide=dko_guide,
            params=params_d,
            num_samples=config["nSamples"],
            return_sites=sites_from_model,
        )

        combsDF, singlesDF = sample_posterior(
            predictive,
            config,
            data,
            lengths,
            indices,
            prior_params,
            batch=config["batch_sample"],
            **dko_args,
        )

        stage = "SecondStage"

        save_posterior_samples(
            combsDF, singlesDF, config, outputDir, singlesStage=stage
        )

        annotation = stage if config["fit_mode"] in ("full", "singles-dko") else "Only"

        save_posterior_predictive(
            valinor_model,
            dko_guide,
            params_d,
            config["nSamples"],
            sites_from_model,
            data,
            lengths,
            indices,
            prior_params,
            config,
            annotation,
            **dko_args,
        )


def makeArgs():
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "--no-singletons",
        action="store_true",
        dest="no_singletons",
        default=False,
        help="No singletons.",
    )

    argParser.add_argument(
        "--only-singletons",
        action="store_true",
        dest="only_singletons",
        default=False,
        help="Only singletons.",
    )

    argParser.add_argument(
        "--no-controls",
        action="store_true",
        dest="no_controls",
        default=False,
        help="No controls.",
    )

    argParser.add_argument(
        "--stable-update",
        action="store_true",
        dest="stable_update",
        default=False,
        help="Stable update.",
    )

    argParser.add_argument(
        "--alternateLH",
        action="store_true",
        dest="alternateLH",
        default=False,
        help="Whether to use the alternate approximate likelihood.",
    )

    argParser.add_argument(
        "-n", type=str, dest="name", default=None, help="Output name."
    )

    argParser.add_argument(
        "-o", type=str, dest="outputDir", default="", help="Output directory."
    )

    argParser.add_argument(
        "--paramsFileName",
        type=str,
        dest="paramsFileName",
        default="params.h5",
        help="Parameters file name.",
    )

    argParser.add_argument(
        "--lr", type=float, dest="lr", default=0.01, help="Learning rate."
    )

    argParser.add_argument(
        "--epochs",
        type=int,
        dest="epochs",
        default=50000,
        help="Number of epochs to train for.",
    )

    argParser.add_argument(
        "--nSamples",
        type=int,
        dest="nSamples",
        default=100,
        help="Number of samples to draw from the fitted model.",
    )

    argParser.add_argument(
        "--nParticles",
        type=int,
        dest="n_particles",
        default=1,
        help="Number of particles for the ELBO minimisation",
    )

    argParser.add_argument(
        "--config",
        "-c",
        type=str,
        dest="config",
        default=None,
        help="Config JSON file (overrides all other config)",
    )

    argParser.add_argument(
        "--combinationsFile",
        type=str,
        dest="combinationsFile",
        default=None,
        help="Data combinations file.",
    )

    argParser.add_argument(
        "--singletonsFile",
        type=str,
        dest="singletonsFile",
        default=None,
        help="Data singletons file.",
    )

    argParser.add_argument(
        "--controlsFile",
        type=str,
        dest="controlsFile",
        default=None,
        help="Data controls file.",
    )

    argParser.add_argument(
        "--batch-sample",
        action="store_true",
        dest="batch_sample",
        default=False,
        help="Whether to sample batches of data from the model.",
    )

    argParser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=2**14,
        help="Batch size for batched operations.",
    )

    argParser.add_argument(
        "--guide-config",
        type=str,
        dest="guide_config",
        default="partial_pooling",
        help="Guide pooling type, one of 'no_pooling', 'full_pooling', or 'partial_pooling'.",
    )

    argParser.add_argument(
        "--priorsFile",
        type=str,
        dest="priorsFile",
        default=None,
        help="Prior parameters file.",
    )

    argParser.add_argument(
        "--ZINB",
        dest="zi",
        default=False,
        action="store_true",
        help="Set final distributions to be zero inflated.",
    )

    argParser.add_argument(
        "--ZINBNoSingles",
        dest="zi_ns",
        default=False,
        action="store_true",
        help="Set only the combination distributions to be zero inflated.",
    )

    argParser.add_argument(
        "--reindex",
        dest="reindex",
        default=False,
        action="store_true",
        help="Reindex data types that are only of one class (i.e., only combinations or only singles) in the case where guides are not common.",
    )

    return argParser


def run():
    argParser = makeArgs()
    args = argParser.parse_args()

    # Automatically set flags if files are not provided and display warnings
    if args.singletonsFile is None:
        args.no_singletons = True
        print("Warning: No singletons file provided. --no-singletons flag set.")

    if args.controlsFile is None:
        args.no_controls = True
        print("Warning: No controls file provided. --no-controls flag set.")

    if args.combinationsFile is None:
        args.only_singletons = True
        print("Warning: No combinations file provided. --only-singletons flag set.")

    config = configArgs(args)

    config["fit_mode"] = "full"
    if config["no_singletons"]:
        config["fit_mode"] = "dko"  # Potentially also with controls
    elif config["only_singletons"]:
        config["fit_mode"] = "singles"
    elif config["no_controls"]:
        config["fit_mode"] = "singles-dko"

    data_files = {
        "combinations": config["combinationsFile"],
        "singletons": config["singletonsFile"],
        "controls": config["controlsFile"],
    }

    loaded_priors = loadPriors(args.priorsFile) if args.priorsFile is not None else None

    lengths, indices, prior_params, data = prepareData(
        data_files,
        loaded_priors,
        config["only_singletons"],
        not config["no_singletons"],
        not config["no_controls"],
        config["reindex"],
    )

    # print(lengths)

    # for k, v in indices.items():
    #     print(k, len(v))
    # print('')

    # for c, v in data.items():
    #     for d, vd in v.items():
    #         if not (vd is None):
    #             print(c, d, len(vd))
    # print('')

    # indices_to_subset = [4, 14, 3]

    # target_data, target_lengths, target_indices = subset_for_mcmc(indices_to_subset, data, lengths, indices)

    # for k, v in target_indices.items():
    #     print(k, len(v))
    # print('')

    # for c, v in target_data.items():
    #     for d, vd in v.items():
    #         if not (vd is None):
    #             print(c, d, len(vd))

    # print(target_indices['guide_pair_idx'])
    # print(target_lengths)
    # exit(0)

    runValinor(lengths, indices, prior_params, data, config)


if __name__ == "__main__":
    run()

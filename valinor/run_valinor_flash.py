import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

import preprocessing
import priors

from flash_model import (
    FlashLengths,
    make_flash_lengths,
    make_flash_inputs_dko,
    make_flash_inputs_sko,
    make_flash_inputs_ctrl,
    fit_flash_all,
)


def _maybe_load_initials(path: str):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Initials file not found: {path}")
    if p.suffix == ".npz":
        arrs = np.load(p, allow_pickle=True)
        return {k: jnp.asarray(arrs[k]) for k in arrs.files}
    if p.suffix in (".pkl", ".pickle"):
        with open(p, "rb") as f:
            obj = pickle.load(f)
        return {k: jnp.asarray(v) for k, v in obj.items()}
    if p.suffix == ".json":
        with open(p, "r") as f:
            obj = json.load(f)
        return {k: jnp.asarray(v) for k, v in obj.items()}
    raise ValueError("Unsupported initials format; use .npz, .pkl, or .json")


def _save_outputs(params: dict, out_path: str):
    out = {k: np.asarray(v) for k, v in params.items()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)


def _augment_with_flash_flags(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--init", type=str, default=None)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--zi", action="store_true")
    parser.add_argument("--w-dko", type=float, default=1.0)
    parser.add_argument("--w-sko", type=float, default=1.0)
    parser.add_argument("--w-ctrl", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def build_argparser():
    try:
        import run_valinor as _rv

        if hasattr(_rv, "build_argparser"):
            base = _rv.build_argparser()
        else:
            base = argparse.ArgumentParser("run_valinor_flash")
    except Exception:
        base = argparse.ArgumentParser("run_valinor_flash")
    base = _augment_with_flash_flags(base)
    return base


def _run_preprocessing_with_same_inputs(args):
    if hasattr(preprocessing, "prepare_inputs_from_args"):
        return preprocessing.prepare_inputs_from_args(args)
    if hasattr(preprocessing, "prepare_inputs"):
        try:
            return preprocessing.prepare_inputs(args)
        except TypeError:
            pass
    if hasattr(preprocessing, "prepare_inputs"):
        if hasattr(args, "config"):
            return preprocessing.prepare_inputs(args.config)
    raise RuntimeError("Unable to run preprocessing with provided arguments.")


def run(args):
    data, lengths, indices, prior_params = _run_preprocessing_with_same_inputs(args)

    L: FlashLengths = make_flash_lengths(lengths)
    y_dko, idx_dko = make_flash_inputs_dko(data, lengths, indices)
    y_sko, idx_sko = make_flash_inputs_sko(data, lengths, indices)
    y_ctrl, idx_ctrl, log_init_ctrl = make_flash_inputs_ctrl(data, lengths, indices)

    hyper = priors.flash_hyper(prior_params) if hasattr(priors, "flash_hyper") else None
    init_overrides = _maybe_load_initials(args.init) if args.init else None

    key = jax.random.PRNGKey(args.seed)

    params = fit_flash_all(
        key=key,
        L=L,
        y_dko=y_dko,
        idx_dko=idx_dko,
        y_sko=y_sko,
        idx_sko=idx_sko,
        y_ctrl=y_ctrl,
        idx_ctrl=idx_ctrl,
        log_init_ctrl=log_init_ctrl,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_zi=args.zi,
        w_dko=args.w_dko,
        w_sko=args.w_sko,
        w_ctrl=args.w_ctrl,
        hyper=hyper,
        init=init_overrides,
    )

    _save_outputs(params, args.out)


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)

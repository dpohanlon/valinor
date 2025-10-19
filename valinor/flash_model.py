import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Dict, Any, Optional, Tuple

# ---------- Compact containers ----------


class FlashLengths(NamedTuple):
    n_obs_dko: int
    n_obs_sko: int
    n_obs_ctrl: int
    n_genes: int
    n_pairs: int
    n_guides: int
    n_cells: int


class FlashDKOIdx(NamedTuple):
    pair_idx: jnp.ndarray
    gene1_idx: jnp.ndarray
    gene2_idx: jnp.ndarray
    guide1_idx: jnp.ndarray
    guide2_idx: jnp.ndarray
    cell_idx: jnp.ndarray


class FlashSKOIdx(NamedTuple):
    gene_idx: jnp.ndarray
    guide_idx: jnp.ndarray
    cell_idx: jnp.ndarray


class FlashCTRLIdx(NamedTuple):
    cell_idx: jnp.ndarray


# ---------- Likelihoods ----------


def nb_logpmf(y, mu, phi):
    a = jax.lax.lgamma(y + phi) - jax.lax.lgamma(phi) - jax.lax.lgamma(y + 1.0)
    b = phi * (jnp.log(phi) - jnp.log(phi + mu))
    c = y * (jnp.log(mu) - jnp.log(phi + mu))
    return a + b + c


def zinb_logpmf(y, mu, phi, gate):
    lp = nb_logpmf(y, mu, phi)
    is_zero = (y == 0).astype(mu.dtype)
    l0 = jnp.logaddexp(jnp.log(gate + 1e-8), jnp.log1p(-gate) + nb_logpmf(0.0, mu, phi))
    return is_zero * l0 + (1.0 - is_zero) * (jnp.log1p(-gate) + lp)


# ---------- Priors / params ----------


def _center(x):
    return x - jnp.mean(x)


def init_params(key, L: FlashLengths, init: Optional[Dict[str, jnp.ndarray]] = None):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    P = {
        "log_init_pair": jax.random.normal(k1, (L.n_pairs,)) * 0.1,
        "delta_gene": _center(jax.random.normal(k2, (L.n_genes,)) * 0.1),
        "delta_pair": _center(jax.random.normal(k3, (L.n_pairs,)) * 0.05),
        "alpha": jax.random.normal(k4, (L.n_guides, L.n_cells)) * 0.1,
        "gamma_cell": _center(jax.random.normal(k5, (L.n_cells,)) * 0.05),
        "bias_cell": jnp.zeros((L.n_cells,)),
        "log_phi_cell": jnp.log(jnp.full((L.n_cells,), 50.0)),
        "z_gate": jnp.full((L.n_cells,), -6.0),
    }
    if init:
        for k, v in init.items():
            if k in P and v is not None:
                P[k] = v
    return P


def priors_penalty(P, h):
    s = 0.0
    s += jnp.sum(0.5 * (P["log_init_pair"] / h["sd_log_init"]) ** 2)
    s += jnp.sum(0.5 * (P["delta_gene"] / h["sd_delta_gene"]) ** 2)
    s += jnp.sum(0.5 * (P["delta_pair"] / h["sd_delta_pair"]) ** 2)
    s += jnp.sum(0.5 * (P["alpha"] / h["sd_alpha"]) ** 2)
    s += jnp.sum(0.5 * (P["gamma_cell"] / h["sd_gamma"]) ** 2)
    s += jnp.sum(0.5 * ((P["bias_cell"]) / h["sd_bias"]) ** 2)
    s += jnp.sum(
        0.5 * ((P["log_phi_cell"] - jnp.log(h["phi_loc"])) / (h["phi_scale"])) ** 2
    )
    s += jnp.sum(0.5 * (P["z_gate"] / h["sd_gate"]) ** 2)
    return s


# ---------- Expected-mean surrogates ----------


def mu_dko(P, idx: FlashDKOIdx):
    p1 = jax.nn.sigmoid(P["alpha"][idx.guide1_idx, idx.cell_idx])
    p2 = jax.nn.sigmoid(P["alpha"][idx.guide2_idx, idx.cell_idx])
    log_base = (
        P["log_init_pair"][idx.pair_idx]
        + P["gamma_cell"][idx.cell_idx]
        + P["bias_cell"][idx.cell_idx]
    )
    g1 = P["delta_gene"][idx.gene1_idx]
    g2 = P["delta_gene"][idx.gene2_idx]
    g12 = P["delta_pair"][idx.pair_idx]
    lp00 = jnp.log1p(-p1) + jnp.log1p(-p2)
    lp10 = jnp.log(p1) + jnp.log1p(-p2)
    lp01 = jnp.log(p2) + jnp.log1p(-p1)
    lp11 = jnp.log(p1) + jnp.log(p2)
    w = jax.nn.softmax(jnp.stack([lp00, lp10, lp01, lp11], -1), -1)
    comps = jnp.stack(
        [log_base, log_base + g1, log_base + g2, log_base + g1 + g2 + g12], -1
    )
    mu = jnp.sum(w * jnp.exp(jnp.clip(comps, -20.0, 20.0)), -1)
    return jnp.clip(mu, 1e-6, 1e12)


def mu_sko(P, idx: FlashSKOIdx):
    p = jax.nn.sigmoid(P["alpha"][idx.guide_idx, idx.cell_idx])
    log_base = P["gamma_cell"][idx.cell_idx] + P["bias_cell"][idx.cell_idx]
    g = P["delta_gene"][idx.gene_idx]
    w = jax.nn.softmax(jnp.stack([jnp.log1p(-p), jnp.log(p)], -1), -1)
    comps = jnp.stack([log_base, log_base + g], -1)
    mu = jnp.sum(w * jnp.exp(jnp.clip(comps, -20.0, 20.0)), -1)
    return jnp.clip(mu, 1e-6, 1e12)


def mu_ctrl(P, idx: FlashCTRLIdx, log_init_ctrl):
    log_base = log_init_ctrl + P["gamma_cell"][idx.cell_idx]
    mu = jnp.exp(jnp.clip(log_base, -20.0, 20.0))
    return jnp.clip(mu, 1e-6, 1e12)


# ---------- Joint objective ----------


def loss_blocks(
    P,
    y_dko,
    idx_dko: Optional[FlashDKOIdx],
    y_sko,
    idx_sko: Optional[FlashSKOIdx],
    y_ctrl,
    idx_ctrl: Optional[FlashCTRLIdx],
    log_init_ctrl: Optional[jnp.ndarray],
    w_dko,
    w_sko,
    w_ctrl,
    use_zi: bool,
    h,
):
    pen = priors_penalty(P, h)
    ll = 0.0
    if y_dko is not None:
        mu = mu_dko(P, idx_dko)
        phi = jnp.exp(P["log_phi_cell"][idx_dko.cell_idx])
        if use_zi:
            gate = jax.nn.sigmoid(P["z_gate"][idx_dko.cell_idx])
            ll += w_dko * jnp.sum(zinb_logpmf(y_dko, mu, phi, gate))
        else:
            ll += w_dko * jnp.sum(nb_logpmf(y_dko, mu, phi))
    if y_sko is not None:
        mu = mu_sko(P, idx_sko)
        phi = jnp.exp(P["log_phi_cell"][idx_sko.cell_idx])
        if use_zi:
            gate = jax.nn.sigmoid(P["z_gate"][idx_sko.cell_idx])
            ll += w_sko * jnp.sum(zinb_logpmf(y_sko, mu, phi, gate))
        else:
            ll += w_sko * jnp.sum(nb_logpmf(y_sko, mu, phi))
    if y_ctrl is not None:
        mu = mu_ctrl(P, idx_ctrl, log_init_ctrl)
        phi = jnp.exp(P["log_phi_cell"][idx_ctrl.cell_idx])
        if use_zi:
            gate = jax.nn.sigmoid(P["z_gate"][idx_ctrl.cell_idx])
            ll += w_ctrl * jnp.sum(zinb_logpmf(y_ctrl, mu, phi, gate))
        else:
            ll += w_ctrl * jnp.sum(nb_logpmf(y_ctrl, mu, phi))
    return -(ll) + pen


@jax.jit
def _step(P, opt_state, tx, args):
    f = lambda Q: loss_blocks(Q, *args)
    val, grads = jax.value_and_grad(f)(P)
    updates, opt_state = tx.update(grads, opt_state, P)
    P = optax.apply_updates(P, updates)
    return P, opt_state, val


def fit_flash_all(
    key,
    L: FlashLengths,
    y_dko: Optional[jnp.ndarray],
    idx_dko: Optional[FlashDKOIdx],
    y_sko: Optional[jnp.ndarray],
    idx_sko: Optional[FlashSKOIdx],
    y_ctrl: Optional[jnp.ndarray],
    idx_ctrl: Optional[FlashCTRLIdx],
    log_init_ctrl: Optional[jnp.ndarray],
    steps=800,
    lr=5e-2,
    weight_decay=1e-4,
    use_zi=False,
    w_dko=1.0,
    w_sko=1.0,
    w_ctrl=1.0,
    hyper: Optional[Dict[str, float]] = None,
    init: Optional[Dict[str, jnp.ndarray]] = None,
):
    if hyper is None:
        hyper = {
            "sd_log_init": 1.0,
            "sd_delta_gene": 0.5,
            "sd_delta_pair": 0.2,
            "sd_alpha": 0.75,
            "sd_gamma": 0.5,
            "sd_bias": 0.1,
            "phi_loc": 50.0,
            "phi_scale": 1.0,
            "sd_gate": 1.0,
        }
    P = init_params(key, L, init=init)
    tx = optax.chain(optax.add_decayed_weights(weight_decay), optax.adam(lr))
    opt_state = tx.init(P)
    args = (
        y_dko,
        idx_dko,
        y_sko,
        idx_sko,
        y_ctrl,
        idx_ctrl,
        log_init_ctrl,
        w_dko,
        w_sko,
        w_ctrl,
        use_zi,
        hyper,
    )
    best = P
    best_obj = jnp.inf
    for _ in range(steps):
        P, opt_state, obj = _step(P, opt_state, tx, args)
        better = obj < best_obj
        best = jax.tree_util.tree_map(lambda a, b: jnp.where(better, a, b), P, best)
        best_obj = jnp.minimum(best_obj, obj)
    return best


# ---------- Adapters from existing preprocessing ----------


def _as(x):
    return jnp.asarray(x) if x is not None else None


def make_flash_lengths(lengths: Dict[str, int]) -> FlashLengths:
    return FlashLengths(
        n_obs_dko=lengths.get("len_guide_pairs", 0),
        n_obs_sko=lengths.get("len_guides", 0),
        n_obs_ctrl=lengths.get("len_guide_pairs_c", 0),
        n_genes=lengths["len_genes"],
        n_pairs=lengths["len_gene_pairs"],
        n_guides=lengths["len_guides"],
        n_cells=lengths["len_cell_lines"],
    )


def make_flash_inputs_dko(
    data: Dict[str, Any], lengths: Dict[str, int], indices: Dict[str, Any]
) -> Tuple[Optional[jnp.ndarray], Optional[FlashDKOIdx]]:
    y = _as(data["final"].get("combinations", None)) if "final" in data else None
    if y is None or lengths.get("len_guide_pairs", 0) == 0:
        return None, None
    gene1 = (
        indices["gene_1_idx"]
        if "gene_1_idx" in indices
        else indices["gene_1_common_idx"]
    )
    gene2 = (
        indices["gene_2_idx"]
        if "gene_2_idx" in indices
        else indices["gene_2_common_idx"]
    )
    idx = FlashDKOIdx(
        pair_idx=_as(indices["gene_pair_idx"]),
        gene1_idx=_as(gene1),
        gene2_idx=_as(gene2),
        guide1_idx=_as(indices["guide_1_idx"]),
        guide2_idx=_as(indices["guide_2_idx"]),
        cell_idx=_as(indices["cell_line_idx"]),
    )
    return y, idx


def make_flash_inputs_sko(
    data: Dict[str, Any], lengths: Dict[str, int], indices: Dict[str, Any]
) -> Tuple[Optional[jnp.ndarray], Optional[FlashSKOIdx]]:
    y = _as(data["final"].get("singletons", None)) if "final" in data else None
    if y is None or lengths.get("len_guides", 0) == 0:
        return None, None
    gene = (
        indices["gene_s_idx"]
        if "gene_s_idx" in indices
        else indices["gene_s_common_idx"]
    )
    idx = FlashSKOIdx(
        gene_idx=_as(gene),
        guide_idx=_as(indices["guide_s_idx"]),
        cell_idx=_as(indices["cell_line_s_idx"]),
    )
    return y, idx


def make_flash_inputs_ctrl(
    data: Dict[str, Any], lengths: Dict[str, int], indices: Dict[str, Any]
) -> Tuple[Optional[jnp.ndarray], Optional[FlashCTRLIdx], Optional[jnp.ndarray]]:
    y = _as(data["final"].get("controls", None)) if "final" in data else None
    if y is None or lengths.get("len_guide_pairs_c", 0) == 0:
        return None, None, None
    idx = FlashCTRLIdx(
        cell_idx=_as(indices["cell_line_c_idx"]),
    )
    init_ctrl = (
        _as(data["initial"].get("controls", None)) if "initial" in data else None
    )
    log_init_ctrl = (
        jnp.log(jnp.clip(init_ctrl, 1e-6, 1e12)) if init_ctrl is not None else None
    )
    return y, idx, log_init_ctrl


# ---------- Optional: map initials from your full model ----------


def map_initials_from_full(numpyro_params: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    out = {}
    if "dko.gene_pair_ko_growth" in numpyro_params:
        out["delta_pair"] = _as(numpyro_params["dko.gene_pair_ko_growth"])
    if "gene_ko_growth" in numpyro_params:
        out["delta_gene"] = _as(numpyro_params["gene_ko_growth"])
    if "guides_shared.guide_eff" in numpyro_params:
        g = jnp.clip(_as(numpyro_params["guides_shared.guide_eff"]), 1e-6, 1 - 1e-6)
        out["alpha"] = jnp.log(g) - jnp.log1p(-g)
    if "cell_line_growth" in numpyro_params:
        out["gamma_cell"] = _as(numpyro_params["cell_line_growth"])
    if "library_bias" in numpyro_params:
        out["bias_cell"] = _as(numpyro_params["library_bias"])
    if "raw_mv_cell_line" in numpyro_params:
        mv = (
            jnp.exp(jnp.clip(_as(numpyro_params["raw_mv_cell_line"]), -20.0, 20.0))
            + 1.0
        )
        out["log_phi_cell"] = jnp.log(jnp.clip(mv, 1e-6, 1e12))
    if "p_zi" in numpyro_params:
        p = jnp.clip(_as(numpyro_params["p_zi"]), 1e-6, 1 - 1e-6)
        out["z_gate"] = jnp.log(p) - jnp.log1p(-p)
    if "guide_init_count" in numpyro_params:
        out["log_init_pair"] = jnp.log(
            jnp.clip(_as(numpyro_params["guide_init_count"]), 1e-6, 1e12)
        )
    return out

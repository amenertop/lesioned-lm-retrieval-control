"""Explicit lesion states for the title-gap experiments. No scoring or item selection.

Each state's gates map component keys to retained output fractions. Uniform and learned continuous lesions
therefore use the same existing mean-ablation formula. State generation never examines test scores.
"""
from contextlib import contextmanager
import re

import numpy as np

from . import baselines, gates, hooks, weight_masks
from .components import Component, all_components


def validate_states(states, config):
    available = {c.key for kind in ("head", "expert", "mlp") for c in all_components(config, kind)}
    if not states or states[0].get("name") != "intact" or states[0].get("gates") or states[0].get("weight_mask"):
        raise ValueError("states must start with an empty intact state")
    if len({s["name"] for s in states}) != len(states):
        raise ValueError("duplicate state names")
    for state in states:
        if state.get("region") not in (None, "all", "answer", "prompt"):
            raise ValueError("unknown lesion region")
        if state.get("region") in ("answer", "prompt") and (
                state.get("kind") == "weight_mask" or len(set(state["gates"].values())) > 1):
            raise ValueError("position restriction requires uniform activation gates")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", state["name"]):
            raise ValueError("state names must be simple directory names")
        if set(state["gates"]) - available:
            raise ValueError(f"{state['name']}: component outside this model")
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in state["gates"].values()):
            raise ValueError("retention must lie in [0, 1]")
        if state.get("kind") == "weight_mask" or "weight_mask" in state:
            if state.get("kind") != "weight_mask" or state["gates"] or not state.get("weight_mask"):
                raise ValueError("weight lesions need one explicit mask and no activation gates")
            weight_masks.validate(config, **state["weight_mask"])


@contextmanager
def apply_state(model, means, state):
    values = state["gates"]
    if state.get("kind") == "weight_mask":
        if values:
            raise ValueError("do not combine weight masks and activation gates")
        with weight_masks.masked(model, **state["weight_mask"]):
            yield
    elif not values:
        yield
    elif len(set(values.values())) == 1:
        with hooks.ablate(model, [Component.parse(c) for c in values], means, alpha=next(iter(values.values()))):
            yield
    else:
        g = gates.Gates(list(values)).to(next(model.parameters()).device)
        g.set_values(list(values.values()))
        with gates.gated(model, g, means):
            yield


def layer_bands(config):
    return {name: tuple(int(x) for x in band) for name, band in
            zip(("early", "middle", "late"), np.array_split(np.arange(config.num_hidden_layers), 3)) if len(band)}


def broad_states(config, fractions=(0.25, 0.5, 1.0), alphas=(0.9, 0.75, 0.5), seeds=range(5)):
    """Nested expert membership within each fixed layer band and seed, with separate dose and membership axes."""
    experts = all_components(config, "expert")
    if not experts:
        raise ValueError("broad expert lesions require an MoE model")
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError("expert fractions must lie in (0, 1]")
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one mask seed is required")
    states = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        orders = {layer: list(rng.permutation([c.key for c in experts if c.layer == layer]))
                  for layer in range(config.num_hidden_layers)}
        for band, layers in layer_bands(config).items():
            for fraction in fractions:
                keys = [k for layer in layers for k in orders[layer][:max(1, int(np.ceil(len(orders[layer]) * fraction)))]]
                # Full membership has no mask uncertainty, so keep one copy rather than five identical states.
                if fraction == 1 and seed != min(seeds):
                    continue
                for alpha in alphas:
                    states.append(dict(name=f"broad_{band}_f{fraction:g}_a{alpha:g}_s{seed}", kind="broad",
                                       band=band, layers=list(layers), fraction=fraction, seed=int(seed),
                                       alpha=alpha, gates=dict.fromkeys(keys, alpha)))
    return states


def reference_states(groups, alphas=(0.0, 0.5, 0.75, 0.9)):
    return [dict(name="intact", kind="intact", gates={})] + [
        dict(name=f"{name}_a{alpha:g}", kind="reference", group=name, alpha=alpha, gates=dict.fromkeys(keys, alpha))
        for name, keys in groups.items() for alpha in alphas]


def combination_states(retrieval, comprehension, alpha_retrieval=0.0, alpha_comprehension=0.0):
    if set(retrieval) & set(comprehension):
        raise ValueError("the two factors must be disjoint to make an interpretable four-cell design")
    a, b = dict.fromkeys(retrieval, alpha_retrieval), dict.fromkeys(comprehension, alpha_comprehension)
    return [dict(name=name, kind="factorial", gates=g) for name, g in
            (("intact", {}), ("retrieval_only", a), ("comprehension_only", b), ("joint", {**a, **b}))]


def matched_states(state, config, means, n=200, seed=1910):
    targets = [Component.parse(k) for k in state["gates"]]
    if not targets or n < 1:
        raise ValueError("random matching needs a nonempty target and positive count")
    candidates = all_components(config, "expert") + all_components(config, "head")
    freq = means.routing_frequency()
    routing = {c: float(freq[c.layer, c.index]) for c in candidates if c.kind == "expert"}
    if state.get("kind") == "learned":
        return shuffled_gates(state, routing, n, seed)
    sets = baselines.matched_random_sets(targets, candidates, n, seed=seed, routing_freq=routing)
    return [dict(name=f"random_{state['name']}_{i:03d}", kind="matched_random", target=state["name"],
                 seed=seed, random_index=i, region=state.get("region"),
                 gates={c.key: state["gates"][t.key] for c, t in zip(cs, targets)})
            for i, cs in enumerate(sets)]


def shuffled_gates(state, routing, n=200, seed=1910):
    """Dense soft gates cover the candidate band. Shuffle retention within kind/layer/routing quartile.

    This controls placement of gate strengths, not a disjoint lesion set. Hard sparse groups still use
    matched_random_sets. Expert routing percentiles within one quartile differ by less than 0.25.
    """
    pct = baselines.layer_percentiles(routing)
    groups = {}
    for key in state["gates"]:
        c = Component.parse(key)
        band = min(3, int(pct[c] * 4)) if c.kind == "expert" else 0
        groups.setdefault((c.kind, c.layer, band), []).append(key)
    movable = sum(len(keys) for keys in groups.values() if len({state["gates"][k] for k in keys}) > 1)
    if movable == 0:
        raise ValueError("no variable gates can move within the matched strata")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        values = {k: float(v) for keys in groups.values() for k, v in
                  zip(keys, rng.permutation([state["gates"][k] for k in keys]))}
        out.append(dict(name=f"shuffle_{state['name']}_{i:03d}", kind="matched_gate_shuffle", target=state["name"],
                        seed=seed, random_index=i, movable_components=movable, region=state.get("region"), gates=values))
    return out


def check_source(payload, model, revision, tiny=False):
    if payload.get("model") != model or payload.get("revision") != revision:
        raise ValueError("source belongs to a different model revision")
    if bool(payload.get("tiny", False)) != tiny:
        raise ValueError("tiny and real model sources cannot be mixed")


def variants(state, extra=()):
    """Every leave-one-out and add-one state retains all tasks when passed to the common scorer."""
    values = state["gates"]
    if not values:
        raise ValueError("variants need a nonempty lesion")
    out = [dict(name=f"{state['name']}_minus_{key.replace(':', '_')}", kind="minus_one", removed=key, region=state.get("region"),
                gates={k: v for k, v in values.items() if k != key}) for key in values]
    alpha = min(values.values())
    out += [dict(name=f"{state['name']}_plus_{key.replace(':', '_')}", kind="plus_one", added=key, region=state.get("region"),
                 gates={**values, key: alpha}) for key in extra if key not in values]
    return out


def match_language_loss(table, target_loss, tolerance=0.02):
    """Choose a precomputed development dose closest to the target's relative LM-loss change.

    Rows need split, state, lm_change, kind and n_components. Caller first restricts to the same intervention
    kind and component count. Failure to find a close dose is reported, never called a matched control.
    """
    if set(table.split) != {"dev"}:
        raise ValueError("language-loss matching must not use held-out scores")
    if table.kind.nunique() != 1 or table.n_components.nunique() != 1:
        raise ValueError("restrict matching to one intervention kind and component count")
    if not np.isfinite(table.lm_change.to_numpy(float)).all() or not np.isfinite(target_loss):
        raise ValueError("non-finite language-loss measurement")
    order = table.assign(gap=abs(table.lm_change - target_loss)).sort_values(["gap", "state"])
    if order.empty or order.iloc[0].gap > tolerance:
        return dict(status="no_match", tolerance=tolerance, target_loss=target_loss)
    row = order.iloc[0]
    return dict(status="matched", state=row.state, gap=float(row.gap), target_loss=target_loss, tolerance=tolerance)

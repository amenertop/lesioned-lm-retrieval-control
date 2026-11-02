"""MVP component screen: single-component loop, selection, joint lesions vs random sets (docs/plans/mvp.md; pilot 11).

This module does not import the ablation code. Callers pass `ablate`, a function that takes a list of
Components and returns a context manager; the entry scripts use
functools.partial(hooks.ablate, model, means=means, alpha=0.0). Tests pass a fake.
"""
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

from . import damage, scoring
from .components import Component

THRESHOLD = 0.20   # logit; mvp.md §5 and G2
PRESERVED = 0.10   # logit; G2 upper bound for the effect a component class should leave intact
K = 4              # components per direction in the joint lesion
SCORE_COLS = ["item_id", "order", "logit_A", "logit_B", "logit_correct", "logit_wrong"]
PRIMARY = {"control": "control_effect", "knowledge": "knowledge_effect"}


def make_scorer(model, tokenizer, prompts, chat, ablate, batch_size=32, score_prompts=None):
    """score(components) -> label scores with those components lesioned; score() is the intact model.

    score_prompts defaults to lesion.scoring.score_label_prompts; lesion.prefix_scoring.score_label_prompts
    gives the same scores faster. Use one of them for every stage of a run.
    """
    score_prompts = score_prompts or scoring.score_label_prompts

    def score(components=()):
        with ablate(list(components)) if components else nullcontext():
            return score_prompts(model, tokenizer, prompts, chat, batch_size)
    return score


# --- single-component loop with resumable shards ---------------------------------------------------

def load_shards(shard_dir):
    files = sorted(Path(shard_dir).glob("shard_*.parquet"))
    if not files:
        return pd.DataFrame(columns=["component", *SCORE_COLS])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run_screen(score, components, shard_dir, flush_every=32, log=print):
    """Lesion each component alone and store per-prompt logits in shard_dir; components already stored are skipped."""
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = set(load_shards(shard_dir).component)
    todo = [c for c in components if c.key not in done]
    n_shard = len(list(shard_dir.glob("shard_*.parquet")))
    buf, t0 = [], time.time()
    for i, c in enumerate(todo, start=1):
        s = score([c])[SCORE_COLS].copy()
        s.insert(0, "component", c.key)
        buf.append(s)
        if len(buf) == flush_every or i == len(todo):
            tmp = shard_dir / f"shard_{n_shard:05d}.parquet.tmp"
            pd.concat(buf, ignore_index=True).astype({k: "float32" for k in SCORE_COLS[2:]}).to_parquet(tmp, index=False)
            tmp.replace(tmp.with_suffix(""))
            n_shard, buf = n_shard + 1, []
            rate = i / (time.time() - t0)
            log(f"{len(done) + i}/{len(components)} components, {rate:.2f}/s, about {(len(todo) - i) / rate / 60:.0f} min left")
    return load_shards(shard_dir)


def summarize(intact, shards, items, freq_median):
    """One row per component: the six effects of lesioning it alone (lesion.damage.effects)."""
    rows = []
    for key, s in shards.groupby("component", sort=False):
        t = damage.item_frame(damage.item_damage(intact, s), items, freq_median)
        c = Component.parse(key)
        rows.append(dict(component=key, kind=c.kind, layer=c.layer, index=c.index, **damage.effects(damage.columns(t))))
    return pd.DataFrame(rows).sort_values(["kind", "layer", "index"]).reset_index(drop=True)


def domain_general(summary):
    """Components that hurt string matching more than they raise the control effect (user decision 2026-09-28,
    ADR 0005): damage that is not specific to semantic control."""
    return summary.string_damage > summary.control_effect


def select(summary, k=K, exclude_domain_general=False):
    """Top-k components per direction, with a status per direction.

    Each component is assigned to the direction where its standardised effect (z across all components) is
    larger, so no component lands in both sets; within a direction the k largest raw effects are taken.
    With exclude_domain_general, control candidates that are domain_general() are skipped (they are not
    moved to the knowledge direction). A direction with fewer than k eligible components is "not_found" and
    gets no components, so the G2 branch handles it instead of a short set being tested as if it were complete.
    """
    s = summary.dropna(subset=list(PRIMARY.values())).copy()
    for e in PRIMARY.values():
        if len(s) < 2 or not s[e].std() > 0:
            raise ValueError(f"cannot standardise {e}: {len(s)} components with a finite value, spread {s[e].std()}")
    z = {d: (s[e] - s[e].mean()) / s[e].std() for d, e in PRIMARY.items()}
    s["direction"] = np.where(z["control"] >= z["knowledge"], "control", "knowledge")
    out, status = {}, {}
    for d, e in PRIMARY.items():
        pool = s[s.direction == d]
        if d == "control" and exclude_domain_general:
            pool = pool[~domain_general(pool)]
        n = len(pool)
        status[d] = "ok" if n >= k else f"not_found ({n} of {k} eligible)"
        out[d] = pool.nlargest(k, e).component.tolist() if n >= k else []
    out["status"] = status
    return out


def select_semantic_control(summary, k=K):
    """Exploratory, stricter than the domain-general exclusion (user decision 2026-10-13): control-direction
    components (as assigned by select) whose own string damage is not above 0, the k largest control effects.
    C4 itself hurts string matching jointly (0.85 logit on the held-out items); this asks whether a set exists
    that is specific to semantic control, as the IFG and pMTG are in healthy-participant TMS."""
    s = summary.dropna(subset=list(PRIMARY.values())).copy()
    z = {d: (s[e] - s[e].mean()) / s[e].std() for d, e in PRIMARY.items()}
    pool = s[(z["control"] >= z["knowledge"]) & (s.string_damage <= 0)]
    n = len(pool)
    return dict(control=pool.nlargest(k, PRIMARY["control"]).component.tolist() if n >= k else [],
                status="ok" if n >= k else f"not_found ({n} of {k} eligible)",
                rule="control direction as in select(); own string damage <= 0; the k largest control effects")


def select_with_supplement(summary, k=K):
    """Main selection excludes domain-general control components; the selection without that exclusion is
    kept under "supplementary" and reported alongside (ADR 0005)."""
    main = select(summary, k, exclude_domain_general=True)
    plain = select(summary, k, exclude_domain_general=False)
    main["supplementary"] = dict(control_unfiltered=plain["control"], status=plain["status"]["control"],
                                 excluded_domain_general=sorted(summary.component[domain_general(summary)]))
    return main


# --- joint lesions against random sets ------------------------------------------------------------

def compare_sets(intact, lesioned, items, freq_median, n_boot=damage.N_BOOT, seed=damage.SEED):
    """Effects of each lesioned set on the same items.

    lesioned: {set name: label scores}; the target sets are named "control" and "knowledge", random sets
    "random_<direction>_<i>". Returns (points, intervals, random_rank):
      points: {set name: six effects}
      intervals: cluster-bootstrap intervals for both target sets and for the between-class differences
        delta_C = control effect of control set - of knowledge set, delta_F = knowledge effect of knowledge set - of control set
      random_rank: per direction, the share of its random sets whose primary effect is at least the target's
    """
    h = pd.concat({name: damage.item_damage(intact, s) for name, s in lesioned.items()}, axis=1)
    frame = damage.item_frame(h, items, freq_median)
    cols = damage.columns(frame)
    points = {name: damage.effects(cols, h=name) for name in lesioned}
    targets = [d for d in PRIMARY if d in lesioned]

    def stat(c):
        e = {d: damage.effects(c, h=d) for d in targets}
        out = {f"{d}.{name}": v for d in targets for name, v in e[d].items()}
        if len(targets) == 2:
            out["delta_C"] = e["control"]["control_effect"] - e["knowledge"]["control_effect"]
            out["delta_F"] = e["knowledge"]["knowledge_effect"] - e["control"]["knowledge_effect"]
        return out

    intervals = damage.bootstrap(frame, stat, n_boot, seed) if targets else {}
    random_rank = {}
    for d in targets:
        rand = [points[n][PRIMARY[d]] for n in lesioned if n.startswith(f"random_{d}_")]
        if rand:
            random_rank[d] = dict(n_random=len(rand), share_at_least_target=float(np.mean(np.array(rand) >= points[d][PRIMARY[d]])))
    return points, intervals, random_rank


def mvp_verdict(g1_decision, intervals):
    """docs/plans/mvp.md §5, using the dev-set point estimates; spec G2's preserved-effect bound reported alongside."""
    signal = {d: intervals.get(f"{d}.{e}", {}).get("estimate", np.nan) > THRESHOLD for d, e in PRIMARY.items()}
    other = {"control": "knowledge_effect", "knowledge": "control_effect"}
    preserved = {d: intervals.get(f"{d}.{other[d]}", {}).get("ci_high", np.nan) < PRESERVED for d in PRIMARY}
    if g1_decision != "pass":
        outcome = "g1_fail"
    elif any(signal.values()):
        outcome = "signal"
    else:
        outcome = "no_signal"
    return dict(outcome=outcome, g1_decision=g1_decision, above_threshold=signal, g2_preserved_below=preserved)

"""Development-only family split, context discrimination and selection for production control."""
import re

import numpy as np
import pandas as pd

from . import homonym_items, naming, title_gap_analysis


def split_families(items, seed=1920):
    if set(items.split) != {"dev"}:
        raise ValueError("production search accepts development families only")
    families = sorted(items.family.unique())
    order = np.random.default_rng(seed).permutation(families).tolist()
    if len(order) < 4 or len(order) % 2:
        raise ValueError("need an even number of complete families")
    return dict(seed=seed, screening=sorted(order[:len(order) // 2]), dev=sorted(order[len(order) // 2:]))


def attribution_prompts(items):
    prompts, _ = homonym_items.prompts(items)
    selected = [p for p in prompts if p.condition.endswith("_none")]
    jobs, weights = [], []
    for p in selected:
        for role, word in p.candidates.items():
            jobs.append(naming.NamingPrompt(p.item_id, word, p.text))
            weights.append((-1 if role == "target" else 1) / len(selected))
    return jobs, {"context_damage": np.array(weights)}


def context_scores(table, items):
    """The same word pair changes roles across meanings, cancelling fixed word preferences."""
    table = table[table.condition.str.endswith("_none")]
    pivot = table.pivot(index=["state", "item_id", "condition"], columns="role", values="logprob")
    result = (pivot.target - pivot.other_meaning).rename("margin").reset_index()
    result = result.merge(items[["item_id", "family", "dominance"]], on="item_id", validate="many_to_one")
    expected = len(items) * len(homonym_items.CONTEXTS)
    if not result.groupby("state").size().eq(expected).all() or not np.isfinite(result.margin).all():
        raise ValueError("context scores do not contain both meanings and both contexts")
    return result


def screen_summary(run, items):
    tables, plan = title_gap_analysis.read_run(run)
    scores = context_scores(tables["candidates"], items)
    base = scores.query("state == 'intact'").set_index(["item_id", "condition"]).margin
    generated = tables["generated"].query("condition.str.endswith('_none')", engine="python").copy()
    generated["valid"] = generated.answer.map(lambda x: bool(re.fullmatch("[a-z]+", str(x).strip().lower().strip('.!,?\"\''))))
    valid = generated.groupby("state").valid.mean()
    language = tables["language"].set_index("state").loss
    rows = []
    for state, group in scores.groupby("state", sort=False):
        frame = group.set_index(["item_id", "condition"])
        frame["damage"] = base - frame.margin
        by = frame.groupby("dominance").damage.mean()
        lm_change = float(language[state] / language["intact"] - 1)
        format_drop = float(valid["intact"] - valid[state])
        rows.append(dict(state=state, context_damage=float(frame.damage.mean()),
                         subordinate_extra=float(by.subordinate - by.dominant),
                         lm_change=lm_change, format_drop=format_drop,
                         eligible=bool(lm_change <= .1 and format_drop <= .05)))
    return pd.DataFrame(rows), plan


def joint_states(summary, single_states):
    lookup = {s["name"]: s for s in single_states}
    ranked = summary.query("eligible and context_damage > 0").sort_values(["context_damage", "state"], ascending=[False, True])
    keys = [next(iter(lookup[name]["gates"])) for name in ranked.state]
    states = [dict(name="intact", kind="intact", gates={})]
    for size in (2, 4, 8):
        if len(keys) < size:
            continue
        for region in ("all", "answer"):
            for alpha in (0., .5):
                states.append(dict(name=f"PC{size}_{region}_a{alpha:g}", kind="production_candidate", region=region,
                                   gates=dict.fromkeys(keys[:size], alpha), members=size, alpha=alpha))
    return states, keys


def rank_joint(summary, states):
    meta = pd.DataFrame([dict(state=s["name"], members=s["members"], alpha=s["alpha"]) for s in states if s["gates"]])
    if meta.empty:
        return meta
    return (summary.merge(meta, on="state", validate="one_to_one")
            .query("eligible and context_damage > 0 and subordinate_extra > 0")
            .sort_values(["context_damage", "members", "alpha", "state"], ascending=[False, True, False, True]))

"""Read complete common-battery states and retain effects in their original units."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import damage, naming, p14, runinfo, title_gap_scoring


def analyze(directory, word_items, control_items, partial=False):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if title_gap_scoring._fingerprint({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"]:
        raise ValueError("run plan changed")
    if not partial and not (directory / "complete.json").exists():
        raise ValueError("run is incomplete")
    frames, metadata = {}, {}
    for state in plan["states"]:
        folder = directory / state["name"]
        marker = folder / "complete.json"
        if partial and not marker.exists():
            continue
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record["fingerprint"] != plan["fingerprint"]:
            raise ValueError("mixed state provenance")
        if set(record["files"]) != {f"{name}.parquet" for name in ("naming", "control", "comprehension")}:
            raise ValueError("missing state table")
        for name, digest in record["files"].items():
            if runinfo.sha256(folder / name) != digest:
                raise ValueError("state table changed")
        frames[state["name"]] = {name: pd.read_parquet(folder / f"{name}.parquet")
                                 for name in ("naming", "control", "comprehension")}
        metadata[state["name"]] = record
    base = frames["intact"]
    fm = damage.frequency_median(control_items)
    by_word = word_items.set_index("item_id")
    median = naming.zipf_median(word_items)
    rows = []
    for name, tables in frames.items():
        n = naming.item_damage(base["naming"], tables["naming"])
        tokens = tables["naming"].set_index("item_id").n_tokens.reindex(n.index).to_numpy()
        rare = (by_word.zipf.reindex(n.index) < median).to_numpy()
        c = damage.item_frame(damage.item_damage(base["control"], tables["control"]), control_items, fm)
        a, b = [t.groupby("item_id").margin.mean() for t in (base["comprehension"], tables["comprehension"])]
        if set(a.index) != set(b.index):
            raise ValueError("comprehension observations differ")
        h = a - b.reindex(a.index)
        lm = metadata[name]["language"]["loss"]
        lm_base = metadata["intact"]["language"]["loss"]
        rows.append(dict(state=name, naming_damage=float(n.mean()),
                         naming_frequency=p14.rare_coefficient_np(n.to_numpy(), rare, tokens),
                         comprehension_damage=float(h.mean()), comprehension_accuracy=float((b > 0).mean()),
                         language_loss=lm, lm_change=lm / lm_base - 1 if lm_base else None,
                         seconds=metadata[name]["seconds"], **damage.effects(damage.columns(c))))
    return pd.DataFrame(rows), plan


def ledger_summary(table, plans, control_items):
    """Keep scientific point estimates in original units and operational records outside ledger metrics."""
    plan = plans[0]
    ids = {p["item_id"] for p in plan["control"]}
    control = control_items[control_items.item_id.isin(ids)]
    counts = dict(naming=len(plan["words"]), comprehension=len({p["item_id"] for p in plan["comprehension"]}),
                  control=int(control.task.isin(damage.MATCHING).sum()),
                  synonym=int((control.task == "synonym").sum()),
                  semantic=int(control.task.isin(damage.SEMANTIC).sum()), string=int((control.task == "string").sum()),
                  language=len(plan["neutral_sequences"]))
    scales = dict(naming_damage=("nat", "naming"), naming_frequency=("nat", "naming"),
                  comprehension_damage=("logit", "comprehension"), comprehension_accuracy=("proportion", "comprehension"),
                  control_effect=("logit", "control"), knowledge_effect=("logit", "synonym"),
                  control_slope=("logit / TDS SD", "control"), knowledge_slope=("logit / frequency SD", "synonym"),
                  semantic_damage=("logit", "semantic"), string_damage=("logit", "string"),
                  language_loss=("nat / token", "language"), lm_change=("relative change", "language"))
    states = {}
    for row in table.to_dict("records"):
        states[row["state"]] = {metric: dict(estimate=row[metric], unit=unit, n=counts[task],
            n_unit="中性文本" if task == "language" else "概念/题目") for metric, (unit, task) in scales.items()}
    return dict(meta=dict(status="development_grid_point_estimates", split=plan["provenance"]["split"],
                         control_scorer=plan['provenance']['scorer'],
                         naming_and_comprehension_batches=plan['batches'],
                         source_fingerprints=[p["fingerprint"] for p in plans],
                         analysis_code_sha256=runinfo.sha256(Path(__file__)), counts=counts,
                         state_definitions=[s for p in plans for s in p["states"] if s["name"] != "intact"],
                         total_scoring_seconds=float(table.seconds.sum())), states=states)

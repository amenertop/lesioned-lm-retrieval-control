"""P13 report (docs/plans/p13-knowledge-breadth.md) from run_breadth.py's outputs, exploratory. No GPU:

    python experiments/e3_errors/analyze_breadth.py [--tiny] [--out reports/explore/breadth.json]

Per state: the language-model loss change, (lesioned - intact) / intact; naming on the 240 knowledge test concepts
(knowledge effect, mean damage); experiment 3's three formats on its 160 test concepts (mean damage of unhinted
naming, word-to-definition and six-choice, as in H3c, analyze_consistency.damage) and their cross-format consistency
(mean of the three pairwise Spearman correlations; empty below 100 concepts or with a constant format); M1 semantic and
string damage and the margin-scaling slope.
Per size (the anchor set and the S sets): the knowledge set's value, its random sets' minimum, median and maximum,
whether it exceeds their maximum by EXCEED (consistency: by 0), and whether its language loss change passes
COLLAPSE; then the reading written in the plan before the run: progression, partial, two routes or other.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import analyze_consistency as ac  # noqa: E402
from lesion import damage, m3_items, naming, scoring  # noqa: E402

Q = "qwen3-30b-a3b-instruct"
EXCEED, COLLAPSE, MIN_CONCEPTS = 0.2, 0.10, 100
COMPREHENSION = ("word_to_definition", "options_letter")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def per_state(d, kitems, median, m1_items, fm, m3_ids):
    sets = read(d / "sets.json")
    states = [s for s in sets if s != "intact"]
    loss = read(d / "lm_loss.json")
    nm = pd.read_parquet(d / "naming.parquet").astype({"item_id": str})
    m1 = pd.read_parquet(d / "m1.parquet").astype({"item_id": str})
    h3 = ac.damage(pd.read_parquet(d / "candidates.parquet"), pd.read_parquet(d / "labels.parquet"), m3_ids, lesions=states)
    base_n, base_m = nm[nm.state == "intact"], m1[m1.state == "intact"]
    out = {}
    for s in states:
        frame = naming.item_frame(naming.item_damage(base_n, nm[nm.state == s]), kitems, median)
        ne = naming.effects({c: frame[c].to_numpy() for c in frame.columns})
        me = damage.effects(damage.columns(damage.item_frame(damage.item_damage(base_m, m1[m1.state == s]), m1_items, fm)))
        slope = damage.margin_scaling(base_m, m1[m1.state == s], m1_items)
        h = h3[s]
        usable = len(h) >= MIN_CONCEPTS and (h.std() > 0).all()
        out[s] = dict(n_components=len(sets[s]), lm_change=loss[s] / loss["intact"] - 1,
                      knowledge_effect=ne["knowledge_effect"], naming_damage=ne["naming_damage"],
                      **{f"m3_{f}": float(h[f].mean()) for f in ac.FORMATS},
                      consistency=ac.consistency(h)[0] if usable else None,
                      m1_semantic=me["semantic_damage"], m1_string=me["string_damage"],
                      m1_slope=float(slope[slope.task == "all"].slope.iloc[0]))
    return out


def by_size(states, sizes):
    """Each size's knowledge value against its random sets."""
    metrics = ("naming_damage", "knowledge_effect", *[f"m3_{f}" for f in ac.FORMATS], "consistency", "lm_change",
               "m1_semantic", "m1_string", "m1_slope")
    out = {}
    for name in sizes:
        rand = [v for k, v in states.items() if k.startswith(f"{name}_random_")]
        row = dict(n_components=states[name]["n_components"], collapsed=bool(states[name]["lm_change"] > COLLAPSE))
        for m in metrics:
            value, values = states[name][m], [r[m] for r in rand if r[m] is not None]
            if value is None or not values:
                row[m] = dict(value=value, random=None, exceeds=None)
                continue
            margin = 0.0 if m == "consistency" else EXCEED
            row[m] = dict(value=round(float(value), 4),
                          random=dict(n=len(values), min=round(float(min(values)), 4),
                                      median=round(float(np.median(values)), 4), max=round(float(max(values)), 4)),
                          exceeds=bool(value > max(values) + margin),
                          excess_over_median=round(float(value - np.median(values)), 4))
        out[name] = row
    return out


def reading(rows, sizes):
    """The plan's pre-written reading over the sizes whose language loss change is at most COLLAPSE."""
    ok = [s for s in sizes if not rows[s]["collapsed"]]
    comp = {f: [s for s in ok if rows[s][f"m3_{f}"]["exceeds"]] for f in COMPREHENSION}

    def rising(f):
        # three adjacent sizes (in the order given, all below the collapse line) whose excess does not fall
        for i in range(len(sizes) - 2):
            window = sizes[i:i + 3]
            if all(s in ok for s in window):
                e = [rows[s][f"m3_{f}"].get("excess_over_median") for s in window]
                if None not in e and e[0] <= e[1] <= e[2]:
                    return True
        return False

    both = [s for s in ok if all(rows[s][f"m3_{f}"]["exceeds"] for f in COMPREHENSION)]
    consistent = [s for s in both if rows[s]["consistency"].get("exceeds")]
    if all(comp[f] for f in COMPREHENSION) and all(rising(f) for f in COMPREHENSION) and consistent:
        verdict = "progression"
    elif any(comp[f] for f in COMPREHENSION):
        verdict = "partial"
    elif ok and all(rows[s]["naming_damage"]["exceeds"] for s in ok):
        verdict = "two routes"
    else:
        verdict = "other"
    return dict(verdict=verdict, overall_damage_sizes=[s for s in sizes if rows[s]["collapsed"]], sizes_below_collapse=ok,
                comprehension_exceeds=comp,
                rising={f: rising(f) for f in COMPREHENSION}, consistency_exceeds_where_both=consistent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--knowledge-meta", default=None,
                    help="the knowledge screen's meta.json (for the Zipf median); default: the run's own folder, else the fetched copy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suf = "_tiny" if args.tiny else ""
    d = Path(args.dir) if args.dir else ROOT / f"outputs/e3_errors{suf}" / Q / "breadth"
    own = ROOT / f"outputs/e3_knowledge{suf}" / Q / "screen/meta.json"          # on the server, or a tiny run
    fetched = ROOT / "outputs/server/e3_knowledge" / Q / "screen/meta.json"     # a local copy fetched from the server
    kmeta = Path(args.knowledge_meta) if args.knowledge_meta else (own if own.exists() or args.tiny else fetched)
    kitems = pd.read_csv(ROOT / "data/materials/m3_knowledge/items.csv", dtype={"item_id": str})
    m1_items = scoring.load_materials()[0].astype({"item_id": str})
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    m3_ids = list(final.item_id[final.role == "test"])
    meta = read(d / "meta.json")
    states = per_state(d, kitems, read(kmeta)["params"]["zipf_median"], m1_items, damage.frequency_median(m1_items), m3_ids)
    sizes = [f"K{meta['anchor']}" if f"K{meta['anchor']}" in states else next(k for k in states if k.startswith("K") and k != "K8"),
             *[f"S{n}" for n in meta["sizes"]]]
    rows = by_size(states, sizes)
    report = dict(status="exploratory (P13, docs/plans/p13-knowledge-breadth.md)", source=str(d), exceed=EXCEED,
                  collapse=COLLAPSE, states={k: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in s.items()}
                                             for k, s in states.items()},
                  by_size=rows, reading=reading(rows, sizes))
    out = Path(args.out) if args.out else ROOT / ("outputs/breadth_tiny.json" if args.tiny else "reports/explore/breadth.json")
    out.write_bytes((json.dumps(report, indent=1, ensure_ascii=False, default=float) + "\n").encode("utf-8"))
    print(json.dumps(report["reading"], ensure_ascii=False))
    print(f"written to {out}")


if __name__ == "__main__":
    main()

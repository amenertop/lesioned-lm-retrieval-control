"""GPU session 13 (docs/plans/gpu-session13.md), exploratory: one report from the run outputs. No GPU:

    python experiments/analyze_session13.py [--tiny] [--out reports/explore/session13.json]

Item 1, K8b taken apart (K8b, K8b without the pair it shares with S8, the pair alone, S8 without the pair):
- language-model loss change per state; K8b without the pair and the pair against their matched random sets;
- held-out M1 with one scoring function: semantic and string damage and the margin-scaling slope per state, the same
  two sets against their matched random sets;
- the three held-out dissociation tests (naming.* and m1.* intervals, random sets, h2), and the paired naming
  differences K8b minus each part on the same items (K8b from session 12's dissociation_k8b; the intact scores must be
  identical);
- rare or long words on the unused words (session 12's length_table);
- the target's place in the whole vocabulary's change: the median share of tokens that lost at least as much, and the
  share of items where the target lost the most among the intact top 50, concept-resampled.
Item 2, where the control heads act: C4's control effect in each of seven disjoint parts of the prompt and everywhere
(intervals), the 50 random head sets in each part (median, max, C4's quantile), each part's share of the whole, and the
sum of the seven parts over the whole (interval).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_session12 import both_parts, interval, length_table, read, resample  # noqa: E402
from lesion import damage, naming, probes, runinfo, scoring  # noqa: E402

Q = "qwen3-30b-a3b-instruct"
N_BOOT = 4000
TESTS = ("dissociation_k8b_minus_shared", "dissociation_shared_pair", "dissociation_s8_minus_pair")
MATCHED = {"K8b_minus_shared": "rand_k8bm_", "shared_pair": "rand_pair_"}


def among(value, values):
    """Where value falls among values: their range and the share below it."""
    v = np.asarray(values, dtype=float)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 4), min=round(float(v.min()), 4),
                max=round(float(v.max()), 4), quantile=round(float((v < value).mean()), 4))


def language(split):
    lm = read(split / "lm_loss.json")["lm_loss"]
    change = {k: round(v / lm["intact"] - 1, 5) for k, v in lm.items()}
    out = dict(change={k: v for k, v in change.items() if not k.startswith("rand_")})
    for name, prefix in MATCHED.items():
        out[f"{name}_vs_random"] = among(change[name], [v for k, v in change.items() if k.startswith(prefix)])
    return out


def m1(split, items, fm):
    s = pd.read_parquet(split / "m1_scores.parquet").astype({"item_id": str})
    base = s[s.state == "intact"]
    point, out = {}, {}
    for state, les in s.groupby("state"):
        if state == "intact":
            continue
        frame = damage.item_frame(damage.item_damage(base, les), items, fm)
        e = damage.effects(damage.columns(frame))
        slope = damage.margin_scaling(base, les, items)
        point[state] = dict(semantic_damage=e["semantic_damage"], string_damage=e["string_damage"],
                            control_effect=e["control_effect"], slope=float(slope[slope.task == "all"].slope.iloc[0]))
        if not state.startswith("rand_"):
            boot = damage.bootstrap(frame, lambda c: {k: v for k, v in damage.effects(c).items()
                                                      if k in ("semantic_damage", "string_damage", "control_effect")}, N_BOOT)
            out[state] = {**{k: interval(v) for k, v in boot.items()}, "slope": round(point[state]["slope"], 3)}
    for name, prefix in MATCHED.items():
        rand = {k: v for k, v in point.items() if k.startswith(prefix)}
        out[f"{name}_vs_random"] = {m: among(point[name][m], [v[m] for v in rand.values()])
                                    for m in ("semantic_damage", "string_damage", "slope")}
    return out


def held_out(held, k8b_dir, kitems, median):
    out = {}
    full = None
    if (k8b_dir / "naming_scores.parquet").exists():
        full = pd.read_parquet(k8b_dir / "naming_scores.parquet").astype({"item_id": str})
        full = full[full.scoring == "averaged"]
    for name in TESTS:
        d = held / name
        if not (d / "results.json").exists():
            continue
        r = read(d / "results.json")
        out[name] = dict(knowledge_set=r["knowledge_sets"]["main"], control=r["control"], h2=r["h2"], random=r["random"],
                         sets=both_parts(r))
        if full is not None and name != "dissociation_s8_minus_pair":
            part = pd.read_parquet(d / "naming_scores.parquet").astype({"item_id": str})
            part = part[part.scoring == "averaged"]
            fi, pi = (x[x.set == "intact"].set_index("item_id").logprob for x in (full, part))
            same = bool(np.allclose(fi.sort_index().to_numpy(), pi.reindex(fi.sort_index().index).to_numpy()))
            h = pd.DataFrame({"full": naming.item_damage(full[full.set == "intact"], full[full.set == "main"]),
                              "part": naming.item_damage(part[part.set == "intact"], part[part.set == "main"])}).dropna()
            if not same:                       # a paired difference means nothing when the intact scores differ
                print(f"{name}: intact naming scores differ from dissociation_k8b; paired difference skipped")
                out[name]["k8b_minus_this"] = dict(knowledge_effect=None, intact_identical=False)
                continue
            frame = naming.item_frame(h, kitems, median)
            diff = damage.bootstrap(frame, lambda c: {"d": naming.effects(c, "full")["knowledge_effect"]
                                                      - naming.effects(c, "part")["knowledge_effect"]}, N_BOOT)["d"]
            out[name]["k8b_minus_this"] = dict(knowledge_effect=interval(diff), intact_identical=True)
    return out


def target_rank(split):
    t = pd.read_parquet(split / "target_rank.parquet")
    out = {}
    for state, g in t.groupby("state", sort=False):
        f = g.assign(group=g.item_id)
        top = f[f.rank_in_top.notna()]
        out[state] = dict(
            median_share_at_least_as_negative=resample(f, lambda x: float(x.share_at_least_as_negative.median()), N_BOOT),
            lost_most_in_top=resample(top, lambda x: float((x.rank_in_top == 1).mean()), N_BOOT) if len(top) else None,
            n_target_in_top=int(len(top)), mean_target_change=round(float(g.target_change.mean()), 4))
    return out


def positions(held, old_control, items, fm, name="control_positions", parts=probes.PARTS):
    """name, parts: the run folder and its parts (session 14b: control_positions_labels, probes.LABEL_PARTS)."""
    d = held / name
    if not (d / "scores.parquet").exists():
        return None
    s = pd.read_parquet(d / "scores.parquet").astype({"item_id": str})
    base = s[s.state == "intact"]
    h = {state: damage.item_damage(base, les) for state, les in s.groupby("state") if state != "intact"}
    effect = {state: damage.effects(damage.columns(damage.item_frame(v, items, fm)))["control_effect"] for state, v in h.items()}
    c4 = [f"C4_{p}" for p in ("all", *parts)]
    frame = damage.item_frame(pd.DataFrame({k: h[k] for k in c4}), items, fm)

    def stat(c):
        e = {k: damage.effects(c, k)["control_effect"] for k in c4}
        whole = e["C4_all"]
        return {**{f"{k}.control_effect": v for k, v in e.items()},
                **{f"{k}.share_of_all": e[k] / whole for k in c4[1:]},
                "sum_of_parts_over_all": sum(e[k] for k in c4[1:]) / whole}

    boot = damage.bootstrap(frame, stat, N_BOOT)
    out = dict(intervals={k: interval(v) for k, v in boot.items()}, random={},
               # the share and sum-over-whole ratios are descriptive, and only readable when the whole effect is clearly > 0
               whole_interval_above_zero=bool(boot["C4_all.control_effect"]["ci_low"] > 0))
    for part in parts:
        rand = [v for k, v in effect.items() if k.startswith("random_") and k.endswith(f"_{part}")]
        out["random"][part] = {**among(effect[f"C4_{part}"], rand),
                               "rank_from_top": int(1 + sum(v >= effect[f"C4_{part}"] for v in rand))}
    if old_control is not None and (old_control / "results.json").exists():
        r = read(old_control / "results.json")
        # historical reference only: session 10 scored with prefix_avg3, session 13 without the prefix cache
        out["session10_prefix_avg3_reference"] = dict(control_effect=interval(r["intervals"]["control_effect"]),
                                                      random=r.get("random"))
    tokens = pd.DataFrame(read(d / "parts.json"))
    out["tokens_per_part"] = {p: dict(median=float(tokens[p].median()), min=int(tokens[p].min()), max=int(tokens[p].max()))
                              for p in parts}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suf = "_tiny" if args.tiny else ""
    split = ROOT / f"outputs/e3_knowledge{suf}" / Q / "k8b_split"
    held = ROOT / f"outputs/e2_confirm{suf}" / Q
    old = held if args.tiny else ROOT / "outputs/server/e2_confirm" / Q          # the session 10 runs, fetched earlier
    kmeta = (ROOT / f"outputs/e3_knowledge{suf}" / Q if args.tiny else ROOT / "outputs/server/e3_knowledge" / Q) / "screen/meta.json"
    items = scoring.load_materials()[0].astype({"item_id": str})
    fm = damage.frequency_median(items)
    kitems = pd.read_csv(ROOT / "data/materials/m3_knowledge/items.csv", dtype={"item_id": str})
    median = read(kmeta)["params"]["zipf_median"]
    report = dict(status="exploratory (GPU session 13, docs/plans/gpu-session13.md)", commit=runinfo.commit())
    if (split / "lm_loss.json").exists():
        report["item1_k8b_split"] = dict(sets={k: v for k, v in read(split / "sets.json").items()
                                               if not k.startswith(("rand_", "random_"))},
                                         language=language(split), m1=m1(split, items, fm),
                                         held_out=held_out(held, held / "dissociation_k8b", kitems, median),
                                         word_length=length_table(split / "naming_length.parquet"),
                                         target_rank=target_rank(split))
    report["item2_control_positions"] = positions(held, old / "control", items, fm)
    out = Path(args.out) if args.out else ROOT / ("outputs/session13_tiny.json" if args.tiny else "reports/explore/session13.json")
    out.write_bytes((json.dumps(report, indent=1, ensure_ascii=False, default=float) + "\n").encode("utf-8"))
    print(f"written to {out}")


if __name__ == "__main__":
    main()

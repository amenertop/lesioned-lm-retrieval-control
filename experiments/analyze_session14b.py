"""GPU session 14b, exploratory: one report from the run outputs. No GPU:

    python experiments/analyze_session14b.py [--tiny] [--out reports/explore/session14b.json]

Step 1, K8 and K8b-2 lesioned together (outputs/e3_knowledge/<model>/joint_sets):
- held-out naming per state: knowledge effect and naming damage (intervals, items resampled; scores averaged over
  three batch compositions); the joint set among its matched random sets (one composition for both, as the
  dissociation test);
- additivity on the same items: joint - (K8 + K8b-2) for both measures (interval), and joint / (K8 + K8b-2)
  (descriptive);
- which words each set hurts: on the rare words, the correlation of K8's and K8b-2's item damage, and how many of
  K8's 25% most damaged rare words are also among K8b-2's (chance: 25%);
- held-out M1 per state (semantic, string and control damage, margin-scaling slope); the language-model loss change
  per state; the joint set among its random sets on both.
Reading, written before the run: the interaction's interval above 0 means more than additive (each set makes up part
of what the other loses: a backup); containing 0, additive (two independent contributions); below 0, less than
additive (an overlap: the same work done twice, or a floor on the scores). A floor is checked separately: the share
of rare words whose intact answer is already improbable cannot explain a shortfall if it is small.
Step 2, C4 in the three label parts (outputs/e2_confirm/<model>/control_positions_labels): as session 13's positions
(analyze_session13.positions) for label_a, label_b and label_answer, plus session 13's whole labels part on the same
prompts for reference.
Step 3, the taxonomy (outputs/e3_knowledge/<model>/profile_pool/profiles.csv): Ward clustering of the components with
a complete profile, on the seven features ranked and standardised (ranks, so that a few destructive components do not
decide the scale), the number of clusters by mean silhouette (2 to 6); each cluster's median profile and members,
and the cluster of every member of K8, K8b-2, the shared pair and S8.
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

from analyze_session12 import interval, read  # noqa: E402
from analyze_session13 import among, positions  # noqa: E402
from lesion import damage, naming, probes, runinfo, scoring  # noqa: E402

Q = "qwen3-30b-a3b-instruct"
N_BOOT = 4000
SETS = ("K8", "K8b_minus_shared", "joint")
FEATURES = ("knowledge_effect", "naming_damage", "m1_semantic_damage", "m1_string_damage", "m1_control_effect",
            "m1_slope", "lm_loss_change")
FLOOR_LOGPROB = -10.0   # an intact answer this improbable has little left to lose (descriptive)
MIN_SUM = 0.1           # K8 + K8b-2 below this (log-probability units), the ratio joint / sum is not reported


def joint(d, kitems, median, items, fm):
    every = pd.read_parquet(d / "naming.parquet").astype({"item_id": str})
    n = every[every.scoring == "averaged"]
    base = n[n.state == "intact"]
    h = {s: naming.item_damage(base, g) for s, g in n.groupby("state") if s != "intact"}
    point = {s: naming.effects(damage.columns(naming.item_frame(v, kitems, median))) for s, v in h.items()}
    one = every[every.scoring == "single"]
    one_base = one[one.state == "intact"]
    single = {s: naming.effects(damage.columns(naming.item_frame(naming.item_damage(one_base, g), kitems, median)))
              for s, g in one.groupby("state") if s != "intact"}
    frame = naming.item_frame(pd.DataFrame({s: h[s] for s in SETS}), kitems, median)

    def stat(c):
        e = {s: naming.effects(c, s) for s in SETS}
        out = {f"{s}.{m}": e[s][m] for s in SETS for m in ("knowledge_effect", "naming_damage")}
        for m in ("knowledge_effect", "naming_damage"):
            out[f"interaction.{m}"] = e["joint"][m] - e["K8"][m] - e["K8b_minus_shared"][m]
        return out

    boot = damage.bootstrap(frame, stat, N_BOOT)
    rand = {s: v for s, v in single.items() if s.startswith("random_")}
    total = {m: point["K8"][m] + point["K8b_minus_shared"][m] for m in ("knowledge_effect", "naming_damage")}
    rare = frame[frame.rare]
    top = {s: set(rare.nlargest(max(1, len(rare) // 4), s).item_id) for s in ("K8", "K8b_minus_shared")}
    intact_rare = base.set_index("item_id").logprob.reindex(rare.item_id)
    out = dict(intervals={k: interval(v) for k, v in boot.items()},
               # descriptive; not given when the sum is too small for a ratio to mean anything
               ratio_joint_over_sum={m: round(point["joint"][m] / total[m], 3) if abs(total[m]) >= MIN_SUM else None
                                     for m in total},
               joint_vs_random={m: among(single["joint"][m], [v[m] for v in rand.values()])
                                for m in ("knowledge_effect", "naming_damage")},
               single_scoring={s: {m: round(single[s][m], 4) for m in ("knowledge_effect", "naming_damage")} for s in SETS},
               rare_words=dict(n=int(len(rare)),
                               damage_correlation=round(float(np.corrcoef(rare.K8, rare.K8b_minus_shared)[0, 1]), 3),
                               top_quarter_overlap=round(len(top["K8"] & top["K8b_minus_shared"]) / len(top["K8"]), 3),
                               chance=0.25,
                               share_intact_below_floor=round(float((intact_rare < FLOOR_LOGPROB).mean()), 3)))
    m = pd.read_parquet(d / "m1.parquet").astype({"item_id": str})
    mbase = m[m.state == "intact"]
    mpoint = {}
    for s, les in m.groupby("state"):
        if s == "intact":
            continue
        e = damage.effects(damage.columns(damage.item_frame(damage.item_damage(mbase, les), items, fm)))
        slope = damage.margin_scaling(mbase, les, items)
        mpoint[s] = dict(semantic_damage=e["semantic_damage"], string_damage=e["string_damage"],
                         control_effect=e["control_effect"], slope=float(slope[slope.task == "all"].slope.iloc[0]))
    mrand = [v for s, v in mpoint.items() if s.startswith("random_")]
    out["m1"] = {s: {k: round(v, 4) for k, v in mpoint[s].items()} for s in SETS}
    out["m1"]["joint_vs_random"] = {k: among(mpoint["joint"][k], [v[k] for v in mrand]) for k in ("semantic_damage", "slope")}
    lm = read(d / "lm_loss.json")
    change = {s: v / lm["intact"] - 1 for s, v in lm.items() if s != "intact"}
    out["lm_loss_change"] = {s: round(change[s], 5) for s in SETS}
    out["lm_loss_change"]["joint_vs_random"] = among(change["joint"], [v for s, v in change.items() if s.startswith("random_")])
    out["n_random"] = len(rand)
    return out


def labels(held, items, fm):
    d = held / "control_positions_labels"
    if not (d / "scores.parquet").exists():
        return None
    # no session 10 reference here (old_control None); session 13's labels part is added below as the reference
    out = positions(held, None, items, fm, name="control_positions_labels", parts=probes.LABEL_PARTS)
    old = held / "control_positions" / "scores.parquet"
    if old.exists():
        new = pd.read_parquet(d / "scores.parquet").astype({"item_id": str})
        s13 = pd.read_parquet(old).astype({"item_id": str})
        a, b = (x[x.state == "intact"].sort_values(["item_id", "order"]).reset_index(drop=True) for x in (new, s13))
        same = len(a) == len(b) and bool(np.allclose(a.drop(columns=["state", "set", "part"]).select_dtypes("number"),
                                                     b.drop(columns=["state", "set", "part"]).select_dtypes("number")))
        h = damage.item_damage(s13[s13.state == "intact"], s13[s13.state == "C4_labels"])
        boot = damage.bootstrap(damage.item_frame(h, items, fm),
                                lambda c: {"control_effect": damage.effects(c)["control_effect"]}, N_BOOT)
        out["session13_labels_part"] = dict(control_effect=interval(boot["control_effect"]), intact_identical=same)
    return out


def taxonomy(path, groups):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform

    t = pd.read_csv(path)
    full = t.dropna(subset=list(FEATURES)).reset_index(drop=True)
    x = full[list(FEATURES)].rank().to_numpy()
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    tree = linkage(x, method="ward")
    dist = squareform(pdist(x))

    def silhouette(labels_):
        s = []
        for i in range(len(labels_)):
            same = labels_ == labels_[i]
            if same.sum() < 2:
                s.append(0.0)
                continue
            a = dist[i, same].sum() / (same.sum() - 1)
            b = min(dist[i, labels_ == k].mean() for k in set(labels_) if k != labels_[i])
            s.append((b - a) / max(a, b))
        return float(np.mean(s))

    sil = {k: silhouette(fcluster(tree, k, criterion="maxclust")) for k in range(2, min(6, len(full) - 1) + 1)}
    k = max(sil, key=sil.get)
    full["cluster"] = fcluster(tree, k, criterion="maxclust")
    clusters = {int(c): dict(n=int(len(g)), median_profile={f: round(float(g[f].median()), 4) for f in FEATURES},
                             members=sorted(g.component))
                for c, g in full.groupby("cluster")}
    where = {name: {m: (int(full.cluster[full.component == m].iloc[0]) if (full.component == m).any() else None)
                    for m in members} for name, members in groups.items()}
    return dict(n_rows=int(len(t)), n_complete=int(len(full)), features=list(FEATURES), transform="rank, then z-score",
                silhouette={int(kk): round(v, 3) for kk, v in sil.items()}, n_clusters=int(k), clusters=clusters,
                sets=where)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suf = "_tiny" if args.tiny else ""
    kroot = ROOT / f"outputs/e3_knowledge{suf}" / Q
    held = ROOT / f"outputs/e2_confirm{suf}" / Q
    items = scoring.load_materials()[0].astype({"item_id": str})
    fm = damage.frequency_median(items)
    kitems = pd.read_csv(ROOT / "data/materials/m3_knowledge/items.csv", dtype={"item_id": str})
    median = naming.zipf_median(kitems)
    report = dict(status="exploratory (GPU session 14b, experiments/gpu_session14b.sh)", commit=runinfo.commit())
    d = kroot / "joint_sets"
    if (d / "meta.json").exists() and read(d / "meta.json").get("status") == "done":
        sets = read(d / "sets.json")
        report["step1_joint_sets"] = dict(sets={s: sets[s] for s in SETS}, **joint(d, kitems, median, items, fm))
    report["step2_label_parts"] = labels(held, items, fm)
    p = kroot / "profile_pool" / "profiles.csv"
    if p.exists():
        groups = {}
        if (kroot / "k8b_split/sets.json").exists():
            split = read(kroot / "k8b_split/sets.json")
            groups.update({k: split[k] for k in ("K8b_minus_shared", "shared_pair") if k in split})
            if "S8_minus_pair" in split and "shared_pair" in split:
                groups["S8"] = split["S8_minus_pair"] + split["shared_pair"]      # session 11's store set
        if (d / "sets.json").exists():
            groups["K8"] = read(d / "sets.json")["K8"]
        report["step3_taxonomy"] = taxonomy(p, groups)
    out = Path(args.out) if args.out else ROOT / ("outputs/session14b_tiny.json" if args.tiny else "reports/explore/session14b.json")
    out.write_bytes((json.dumps(report, indent=1, ensure_ascii=False, default=float) + "\n").encode("utf-8"))
    print(f"written to {out}")


if __name__ == "__main__":
    main()

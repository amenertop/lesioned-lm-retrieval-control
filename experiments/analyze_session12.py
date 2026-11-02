"""GPU session 12 (docs/plans/gpu-session12.md), exploratory: one report of the five items, from the run outputs. No GPU:

    python experiments/analyze_session12.py [--tiny] [--out reports/explore/session12.json]

1. Phi's knowledge set taken apart: LM loss change of every state, the two-by-two (mean or zero, first or other
   tokens), naming damage and frequency effect, the two experts' output lengths; item 1b's held-out test if it ran.
2. Rare words and K8, main analysis on the 240 held-out targets: the change of the target's first-token
   log-probability regressed on -Zipf (standardised), controlling the word's token count and the intact
   log-probability, concept-resampled; mean change by Zipf fifth. Exploratory: the whole-vocabulary Spearman
   correlation with word frequency at the naming answer and in neutral text, mean and interval over positions.
3. Shared components removed: each control test's effect, the random sets' median, 95% range and the target's
   quantile among them, and the paired difference to the full set (C4 from session 10, C4s from session 11) on the
   same held-out items; the dissociation test without the shared components.
4. Task vectors: control effect and mean damage per state, and the paired differences other task minus same task
   and other task minus random direction.
5. K8b: the selection with its exclusions, the dev test, the held-out dissociation test, and K8b's margin scaling.
6. Rare words or long words (added after agent4's review): on the 645 unused words, the damage regressed on -Zipf, the
   token count and the intact log-probability; per-token and two-token-only rare-minus-common damage.
7. Rerouting instead of ablation for Phi's experts: in item 1's states (super_both_reroute, K8_experts_mean, K8_experts_reroute).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lesion import damage, runinfo, scoring  # noqa: E402

Q, P = "qwen3-30b-a3b-instruct", "phi35-moe-instruct"
N_BOOT, SEED = 4000, 0
BIG_NORM = 100.0


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def interval(v):
    return [round(v["estimate"], 4), round(v["ci_low"], 4), round(v["ci_high"], 4)]


def resample(values_by_group, stat, n=N_BOOT, seed=SEED):
    """Point estimate and 95% interval of stat(rows) resampling groups (a DataFrame with a 'group' column)."""
    frame = values_by_group.reset_index(drop=True)
    codes, groups = pd.factorize(frame.group)
    rows = [np.flatnonzero(codes == g) for g in range(len(groups))]
    rng = np.random.default_rng(seed)
    point = stat(frame)
    draws = [stat(frame.iloc[np.concatenate([rows[g] for g in rng.integers(len(groups), size=len(groups))])])
             for _ in range(n)]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return [round(float(point), 4), round(float(lo), 4), round(float(hi), 4)]


def phi_split(root):
    d = root["k"] / P / "split"
    if not (d / "results.json").exists():
        return None
    r = read(d / "results.json")
    intact = r["lm_loss"]["intact"]
    out = dict(lm_loss_change={k: round(v / intact - 1, 4) for k, v in r["lm_loss"].items()},
               naming={k: {e: interval(v[e]) for e in ("naming_damage", "knowledge_effect")} for k, v in r["naming"].items()},
               norms=r["norms"], greedy={k: v[:6] for k, v in r["greedy"].items()})
    if (d / "norms.parquet").exists():
        # which tokens get the very large outputs after the first position (added after the first reading: expert:5:10
        # has a 99th percentile of about 1,670 there); token ids only, decoding needs the Phi tokenizer
        n = pd.read_parquet(d / "norms.parquet")
        big = n[(n.position > 0) & (n.norm > BIG_NORM)]
        out["large_outputs_after_first"] = {
            e: dict(n_rows=int((g.position > 0).sum()), n_large=int(len(big[big.expert == e])),
                    by_token_id={int(i): dict(n_large=int(c), n_routed=int(((g.position > 0) & (g.token_id == i)).sum()))
                                 for i, c in big[big.expert == e].token_id.value_counts().head(5).items()})
            for e, g in n.groupby("expert")}
    held = root["c"] / P / "dissociation_split" / "results.json"
    if held.exists():
        h = read(held)
        out["item_1b"] = dict(h2=h.get("h2"), random=h.get("random"),
                              naming={k: interval(v) for k, v in h["sets"]["main"]["naming"].items() if isinstance(v, dict) and "estimate" in v})
    return out


def vocab(root, items):
    d = root["e"] / Q / "mechanisms"
    if not (d / "naming_first.parquet").exists():
        return None
    t = pd.read_parquet(d / "naming_first.parquet").merge(items[["item_id", "zipf"]].astype({"item_id": str}), on="item_id")
    base = t[t.state == "intact"].set_index("item_id")
    out = {}
    for state in t.state.unique():
        if state == "intact":
            continue
        s = t[t.state == state].set_index("item_id")
        f = pd.DataFrame(dict(change=s.first_logprob - base.first_logprob.reindex(s.index), zipf=base.zipf, n_tokens=base.n_tokens,
                              intact=base.first_logprob)).reset_index().rename(columns={"item_id": "group"})
        z = lambda v: (v - v.mean()) / v.std() if v.std() > 0 else v * 0  # noqa: E731

        def rarity(x):
            X = np.column_stack([np.ones(len(x)), z(-x.zipf), z(x.n_tokens), z(x.intact)])
            return np.linalg.lstsq(X, x.change.to_numpy(), rcond=None)[0][1]

        fifths = pd.qcut(f.zipf, 5, labels=False, duplicates="drop")
        out[state] = dict(rarity_coef=resample(f, rarity),
                          mean_change_by_zipf_fifth=[round(float(v), 4) for v in f.groupby(fifths).change.mean()])
    res = {"targets": out}
    for name in ("vocab_naming", "vocab_neutral"):
        v = pd.read_parquet(d / f"{name}.parquet")
        res[name] = {}
        for state, g in v.groupby("state"):
            rho = g.spearman.dropna().to_numpy()
            rng = np.random.default_rng(SEED)
            boot = [rng.choice(rho, len(rho)).mean() for _ in range(N_BOOT)] if len(rho) else [np.nan]
            res[name][state] = dict(spearman=[round(float(rho.mean()), 4) if len(rho) else None,
                                              *[round(float(q), 4) for q in np.nanpercentile(boot, [2.5, 97.5])]],
                                    mean_change_by_bin=[round(float(g[f"bin{k}"].mean()), 4) for k in range(5)])
    return res


def length_check(root):
    """Item 6: is K8's frequency effect the number of tokens? On the unused words: the damage regressed on -Zipf, the
    word's token count and the intact log-probability (standardised), concept-resampled; the rare-minus-common damage
    per token; and the rare-minus-common damage among two-token words only (Zipf below 3 is rare, as the items' median)."""
    d = root["e"] / Q / "mechanisms" / "naming_length.parquet"
    return length_table(d) if d.exists() else None


def length_table(path):
    """length_check on one naming_length.parquet (states: intact and lesions); also used for session 13's K8b."""
    t = pd.read_parquet(path)
    base = t[t.state == "intact"].set_index("item_id")
    out = dict(n_words=int(len(base)), n_tokens=base.n_tokens.value_counts().sort_index().to_dict(),
               corr_zipf_tokens=round(float(np.corrcoef(base.zipf, base.n_tokens)[0, 1]), 3))
    z = lambda v: (v - v.mean()) / v.std() if v.std() > 0 else v * 0  # noqa: E731
    for state in t.state.unique():
        if state == "intact":
            continue
        s = t[t.state == state].set_index("item_id").reindex(base.index)
        f = pd.DataFrame(dict(h=base.logprob - s.logprob, zipf=base.zipf, n_tokens=base.n_tokens, intact=base.logprob)
                         ).reset_index().rename(columns={"item_id": "group"})
        f["rare"] = f.zipf < 3.0
        f["h_token"] = f.h / (f.n_tokens + 1)        # the word's tokens plus the end-of-turn token

        def coef(k):
            def stat(x):
                X = np.column_stack([np.ones(len(x)), z(-x.zipf), z(x.n_tokens), z(x.intact)])
                return np.linalg.lstsq(X, x.h.to_numpy(), rcond=None)[0][k]
            return stat

        rare_minus_common = lambda col: lambda x: x[col][x.rare].mean() - x[col][~x.rare].mean()  # noqa: E731
        two = f[f.n_tokens == 2]
        out[state] = dict(rarity_coef=resample(f, coef(1)), n_tokens_coef=resample(f, coef(2)),
                          per_token_rare_minus_common=resample(f, rare_minus_common("h_token")),
                          two_token_rare_minus_common=resample(two, rare_minus_common("h")) if two.rare.any() and (~two.rare).any() else None,
                          n_two_token=[int((~two.rare).sum()), int(two.rare.sum())])
    return out


def control_profile(folder, items, fm, full=None):
    """A control test's effect, its random sets' distribution, and (with `full`, another control test folder) the
    paired difference full minus this one on the same items."""
    r = read(folder / "results.json")
    intact = pd.read_csv(folder / "intact.csv", dtype={"item_id": str})
    les = pd.read_parquet(folder / "lesioned.parquet").astype({"item_id": str})

    def effect(scores):
        frame = damage.item_frame(damage.item_damage(intact, scores), items, fm)
        return damage.effects(damage.columns(frame))["control_effect"]

    target = r["intervals"]["control_effect"]["estimate"]
    rand = np.array([effect(g) for s, g in les.groupby("set") if s != "control"])
    out = dict(control_effect=interval(r["intervals"]["control_effect"]),
               string_damage=interval(r["intervals"]["string_damage"]),
               random=dict(n=len(rand), median=round(float(np.median(rand)), 4) if len(rand) else None,
                           range95=[round(float(q), 4) for q in np.percentile(rand, [2.5, 97.5])] if len(rand) else None,
                           target_quantile=round(float((rand < target).mean()), 4) if len(rand) else None))
    if full is not None and (full / "lesioned.parquet").exists():
        fi = pd.read_csv(full / "intact.csv", dtype={"item_id": str})
        fl = pd.read_parquet(full / "lesioned.parquet").astype({"item_id": str})
        fl = fl[fl.set == "control"]
        mine = les[les.set == "control"]
        h = pd.DataFrame({"h_full": damage.item_damage(fi, fl), "h_part": damage.item_damage(intact, mine)}).dropna()
        frame = damage.item_frame(h, items, fm)
        out["paired_full_minus_this"] = interval(damage.bootstrap(
            frame, lambda c: {"d": damage.effects(c, "h_full")["control_effect"] - damage.effects(c, "h_part")["control_effect"]},
            N_BOOT)["d"])
        out["intact_identical_to_full"] = bool(np.allclose(fi.sort_values(["item_id", "order"]).logit_correct.to_numpy(),
                                                           intact.sort_values(["item_id", "order"]).logit_correct.to_numpy()))
    return out


def shared(root, items, fm):
    base = root["c"] / Q
    full_c4 = ROOT / "outputs/server/e2_confirm" / Q / "control" if not root["tiny"] else None
    full_c4s = base / "control_semantic" if not root["tiny"] else None
    out = {}
    for name, full in (("c4_minus_shared", full_c4), ("shared_head_alone", full_c4), ("c4s_minus_shared", full_c4s)):
        if (base / name / "results.json").exists():
            out[name] = control_profile(base / name, items, fm, full)
    d = base / "dissociation_minus_shared"
    if (d / "results.json").exists():
        r = read(d / "results.json")
        m1 = pd.read_parquet(d / "m1_scores.parquet").astype({"item_id": str})
        out["dissociation_minus_shared"] = dict(
            h2=r["h2"], random=r["random"],
            sets=both_parts(r),
            knowledge_set_scaling=damage.margin_scaling(m1[m1.set == "intact"], m1[m1.set == "main"], items).round(3).to_dict("records"))
    return out


def both_parts(r):
    """A dissociation test's intervals for naming and M1, keyed part.statistic (both parts use the same statistic names)"""
    return {f"{part}.{k}": interval(v) for part in ("naming", "m1") for k, v in r["sets"]["main"][part].items()
            if isinstance(v, dict) and "estimate" in v}


def patch(root, items, fm):
    d = root["e"] / Q / "mechanisms"
    if not (d / "patch_scores.parquet").exists():
        return None
    t = pd.read_parquet(d / "patch_scores.parquet").astype({"item_id": str})
    base = t[t.state == "intact"]
    hs = {s: damage.item_damage(base[base.item_id.isin(set(g.item_id))], g) for s, g in t.groupby("state") if s != "intact"}
    frame = damage.item_frame(pd.DataFrame({f"h_{s}": h for s, h in hs.items()}), items, fm)
    stat = lambda c: {**{f"{s}.control_effect": damage.effects(c, f"h_{s}")["control_effect"] for s in hs},  # noqa: E731
                      **{f"{s}.mean_damage": float(np.mean(c[f"h_{s}"])) for s in hs},
                      **({"other_minus_same": float(np.mean(c["h_other_task"]) - np.mean(c["h_same_task"])),
                          "other_minus_random_direction": float(np.mean(c["h_other_task"]) - np.mean(c["h_random_direction"]))}
                         if {"other_task", "same_task", "random_direction"} <= set(hs) else {})}
    out = {k: interval(v) for k, v in damage.bootstrap(frame, stat, N_BOOT).items()}
    if (d / "task_vectors.pt").exists():
        out["vector_geometry"] = task_vector_geometry(d / "task_vectors.pt")
    return out


def task_vector_geometry(path):
    """Per head: the lengths of the colour and size vectors, their cosine and |colour - size| / |colour| (added after
    the first reading: other-task and same-task patches were equally harmless)"""
    import torch
    v = torch.load(path, map_location="cpu", weights_only=False)["vectors"]
    a, b = v["colour"], v["size"]
    return {h: dict(norm_colour=round(float(a[h].norm()), 3), norm_size=round(float(b[h].norm()), 3),
                    cosine=round(float(torch.nn.functional.cosine_similarity(a[h], b[h], dim=0)), 4),
                    relative_difference=round(float((a[h] - b[h]).norm() / a[h].norm()), 4)) for h in a}


def k8b(root, items):
    d = root["k"] / Q / "k8b"
    if not (d / "selection.json").exists():
        return None
    out = dict(selection=read(d / "selection.json"))
    if (d / "joint.json").exists():
        j = read(d / "joint.json")
        out["dev"] = dict(knowledge_effect=interval(j["k8b_on_naming"]["knowledge_effect"]) if "k8b_on_naming" in j else None,
                          m1=j.get("k8b_on_m1"), lm_loss_change=j.get("k8b_lm_loss_change"), specific=j.get("specific"))
    h = root["c"] / Q / "dissociation_k8b"
    if (h / "results.json").exists():
        r = read(h / "results.json")
        m1 = pd.read_parquet(h / "m1_scores.parquet").astype({"item_id": str})
        out["held_out"] = dict(h2=r["h2"], random=r["random"],
                               sets=both_parts(r),
                               scaling=damage.margin_scaling(m1[m1.set == "intact"], m1[m1.set == "main"], items).round(3).to_dict("records"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suf = "_tiny" if args.tiny else ""
    root = dict(tiny=args.tiny, k=ROOT / f"outputs/e3_knowledge{suf}", c=ROOT / f"outputs/e2_confirm{suf}",
                e=ROOT / f"outputs/e3_errors{suf}")
    items = scoring.load_materials()[0].astype({"item_id": str})
    fm = damage.frequency_median(items)
    kitems = pd.read_csv(ROOT / "data/materials/m3_knowledge/items.csv", dtype={"item_id": str})
    report = dict(status="exploratory (GPU session 12, docs/plans/gpu-session12.md)", commit=runinfo.commit(),
                  item1_phi_split=phi_split(root), item2_rare_words=vocab(root, kitems), item6_word_length=length_check(root),
                  item3_shared_removed=shared(root, items, fm), item4_task_vectors=patch(root, items, fm),
                  item5_k8b=k8b(root, items))
    out = Path(args.out) if args.out else ROOT / ("outputs/session12_tiny.json" if args.tiny else "reports/explore/session12.json")
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()

"""Analyse the cross-format mechanism battery (run_battery.py output; no GPU):

    python experiments/e3_errors/analyze_battery.py [--dir outputs/e3_errors/<model>/battery]

Per lesion state, changes from intact, per concept, with percentile intervals over concepts (5,000 resamples):
- name_free: the target's log-probability (naming loss) and the five-candidate shares;
- greedy: the share of answers that are the target, and the kinds of the others;
- options_word, copy: the target's log-probability;
- options_letter, word_to_definition: the margin of the correct label (mean over rotations or orders) and the
  share of concepts still correct in every rotation or order;
- hints: cue benefit B and miscue harm H against the hinted-demo baseline, A = B - H, and the evidence weights
  (a, b_cue, b_miscue), with the plain no-hint baseline as a check;
- item consistency: across concepts, naming loss against the comprehension (word_to_definition), six-choice and
  copy changes (partial semantic loss predicts a positive relation with comprehension loss);
- every change also in units of the intact between-concept SD of that score, so formats can be compared;
- the answer- and prompt-restricted lesions as shares of the whole lesion's naming loss;
- logit lens: mean target log-probability per layer, and where the intact model first ranks the target first;
- C4 heads' attention to the hint letters, cue against miscue.
Writes reports/explore/qwen_m3_battery.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import e3_analysis as ea, m3_items, m3_scoring as ms  # noqa: E402

N_BOOT, SEED = 5000, 0


def ci(x, rng):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(mean=None, ci=[None, None], n=0)
    b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return dict(mean=round(float(x.mean()), 3), ci=[round(float(np.quantile(b, q)), 3) for q in (0.025, 0.975)], n=int(len(x)))


def spearman(x, y):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def corr_ci(x, y, rng):
    x, y = np.asarray(x), np.asarray(y)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(rho=None, ci=[None, None], n=int(len(x)))
    r = [spearman(x[i], y[i]) for i in rng.integers(0, len(x), (N_BOOT, len(x)))]
    return dict(rho=round(spearman(x, y), 3), ci=[round(float(np.nanquantile(r, q)), 3) for q in (0.025, 0.975)], n=int(len(x)))


def per_concept(cand, labels):
    """Per state, per concept scores of every format (index item_id)."""
    out = {}
    for s, c in cand.groupby("state"):
        t = {}
        free = c[c.condition == "name_free"]
        e = ms.naming_errors(free).set_index("item_id")
        t["name_target_logprob"] = e.target_logprob
        t["name_p_target"] = e.p_target
        t["name_S"] = e.S
        for fmt in ("options_word", "copy"):
            f = c[(c.condition == fmt) & (c.role == "target")].set_index("item_id").logprob
            t[f"{fmt}_target_logprob"] = f
        d = {}
        # the plain no-hint baseline is name_free (same prompt, five candidates including target and associate)
        for cond, src in (("none", "name_free"), ("none_hinted_demos", "none_hinted_demos"), ("cue", "cue"), ("miscue", "miscue")):
            g = c[c.condition == src].pivot_table(index="item_id", columns="role", values="logprob")
            if len(g):
                d[cond] = g.target - g.associate
        for cond, v in d.items():
            t[f"delta_{cond}"] = v
        lab = labels[labels.state == s]
        for fmt in ("options_letter", "word_to_definition"):
            g = lab[lab.fmt == fmt].groupby("item_id")
            t[f"{fmt}_margin"] = g.margin.mean()
            t[f"{fmt}_all_correct"] = g.top_correct.all().astype(float)
        out[s] = pd.DataFrame(t)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "outputs/e3_errors/qwen3-30b-a3b-instruct/battery"))
    ap.add_argument("--out", default=str(ROOT / "reports/explore/qwen_m3_battery.json"))
    args = ap.parse_args(argv)
    d = Path(args.dir)
    rng = np.random.default_rng(SEED)
    cand = pd.read_parquet(d / "candidates.parquet")
    labels = pd.read_parquet(d / "labels.parquet")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    pc = per_concept(cand, labels)
    I = all_intact = pc["intact"]
    out = dict(meta=dict(states=meta["states"], positional=meta["positional"], commit=meta.get("git_commit")),
               intact={c: round(float(I[c].mean()), 3) for c in I.columns}, states={})
    scores = ["name_target_logprob", "name_p_target", "name_S", "options_word_target_logprob", "copy_target_logprob",
              "options_letter_margin", "word_to_definition_margin"]
    for s, T in pc.items():
        if s == "intact":
            continue
        I = all_intact
        if s.endswith("_taskmean"):
            roles = meta.get("item_roles", {})
            b_ids = [i for i in I.index if roles.get(i) == "calibration_b"]
            if not b_ids:
                raise ValueError("task-mean analysis requires calibration_b item_roles in metadata")
            I = I.loc[b_ids]
        sd = I.std()
        T = T.reindex(I.index)
        r = dict(change={}, change_in_intact_sd={}, still_all_correct={}, n_concepts=len(I),
                 cohort="calibration_b" if s.endswith("_taskmean") else "calibration_a_and_b",
                 reference_intact={c: round(float(I[c].mean()), 3) for c in scores})
        for c in scores:
            r["change"][c] = ci(T[c] - I[c], rng)
            r["change_in_intact_sd"][c] = round(float((T[c] - I[c]).mean() / sd[c]), 3) if sd[c] > 0 else None
        if s.endswith("_taskmean"):
            wiki = pc[s.removesuffix("_taskmean")].reindex(I.index)
            r["mean_reference_comparison"] = {
                c: dict(wiki_change=ci(wiki[c] - I[c], rng), task_minus_wiki=ci(T[c] - wiki[c], rng))
                for c in scores}
        for fmt in ("options_letter", "word_to_definition"):
            r["still_all_correct"][fmt] = dict(intact=round(float(I[f"{fmt}_all_correct"].mean()), 3),
                                               lesion=round(float(T[f"{fmt}_all_correct"].mean()), 3))
        if "foil_match" in labels.columns:
            strict_ids = labels.loc[(labels.state == "intact") & (labels.fmt == "word_to_definition") &
                                   (labels.foil_match == "category"), "item_id"].unique()
            strict_ids = I.index.intersection(strict_ids)
            r["same_category_definition"] = dict(
                n_concepts=len(strict_ids),
                margin_change=ci(T.loc[strict_ids, "word_to_definition_margin"] -
                                 I.loc[strict_ids, "word_to_definition_margin"], rng),
                naming_loss_vs_margin_change=corr_ci(
                    I.loc[strict_ids, "name_target_logprob"] - T.loc[strict_ids, "name_target_logprob"],
                    T.loc[strict_ids, "word_to_definition_margin"] - I.loc[strict_ids, "word_to_definition_margin"], rng))
        # hints: B, H, A against the hinted-demo baseline (and the plain baseline), evidence weights
        for base in ("none_hinted_demos", "none"):
            cols = [f"delta_{base}", "delta_cue", "delta_miscue"]
            ok = I[cols].notna().all(axis=1) & T[cols].notna().all(axis=1)
            Ii, Tt = I.loc[ok, cols].to_numpy(), T.loc[ok, cols].to_numpy()
            B = (Tt[:, 1] - Tt[:, 0]) - (Ii[:, 1] - Ii[:, 0])
            H = -((Tt[:, 2] - Tt[:, 0]) - (Ii[:, 2] - Ii[:, 0]))
            r[f"hints_vs_{base}"] = dict(cue_benefit_change=ci(B, rng), miscue_harm_change=ci(H, rng), A=ci(B - H, rng),
                                         R_change=ci(B + H, rng),
                                         evidence_weights=ea.evidence_weights_bootstrap(Ii, Tt, n_boot=2000, seed=SEED))
        # item consistency: naming loss against the other formats
        loss = I["name_target_logprob"] - T["name_target_logprob"]
        r["item_consistency"] = {f"naming_loss_vs_{c}_change": corr_ci(loss, T[c] - I[c], rng)
                                 for c in ("word_to_definition_margin", "options_letter_margin", "copy_target_logprob",
                                           "options_word_target_logprob")}
        out["states"][s] = r

    # positional shares of the whole lesion's naming loss
    I = all_intact
    for k in ("K8", "C4"):
        if f"{k}_answer" in pc:
            loss = I.name_target_logprob - pc[k].reindex(I.index).name_target_logprob
            whole = loss.mean()
            whole_ci = ci(loss, rng)
            out.setdefault("positional", {})[k] = {
                r: dict(naming_loss=round(float((I.name_target_logprob - pc[f"{k}_{r}"].name_target_logprob).mean()), 3),
                        whole_naming_loss=whole_ci,
                        share_of_whole=round(float((I.name_target_logprob - pc[f"{k}_{r}"].name_target_logprob).mean() / whole), 3)
                        if whole_ci["ci"][0] > 0 else None)
                for r in ("answer", "prompt")}

    if (d / "greedy.parquet").exists():
        g = pd.read_parquet(d / "greedy.parquet")
        out["greedy"], out["greedy_n"] = {}, {}
        for s, x in g.groupby("state"):
            if s.endswith("_taskmean"):
                ids = {i for i, role in meta["item_roles"].items() if role == "calibration_b"}
                x = x[x.item_id.isin(ids)]
                out.setdefault("greedy_reference_comparison", {})[s] = {
                    ref: g[(g.state == ref) & g.item_id.isin(ids)].kind.value_counts(normalize=True).round(3).to_dict()
                    for ref in ("intact", s.removesuffix("_taskmean"))}
            out["greedy"][s] = x.kind.value_counts(normalize=True).round(3).to_dict()
            out["greedy_n"][s] = len(x)
        ex = g[(g.state == "K8") & (g.kind != "target")].merge(g[g.state == "intact"][["item_id", "kind"]], on="item_id",
                                                               suffixes=("", "_intact"))
        out["greedy_examples_K8"] = ex[ex.kind_intact == "target"].head(15)[["item_id", "answer", "kind"]].to_dict("records")
    if (d / "lens.parquet").exists():
        lens = pd.read_parquet(d / "lens.parquet")
        prof = lens.groupby(["state", "layer"]).logprob.mean().unstack(0)
        out["lens_mean_target_logprob"] = {s: [round(float(v), 2) for v in prof[s]] for s in prof.columns}
        first = lens[lens.state == "intact"].groupby("layer")["rank"].apply(lambda x: float((x == 0).mean()))
        out["lens_intact_share_target_top1_by_layer"] = [round(float(v), 3) for v in first]
    if (d / "hint_attention.parquet").exists():
        att = pd.read_parquet(d / "hint_attention.parquet")
        out["hint_attention"] = att.groupby(["state", "head", "condition"]).to_hint.mean().round(4).unstack().to_dict("index")
        out["hint_attention"] = {f"{k[0]} {k[1]}": v for k, v in out["hint_attention"].items()}

    path = Path(args.out)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for s, r in out["states"].items():
        ch = r["change"]
        print(f"{s:12s} naming {ch['name_target_logprob']['mean']:+.2f} | w2d {ch['word_to_definition_margin']['mean']:+.2f} "
              f"| letter {ch['options_letter_margin']['mean']:+.2f} | word-shown {ch['options_word_target_logprob']['mean']:+.2f} "
              f"| copy {ch['copy_target_logprob']['mean']:+.2f} | A {r['hints_vs_none_hinted_demos']['A']['mean']:+.2f}")
    print(f"written to {path}")


if __name__ == "__main__":
    main()

"""P14 step 2 (docs/plans/p14-learned-gates.md), exploratory: pick each objective's set from the training runs, then
test whether it is minimal and sufficient, on items no choice was made on.

    python experiments/p14/confirm.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16 \
        --freeze-tag analysis-frozen-session15

1. The set (no GPU): for each objective (knowledge, hub, control), among the runs on its largest pool, the run whose
   chosen size (train_gates.py, on the select split) reaches the highest select-split selectivity; the set is that
   run's first k components. Stability across the runs (seeds and sparsity weights): every component's selection
   frequency, the mean Jaccard index, the Spearman correlation of g between runs and the overlap of their first
   |set| components; and, for the knowledge objective, the sets grown on the smaller pool against the main one.
2. Minimality, real mean ablation (lesion.hooks), paired over the same items:
   - knowledge and hub: naming of the confirm half of the unused words (averaged over three batch compositions);
     the selectivity is the rare-word coefficient (lesion.p14.rare_coefficient), the hub's comprehension half is
     tested with the sealed M1 items in the dissociation test (gpu_session15.sh);
   - control: the sealed M1 items (one batch composition), the strong-minus-weak distractor damage within colour and
     size. M1 has no fourth split, so this is descriptive: the set is fixed on the dev items before the sealed items
     are opened, nothing is chosen from these tests, and the final test on the same sealed items is therefore not an
     independent confirmation of the minimality, only of the set's effect (agent2's code review, 2026-10-15).
   delta is the bootstrap standard error of the whole set's selectivity. "Necessary": removing the component lowers
   the selectivity by more than delta and the paired 95% interval of the drop excludes 0. Pairs of members that are
   not necessary alone are removed together (substitutes show up as a pair that is); all such pairs up to MAX_PAIRS,
   and groups.json records how many were tested of how many ("partial" when cut). "Saturated": adding one of the
   next 10 components of the run's order raises the selectivity by less than delta; the language-model loss change
   is recorded for each added state.
Writes outputs/p14/<model>/confirm/{groups.json, minimality.csv, stability.json, meta.json}.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))

import common  # noqa: E402
from lesion import damage, knowledge_items as ki, materials, naming, neutral_text, p14, probes, runinfo, scoring  # noqa: E402
from lesion.components import Component  # noqa: E402

BATCHES = (16, 11, 7)
N_ADD = 10
MAX_PAIRS = 45           # all pairs of up to 10 members that are not necessary alone
N_BOOT = 2000
SEED = 1415


def runs_table(root):
    rows = []
    for meta_file in sorted((root / "runs").glob("*/meta.json")):
        m = json.loads(meta_file.read_text(encoding="utf-8"))
        if m.get("status") != "done":
            continue
        rows.append(dict(run=meta_file.parent.name, dir=meta_file.parent, objective=m["side"], pool=m["pool"],
                         lambda_sp=m["lambda_sp"], seed=m["seed"], chosen_k=m.get("chosen_k"), delta=m.get("delta")))
    return pd.DataFrame(rows)


def chosen_group(run_dir, k):
    curve = pd.read_csv(run_dir / "curve.csv")
    row = curve[curve.k == k].iloc[0]
    return row.members.split(" "), float(row.selectivity)


def pick(runs, pools):
    """{objective: dict(run, pool, k, members, select_selectivity)} and the stability record."""
    groups, stab = {}, {}
    for objective, g in runs.groupby("objective"):
        side = "knowledge" if objective in ("knowledge", "hub") else "control"
        sizes = {p: pools[p]["top"] for p in g.pool.unique() if pools[p]["side"] == side}
        main_pool = max(sizes, key=sizes.get)
        main = g[(g.pool == main_pool) & g.chosen_k.notna()]
        if main.empty:
            continue
        cand = [(r.run, int(r.chosen_k), *chosen_group(r.dir, int(r.chosen_k))) for r in main.itertuples()]
        best = max(cand, key=lambda c: (c[3], -c[1]))
        groups[objective] = dict(run=best[0], pool=main_pool, k=best[1], members=best[2], select_selectivity=best[3])
        sets = [c[2] for c in cand]
        gates = {r.run: pd.read_csv(r.dir / "gates.csv").set_index("component").g for r in main.itertuples()}
        names = list(gates)
        pairs = list(itertools.combinations(names, 2))
        size = best[1]
        top = {n: set(gates[n].sort_values().index[:size]) for n in names}
        s = p14.stability(sets)
        s.update(runs=[c[0] for c in cand], chosen_k={c[0]: c[1] for c in cand},
                 select_selectivity={c[0]: c[3] for c in cand},
                 spearman_g={f"{a}|{b}": p14.spearman(gates[a], gates[b].reindex(gates[a].index)) for a, b in pairs},
                 top_overlap={f"{a}|{b}": len(top[a] & top[b]) / size for a, b in pairs})
        smaller = g[(g.pool != main_pool) & g.chosen_k.notna()]
        if len(smaller):
            s["smaller_pool"] = {r.run: dict(k=int(r.chosen_k), jaccard_with_set=len(set(chosen_group(r.dir, int(r.chosen_k))[0]) & set(best[2]))
                                             / len(set(chosen_group(r.dir, int(r.chosen_k))[0]) | set(best[2])),
                                             select_selectivity=chosen_group(r.dir, int(r.chosen_k))[1])
                                 for r in smaller.itertuples()}
        stab[objective] = s
    return groups, stab


def order_of(run_dir):
    t = pd.read_csv(run_dir / "gates.csv")
    return p14.matched_order(dict(zip(t.component, t.g)))


def paired(stat_full, stat_other, n, n_boot=N_BOOT, seed=SEED):
    """Point and 95% interval of stat_full(ix) - stat_other(ix) over item resamples (the same draws for both)."""
    rng = np.random.default_rng(seed)
    point = stat_full(np.arange(n)) - stat_other(np.arange(n))
    draws = [stat_full(ix) - stat_other(ix) for ix in (rng.integers(n, size=n) for _ in range(n_boot))]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def minimality(score_fn, members, extra, lm_fn):
    """Leave-one-out, leave-two-out among the members not necessary alone, and add-one; score_fn(keys) -> (stat, n)
    where stat(ix) is the selectivity on the items ix."""
    full_stat, n = score_fn(members)
    se = p14.bootstrap_se(full_stat, n, n=N_BOOT)
    rows = [dict(state="set", removed="", added="", selectivity=full_stat(np.arange(n)), delta=se, lm_change=lm_fn(members))]
    single = {}
    for c in members:
        stat, _ = score_fn([m for m in members if m != c])
        d, lo, hi = paired(full_stat, stat, n)
        single[c] = d > se and lo > 0
        rows.append(dict(state="minus_one", removed=c, added="", selectivity=stat(np.arange(n)), change=-d, ci_low=-hi,
                         ci_high=-lo, necessary=single[c]))
    weak = [c for c in members if not single[c]]
    pairs = list(itertools.combinations(weak, 2))
    rows[0].update(pairs_total=len(pairs), pairs_tested=min(len(pairs), MAX_PAIRS))
    for a, b in pairs[:MAX_PAIRS]:
        stat, _ = score_fn([m for m in members if m not in (a, b)])
        d, lo, hi = paired(full_stat, stat, n)
        rows.append(dict(state="minus_two", removed=f"{a} {b}", added="", selectivity=stat(np.arange(n)), change=-d,
                         ci_low=-hi, ci_high=-lo, necessary=d > se and lo > 0))
    for c in extra:
        stat, _ = score_fn(members + [c])
        d, lo, hi = paired(stat, full_stat, n)
        rows.append(dict(state="plus_one", removed="", added=c, selectivity=stat(np.arange(n)), change=d, ci_low=lo,
                         ci_high=hi, saturated=d < se, lm_change=lm_fn(members + [c])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--freeze-tag", required=True)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N confirm words (tiny runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    commit = runinfo.check_analysis_freeze(args.freeze_tag)
    suf = "_tiny" if args.tiny else ""
    root = ROOT / f"outputs/p14{suf}" / args.model
    out = Path(args.out) if args.out else root / "confirm"
    pools = json.loads((root / "pools.json").read_text(encoding="utf-8"))["pools"]
    runs = runs_table(root)
    groups, stab = pick(runs, pools)
    if not groups:
        raise SystemExit("no finished run with a feasible set")
    fields = dict(stage="p14_confirm", model=args.model, tiny=args.tiny, freeze_tag=args.freeze_tag,
                  dtype=args.dtype, means_sha256=runinfo.sha256(args.means), batches=list(BATCHES), n_add=N_ADD,
                  max_pairs=MAX_PAIRS, n_boot=N_BOOT, runs=runs.run.tolist(), limit=args.limit,
                  pools_sha256=runinfo.sha256(root / "pools.json"),
                  sets={k: v["members"] for k, v in groups.items()}, **common.moe_fields(args))
    if (out / "meta.json").exists():
        old = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        if old.get("status") == "done" and not common.moe_mismatch(old, fields) \
                and all(old.get(k) == v for k, v in json.loads(json.dumps(fields)).items()):
            print(f"{out}: already done with the same runs and sets")
            return
        print(f"{out}: an earlier confirmation with other runs, sets or settings is replaced", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stability.json").write_text(json.dumps(stab, indent=1, default=float), encoding="utf-8")
    runinfo.write_meta(out, status="running", commit=commit, moe_forward=args.moe, **fields)
    t0 = time.time()

    model, tok, chat, means, ablate = common.load(args)
    pad = tok.eos_token_id or 0
    all_items = pd.read_csv(ki.OUT / "items.csv")
    median = naming.zipf_median(all_items)
    demos = pd.read_csv(ki.OUT / "demos.csv")
    words = p14.split_unused(p14.unused_words(all_items, demos, pd.read_csv(ki.OUT / "eligible.csv")))["confirm"]
    if args.limit:
        words = words.head(args.limit)
    prompts = naming.build_prompts(words, demos, chat)
    rare = (words.zipf < median).to_numpy()
    lm_seqs = [tok(t, add_special_tokens=False).input_ids[:128] for t in neutral_text.load()[:40 if not args.limit else 4]]
    lm_base = probes.lm_loss_region(model, lm_seqs, [], means, pad_id=pad)

    def lm_fn(keys):
        return probes.lm_loss_region(model, lm_seqs, [Component.parse(k) for k in keys], means, pad_id=pad) / lm_base - 1

    base_n = naming.averaged_logprobs(model, tok, prompts, chat, BATCHES)

    def naming_score(keys):
        with ablate([Component.parse(k) for k in keys]):
            les = naming.averaged_logprobs(model, tok, prompts, chat, BATCHES)
        h = naming.item_damage(base_n, les).reindex(words.item_id).to_numpy()
        ntok = les.set_index("item_id").n_tokens.reindex(words.item_id).to_numpy()
        return (lambda ix: p14.rare_coefficient_np(h[ix], rare[ix], ntok[ix])), len(h)

    all_m1 = scoring.load_materials()[0].astype({"item_id": str})
    fm = damage.frequency_median(all_m1)
    m_test = materials.load_split("test", freeze_tag=args.freeze_tag)
    _, m1_demos, templates = scoring.load_materials()
    m_prompts = scoring.build_label_prompts(m_test, m1_demos, templates, variant="main", chat=chat)
    base_m = None

    def control_score(keys):
        nonlocal base_m
        if base_m is None:
            base_m = probes.label_scores_in_positions(model, tok, m_prompts, chat, None, 16)
        les = probes.label_scores_in_positions(model, tok, m_prompts, chat, None, 16, ([Component.parse(k) for k in keys], means))
        frame = damage.item_frame(damage.item_damage(base_m, les), all_m1, fm).reset_index(drop=True)
        cols = damage.columns(frame)
        return (lambda ix: damage.effects({k: v[ix] for k, v in cols.items()})["control_effect"]), len(frame)

    table = []
    for objective, grp in groups.items():
        run_dir = root / "runs" / grp["run"]
        order = order_of(run_dir)
        extra = [c for c in order if c not in grp["members"]][:N_ADD]
        score = control_score if objective == "control" else naming_score
        rows = minimality(score, grp["members"], extra, lm_fn)
        table += [dict(objective=objective, **r) for r in rows]
        grp["delta"] = rows[0]["delta"]
        grp["confirm_selectivity"] = rows[0]["selectivity"]
        grp["necessary"] = [r["removed"] for r in rows if r["state"] == "minus_one" and r["necessary"]]
        grp["redundant"] = [r["removed"] for r in rows if r["state"] == "minus_one" and not r["necessary"]]
        grp["saturated_share"] = float(np.mean([r["saturated"] for r in rows if r["state"] == "plus_one"])) if extra else None
        grp["pairs"] = dict(tested=rows[0]["pairs_tested"], total=rows[0]["pairs_total"],
                            partial=rows[0]["pairs_tested"] < rows[0]["pairs_total"])
        grp["minimality_items"] = "sealed M1 (descriptive)" if objective == "control" else "confirm half of the unused words"
        print(f"{objective}: set of {len(grp['members'])} from {grp['run']}; {len(grp['necessary'])} necessary; "
              f"({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(table).to_csv(out / "minimality.csv", index=False)
    (out / "groups.json").write_text(json.dumps(groups, indent=1, default=float), encoding="utf-8")
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), commit=commit, moe_forward=args.moe,
                       **fields)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

"""Optogenetics-inspired states on the M3 concepts (docs/plans/optogenetics-inspired-experiments.md §9; user
decisions Q23, 2026-10-11, and P7, 2026-10-12).

--role calibration (default): the 80 calibration concepts, exploratory (GPU session 9); --role test: the 160 sealed
test concepts, once, at the analysis-plan freeze tag (docs/analysis-plan/optogenetics.md), states and router counts
only (the implant stays on calibration concepts).

    python experiments/e3_errors/run_optogenetics.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16 \
        --role test --freeze-tag analysis-frozen-optogenetics --out-name optogenetics_confirm
    python experiments/e3_errors/run_optogenetics.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16 \
        --only implant --out-name optogenetics_implant_v2        (O3 again, implant v2, calibration concepts)
    python experiments/e3_errors/run_optogenetics.py --model qwen3-30b-a3b-instruct --means none --tiny [--role test]   (local smoke run)

Two GPUs (user decision 2026-10-12): --shard 0/2 and --shard 1/2, one process per GPU, each running every second
state of the same plan into <out-name>.shard<i>of<n>; the router part runs in shard 0 only. Then
merge_optogenetics_shards.py checks that the shards share one plan and code and together hold every state once, and
writes <out-name>.

Part 1, states (all mean ablation; alpha scales the component's deviation from its mean, 0 = lesion, 1 = intact):
- intact, K8, C4 (references within this session) and K8+C4 together (O5, interaction);
- K8 and C4 at alpha 0.25, 0.5, 0.75, 1.5, 2; the first matched random sets of the battery's random controls,
  randomK_00 and randomC_00, at alpha 0.5 and 2 (O1, dose and gain);
- K8 at the answer positions only at alpha 0.25, 0.5, 0.75 (O2, rescue curve);
- C4 on every combination of the probe, middle and answer segments plus the context alone, and K8 on the context,
  probe and middle alone (O4, epoch map; segments as in lesion.m3_battery.prompt_segments);
- test role only: randomK_00..02 and randomC_00..02 alone, K8 with each randomC and each randomK with C4 (random
  double lesions and the single lesions their interaction needs; exploratory check of the O5 interaction's
  specificity, agent2 2026-10-12): 45 states.
Per state the battery tables of run_battery.run_state (candidates, labels; greedy answers for whole-sequence states;
no logit lens or attention) and, for whole-sequence states, the language-model loss on 40 neutral passages.
Part 2, implant (O3, calibration only): lesion.m3_optogenetics.implant_scores (implant v2) for K8, randomK_00 and C4
with self, lesion-only, near and far donors, plus the intact reference.
Part 3, router (test role): lesion.m3_optogenetics.router_selection of the K8 experts on the free-naming prompts
(is L45 E48 ever routed at the answer position?).
--only intact-check: the intact state alone, on this shard's GPU, into <out-name>.intactcheck<i>of<n> (the
cross-GPU consistency check of the merged run; analysis plan §8).
Writes outputs/e3_errors/<model>/<out-name>/{candidates,labels,greedy,lm_loss,implant,router}.parquet, pairs.csv and
meta.json; never overwrites a finished run. Each state, implant condition and the router part is written as its own
unit (lesion.state_store: tables first, a marker with row counts and hashes last), so a stopped run resumes from
<out-name>.partial by redoing only unfinished units, provided everything in the plan fingerprint is the same. Before
the folder is renamed, lesion.m3_optogenetics_plan.check_run_tables verifies every table against the plan and the
materials. The plan itself (states, components, inputs, expected tables) lives in lesion.m3_optogenetics_plan (op).
Test role: the inputs and components must be the ones the analysis plan names (op.FROZEN, op.FROZEN_SETS) and the
tag must be op.FREEZE_TAG.
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import common  # noqa: E402
import run_battery  # noqa: E402
from lesion import (hooks, m3_battery as mb, m3_dla, m3_items, m3_optogenetics as mo, m3_optogenetics_plan as op,  # noqa: E402
                    models, neutral_text, runinfo, state_store)
from lesion.components import Component  # noqa: E402

# options of common.add_args this runner does not read
UNUSED_DEFAULTS = dict(batch_size=32, n_boot=10_000, scorer="prefix", ablation_mode=op.ABLATION_MODE)
TABLES = ("candidates", "labels", "greedy")


def real_sets(args, n_random=1):
    """The frozen K8 and C4 and the battery's first n_random matched random sets of each side
    (outputs/.../random_controls/sets.json), after checking that the random-control run was a complete
    mean-ablation run of this model, dtype, means file and item file. Returns (sets as component keys, provenance of
    the random sets: the run's commit and the hashes of its sets.json and meta.json). Test role: the sets and their
    provenance must be the ones the analysis plan names."""
    import probe_lesions
    import run_dissociation_test as rdt

    m1_dir = common.out_dir(args, "screen")
    control = list(common.load_selection(m1_dir, json.loads((m1_dir / "meta.json").read_text(encoding="utf-8")))["control"])
    keys = probe_lesions.dose_sets(pd.read_csv(m1_dir / "screen.csv"),
                                   pd.read_csv(ROOT / "outputs/e3_knowledge" / args.model / "screen/screen.csv"))
    if keys["C4"] != control or keys["K8"] != list(rdt.K8):
        raise SystemExit("the rankings do not start with the frozen sets")
    rc = ROOT / "outputs/e3_errors" / args.model / "random_controls"
    meta = json.loads((rc / "meta.json").read_text(encoding="utf-8"))
    expected = dict(model=args.model, dtype=args.dtype, means_sha256=runinfo.sha256(args.means), complete=True,
                    final_items_sha256=runinfo.sha256(m3_items.OUT / "final_items.csv"))
    wrong = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
    if wrong or meta.get("ablation", {}).get("mode") != "mean":
        raise SystemExit(f"random_controls/meta.json does not match this run: {wrong or meta.get('ablation')}")
    rand = json.loads((rc / "sets.json").read_text(encoding="utf-8"))
    names = [f"random{side}_{i:02d}" for side in "KC" for i in range(n_random)]
    keys.update({n: rand[n] for n in names})
    sets = {k: [Component.parse(x).key for x in keys[k]] for k in ("K8", "C4", *names)}
    provenance = dict(commit=meta.get("git_commit"), sets_sha256=runinfo.sha256(rc / "sets.json"),
                      meta_sha256=runinfo.sha256(rc / "meta.json"))
    if args.role == "test":
        wrong = op.input_problems(dict(random_sets_sha256=provenance["sets_sha256"],
                                       random_meta_sha256=provenance["meta_sha256"], random_commit=provenance["commit"]))
        if wrong or sets != op.frozen_sets(n_random):
            raise SystemExit(f"the components or the random sets' provenance differ from the analysis plan: {wrong or 'sets'}")
    return sets, provenance


def load(args):
    """(model, tokenizer, chat, means, means fingerprint, the dtype actually used)."""
    if args.tiny:
        from lesion import means as means_mod

        model, tok = models.load_tiny(args.model)
        m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3)
        return model, tok, models.uses_chat(args.model), m, "tiny", str(next(model.parameters()).dtype)
    model, tok, chat, m, _ = common.load(args)
    return model, tok, chat, m, runinfo.sha256(args.means), args.dtype


def check_frozen(args):
    """The confirmatory run's inputs must be the ones written into the analysis plan (op.FROZEN; agent2 2026-10-12):
    model, dtype, revision, means file and hash, item file, word-to-definition foils; the random sets and the
    components are checked in real_sets."""
    got = dict(revision=models.MODELS[args.model][1], means_sha256=runinfo.sha256(args.means),
               final_items_sha256=runinfo.sha256(m3_items.OUT / "final_items.csv"), w2d_foils_sha256=runinfo.sha256(op.FOILS))
    wrong = op.input_problems(got)
    if args.model != op.MODEL or args.dtype != op.DTYPE or (ROOT / args.means).resolve() != (ROOT / op.MEANS).resolve():
        wrong.append("model, dtype or means file")
    if wrong:
        raise SystemExit(f"inputs differ from the analysis plan: {wrong}")
    if args.freeze_tag != op.FREEZE_TAG:
        raise SystemExit(f"the confirmatory run uses the tag {op.FREEZE_TAG}")


def check_unused_args(args):
    """Options of common.add_args this runner does not use must stay at their defaults, so a command line never
    seems to change a result it cannot change (agent2 2026-10-12)."""
    changed = sorted(k for k, v in UNUSED_DEFAULTS.items() if getattr(args, k) != v)
    if changed:
        raise SystemExit(f"this runner does not use {changed}; leave them at their defaults")


def main(argv=None):
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--role", choices=("calibration", "test"), default="calibration")
    ap.add_argument("--freeze-tag", default=None, help="analysis-plan freeze tag; required with --role test")
    ap.add_argument("--out-name", default="optogenetics")
    ap.add_argument("--only", choices=("all", "states", "implant", "router", "intact-check"), default="all")
    ap.add_argument("--implant-batch", type=int, default=8)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th state from i (one process per GPU)")
    args = ap.parse_args(argv)
    check_unused_args(args)
    test = args.role == "test"
    _, shard_i, shard_n = op.shard_states([], args.shard)
    if args.only == "intact-check":            # the intact state once more on this shard's GPU (cross-GPU check)
        parts = ("states",)
    else:
        parts = {"all": ("states", "router") if test else ("states", "implant")}.get(args.only, (args.only,))
        if shard_n > 1:
            if "implant" in parts:
                raise SystemExit("the implant is not sharded; run it in one process")
            parts = tuple(x for x in parts if x != "router" or shard_i == 0)
    if test and "implant" in parts:
        raise SystemExit("the implant (O3) runs on calibration concepts only")
    base = ROOT / ("outputs/e3_errors_tiny" if args.tiny else "outputs/e3_errors") / args.model
    suffix = f".intactcheck{shard_i}of{shard_n}" if args.only == "intact-check" else (f".shard{shard_i}of{shard_n}" if shard_n > 1 else "")
    out = base / f"{args.out_name}{suffix}"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite a finished run: {out}")
    if not hasattr(hooks, "position_mask"):
        raise SystemExit("this run requires hooks.position_mask")
    if test and not args.tiny:
        # held-out concepts: only at the freeze tag, no tracked file changed, the tagged plan and protocol (agent3 N2)
        commit = runinfo.check_analysis_freeze(args.freeze_tag, whole_tree=True)
        check_frozen(args)
        wrong = op.tagged_plan_problems()
        if wrong:
            raise SystemExit("; ".join(wrong))
    else:
        commit = common.guard(args)
    items, pool = op.select_items(op.read_final(), args.role, args.tiny)
    if test and not args.tiny:
        import run_experiment3

        if tuple(run_experiment3.POOL_ROLES) != op.TEST_POOL_ROLES:
            raise SystemExit("the foil pool differs from experiment 3's")
        run_experiment3.check_foils(items, pool)               # the foils built now are the ones written down
    doubles = op.doubles_for(args.role)
    n_random = len(doubles) or 1
    keys, random_prov = (op.tiny_sets(n_random), None) if args.tiny else real_sets(args, n_random)
    sets = {k: [Component.parse(x) for x in v] for k, v in keys.items()}
    all_states = op.plan(sets, doubles)
    if test and len(all_states) != op.N_STATES:
        raise SystemExit(f"the confirmatory run has {op.N_STATES} states, not {len(all_states)}")
    states = [s for s in all_states if s[0] == "intact"] if args.only == "intact-check" else op.shard_states(all_states, args.shard)[0]
    whole = [s[0] for s in states if s[4] is None]
    # donor pairs only for the implant (calibration concepts)
    pairs = mo.donor_pairs(items, seed=0) if "implant" in parts else pd.DataFrame(columns=["recipient", "near", "near_kind", "far"])
    router_experts = [Component.parse(k) for k in op.router_components(keys)]
    n_texts, neutral_tokens = op.TINY_NEUTRAL if args.tiny else (op.N_NEUTRAL, op.NEUTRAL_TOKENS)
    texts = neutral_text.load()[:n_texts]

    model, tok, chat, means, means_fp, dtype = load(args)
    # everything that must match before a stopped run may be resumed into the same folder (agent2 reviews 2026-10-11, -12)
    fingerprint = dict(states=[s[0] for s in states], n_states=len(states), shard=args.shard, only=args.only,
                       all_states=[s[0] for s in all_states], whole_states=whole,
                       sets=keys, role=args.role, parts=list(parts),
                       freeze_tag=args.freeze_tag if test else None,
                       items=list(items.item_id), means=means_fp, pairs=pairs.to_dict("records"), commit=commit,
                       model=args.model, revision=models.MODELS[args.model][1], dtype=dtype, tiny=args.tiny,
                       ablation_mode=op.ABLATION_MODE, implant_batch=args.implant_batch,
                       final_items_sha256=runinfo.sha256(m3_items.OUT / "final_items.csv"),
                       analysis_plan_sha256=runinfo.analysis_plan_sha256(),
                       protocol_sha256=runinfo.sha256(ROOT / op.PROTOCOL), random_controls=random_prov,
                       neutral_texts=len(texts), neutral_tokens=neutral_tokens)
    partial = out.with_name(out.name + ".partial")
    if state_store.begin(partial, fingerprint):      # same plan: resume; plan.json and timing.json written atomically
        print(f"resuming {partial}", flush=True)
    pairs.to_csv(partial / "pairs.csv", index=False)
    # a unit counts as done only if its tables are what the plan says (agent3 2026-10-13, N8)
    units = state_store.Store(partial / "units", required=lambda name: op.unit_tables(name, set(whole)),
                              validate=lambda name, d: op.check_unit(d, name, args.role, args.tiny, pairs))
    removed = units.clean()                                   # unfinished units of a stopped run are redone
    if removed:
        print(f"redoing unfinished units: {removed}", flush=True)
    t0 = time.time()
    # a damaged timing file, or one that is not {name: seconds}, is dropped and timing starts again (agent3 F5)
    timing = state_store.read_json(partial / "timing.json", {},
                                   ok=lambda t: all(type(v) in (int, float) for v in t.values()))

    def note(key, seconds, extra=""):
        timing[key] = round(seconds, 1)
        state_store.write_json(partial / "timing.json", timing)
        print(f"{key}: {timing[key]} s{extra}", flush=True)

    def write(name, tables):
        """A unit that fails the plan's check stops the run at once, its tables kept for diagnosis; rerunning
        computes it again (the same inputs give the same numbers), so the cause must be found first."""
        try:
            units.write(name, tables)
        except ValueError as e:
            raise SystemExit(f"STOPPED: {e}")

    if "states" in parts:
        finished = units.done()
        for name, set_name, comps, alpha, region in states:
            if f"state={name}" in finished:
                continue
            t1 = time.time()
            ablate = functools.partial(hooks.ablate, model, means=means, alpha=alpha, mode="mean")
            res = run_battery.run_state(model, tok, chat, items, pool, comps, ablate, region, name, greedy=True, readouts=False)
            loss = None
            if region is None:
                with (ablate(comps) if comps is not None else _null()):
                    loss = mo.lm_loss(model, tok, texts, max_tokens=neutral_tokens)
            meta_cols = op.columns(set_name, alpha, region)
            tables = {t: (res[t].assign(**meta_cols) if res[t] is not None else None) for t in TABLES}
            tables["lm_loss"] = None if loss is None else pd.DataFrame([dict(state=name, lm_loss=loss, **meta_cols)])
            write(f"state={name}", tables)
            note(name, time.time() - t1, f", lm_loss {loss:.4f}" if loss is not None else "")

    if "implant" in parts:
        finished = units.done()
        name_free = [p for p in mb.scored_prompts(items, chat) if p.condition == "name_free"]
        prompts = {p.item_id: p for p in name_free}
        # self-implants must be batched exactly as the vectors were recorded (row order decides float rounding)
        if list(prompts) != [r for r, _ in op.implant_plan(pairs)[0][3]]:
            raise SystemExit("self-implant order differs from the recording order")
        vectors = {}
        for set_name, condition, add, pair_list in op.implant_plan(pairs):
            if f"implant={set_name}__{condition}" in finished:
                continue
            t1 = time.time()
            if set_name == "intact":
                df = mo.intact_first_tokens(model, tok, prompts, pair_list, chat, batch_size=args.implant_batch)
            else:
                comps = sets[set_name]
                if set_name not in vectors:
                    vectors[set_name], _ = m3_dla.contributions(model, means, comps, name_free, tok, chat,
                                                                batch_size=args.implant_batch, model_dtype=True)
                df = mo.implant_scores(model, tok, means, comps, prompts, vectors[set_name], pair_list, chat,
                                       batch_size=args.implant_batch, add=add)
            write(f"implant={set_name}__{condition}", dict(implant=df.assign(set=set_name, condition=condition)))
            note(f"implant_{set_name}_{condition}", time.time() - t1)

    if "router" in parts and "router" not in units.done():
        t1 = time.time()
        name_free = [p for p in mb.scored_prompts(items, chat) if p.condition == "name_free"]
        write("router", dict(router=mo.router_selection(model, tok, name_free, router_experts, chat,
                                                        batch_size=args.implant_batch)))
        note("router", time.time() - t1)

    # the units stay in the finished folder: the confirmatory analysis checks the tables against them (agent3 F2)
    order = op.unit_names([s[0] for s in states], parts, pairs)
    missing = sorted(set(order) - units.done())
    if missing:
        print(f"partial results in {partial}; rerun to finish {len(missing)} units", flush=True)
        return
    state_store.assemble(units, order, (*TABLES, "lm_loss", "implant", "router"), partial)
    try:
        op.check_run_tables(partial, args.role, args.tiny, [s[0] for s in states] if "states" in parts else [], parts, pairs)
    except ValueError as e:
        raise SystemExit(f"{partial} is incomplete: {e}")
    self_check = None
    if "implant" in parts:
        imp = pd.read_parquet(partial / "implant.parquet")
        ref = imp[(imp.set == "intact") & (imp.condition == "self")].set_index("recipient").lp_recipient
        self_check = {s: float((imp[(imp.set == s) & (imp.condition == "self")].set_index("recipient").lp_recipient - ref).abs().max())
                      for s in ("K8", "randomK_00", "C4")}
    runinfo.write_meta(partial, model=args.model, revision=fingerprint["revision"], dtype=dtype, tiny=args.tiny,
                       role=args.role, parts=fingerprint["parts"], only=args.only, freeze_tag=fingerprint["freeze_tag"],
                       shard=args.shard, all_states=fingerprint["all_states"], whole_states=whole,
                       router_experts=[c.key for c in router_experts] if "router" in parts else None,
                       implant_version=2 if "implant" in parts else None, ablation_mode=op.ABLATION_MODE,
                       means_path=str(args.means), protocol_sha256=fingerprint["protocol_sha256"],
                       random_controls=random_prov, w2d_foils_sha256=runinfo.sha256(op.FOILS) if test else None,
                       means_sha256=means_fp, sets=fingerprint["sets"], states=fingerprint["states"],
                       item_roles=dict(zip(items.item_id, items.role)), n_items=len(items), n_neutral_texts=len(texts),
                       neutral_tokens=neutral_tokens, implant_self_max_abs_diff=self_check, resume_commit=commit,
                       timing=timing, final_items_sha256=runinfo.sha256(m3_items.OUT / "final_items.csv"),
                       seconds_this_call=round(time.time() - t0, 1), complete=True)
    partial.rename(out)
    print(f"written to {out}; implant self-check max |diff| {self_check}")


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()

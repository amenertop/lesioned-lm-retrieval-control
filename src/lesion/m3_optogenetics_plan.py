"""The plan of the optogenetics-inspired runs on the M3 concepts, in one place for the runner, the shard merger and
the confirmatory analysis, so all three check the same things (agent2 review 2026-10-13): which states, which
concepts and prompts, which inputs, and what every finished table must hold. No model, no GPU.

Exploratory runs on the calibration concepts: docs/plans/optogenetics-inspired-experiments.md §9. The confirmation
on the 160 sealed test concepts: docs/analysis-plan/optogenetics.md, whose §2-3 name the inputs in FROZEN and the
components in FROZEN_SETS (tests/test_optogenetics_runs.py checks that the plan names these same values).
"""
import functools
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import m3_battery as mb, m3_items, runinfo, shards, state_store
from .components import Component

ROOT = Path(__file__).resolve().parents[2]
MODEL, DTYPE = "qwen3-30b-a3b-instruct", "float16"
ALPHAS = (0.25, 0.5, 0.75, 1.5, 2.0)
RANDOM_ALPHAS = (0.5, 2.0)
RESCUE_ALPHAS = (0.25, 0.5, 0.75)
C4_SEGMENTS = (("probe",), ("middle",), ("answer",), ("probe", "middle"), ("probe", "answer"), ("middle", "answer"),
               ("probe", "middle", "answer"), ("context",))
K8_SEGMENTS = (("context",), ("probe",), ("middle",), ("answer",))
RANDOM_DOUBLES = ("00", "01", "02")       # test role: K8 + randomC_xx and randomK_xx + C4
N_TEST, N_STATES = 160, 45
N_NEUTRAL, NEUTRAL_TOKENS = 40, 256       # language-model loss: the first 40 WikiText passages, 256 tokens each
TINY_NEUTRAL = (3, 32)                    # the same on the tiny random model (pipeline check)
ABLATION_MODE = "mean"
TEST_POOL_ROLES = ("test", "calibration_a", "calibration_b", "reserve")   # the foil pool of experiment 3 (run_experiment3.POOL_ROLES)
FREEZE_TAG = "analysis-frozen-optogenetics"
FOILS = m3_items.OUT / "w2d_foils_test.csv"
PROTOCOL = "docs/plans/optogenetics-inspired-experiments.md"
MEANS = "outputs/means/qwen3-30b-a3b-instruct@0d7cf23__42ae809__fp16.pt"
RANDOM_CONTROLS = ROOT / "outputs/e3_errors" / MODEL / "random_controls"
FROZEN = dict(revision="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
              means_sha256="98b3865a71b0c8e85c530cb78d5438f9223a828fdf77c907315785b1f1de0d52",
              final_items_sha256="d539310755a31f8155cda5db7ae4920760dd36bbde97442800fa0b17df16b98d",
              w2d_foils_sha256="f703c084e27be238aa181a90cb5ef3952bd2bffab7b429f5762d803364b207b8",
              random_sets_sha256="e7a4bdb97d643c9221f270307bd9db9546690ab0d702209f2b5fd06aafac7d78",
              random_meta_sha256="2c2a096494a8af8ace5dbdd17e9cb469e5c44a61dc9dca88be068a00bf8eef60",
              random_commit="8ec63cd")
FROZEN_SETS = dict(
    K8=["expert:46:61", "expert:46:101", "head:37:24", "expert:45:32", "expert:45:48", "expert:43:70", "expert:42:115", "expert:45:93"],
    C4=["head:9:15", "head:27:9", "head:25:24", "head:24:5"],
    randomK_00=["expert:46:29", "expert:46:82", "head:37:10", "expert:45:4", "expert:45:54", "expert:43:85", "expert:42:8", "expert:45:95"],
    randomK_01=["expert:46:35", "expert:46:10", "head:37:21", "expert:45:22", "expert:45:51", "expert:43:18", "expert:42:80", "expert:45:127"],
    randomK_02=["expert:46:124", "expert:46:117", "head:37:22", "expert:45:37", "expert:45:99", "expert:43:111", "expert:42:75", "expert:45:67"],
    randomC_00=["head:9:10", "head:27:8", "head:25:13", "head:24:12"],
    randomC_01=["head:9:23", "head:27:25", "head:25:30", "head:24:31"],
    randomC_02=["head:9:20", "head:27:10", "head:25:29", "head:24:15"])


# --- states ---------------------------------------------------------------------------------------------------------

def plan(sets, doubles=()):
    """(name, set name, components, alpha, region) for every state, in running order; doubles: the random sets'
    numbers for the random double lesions (test role). Whole-sequence states have region None."""
    s = sets
    out = [("intact", None, None, 0.0, None), ("K8", "K8", s["K8"], 0.0, None), ("C4", "C4", s["C4"], 0.0, None),
           ("K8+C4", "K8+C4", s["K8"] + s["C4"], 0.0, None)]
    for n in doubles:
        out += [(f"randomK_{n}", f"randomK_{n}", s[f"randomK_{n}"], 0.0, None),
                (f"randomC_{n}", f"randomC_{n}", s[f"randomC_{n}"], 0.0, None),
                (f"K8+randomC_{n}", f"K8+randomC_{n}", s["K8"] + s[f"randomC_{n}"], 0.0, None),
                (f"randomK_{n}+C4", f"randomK_{n}+C4", s[f"randomK_{n}"] + s["C4"], 0.0, None)]
    out += [(f"{k}@{a}", k, s[k], a, None) for k in ("K8", "C4") for a in ALPHAS]
    out += [(f"{k}@{a}", k, s[k], a, None) for k in ("randomK_00", "randomC_00") for a in RANDOM_ALPHAS]
    out += [(f"K8_answer@{a}", "K8", s["K8"], a, "answer") for a in RESCUE_ALPHAS]
    out += [(f"C4_seg_{'+'.join(g)}", "C4", s["C4"], 0.0, g) for g in C4_SEGMENTS]
    out += [(f"K8_seg_{'+'.join(g)}", "K8", s["K8"], 0.0, g) for g in K8_SEGMENTS]
    return out


def columns(set_name, alpha, region):
    """The lesion every table row of a state carries: set, alpha (1.0 for the intact model) and region ("all" for
    the whole sequence, else the segments joined by +)."""
    return dict(set=set_name or "intact", alpha=1.0 if set_name is None else float(alpha),
                region="all" if region is None else "+".join((region,) if isinstance(region, str) else region))


def describe(role):
    """{state name: (set, alpha, region)} of the full plan of a role, as columns() writes them."""
    return {name: tuple(columns(set_name, alpha, region).values())
            for name, set_name, _, alpha, region in plan(defaultdict(list), doubles_for(role))}


def doubles_for(role):
    return RANDOM_DOUBLES if role == "test" else ()


def state_names(role):
    """{state name: whole-sequence?} of the full plan of a role, in running order; independent of the components."""
    return {name: region is None for name, _, _, _, region in plan(defaultdict(list), doubles_for(role))}


def shard_states(states, shard):
    """The states one shard runs ("i/n": every n-th state starting at i, so long and short states mix evenly)."""
    i, n = (int(x) for x in shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"bad --shard {shard}")
    return states[i::n], i, n


def unit_tables(name, whole):
    """The tables a unit of lesion.state_store must hold; whole: the whole-sequence state names."""
    kind, _, rest = name.partition("=")
    if kind == "state":
        return {"candidates", "labels"} | ({"greedy", "lm_loss"} if rest in whole else set())
    return {kind}                               # implant=<set>__<condition> -> implant; router -> router


def unit_names(states, parts, pairs=None):
    """The units of a run in the order its tables are assembled: its states, the implant conditions, the router
    counts (the parts it runs)."""
    names = [f"state={s}" for s in states] if "states" in parts else []
    if "implant" in parts:
        names += [f"implant={s}__{c}" for s, c, _, _ in implant_plan(pairs)]
    return names + (["router"] if "router" in parts else [])


# --- implant (O3, calibration concepts) -----------------------------------------------------------------------------

def implant_plan(pairs):
    """(set name, condition, add, [(recipient, donor)]) for every implant condition."""
    self_pairs = [(r, r) for r in pairs.recipient]
    near = [(r.recipient, r.near) for r in pairs.itertuples() if r.near]
    far = [(r.recipient, r.far) for r in pairs.itertuples()]
    out = [("intact", c, False, p) for c, p in (("self", self_pairs), ("near", near), ("far", far))]
    for s in ("K8", "randomK_00", "C4"):
        out += [(s, "self", True, self_pairs), (s, "lesion", False, self_pairs), (s, "near", True, near), (s, "far", True, far)]
    return out


def implant_rows(pairs):
    return {(s, c): len(p) for s, c, _, p in implant_plan(pairs)}


# --- components -----------------------------------------------------------------------------------------------------

def frozen_sets(n_random):
    """K8, C4 and the first n_random random sets of each side, as component keys."""
    names = ["K8", "C4"] + [f"random{side}_{i:02d}" for side in "KC" for i in range(n_random)]
    return {k: FROZEN_SETS[k] for k in names}


def tiny_sets(n_random=1):
    """Stand-ins on the 2-layer random model (pipeline check only), as component keys."""
    # the tiny Qwen3-MoE config has 8 experts and 4 heads per layer
    out = dict(K8=["expert:1:3", "expert:1:5", "head:1:2"], C4=["head:0:1", "head:0:3"])
    for i in range(n_random):
        out[f"randomK_{i:02d}"] = [f"expert:1:{(6 + i) % 8}", f"expert:1:{(7 + i) % 8}", f"head:1:{i % 2}"]
        out[f"randomC_{i:02d}"] = ["head:0:0", f"head:0:{2 if i % 2 == 0 else 1}"]
    return out


def router_components(sets):
    """The K8 experts whose routing the router part counts."""
    return [k for k in sets["K8"] if Component.parse(k).kind == "expert"]


# --- concepts and what every table must hold -----------------------------------------------------------------------

def select_items(final, role, tiny):
    """(items, pool): the concepts of a run and the pool its word-to-definition foils come from. A tiny test-role
    run keeps two free-naming concepts and two others (the hint measures need the first); a tiny calibration run
    the first four."""
    test = role == "test"
    items = final[final.role.isin(("test",) if test else ("calibration_a", "calibration_b"))].reset_index(drop=True)
    pool = final[final.role.isin(TEST_POOL_ROLES if test else mb.POOL_ROLES)].reset_index(drop=True)
    if tiny:
        fn = items.free_naming.astype(str).str.lower() == "true"
        keep = set(items[fn].item_id[:2]) | set(items[~fn].item_id[:2]) if test else set(items.item_id[:4])
        items = items[items.item_id.isin(keep)].reset_index(drop=True)
    return items, pool


def read_final():
    return pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)


def expected_keys(items, pool):
    """The rows every state must have, built from the prompts themselves: candidates (item, condition, role, word),
    and labels (item, format, variant) -> (the correct label, the labels read)."""
    cand = {(p.item_id, p.condition, role, word) for p in mb.scored_prompts(items, True) for role, word in p.candidates.items()}
    lab = {(it.item_id, it.fmt, it.variant): (it.correct, tuple(it.labels)) for it in mb.label_items(items, pool, True)}
    return dict(candidates=cand, labels=lab)


@functools.lru_cache(maxsize=4)
def _expected(role, tiny):
    items, pool = select_items(read_final(), role, tiny)
    return tuple(items.item_id), expected_keys(items, pool)


def check_run_tables(d, role, tiny, states, parts, pairs=None, implant=None):
    """lesion.state_store.check_tables with every expectation taken from the plan and the materials, none from the
    run's own record: the given states (a subset of the role's plan), their whole-sequence states and lesions, the
    prompt rows of the role's concepts, the router components, the implant conditions (all of them, or only the
    (set, condition) pair `implant`). Raises ValueError."""
    names = state_names(role)
    unknown = [s for s in states if s not in names]
    if unknown:
        raise ValueError(f"states outside the plan: {unknown}")
    item_ids, keys = _expected(role, bool(tiny))
    sets = tiny_sets(len(doubles_for(role)) or 1) if tiny else frozen_sets(len(doubles_for(role)) or 1)
    states_part = "states" in parts
    rows = None
    if "implant" in parts:
        rows = implant_rows(pairs)
        rows = {implant: rows[implant]} if implant is not None else rows
    state_store.check_tables(
        d, states=states if states_part else None, whole_states=[s for s in states if names[s]] if states_part else None,
        keys=keys if states_part else None, describe=describe(role) if states_part else None, item_ids=item_ids,
        router=router_components(sets) if "router" in parts else None, implant_rows=rows)


def check_unit(d, name, role, tiny, pairs=None):
    """The content check of one unit of lesion.state_store (state=<name>, implant=<set>__<condition>, router) in
    folder d, so a unit that is not what the plan says is never marked done (agent3 2026-10-13, N8)."""
    kind, _, rest = name.partition("=")
    if kind == "state":
        check_run_tables(d, role, tiny, [rest], ("states",))
    elif kind == "implant":
        check_run_tables(d, role, tiny, [], ("implant",), pairs, implant=tuple(rest.split("__")))
    elif name == "router":
        check_run_tables(d, role, tiny, [], ("router",))
    else:
        raise ValueError(f"unit {name} is not in the plan")


def confirm_ids(tiny=False):
    """The concepts of the confirmatory analysis in item-file order: the order fixes which bootstrap counts go with
    which concept, so it comes from the frozen item file, never from a run's record (agent3 2026-10-13, N5)."""
    return list(_expected("test", bool(tiny))[0])


# --- the confirmatory run's inputs and the confirmatory analysis's guard -------------------------------------------

def input_problems(got):
    """Names of the inputs (a dict over FROZEN's keys, or some of them) that differ from the analysis plan."""
    return sorted(k for k, v in got.items() if (not str(v or "").startswith(FROZEN[k]) if k == "random_commit" else v != FROZEN[k]))


def current_inputs():
    """The inputs of the confirmatory run as the files are now."""
    return dict(means_sha256=runinfo.sha256(ROOT / MEANS), final_items_sha256=runinfo.sha256(m3_items.OUT / "final_items.csv"),
                w2d_foils_sha256=runinfo.sha256(FOILS), random_sets_sha256=runinfo.sha256(RANDOM_CONTROLS / "sets.json"),
                random_meta_sha256=runinfo.sha256(RANDOM_CONTROLS / "meta.json"))


def tagged_plan_problems():
    """Before the confirmatory run loads the model: the analysis plan and the protocol in the working tree must be
    the ones committed at the freeze tag (agent3 2026-10-13, N2), not only recorded and compared afterwards."""
    try:
        same = (runinfo.analysis_plan_sha256() == runinfo.analysis_plan_sha256_at(FREEZE_TAG)
                and runinfo.sha256(ROOT / PROTOCOL) == runinfo.sha256_at(FREEZE_TAG, PROTOCOL))
    except (OSError, ValueError) as e:
        return [f"cannot compare the plan with the tag: {e}"]
    return [] if same else ["the analysis plan or the protocol differs from the tagged version"]


def code_problems():
    """The analysis must run the frozen code: no tracked file under code, materials, the analysis plan or the
    protocol differs from the freeze tag and nothing untracked is there (agent3 2026-10-13, N2 and F4). Later commits
    may reanalyse as long as these files are unchanged."""
    differ = runinfo.differs_from(FREEZE_TAG, (*runinfo.TRACKED, PROTOCOL))
    return [f"the code or plan differs from {FREEZE_TAG}: {differ[:5]}"] if differ else []


FINISHED_NAME = (".partial", ".crossgpu_failed", ".shard", ".intactcheck", ".tmp")
# plan.json (the runner's fingerprint, written when a run starts) -> the meta.json field written when it ends: what a
# process computed, so a record cannot claim another run than the one its units were computed under
RUN_IDENTITY = dict(states="states", all_states="all_states", whole_states="whole_states", shard="shard", only="only",
                    parts="parts", sets="sets", role="role", tiny="tiny", freeze_tag="freeze_tag", model="model",
                    revision="revision", dtype="dtype", means="means_sha256", ablation_mode="ablation_mode",
                    final_items_sha256="final_items_sha256", analysis_plan_sha256="analysis_plan_sha256",
                    protocol_sha256="protocol_sha256", random_controls="random_controls",
                    neutral_texts="n_neutral_texts", neutral_tokens="neutral_tokens", commit="resume_commit")
MERGE_FIELDS = ("shards", "intact_checks", "cross_gpu_check", "tables_sha256", "source_tables_sha256", "merged_by_commit")


def unit_problems(d, meta):
    """What keeps folder d, the run of one process (a whole run, a shard or an intact check), from being what its
    record says (agent3 2026-10-13, F2): the plan.json the runner wrote at the start must agree with the record on
    what was computed (RUN_IDENTITY), the run must have started, resumed and finished at one commit (G2), its
    concepts must be the planned test concepts in item-file order (compared as a set in the record, whose key
    order carries no meaning; G3), and the tables must be exactly the concatenation of the run's finished units,
    each as the plan says (lesion.state_store.assembled_problems). Test-role parts only (the implant is not read
    here)."""
    try:
        plan_json = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        plan_json = None
    if not isinstance(plan_json, dict):
        return [f"{d.name}: no readable plan.json, so nothing shows which run computed its tables"]
    differ = [k for k, m in RUN_IDENTITY.items() if plan_json.get(k) != meta.get(m)]
    ids = confirm_ids(bool(meta.get("tiny")))
    if plan_json.get("items") != ids or set(meta.get("item_roles") or {}) != set(ids):
        differ.append("items")
    if differ:
        return [f"{d.name}: the record and the run's plan.json differ in {differ}"]
    if meta.get("git_commit") != plan_json.get("commit"):
        return [f"{d.name}: started at commit {plan_json.get('commit')}, finished at {meta.get('git_commit')}"]
    if "implant" in plan_json["parts"]:
        return [f"{d.name}: an implant run, not a run of the confirmation"]
    role, tiny = meta.get("role"), bool(meta.get("tiny"))
    return state_store.assembled_problems(d, unit_names(plan_json["states"], plan_json["parts"]),
                                          lambda name, u: check_unit(u, name, role, tiny))


def evidence_problems(d, meta):
    """What the saved evidence says against reading run folder d as one run of the plan, whatever its record says
    (agent3 reviews 2026-10-13, N1, F1, F2):
    - one process (no record of shards): shard 0/1, none of the merge's fields, no shard or intact-check folder
      beside it, and unit_problems;
    - a merged run: its n shards and n - 1 intact checks beside it, complete and of the same plan as the record;
      shard i holds exactly the plan's states i, i + n, ... (and the router counts if i = 0), intact check i the
      intact state on GPU i; every part passes unit_problems; the merged tables are the shards' tables joined, by
      content (lesion.shards.joined_problems), and they and the parts' tables have the fingerprints the merge
      recorded; the tolerance is the plan's; and the cross-GPU check, recomputed from the intact state of the shard
      that ran it against each intact check, passes.
    Editing the record, or the record and some of the tables, cannot turn a failed merge into a run."""
    base, name = d.parent, d.name
    beside = sorted(p.name for p in base.glob(f"{name}.shard*of*")) + sorted(p.name for p in base.glob(f"{name}.intactcheck*of*"))
    info = meta.get("shards")
    if info is None:
        merged = [k for k in MERGE_FIELDS if k in meta]
        if meta.get("shard") != "0/1" or beside or merged:
            return [f"no record of the shards, yet not a single-process run (shard {meta.get('shard')!r}, folders "
                    f"beside it {beside[:3]}, fields of a merge {merged})"]
        return unit_problems(d, meta)
    try:
        n = len(info)
        parts = [base / f"{name}.shard{i}of{n}" for i in range(n)]
        checks = [base / f"{name}.intactcheck{i}of{n}" for i in range(1, n)]
        if n < 2 or [s.get("dir") for s in info] != [p.name for p in parts] or meta.get("intact_checks") != [c.name for c in checks]:
            return ["the record of the shards is not the n shards and n - 1 intact checks of one run"]
        metas, check_metas = [shards.read_meta(p) for p in parts], [shards.read_meta(c) for c in checks]
        shards.same_plan([meta] + metas + check_metas)
        all_states = list(meta.get("all_states") or [])
        for i, (p, m, rec) in enumerate(zip(parts, metas, info)):
            got = (m.get("shard"), m.get("only"), m.get("states"), set(m.get("parts") or []), rec.get("states"))
            if got != (f"{i}/{n}", "all", all_states[i::n], {"states", "router"} if i == 0 else {"states"}, m.get("states")):
                return [f"{p.name} is not shard {i} of {n} of the plan (its place, states or parts)"]
        for i, (c, m) in enumerate(zip(checks, check_metas), start=1):
            if (m.get("shard"), m.get("only"), m.get("states"), m.get("parts")) != (f"{i}/{n}", "intact-check", ["intact"], ["states"]):
                return [f"{c.name} is not the intact rerun on GPU {i}"]
        problems = [x for p, m in zip(parts + checks, metas + check_metas) for x in unit_problems(p, m)]
        problems += shards.joined_problems(d, parts)
        if problems:
            return problems
        recorded = meta.get("source_tables_sha256") or {}
        if not recorded or {p.name: state_store.tables_sha256(p) for p in parts + checks} != recorded:
            return ["a shard's or intact check's tables differ from those merged"]
        if state_store.tables_sha256(d) != meta.get("tables_sha256"):
            return ["the run's tables differ from those the merge wrote"]
        tolerance = (meta.get("cross_gpu_check") or {}).get("tolerance")
        if tolerance != dict(max=shards.CROSS_GPU_MAX, median=shards.CROSS_GPU_MEDIAN):
            return [f"cross-GPU tolerance {tolerance}, the plan says {shards.CROSS_GPU_MAX} and {shards.CROSS_GPU_MEDIAN}"]
        holder = [p for p, m in zip(parts, metas) if "intact" in m["states"]]
        result = shards.cross_gpu(holder[0], checks)
    except (SystemExit, OSError, ValueError, KeyError, TypeError, IndexError) as e:
        return [f"shard evidence: {e}"]
    failed = {k: v for k, v in result.items() if not v["passed"] or not v["n"]}
    return [f"cross-GPU check fails when recomputed from the shards: {failed}"] if failed else []


def confirm_problems(d, meta, smoke=False):
    """Everything that keeps the confirmatory analysis from reading run folder d with record meta (agent2 reviews
    2026-10-12, -13; agent3 2026-10-13); an empty list when it may. Checked against the plan and the files, not the
    record alone:
    - a finished run folder (not a partial, failed, shard or intact-check folder) of a complete test-role run; a
      tiny run only with smoke (pipeline check);
    - real runs: model, dtype, freeze tag, the run's commit is the tag's and its tree was clean, the analysis code
      now equals the tag's; revision, means, items, foils, random sets and their provenance as recorded equal the
      plan's values, and so do the files now; the recorded analysis-plan and protocol hashes equal those committed
      at the tag; the components are the frozen sets, the router counts the K8 experts;
    - every run: the concepts are exactly the planned test concepts (all 160, or the tiny run's four); mean
      ablation and the planned neutral texts; the planned states in the plan's order, the states and router
      parts; the evidence of the run (evidence_problems: each process's plan and units, a merge's shards and
      intact checks, the cross-GPU check recomputed from them); every table as check_run_tables requires."""
    problems = []
    if any(x in Path(d).name for x in FINISHED_NAME):
        problems.append(f"{Path(d).name} is not a finished run folder")
    if meta.get("role") != "test" or not meta.get("complete"):
        problems.append("not a complete test-role run")
    tiny = bool(meta.get("tiny"))
    if tiny and not smoke:
        problems.append("a tiny run (use --smoke for a pipeline check)")
    if meta.get("ablation_mode") != ABLATION_MODE:
        problems.append(f"ablation mode {meta.get('ablation_mode')!r}, the plan says {ABLATION_MODE}")
    neutral = TINY_NEUTRAL if tiny else (N_NEUTRAL, NEUTRAL_TOKENS)
    if (meta.get("n_neutral_texts"), meta.get("neutral_tokens")) != neutral:
        problems.append(f"neutral texts {meta.get('n_neutral_texts')} x {meta.get('neutral_tokens')} tokens, "
                        f"the plan says {neutral[0]} x {neutral[1]}")
    if not tiny:
        if meta.get("model") != MODEL or meta.get("dtype") != DTYPE:
            problems.append(f"model {meta.get('model')} in {meta.get('dtype')}, the plan says {MODEL} in {DTYPE}")
        if meta.get("freeze_tag") != FREEZE_TAG:
            problems.append(f"freeze tag {meta.get('freeze_tag')!r}")
        tagged = runinfo.git("rev-parse", "--verify", "--quiet", f"{FREEZE_TAG}^{{commit}}")
        if not tagged or meta.get("git_commit") != tagged or not meta.get("clean_tree"):
            problems.append("the run's commit is not the freeze tag on a clean tree")
        if tagged:
            problems += code_problems()
        rc = meta.get("random_controls") or {}
        recorded = dict(revision=meta.get("revision"), means_sha256=meta.get("means_sha256"),
                        final_items_sha256=meta.get("final_items_sha256"), w2d_foils_sha256=meta.get("w2d_foils_sha256"),
                        random_sets_sha256=rc.get("sets_sha256"), random_meta_sha256=rc.get("meta_sha256"),
                        random_commit=rc.get("commit"))
        problems += [f"recorded {k} differs from the plan" for k in input_problems(recorded)]
        if (ROOT / str(meta.get("means_path"))).resolve() != (ROOT / MEANS).resolve():
            problems.append(f"means file {meta.get('means_path')}, the plan names {MEANS}")
        try:
            problems += [f"current {k} differs from the plan" for k in input_problems(current_inputs())]
        except OSError as e:
            problems.append(f"an input file is missing: {e}")
        if tagged:
            try:
                at_tag = dict(analysis_plan_sha256=runinfo.analysis_plan_sha256_at(FREEZE_TAG),
                              protocol_sha256=runinfo.sha256_at(FREEZE_TAG, PROTOCOL))
                problems += [f"recorded {k} differs from the one committed at the tag" for k, v in at_tag.items() if meta.get(k) != v]
            except ValueError as e:
                problems.append(f"cannot read the tagged plan: {e}")
        if meta.get("sets") != frozen_sets(len(RANDOM_DOUBLES)):
            problems.append("components differ from the frozen sets")
        if meta.get("router_experts") != router_components(FROZEN_SETS):
            problems.append("router components differ from the K8 experts")
    ids = set(confirm_ids(tiny))
    roles = meta.get("item_roles") or {}
    if set(roles) != ids or (not tiny and len(ids) != N_TEST) or set(roles.values()) != {"test"}:
        problems.append(f"concepts are not the {'tiny run' if tiny else N_TEST} test concepts")
    names = state_names("test")
    for k in ("states", "all_states"):
        got = meta.get(k) or []
        if len(got) != N_STATES or len(set(got)) != len(got) or set(got) != set(names):
            problems.append(f"{k}: not the {N_STATES} planned states")
    if list(meta.get("all_states") or []) != list(names):          # the shards' states follow from this order
        problems.append("all_states: not in the plan's order")
    if not {"states", "router"} <= set(meta.get("parts") or []):
        problems.append("states or router part missing")
    problems += evidence_problems(Path(d), meta)
    if not problems:
        try:
            check_run_tables(d, "test", tiny, list(names), ("states", "router"))
        except ValueError as e:
            problems.append(f"tables: {e}")
    return problems

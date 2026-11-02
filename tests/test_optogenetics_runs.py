"""Run bookkeeping of the optogenetics confirmation (agent2 reviews 2026-10-12, -13; agent3 review 2026-10-13): the plan
module's inputs, components and states as the analysis plan names them, transactional units and completeness checks
(lesion.state_store), shard merging with the cross-GPU check and its evidence (lesion.shards), the guard of the
confirmatory analysis (lesion.m3_optogenetics_plan.confirm_problems), the confirmatory rules on synthetic losses,
an end-to-end smoke analysis of a merged tiny layout, and the H3c helpers."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "experiments/e1_screen", "experiments/e2_confirm", "experiments/e3_errors"):
    sys.path.insert(0, str(ROOT / p))

import analyze_consistency as ac  # noqa: E402
import analyze_optogenetics as ao  # noqa: E402
import run_optogenetics as ro  # noqa: E402
from lesion import m3_optogenetics_plan as op, shards, state_store  # noqa: E402
from lesion.components import Component  # noqa: E402


# --- the plan module against the analysis plan -----------------------------------------------------------------------

def test_frozen_inputs_and_components_are_the_ones_the_plan_names():
    plan = (ROOT / "docs/analysis-plan/optogenetics.md").read_text(encoding="utf-8")
    for key, value in op.FROZEN.items():
        assert value in plan, key
    for name, members in op.FROZEN_SETS.items():
        assert "、".join(members) in plan, name
    assert op.FREEZE_TAG in plan and op.MEANS in plan
    import run_dissociation_test as rdt
    import run_experiment3

    assert op.FROZEN_SETS["K8"] == list(rdt.K8)
    assert op.TEST_POOL_ROLES == tuple(run_experiment3.POOL_ROLES) and op.FOILS == run_experiment3.FOILS


def test_state_names_and_lesions_follow_the_plan_whatever_the_components():
    names = op.state_names("test")
    assert len(names) == op.N_STATES == 45 and len(op.state_names("calibration")) == 33
    sets = {k: [Component.parse(x) for x in v] for k, v in op.frozen_sets(3).items()}
    assert [s[0] for s in op.plan(sets, op.RANDOM_DOUBLES)] == list(names)
    assert names["K8+randomC_01"] and not names["C4_seg_middle"] and not names["K8_answer@0.5"]
    desc = op.describe("test")
    assert desc["intact"] == ("intact", 1.0, "all") and desc["K8@0.25"] == ("K8", 0.25, "all")
    assert desc["K8_answer@0.5"] == ("K8", 0.5, "answer") and desc["C4_seg_probe+middle"] == ("C4", 0.0, "probe+middle")
    assert op.unit_tables("state=K8", {"K8"}) == {"candidates", "labels", "greedy", "lm_loss"}
    assert op.unit_tables("state=C4_seg_middle", {"K8"}) == {"candidates", "labels"}
    assert op.unit_tables("router", set()) == {"router"} and op.unit_tables("implant=K8__self", set()) == {"implant"}


def test_unused_options_are_refused():
    import argparse
    import common

    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ro.check_unused_args(ap.parse_args(["--model", "m", "--means", "x"]))
    for extra in (["--ablation-mode", "zero"], ["--batch-size", "8"], ["--scorer", "full"], ["--n-boot", "5"]):
        with pytest.raises(SystemExit):
            ro.check_unused_args(ap.parse_args(["--model", "m", "--means", "x", *extra]))


def test_shards_partition_the_plan():
    states = [(f"s{i}",) for i in range(45)]
    a, _, _ = op.shard_states(states, "0/2")
    b, _, _ = op.shard_states(states, "1/2")
    assert len(a) == 23 and len(b) == 22 and sorted(a + b) == sorted(states)
    with pytest.raises(SystemExit):
        op.shard_states(states, "2/2")


def test_the_analysis_order_of_concepts_is_the_item_file_order():
    ids = op.confirm_ids(False)
    final = op.read_final()
    assert ids == list(final[final.role == "test"].item_id) and len(ids) == op.N_TEST
    assert op.confirm_ids(True) == [i for i in ids if i in set(op.confirm_ids(True))]


# --- lesion.state_store: units and bookkeeping files ----------------------------------------------------------------

def test_units_are_written_atomically_and_unfinished_ones_are_redone(tmp_path):
    store = state_store.Store(tmp_path / "units")
    store.write("state=K8@0.5", dict(candidates=pd.DataFrame(dict(a=[1, 2])), greedy=None))
    assert store.done() == {"state=K8@0.5"}
    with pytest.raises(FileExistsError):
        store.write("state=K8@0.5", dict(candidates=pd.DataFrame(dict(a=[1]))))
    # a stop during writing leaves a .tmp folder; a damaged unit fails its marker
    store.write("state=C4", dict(candidates=pd.DataFrame(dict(a=[3]))))
    (tmp_path / "units" / "state=C16.tmp").mkdir()
    pd.DataFrame(dict(a=[9, 9])).to_parquet(tmp_path / "units" / "state=C4" / "candidates.parquet", index=False)
    assert store.done() == {"state=K8@0.5"}
    assert sorted(store.clean()) == ["state=C16.tmp", "state=C4"]
    store.write("state=C4", dict(candidates=pd.DataFrame(dict(a=[3]))))
    state_store.assemble(store, ["state=C4", "state=K8@0.5"], ("candidates", "greedy"), tmp_path)
    assert pd.read_parquet(tmp_path / "candidates.parquet").a.tolist() == [3, 1, 2]
    assert not (tmp_path / "greedy.parquet").exists()


def _marker(d):
    return json.loads((d / state_store.MARKER).read_text(encoding="utf-8"))


@pytest.mark.parametrize("damage", ["bad json", "rows", "fractional rows", "huge rows", "boolean rows", "empty tables",
                                    "no name", "extra table", "missing table"])
def test_a_damaged_marker_makes_the_unit_unfinished_and_is_cleaned(tmp_path, damage):
    store = state_store.Store(tmp_path / "units")
    store.write("state=K8", dict(candidates=pd.DataFrame(dict(a=[1, 2])), labels=pd.DataFrame(dict(b=[1]))))
    d = tmp_path / "units" / "state=K8"
    m = _marker(d)
    text = None
    if damage == "bad json":
        text = "{not json"
    elif damage == "rows":                         # the row count changed, the hash kept
        m["tables"]["candidates"]["rows"] = 3
    elif damage == "fractional rows":              # int(1.9) == 1 must not pass (agent3 N8)
        text = json.dumps(m).replace('"rows": 1', '"rows": 1.9')
    elif damage == "huge rows":                    # a valid JSON number that overflows
        text = json.dumps(m).replace('"rows": 1', '"rows": 1e999')
    elif damage == "boolean rows":
        text = json.dumps(m).replace('"rows": 1', '"rows": true')
    elif damage == "empty tables":
        m["tables"] = {}
    elif damage == "no name":
        del m["name"]
    elif damage == "extra table":
        pd.DataFrame(dict(c=[1])).to_parquet(d / "greedy.parquet", index=False)
    elif damage == "missing table":
        (d / "labels.parquet").unlink()
    if text is not None or damage in ("rows", "empty tables", "no name"):
        (d / state_store.MARKER).write_text(text if text is not None else json.dumps(m), encoding="utf-8")
    assert not store.valid("state=K8") and store.done() == set()
    assert store.clean() == ["state=K8"] and not d.exists()


def test_units_must_hold_the_tables_the_plan_says(tmp_path):
    whole = {"K8"}
    store = state_store.Store(tmp_path / "units", required=lambda n: op.unit_tables(n, whole))
    df = pd.DataFrame(dict(a=[1]))
    with pytest.raises(ValueError):                # a whole-sequence state without its greedy answers and loss
        store.write("state=K8", dict(candidates=df, labels=df))
    with pytest.raises(ValueError):
        store.write("state=C4_seg_middle", dict(candidates=None, labels=None))
    store.write("state=C4_seg_middle", dict(candidates=df, labels=df, greedy=None))
    store.write("state=K8", dict(candidates=df, labels=df, greedy=df, lm_loss=df))
    assert store.done() == {"state=C4_seg_middle", "state=K8"}
    whole.add("C4_seg_middle")                     # a unit written under another plan no longer counts
    assert store.done() == {"state=K8"}


def test_a_unit_that_fails_the_content_check_is_never_done_and_is_kept_for_diagnosis(tmp_path):
    def validate(name, d):
        if not np.isfinite(pd.read_parquet(d / "candidates.parquet").a.to_numpy(dtype=float)).all():
            raise ValueError("non-finite")
    store = state_store.Store(tmp_path / "units", validate=validate)
    with pytest.raises(ValueError, match="kept in"):
        store.write("state=K8", dict(candidates=pd.DataFrame(dict(a=[1.0, np.nan]))))
    assert store.done() == set() and len(list((tmp_path / "invalid").iterdir())) == 1
    store.write("state=C4", dict(candidates=pd.DataFrame(dict(a=[1.0]))))
    assert store.done() == {"state=C4"}
    # a finished unit whose content no longer passes (written by older code) is redone
    plain = state_store.Store(tmp_path / "units")
    plain.write("state=K8", dict(candidates=pd.DataFrame(dict(a=[np.nan]))))
    assert store.done() == {"state=C4"} and store.clean() == ["state=K8"]


@pytest.mark.parametrize("text", ["null", "[]", "1", '"text"'])
def test_a_plan_file_of_another_kind_counts_as_damaged(tmp_path, text):
    """agent3 2026-10-13, F5: valid JSON that is not a plan restarts a folder without units and refuses one with."""
    partial = tmp_path / "run.partial"
    fp = dict(states=["intact", "K8"])
    state_store.begin(partial, fp)
    (partial / "plan.json").write_text(text, encoding="utf-8")
    assert state_store.begin(partial, fp) is False and state_store.begin(partial, fp) is True
    (partial / "units" / "state=K8").mkdir(parents=True)
    (partial / "plan.json").write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit):
        state_store.begin(partial, fp)


def test_the_plan_file_starts_resumes_or_refuses(tmp_path):
    partial = tmp_path / "run.partial"
    fp = dict(states=["intact", "K8"], shard=(0, 2))
    assert state_store.begin(partial, fp) is False
    assert state_store.begin(partial, fp) is True
    with pytest.raises(SystemExit):
        state_store.begin(partial, dict(fp, states=["intact"]))
    (partial / "plan.json").write_text("{", encoding="utf-8")       # damaged before any unit: start afresh
    assert state_store.begin(partial, fp) is False and state_store.begin(partial, fp) is True
    (partial / "units" / "state=K8").mkdir(parents=True)
    (partial / "plan.json").write_text("", encoding="utf-8")        # damaged with units: their plan is unknown
    with pytest.raises(SystemExit):
        state_store.begin(partial, fp)
    (partial / "timing.json").write_text("{", encoding="utf-8")
    assert state_store.read_json(partial / "timing.json", {}) == {}
    # valid JSON of another kind, or entries that are not seconds, count as damaged too (agent3 F5)
    ok = lambda t: all(type(v) in (int, float) for v in t.values())
    for text in ("[]", "null", "1", '"text"', '{"K8": "slow"}', '{"K8": true}'):
        (partial / "timing.json").write_text(text, encoding="utf-8")
        assert state_store.read_json(partial / "timing.json", {}, ok) == {}
    state_store.write_json(partial / "timing.json", dict(K8=1.5))
    assert state_store.read_json(partial / "timing.json", {}) == dict(K8=1.5)
    assert not (partial / "timing.json.tmp").exists()


# --- lesion.state_store.check_tables -----------------------------------------------------------------------------------

ITEMS = ["m1", "m2"]
KEYS = dict(candidates={(i, c, r, "w") for i in ITEMS for c, r in (("name_free", "target"), ("copy", "target"))},
            labels={**{(i, "options_letter", v): ("BA"[v], tuple("ABCDEF")) for i in ITEMS for v in range(2)},
                    **{(i, "word_to_definition", 0): ("A", ("A", "B")) for i in ITEMS}})
DESCRIBE = {"intact": ("intact", 1.0, "all"), "K8": ("K8", 0.0, "all"), "C4": ("C4", 0.0, "all"),
            "K8_seg_answer": ("K8", 0.0, "answer")}
LESION = lambda s: dict(zip(state_store.DESCRIPTION, DESCRIBE[s]))
ROUTER = ["expert:1:1", "expert:1:2"]


def _tables(d, states, whole, items=ITEMS, router=None):
    cand = pd.DataFrame([dict(state=s, item_id=i, condition=c, role=r, word=w, logprob=-1.0, n_tokens=1,
                              first_token_rank=0, **LESION(s))
                         for s in states for i, c, r, w in sorted(KEYS["candidates"]) if i in items])
    lab = pd.DataFrame([dict(state=s, item_id=i, fmt=f, variant=v, correct=c, margin=5.0, top_correct=True, **LESION(s),
                             **{f"logit_{x}": (1.0 if x in read else np.nan) for x in "ABCDEF"})
                        for s in states for (i, f, v), (c, read) in sorted(KEYS["labels"].items()) if i in items])
    cand.to_parquet(d / "candidates.parquet", index=False)
    lab.to_parquet(d / "labels.parquet", index=False)
    if whole:
        pd.DataFrame([dict(state=s, item_id=i, answer="w", kind="target", **LESION(s)) for s in whole for i in items]).to_parquet(d / "greedy.parquet", index=False)
        pd.DataFrame([dict(state=s, lm_loss=2.0, **LESION(s)) for s in whole]).to_parquet(d / "lm_loss.parquet", index=False)
    if router:
        pd.DataFrame([dict(item_id=i, component=c, selected_last=True, n_selected=1, n_positions=3)
                      for i in items for c in router]).to_parquet(d / "router.parquet", index=False)


def _edit(d, name, fn):
    df = pd.read_parquet(d / f"{name}.parquet")
    fn(df).to_parquet(d / f"{name}.parquet", index=False)


def _check(d, **kw):
    args = dict(states=["intact", "K8", "K8_seg_answer"], whole_states=["intact", "K8"], keys=KEYS, describe=DESCRIBE,
                item_ids=ITEMS, router=ROUTER)
    args.update(kw)
    state_store.check_tables(d, **args)


CHANGES = {
    "fewer states": dict(states=["intact", "K8"]),
    "more states": dict(states=["intact", "K8", "K8_seg_answer", "C4"]),
    "fewer concepts": dict(item_ids=["m1"]),
    "other whole states": dict(whole_states=["intact"]),
    "router component": dict(router=["expert:1:2"]),
    "router rows": ("router", lambda df: pd.concat([df, df.head(1)])),
    "duplicate row": ("candidates", lambda df: pd.concat([df, df.head(1)])),
    "extra condition": ("candidates", lambda df: pd.concat([df, df.head(1).assign(condition="cue")])),
    "extra variant": ("labels", lambda df: pd.concat([df, df.head(1).assign(fmt="word_to_definition", variant=1)])),
    "missing variant": ("labels", lambda df: df[~((df.state == "K8") & (df.variant == 1))]),
    "non-finite margin": ("labels", lambda df: df.assign(margin=df.margin.where(df.index != 0, np.nan))),
    "non-finite loss": ("lm_loss", lambda df: df.assign(lm_loss=np.inf)),
    "router count": ("router", lambda df: df.assign(n_selected=4)),
    # agent3 2026-10-13, N3: fields the analysis needs, the lesion each row describes, the labels it reads
    "missing column": ("candidates", lambda df: df.drop(columns=["n_tokens"])),
    "missing lesion columns": ("labels", lambda df: df.drop(columns=["set", "alpha", "region"])),
    "wrong set": ("candidates", lambda df: df.assign(set=np.where(df.state == "K8", "C4", df.set))),
    "wrong alpha": ("labels", lambda df: df.assign(alpha=np.where(df.state == "K8", 0.5, df.alpha))),
    "wrong region": ("greedy", lambda df: df.assign(region="answer")),
    "wrong correct label": ("labels", lambda df: df.assign(correct=np.where(df.index == 0, "Z", df.correct))),
    "used logit missing": ("labels", lambda df: df.assign(logit_F=np.where(df.fmt == "options_letter", np.nan, df.logit_F))),
    "no answer tokens": ("candidates", lambda df: df.assign(n_tokens=0)),
    "text alpha": ("lm_loss", lambda df: df.assign(alpha=df.alpha.astype(str))),
    # N11: router counts and the last position
    "router fractional positions": ("router", lambda df: df.assign(n_positions=3.5)),
    "router last not boolean": ("router", lambda df: df.assign(selected_last="garbage")),
    "router last without selection": ("router", lambda df: df.assign(n_selected=0, selected_last=True)),
    "router all selected but not last": ("router", lambda df: df.assign(n_selected=3, selected_last=False)),
    "router positions differ": ("router", lambda df: df.assign(n_positions=np.where(df.index == 0, 4, 3))),
    "extra table": "extra table",
    "missing router file": "missing router file",
    # agent3 2026-10-13, F3: a nullable column's missing value is of the right type but no value; a read logit column
    "nullable count missing": ("candidates", lambda df: df.assign(n_tokens=pd.array([pd.NA] + [1] * (len(df) - 1), dtype="Int64"))),
    "nullable rank missing": ("candidates", lambda df: df.assign(first_token_rank=pd.array([0] * (len(df) - 1) + [pd.NA], dtype="Int64"))),
    "nullable boolean missing": ("labels", lambda df: df.assign(top_correct=pd.array([pd.NA] + [True] * (len(df) - 1), dtype="boolean"))),
    "missing text": ("greedy", lambda df: df.assign(kind=[None] + ["target"] * (len(df) - 1))),
    "read logit column missing": ("labels", lambda df: df.drop(columns=["logit_A"])),
}


@pytest.mark.parametrize("change", list(CHANGES))
def test_check_tables_finds_what_the_plan_does_not_allow(tmp_path, change):
    states, whole = ["intact", "K8", "K8_seg_answer"], ["intact", "K8"]
    _tables(tmp_path, states, whole, router=ROUTER)
    _check(tmp_path)
    c, kw = CHANGES[change], {}
    if isinstance(c, dict):
        kw = c
    elif isinstance(c, tuple):
        _edit(tmp_path, *c)
    elif c == "extra table":
        pd.DataFrame(dict(a=[1])).to_parquet(tmp_path / "implant.parquet", index=False)
    else:
        (tmp_path / "router.parquet").unlink()
    with pytest.raises(ValueError):
        _check(tmp_path, **kw)


def test_nullable_columns_without_missing_values_pass(tmp_path):
    """F3 rejects missing values, not nullable types (a parquet round trip may give Int64 or boolean)."""
    _tables(tmp_path, ["intact", "K8", "K8_seg_answer"], ["intact", "K8"], router=ROUTER)
    _edit(tmp_path, "router", lambda df: df.astype(dict(n_selected="Int64", n_positions="Int64", selected_last="boolean")))
    _edit(tmp_path, "candidates", lambda df: df.astype(dict(n_tokens="Int64", first_token_rank="Int64")))
    _check(tmp_path)


def test_a_unit_whose_check_fails_on_a_missing_column_is_kept_for_diagnosis(tmp_path):
    """F3: a content check that fails with another error than ValueError still keeps the tables and stops."""
    def validate(name, d):
        pd.read_parquet(d / "labels.parquet")["logit_A"]
    store = state_store.Store(tmp_path / "units", validate=validate)
    with pytest.raises(ValueError, match="kept in"):
        store.write("state=K8", dict(labels=pd.DataFrame(dict(logit_B=[1.0]))))
    kept = list((tmp_path / "invalid").iterdir())
    assert len(kept) == 1 and (kept[0] / "labels.parquet").exists() and store.done() == set()
    assert not list((tmp_path / "units").iterdir())


def test_unused_label_logits_may_be_missing(tmp_path):
    """Word-to-definition reads two labels: its C to F logits are missing by design (agent3 N3)."""
    _tables(tmp_path, ["intact", "K8", "K8_seg_answer"], ["intact", "K8"], router=ROUTER)
    lab = pd.read_parquet(tmp_path / "labels.parquet")
    assert lab[lab.fmt == "word_to_definition"].logit_C.isna().all()
    _check(tmp_path)


def test_the_plan_layout_is_built_from_the_prompts():
    items, pool = op.select_items(op.read_final(), "test", tiny=False)
    assert len(items) == op.N_TEST and set(items.role) == {"test"}
    small, _ = op.select_items(op.read_final(), "test", tiny=True)
    keys = op.expected_keys(small, pool)
    fmts = {(f, v) for _, f, v in keys["labels"]}
    assert fmts == {("options_letter", v) for v in range(6)} | {("word_to_definition", v) for v in range(2)}
    assert {read for _, read in keys["labels"].values()} == {tuple("ABCDEF"), ("A", "B")}
    conds = {c for _, c, _, _ in keys["candidates"]}
    assert {"name_free", "copy", "options_word", "cue", "miscue", "none_hinted_demos"} == conds


# --- a tiny test-role layout of the real plan ------------------------------------------------------------------------

NAMES = op.state_names("test")


TABLES = ("candidates", "labels", "greedy", "lm_loss", "router")


def _units(states, router=False, shift=0.0, seed=0):
    """{unit name: {table: DataFrame}}: the tables of `states` as the plan wants them, on the tiny run's four
    concepts, with random valid numbers (the same numbers for a state whatever folder it is written to); shift
    moves the intact scores."""
    item_ids, keys = op._expected("test", True)
    desc = op.describe("test")
    out = {}
    for s in states:
        rng = np.random.default_rng([seed, list(NAMES).index(s)])
        les = dict(zip(state_store.DESCRIPTION, desc[s]))
        add = shift if s == "intact" else 0.0
        cand = [dict(state=s, item_id=i, condition=c, role=r, word=w, logprob=-float(rng.uniform(0.1, 6)) + add,
                     n_tokens=1, first_token_rank=0, **les) for i, c, r, w in sorted(keys["candidates"])]
        lab = []
        for (i, f, v), (correct, read) in sorted(keys["labels"].items()):
            lg = {x: float(rng.normal()) for x in read}
            lab.append(dict(state=s, item_id=i, fmt=f, variant=v, correct=correct, margin=float(rng.normal()) + add,
                            top_correct=lg[correct] >= max(lg.values()), **les,
                            **{f"logit_{x}": lg.get(x, np.nan) for x in "ABCDEF"}))
        tables = dict(candidates=pd.DataFrame(cand), labels=pd.DataFrame(lab))
        if NAMES[s]:
            tables["greedy"] = pd.DataFrame([dict(state=s, item_id=i, answer="w", kind="other", **les) for i in item_ids])
            tables["lm_loss"] = pd.DataFrame([dict(state=s, lm_loss=2.0 + float(rng.uniform(0, 0.05)), **les)])
        out[f"state={s}"] = tables
    if router:
        comps = op.router_components(op.tiny_sets(3))
        out["router"] = dict(router=pd.DataFrame([dict(item_id=i, component=c, selected_last=True, n_selected=2,
                                                       n_positions=5) for i in item_ids for c in comps]))
    return out


def _layout(d, states, router=False, shift=0.0, seed=0):
    """A finished run folder as the runner leaves it: every unit written through the store with the plan's
    content check, then the tables assembled from the units in plan order."""
    d.mkdir(parents=True, exist_ok=True)
    whole = {s for s in states if NAMES[s]}
    store = state_store.Store(d / "units", required=lambda n: op.unit_tables(n, whole),
                              validate=lambda n, u: op.check_unit(u, n, "test", True))
    for name, tables in _units(states, router, shift, seed).items():
        store.write(name, tables)
    state_store.assemble(store, op.unit_names(states, ["states"] + (["router"] if router else [])), TABLES, d)


def _record(states, parts, **extra):
    return dict(complete=True, git_commit="abc", clean_tree=True, freeze_tag=None, role="test", model=op.MODEL,
                revision="r", dtype="torch.float32", tiny=True, means_path="none", means_sha256="tiny",
                final_items_sha256="f", w2d_foils_sha256=None, analysis_plan_sha256="p", protocol_sha256="q",
                sets=op.tiny_sets(3), random_controls=None, all_states=list(NAMES), only="all", resume_commit="abc",
                whole_states=[s for s in states if NAMES[s]], item_roles={i: "test" for i in op.confirm_ids(True)},
                n_neutral_texts=3, neutral_tokens=32, ablation_mode="mean", states=list(states), parts=list(parts),
                **extra)


def _write_run(d, rec):
    """meta.json, and the plan.json the runner would have started this run with (op.RUN_IDENTITY)."""
    (d / "meta.json").write_text(json.dumps(rec), encoding="utf-8")
    plan = {k: rec.get(m) for k, m in op.RUN_IDENTITY.items()} | dict(items=list(rec["item_roles"]), pairs=[])
    state_store.write_json(d / "plan.json", plan)


def _part(base, name, states, router=False, shift=0.0, **meta):
    _layout(base / name, states, router=router, shift=shift)
    rec = _record(states, ["states"] + (["router"] if router else []),
                  router_experts=op.router_components(op.tiny_sets(3)) if router else None)
    rec.update(meta)
    _write_run(base / name, rec)


def _two_shards(base, shift=0.0, commit="abc"):
    a, b = op.shard_states(list(NAMES), "0/2")[0], op.shard_states(list(NAMES), "1/2")[0]
    _part(base, "run.shard0of2", a, router=True, shard="0/2")
    _part(base, "run.shard1of2", b, shard="1/2", git_commit=commit)
    _part(base, "run.intactcheck1of2", ["intact"], shift=shift, shard="1/2", only="intact-check")


def _merge(base):
    return shards.merge(base, "run", 2, list(NAMES), lambda d, states, parts: op.check_run_tables(d, "test", True, states, parts))


def _meta(d):
    return json.loads((d / "meta.json").read_text(encoding="utf-8"))


def test_a_tiny_layout_passes_the_real_unit_and_table_checks(tmp_path):
    _layout(tmp_path / "all", list(NAMES), router=True)
    op.check_run_tables(tmp_path / "all", "test", True, list(NAMES), ("states", "router"))
    _layout(tmp_path / "unit", ["K8_seg_answer"])
    op.check_unit(tmp_path / "unit", "state=K8_seg_answer", "test", True)
    with pytest.raises(ValueError):                # a unit's lesion must be the one its name says
        op.check_unit(tmp_path / "unit", "state=K8_seg_middle", "test", True)
    _edit(tmp_path / "unit", "candidates", lambda df: df.assign(logprob=np.where(df.index == 0, np.nan, df.logprob)))
    with pytest.raises(ValueError):
        op.check_unit(tmp_path / "unit", "state=K8_seg_answer", "test", True)


# --- lesion.shards: merging, the cross-GPU check and its evidence ---------------------------------------------------

def test_merge_joins_shards_and_records_the_cross_gpu_check_and_fingerprints(tmp_path):
    _two_shards(tmp_path)
    out, check = _merge(tmp_path)
    meta = _meta(out)
    assert out == tmp_path / "run" and check["passed"] and meta["complete"] and meta["parts"] == ["router", "states"]
    assert set(pd.read_parquet(out / "candidates.parquet").state) == set(NAMES)
    assert meta["tables_sha256"] == state_store.tables_sha256(out) and len(meta["source_tables_sha256"]) == 3
    assert op.evidence_problems(out, meta) == []
    assert op.confirm_problems(out, meta, smoke=True) == []


def test_a_failed_cross_gpu_check_never_becomes_the_run_and_can_be_redone(tmp_path):
    _two_shards(tmp_path, shift=0.2)
    out, check = _merge(tmp_path)
    assert not check["passed"] and not (tmp_path / "run").exists() and out.name.startswith("run.crossgpu_failed-")
    assert _meta(out)["complete"] is False
    # after the cause is fixed and the intact check rerun, the merge runs again (a stale .partial is rebuilt)
    shutil.rmtree(tmp_path / "run.intactcheck1of2")
    _part(tmp_path, "run.intactcheck1of2", ["intact"], shard="1/2", only="intact-check")
    (tmp_path / "run.partial").mkdir()
    out, check = _merge(tmp_path)
    assert check["passed"] and out == tmp_path / "run"


def _passed(meta):
    """The record of a failed merge rewritten to say it passed."""
    meta.update(complete=True)
    meta["cross_gpu_check"].update(passed=True)
    for v in meta["cross_gpu_check"]["results"].values():
        v.update(passed=True, max=0.0, median=0.0)


def _fix_intact(d, by):
    """Move the intact scores of folder d's top-level tables by `by` (to match a shifted rerun)."""
    _edit(d, "candidates", lambda df: df.assign(logprob=np.where(df.state == "intact", df.logprob + by, df.logprob)))
    _edit(d, "labels", lambda df: df.assign(margin=np.where(df.state == "intact", df.margin + by, df.margin)))


FAILED = ("flip the record", "flip and rename", "fix the merged intact scores", "fix the intact scores of shard 0 too",
          "pass a failed merge off as one process", "and drop the merge's fields")


@pytest.mark.parametrize("forgery", ["flip the record", "flip and rename", "drop the shard record", "edit a table",
                                     "edit a shard", "lose the intact check", "loosen the tolerance",
                                     # agent3 2026-10-13, F1 and F2: the record and some of the tables edited together
                                     "edit a table and its fingerprint", "fix the merged intact scores",
                                     "fix the intact scores of shard 0 too", "swap the shards' states",
                                     "pass a failed merge off as one process", "and drop the merge's fields"])
def test_changing_the_record_cannot_make_a_failed_or_altered_run_pass(tmp_path, forgery):
    """agent3 2026-10-13, N1, F1, F2: the guard checks each process's units, the merged tables against the shards' by
    content, and recomputes the cross-GPU check from the shard that ran the intact state, so editing meta.json,
    alone or with some of the tables and their fingerprints, never turns a failed or altered merge into a run."""
    _two_shards(tmp_path, shift=0.2 if forgery in FAILED else 0.0)
    out, _ = _merge(tmp_path)
    meta = _meta(out)
    if forgery == "edit a table and its fingerprint":         # K8 naming made worse in the merged table only
        _edit(out, "candidates", lambda df: df.assign(logprob=np.where(df.state == "K8", df.logprob - 10, df.logprob)))
        meta["tables_sha256"] = state_store.tables_sha256(out)
    elif forgery.startswith("fix the"):                       # the merged intact scores moved onto the rerun's
        out = out.rename(tmp_path / "run")
        _fix_intact(out, 0.2)
        if forgery.endswith("too"):
            _fix_intact(tmp_path / "run.shard0of2", 0.2)
            meta["source_tables_sha256"]["run.shard0of2"] = state_store.tables_sha256(tmp_path / "run.shard0of2")
        meta["tables_sha256"] = state_store.tables_sha256(out)
        _passed(meta)
    elif forgery == "swap the shards' states":
        a, b = _meta(tmp_path / "run.shard0of2"), _meta(tmp_path / "run.shard1of2")
        a["states"], b["states"] = b["states"], a["states"]
        for d, m in (("run.shard0of2", a), ("run.shard1of2", b)):
            (tmp_path / d / "meta.json").write_text(json.dumps(m), encoding="utf-8")
        meta["shards"][0]["states"], meta["shards"][1]["states"] = a["states"], b["states"]
    elif forgery in ("pass a failed merge off as one process", "and drop the merge's fields"):
        (tmp_path / "archive").mkdir()
        out = shutil.copytree(out, tmp_path / "archive" / "run")
        meta.pop("shards")
        meta.update(shard="0/1", complete=True)
        if forgery.startswith("and"):
            for k in op.MERGE_FIELDS:
                meta.pop(k, None)
    elif forgery.startswith("flip"):
        meta.update(complete=True)
        meta["cross_gpu_check"].update(passed=True)
        for v in meta["cross_gpu_check"]["results"].values():
            v.update(passed=True, max=0.0, median=0.0)
        if forgery == "flip and rename":
            out = out.rename(tmp_path / "run")
    elif forgery == "drop the shard record":
        meta.pop("shards")
        meta["shard"] = "0/1"
    elif forgery == "edit a table":
        _edit(out, "labels", lambda df: df.assign(margin=df.margin + 1))
    elif forgery == "edit a shard":
        _edit(tmp_path / "run.shard1of2", "labels", lambda df: df.assign(margin=df.margin + 1))
    elif forgery == "lose the intact check":
        shutil.rmtree(tmp_path / "run.intactcheck1of2")
    elif forgery == "loosen the tolerance":
        meta["cross_gpu_check"]["tolerance"] = dict(max=1.0, median=1.0)
    assert op.confirm_problems(out, meta, smoke=True)


def test_merge_refuses_shards_from_different_code_or_outside_the_plan(tmp_path):
    _two_shards(tmp_path, commit="other")
    with pytest.raises(SystemExit):
        _merge(tmp_path)
    shutil.rmtree(tmp_path / "run.shard1of2")
    _part(tmp_path, "run.shard1of2", op.shard_states(list(NAMES), "1/2")[0][:-1], shard="1/2")
    with pytest.raises(SystemExit):                # a state of the plan is missing
        _merge(tmp_path)


def test_merge_checks_each_part_against_the_plan(tmp_path):
    _two_shards(tmp_path)
    _edit(tmp_path / "run.shard1of2", "labels", lambda df: pd.concat([df, df.head(1).assign(variant=7)]))
    with pytest.raises(SystemExit):
        _merge(tmp_path)
    assert not (tmp_path / "run").exists()


def test_the_order_of_concepts_in_a_record_carries_no_meaning(tmp_path):
    """agent3 2026-10-13, G3: reordering item_roles in any record, a process's or the merged one, refuses nothing."""
    meta = _single(tmp_path / "one" / "run")
    meta["item_roles"] = dict(reversed(list(meta["item_roles"].items())))
    assert op.confirm_problems(tmp_path / "one" / "run", meta, smoke=True) == []
    _two_shards(tmp_path)
    out, _ = _merge(tmp_path)
    for d in (tmp_path / "run.shard0of2", tmp_path / "run.shard1of2", tmp_path / "run.intactcheck1of2", out):
        m = _meta(d)
        m["item_roles"] = dict(reversed(list(m["item_roles"].items())))
        (d / "meta.json").write_text(json.dumps(m), encoding="utf-8")
    assert op.confirm_problems(out, _meta(out), smoke=True) == []


def _single(d):
    _layout(d, list(NAMES), router=True)
    meta = _record(NAMES, ["states", "router"], shard="0/1", router_experts=op.router_components(op.tiny_sets(3)))
    _write_run(d, meta)
    return meta


def test_a_single_process_run_needs_no_shards_but_no_shard_folders_beside_it(tmp_path):
    meta = _single(tmp_path / "run")
    assert op.confirm_problems(tmp_path / "run", meta, smoke=True) == []
    assert op.confirm_problems(tmp_path / "run", dict(meta, shard="0/2"), smoke=True)
    (tmp_path / "run.shard0of2").mkdir()
    assert op.confirm_problems(tmp_path / "run", meta, smoke=True)


@pytest.mark.parametrize("change", ["no plan", "plan of another shard", "record of other states", "edit a table",
                                    "edit a unit", "lose a unit", "extra unit", "unit outside the plan",
                                    "finished at another commit", "plan of other concepts"])
def test_a_single_process_run_must_be_what_its_units_add_up_to(tmp_path, change):
    """agent3 2026-10-13, F2: one process's run is read only with the plan.json it started under and the units it
    wrote; its tables must be exactly their concatenation."""
    d = tmp_path / "run"
    meta = _single(d)
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    if change == "no plan":
        (d / "plan.json").unlink()
    elif change == "plan of another shard":
        state_store.write_json(d / "plan.json", dict(plan, shard="0/2"))
    elif change == "record of other states":
        meta["states"] = list(reversed(meta["states"]))
    elif change == "finished at another commit":           # agent3 G2: start, resume and finish at one commit
        meta["git_commit"] = "other"
    elif change == "plan of other concepts":               # G3: the plan's concepts in item-file order
        state_store.write_json(d / "plan.json", dict(plan, items=list(reversed(plan["items"]))))
    elif change == "edit a table":
        _edit(d, "labels", lambda df: df.assign(margin=df.margin + 1))
    elif change == "edit a unit":
        _edit(d / "units" / "state=K8", "labels", lambda df: df.assign(margin=df.margin + 1))
    elif change == "lose a unit":
        shutil.rmtree(d / "units" / "state=C4")
    elif change == "extra unit":
        shutil.copytree(d / "units" / "state=C4", d / "units" / "state=C4.tmp")
    elif change == "unit outside the plan":                   # a unit's rows describe another state than its name
        u = d / "units" / "state=K8@0.25"
        for f in ("candidates", "labels", "greedy", "lm_loss"):
            pd.read_parquet(d / "units" / "state=K8@0.5" / f"{f}.parquet").to_parquet(u / f"{f}.parquet", index=False)
        m = _marker(u)
        for f in m["tables"]:
            m["tables"][f]["sha256"] = state_store._sha(u / f"{f}.parquet")
        (u / state_store.MARKER).write_text(json.dumps(m), encoding="utf-8")
    assert op.confirm_problems(d, meta, smoke=True)


# --- the confirmatory analysis's guard on a real record ----------------------------------------------------------------

def _real_record(monkeypatch, tables_ok=True):
    """A record of a real confirmatory run as the plan wants it, with git, the input files, the shard evidence and
    the table check replaced by stand-ins (no tag, no means file and no run on this machine)."""
    monkeypatch.setattr(op.runinfo, "git", lambda *a: "c0ffee" if a[0] == "rev-parse" else "")
    monkeypatch.setattr(op.runinfo, "analysis_plan_sha256_at", lambda rev: "plan-hash")
    monkeypatch.setattr(op.runinfo, "sha256_at", lambda rev, path: "protocol-hash")
    monkeypatch.setattr(op, "current_inputs", lambda: {k: op.FROZEN[k] for k in
                                                       ("means_sha256", "final_items_sha256", "w2d_foils_sha256", "random_sets_sha256", "random_meta_sha256")})
    monkeypatch.setattr(op, "evidence_problems", lambda d, meta: [])

    def tables(d, role, tiny, states, parts):
        if not tables_ok:
            raise ValueError("labels: state K8 lacks rows")
    monkeypatch.setattr(op, "check_run_tables", tables)
    names = list(op.state_names("test"))
    return dict(role="test", complete=True, tiny=False, model=op.MODEL, dtype=op.DTYPE, freeze_tag=op.FREEZE_TAG,
                git_commit="c0ffee", clean_tree=True, revision=op.FROZEN["revision"], means_path=op.MEANS,
                means_sha256=op.FROZEN["means_sha256"], final_items_sha256=op.FROZEN["final_items_sha256"],
                w2d_foils_sha256=op.FROZEN["w2d_foils_sha256"],
                random_controls=dict(sets_sha256=op.FROZEN["random_sets_sha256"], meta_sha256=op.FROZEN["random_meta_sha256"],
                                     commit=op.FROZEN["random_commit"] + "0123"),
                analysis_plan_sha256="plan-hash", protocol_sha256="protocol-hash", sets=op.frozen_sets(3),
                router_experts=op.router_components(op.FROZEN_SETS), item_roles={i: "test" for i in op.confirm_ids()},
                states=names, all_states=names, parts=["router", "states"], ablation_mode="mean",
                n_neutral_texts=op.N_NEUTRAL, neutral_tokens=op.NEUTRAL_TOKENS)


@pytest.mark.parametrize("change", [
    dict(tiny=True), dict(dtype="bfloat16"), dict(model="other"), dict(freeze_tag="analysis-frozen-other"),
    dict(git_commit="other"), dict(clean_tree=False), dict(means_sha256="x"), dict(means_path="outputs/means/other.pt"),
    dict(analysis_plan_sha256="edited"), dict(protocol_sha256="edited"), dict(random_controls=dict(sets_sha256="x")),
    dict(sets={**op.frozen_sets(3), "C4": ["head:9:15"]}), dict(router_experts=["expert:46:61"]),
    dict(item_roles={"m1": "test"}), dict(states=list(op.state_names("test"))[:44]), dict(parts=["states"]),
    dict(complete=False), dict(ablation_mode="zero"), dict(n_neutral_texts=1), dict(neutral_tokens=32),
    dict(all_states=list(reversed(op.state_names("test"))))])
def test_the_confirmatory_analysis_refuses_runs_outside_the_plan(monkeypatch, change):
    good = _real_record(monkeypatch)
    assert op.confirm_problems(Path("run"), good) == []
    assert op.confirm_problems(Path("run"), {**good, **change})


@pytest.mark.parametrize("folder", ["run.crossgpu_failed-20261014-010203", "run.partial", "run.shard0of2", "run.intactcheck1of2"])
def test_the_confirmatory_analysis_reads_only_a_finished_run_folder(monkeypatch, folder):
    good = _real_record(monkeypatch)
    assert op.confirm_problems(Path(folder), good)


def test_the_confirmatory_analysis_runs_the_frozen_code(monkeypatch):
    """agent3 2026-10-13, N2: the analysis code itself must equal the tag's, not only the run's record."""
    good = _real_record(monkeypatch)
    monkeypatch.setattr(op.runinfo, "differs_from", lambda rev, paths=None: ["experiments/e3_errors/analyze_optogenetics.py"])
    assert any("differs from" in p for p in op.confirm_problems(Path("run"), good))


def test_the_confirmatory_analysis_reads_the_tables_and_the_files_now(monkeypatch):
    good = _real_record(monkeypatch, tables_ok=False)
    assert any("tables" in p for p in op.confirm_problems(Path("run"), good))
    good = _real_record(monkeypatch)
    monkeypatch.setattr(op, "current_inputs", lambda: dict(means_sha256="changed since the run"))
    assert op.confirm_problems(Path("run"), good) == ["current means_sha256 differs from the plan"]
    with pytest.raises(SystemExit):
        ao.check_confirm_run(Path("run"), {**good, "dtype": "bfloat16"})


def test_the_analysis_compares_the_protocol_with_the_tag_too(monkeypatch):
    """agent3 2026-10-13, F4: the protocol file is part of what the analysis must share with the tag."""
    seen = []
    monkeypatch.setattr(op.runinfo, "differs_from", lambda rev, paths=None: seen.append((rev, paths)) or [])
    assert op.code_problems() == []
    assert seen[0][0] == op.FREEZE_TAG and op.PROTOCOL in seen[0][1] and set(op.runinfo.TRACKED) <= set(seen[0][1])


def test_the_run_compares_the_plan_with_the_tag_before_loading(monkeypatch):
    monkeypatch.setattr(op.runinfo, "analysis_plan_sha256", lambda: "a")
    monkeypatch.setattr(op.runinfo, "analysis_plan_sha256_at", lambda rev: "a")
    monkeypatch.setattr(op.runinfo, "sha256_at", lambda rev, path: op.runinfo.sha256(ROOT / op.PROTOCOL))
    assert op.tagged_plan_problems() == []
    monkeypatch.setattr(op.runinfo, "sha256_at", lambda rev, path: "edited protocol")
    assert op.tagged_plan_problems()


# --- the confirmatory analysis end to end on a merged tiny layout ------------------------------------------------------

def test_the_smoke_analysis_ignores_the_order_of_the_record_and_needs_its_own_output(tmp_path):
    """agent3 2026-10-13, N5 and N6: reordering the concepts in meta.json changes nothing; --smoke never writes the
    confirmatory report."""
    _two_shards(tmp_path)
    out, _ = _merge(tmp_path)
    with pytest.raises(SystemExit):
        ao.main(["--confirm", "--smoke", "--dir", str(out)])
    ao.main(["--confirm", "--smoke", "--dir", str(out), "--out", str(tmp_path / "a.json")])
    meta = _meta(out)
    meta["item_roles"] = dict(reversed(list(meta["item_roles"].items())))
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    ao.main(["--confirm", "--smoke", "--dir", str(out), "--out", str(tmp_path / "b.json")])
    a, b = (json.loads((tmp_path / f).read_text(encoding="utf-8")) for f in ("a.json", "b.json"))
    assert a["rules"] == b["rules"] and a["source_tables_sha256"] == state_store.tables_sha256(out)
    assert a["note"].startswith("SMOKE")


def test_a_smoke_analysis_never_writes_into_the_confirmatory_report_folder(tmp_path, monkeypatch):
    """agent3 2026-10-13, F7: not by naming the report, nor by a path that leads there another way."""
    monkeypatch.setattr(ao, "ROOT", tmp_path)                 # the report folder of a stand-in repo
    report = tmp_path / "reports/confirm/optogenetics/result.json"
    report.parent.mkdir(parents=True)
    report.write_text("the confirmatory report", encoding="utf-8")
    for out in (report, report.parent / "other.json", tmp_path / "reports/confirm/x/../optogenetics/result.json"):
        with pytest.raises(SystemExit, match="never writes"):
            ao.main(["--confirm", "--smoke", "--dir", str(tmp_path / "no run"), "--out", str(out)])
    link = tmp_path / "alias"
    try:
        link.symlink_to(report.parent, target_is_directory=True)
    except OSError:
        link = None                                           # no symlinks without privileges on Windows
    if link is not None:
        with pytest.raises(SystemExit, match="never writes"):
            ao.main(["--confirm", "--smoke", "--dir", str(tmp_path / "no run"), "--out", str(link / "result.json")])
    # G4: --smoke without --confirm, and the exploratory analysis, are refused too
    with pytest.raises(SystemExit, match="only with --confirm"):
        ao.main(["--smoke", "--dir", str(tmp_path / "no run"), "--out", str(tmp_path / "elsewhere.json")])
    with pytest.raises(SystemExit, match="only the confirmatory analysis"):
        ao.main(["--dir", str(tmp_path / "no run"), "--out", str(report)])
    assert report.read_text(encoding="utf-8") == "the confirmatory report"


# --- confirmatory rules ------------------------------------------------------------------------------------------------

def _losses(ids, effect):
    """Synthetic per-concept losses in which every planned effect has size `effect` (0 = none at all)."""
    rng = np.random.default_rng(1)
    noise = lambda: rng.normal(0, 0.05, len(ids))
    frame = lambda **cols: pd.DataFrame({m: cols.get(m, noise()) for m in ("naming", "six_choice", "B", "H", "dR")}, index=ids)
    L = {"K8": frame(naming=effect + noise()), "C4": frame(six_choice=effect + noise()),
         "K8@0.75": frame(naming=0.1 * effect + noise()), "C4@0.75": frame(six_choice=0.1 * effect + noise()),
         "K8@2.0": frame(dR=-effect + noise()), "K8_seg_answer": frame(naming=effect + noise(), B=effect + noise()),
         "K8_answer@0.75": frame(naming=0.1 * effect + noise(), B=0.1 * effect + noise()),
         "C4_seg_middle": frame(H=-effect + noise()), "C4_seg_answer": frame(H=effect + noise())}
    for a in ao.DOSE_ALPHAS:
        L[f"K8@{a}"] = frame(naming=effect * (1 - a) + noise())
        L[f"C4@{a}"] = frame(six_choice=effect * (1 - a) + noise())
    L["K8+C4"] = frame(naming=L["K8"].naming + L["C4"].naming, six_choice=L["K8"].six_choice + L["C4"].six_choice + effect)
    return L


def _lm():
    """Language-loss rises of every whole state the rules read, none above 10%."""
    return {s: 0.01 for k in ao.LM_STATES for s in ao.LM_STATES[k]}


def test_rules_support_planted_effects_and_not_their_absence():
    ids = [f"m{i}" for i in range(60)]
    hint = ids[:40]
    got = ao.rules(_losses(ids, 1.0), ids, hint, _lm(), n_boot=500)
    assert all(v["verdict"] == "支持" for v in got.values())
    got = ao.rules(_losses(ids, 0.0), ids, hint, _lm(), n_boot=500)
    assert all(v["verdict"] == "证据不足" for k, v in got.items() if k != "R2")
    got = ao.rules(_losses(ids, 1.0), ids, hint, _lm() | {"K8@2.0": 0.2}, n_boot=500)
    assert got["R3"]["verdict"].startswith("不解释") and got["R1"]["verdict"] == "支持"


def test_segment_rules_check_the_language_loss_of_their_whole_lesions():
    """agent3 2026-10-13, N4, user decision the same day: R4 and R5 use the whole-sequence lesion of the same set and
    alpha; a loss that was not measured is never read as below 10%."""
    ids = [f"m{i}" for i in range(60)]
    L = _losses(ids, 1.0)
    got = ao.rules(L, ids, ids[:40], _lm() | {"K8@0.75": 0.2}, n_boot=500)
    assert got["R4"]["verdict"].startswith("不解释") and got["R1"]["verdict"].startswith("不解释")
    assert got["R5"]["verdict"] == "支持" and got["R4"]["lm_loss_states"] == ["K8", "K8@0.75"]
    lm = _lm()
    del lm["C4"]
    got = ao.rules(L, ids, ids[:40], lm, n_boot=500)
    assert got["R5"]["verdict"].startswith("无法检查") and got["R5"]["lm_loss_missing_states"] == ["C4"]
    assert got["R3"]["verdict"] == "支持"


def test_rules_read_unrounded_intervals():
    """agent3 2026-10-13, N7: an interval just inside the margin, or just above 0, is not rounded onto it."""
    ids = [f"m{i}" for i in range(60)]
    L = _losses(ids, 1.0)
    L["K8+C4"] = L["K8+C4"].assign(naming=L["K8"].naming + L["C4"].naming + 0.19996)
    got = ao.rules(L, ids, ids[:40], _lm(), n_boot=500)
    assert got["R6"]["naming_interaction"]["ci"][1] < ao.RULE_MARGIN and got["R6"]["conditions_met"]
    L["K8@0.75"] = L["K8@0.75"].assign(naming=L["K8"].naming - 0.00004)
    got = ao.rules(L, ids, ids[:40], _lm(), n_boot=500)
    assert 0 < got["R1"]["K8_naming_alpha0_minus_0_75"]["ci"][0] < 0.0001 and got["R1"]["conditions_met"]


# --- H3c helpers ---------------------------------------------------------------------------------------------------

def test_consistency_and_correct_log_odds():
    h = pd.DataFrame(dict(naming=[1, 2, 3, 4.0], word_to_definition=[2, 4, 6, 8.0], options_letter=[10, 20, 30, 40.0]))
    mean, pairs = ac.consistency(h)
    assert mean == pytest.approx(1.0) and all(v == pytest.approx(1.0) for v in pairs.values())
    labels = pd.DataFrame(dict(state="intact", item_id="m1", fmt="word_to_definition", correct="A",
                               logit_A=[2.0, 0.0], logit_B=[0.0, 2.0]))
    lo = ac.correct_log_odds(labels.assign(logit_C=np.nan, logit_D=np.nan, logit_E=np.nan, logit_F=np.nan))
    assert lo.log_odds.iloc[0] == pytest.approx(0.0)          # +2 and -2 average to 0

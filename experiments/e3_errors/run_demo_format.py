"""Demonstration-format check (exploratory; docs/plans/demo-format-check.md, approved by the user 2026-10-28).

    python experiments/e3_errors/run_demo_format.py --model qwen3-30b-a3b-instruct \
        --means outputs/means/qwen3-30b-a3b-instruct@0d7cf23__42ae809__fp16.pt --dtype float16 --moe sorted \
        --also-existing

SC4 barely hurts naming on the M3 naming battery, whose prompts open with three demos (experiment 3, 0.03 nat), but
hurts naming of the final family concepts, whose prompts have none (0.78 nat). This run crosses the format with the
item set: the 160 M3 test concepts without the demos and the 120 final family concepts with the same three demos,
for the intact model, WR8 and SC4 (full mean ablation, WikiText means; the means, precision and component sets of
experiment 3 and the title-gap runs). --also-existing adds the two existing formats through the same code path: a
reproduction check against the stored runs and a within-run format contrast on the same concepts.

Prompts: the existing formats come from the frozen builders (lesion.m3_scoring.build_naming_prompts and
lesion.fine_semantics.naming_prompts, which stay byte-identical because the analysis protocols and the figdata manifest
record their hashes); the new formats from demo_format_prompts, copies of those builders with a demos switch, checked
before every run to reproduce the frozen output in the existing format. The new prompt of each item set is its
existing prompt with exactly the text of the three demo blocks removed (battery) or inserted (families); instruction,
description and answer cue stay as they were. Candidates as in the original runs:
the five free-naming candidates (experiment 3's name_free) and target, synonym and coordinate (family naming); the
reading is the target's summed log-probability. Each cell is scored on its own, with the scorer of its original run
(lesion.m3_battery.candidate_scores for the battery, lesion.m3_scoring.score_naming_prompts for the families; the same
computation), once per batch layout 16, 11 and 7 (the plan; the confirmatory runs' naming layouts). The per-layout
scores are kept; analyze_demo_format.py averages them. Scoring the cells separately means --also-existing leaves the
new cells' batches, and so their scores, unchanged.

Writes outputs/e3_errors/<model>/demo_format/{scores,prompts}.parquet and meta.json (model revision, git commit,
component lists, item and means hashes, timings). The folder is demo_format.partial until every state is done; a rerun
with the same arguments at the same commit resumes after the last finished state; a finished run is never overwritten.
Real runs need a clean committed tree (common.guard).

--tiny: random 2-layer model, stand-in component sets, 4 concepts per item set (local code check; --means
outputs/means_tiny/<model>.pt). --dry-run: build and count the prompts, print one example per cell and exit without
loading a model (the token counts need the tokenizer files).
"""
import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))

sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import common  # noqa: E402
import demo_format_prompts as dfp  # noqa: E402
import run_dissociation_test as rdt  # noqa: E402
from lesion import fine_semantics as fs, m3_battery as mb, m3_items, m3_scoring as ms, models, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402

BATTERY_ITEMS = m3_items.OUT / "final_items.csv"
FAMILY_ITEMS = ROOT / "data/materials/title_gap/fine_final/items.csv"
BATTERY_SHA256 = "d539310755a31f8155cda5db7ae4920760dd36bbde97442800fa0b17df16b98d"   # experiment 3's final items
FAMILY_SHA256 = "cf64b71ebbb6fb6bf62a896ca195c707b5fe30075cffbd2f722d1d7fde30152d"    # title-gap-fine.json items
MEANS_SHA256 = "98b3865a71b0c8e85c530cb78d5438f9223a828fdf77c907315785b1f1de0d52"     # WikiText means of both runs
N_BATTERY, N_FAMILY = 160, 120
MEMBERS = ROOT / "docs/paper-prep/component-members.csv"
FAMILY_STATES = ROOT / "experiments/title_gap/configs/fine-final-core.json"
SET_SOURCES = dict(WR8=("K8", "WR8_a0"), SC4=("C4", "SC4_a0"))   # paper code -> (experiment 3 name, family state)
TINY_SETS = dict(WR8=("expert:1:0", "expert:1:5", "head:1:1"), SC4=("head:0:2", "head:1:3"))
NEW_CELLS = (("battery", "no_demos"), ("family", "demos"))
EXISTING_CELLS = (("battery", "demos"), ("family", "no_demos"))
# --capitalized (2026-11-02): the family prompts do not ask for lowercase, and the intact model capitalizes most of its
# family answers, so the same family prompts are also scored with every candidate's first letter capitalized, in cells
# of their own (their batches never mix with the lowercase cells); the analysis combines log(P(word) + P(Word)).
CAP_CELLS = (("family_cap", "demos"), ("family_cap", "no_demos"))
BATCHES = ms.NAMING_BATCHES          # (16, 11, 7)
TINY_BATCHES = (4, 3, 2)
CONDITION = {("battery", "demos"): "name_free", ("battery", "no_demos"): "name_free_no_demos"}
PROMPT_SOURCES = ("src/lesion/m3_scoring.py", "src/lesion/fine_semantics.py", "experiments/e3_errors/demo_format_prompts.py")


def component_sets(model_name):
    """WR8 and SC4 from the paper's member table, checked against experiment 3 (rdt.K8; the C4 of its meta.json when
    the outputs are present) and the final family states, where both were full mean ablations (alpha 0)."""
    members = pd.read_csv(MEMBERS, keep_default_na=False)
    sets = {code: tuple(members[members.paper_code == code].sort_values("order").component) for code in SET_SOURCES}
    if sets["WR8"] != tuple(rdt.K8):
        raise SystemExit("WR8 in component-members.csv differs from experiment 3's K8")
    states = {s["name"]: s for s in json.loads(FAMILY_STATES.read_text(encoding="utf-8"))["states"]}
    e3_meta = ROOT / "outputs/e3_errors" / model_name / "confirm/meta.json"
    e3_sets = json.loads(e3_meta.read_text(encoding="utf-8"))["sets"] if e3_meta.exists() else None
    for code, (e3_name, family_state) in SET_SOURCES.items():
        gates = states[family_state]["gates"]
        if set(gates) != set(sets[code]) or set(gates.values()) != {0.0} or len(sets[code]) != len(set(sets[code])):
            raise SystemExit(f"{code} differs from the family run's {family_state}")
        if e3_sets is not None and list(sets[code]) != e3_sets[e3_name]:
            raise SystemExit(f"{code} differs from experiment 3's {e3_name}")
    return sets


def battery_items(limit=0):
    """The 160 test concepts of experiment 3, in its order."""
    final = pd.read_csv(BATTERY_ITEMS, keep_default_na=False)
    items = final[final.role == "test"].reset_index(drop=True)
    return items.head(limit) if limit else items


def family_items(limit=0):
    """The 120 final family concepts as fine_semantics.load_items(path, "test") returns them, without its freeze-tag
    check: the split was opened by the frozen runs, and this check reads it again for an exploratory contrast, as the
    session-11 runs did for the M3 test concepts."""
    table = pd.read_csv(FAMILY_ITEMS, keep_default_na=False)
    table = table[table.review != "excluded"].copy()
    fs.validate(table)
    items = table[table.split == "test"].copy()
    if items.empty or not (items.review == "approved").all():
        raise SystemExit("family test concepts missing or not approved")
    return items.head(limit) if limit else items


def cell_prompts(item_set, fmt, items, chat=True):
    """Prompts of one cell (item set, format): the frozen builder for the existing format, its local copy with only
    the demos switched for the new one."""
    if item_set == "battery":
        prompts = (ms.build_naming_prompts(items, ("none",), chat, roles=ms.ERROR_ROLES) if fmt == "demos" else
                   dfp.battery_prompts(items, ("none",), chat, roles=ms.ERROR_ROLES, demos=False))
        for p in prompts:
            p.condition = CONDITION[(item_set, fmt)]
        return prompts
    if item_set == "family":
        return fs.naming_prompts(items) if fmt == "no_demos" else dfp.family_prompts(items, demos=True)
    if item_set == "family_cap":
        out = cell_prompts("family", fmt, items, chat)
        for p in out:
            p.candidates = {f"{r}_cap": w[:1].upper() + w[1:] for r, w in p.candidates.items()}
        return out
    raise ValueError(f"unknown item set {item_set!r}")


def build_cells(cells, battery, family, chat=True):
    dfp.check_against_frozen(battery, family, chat)
    return {cell: cell_prompts(*cell, battery if cell[0] == "battery" else family, chat) for cell in cells}


def prompt_digest(prompts):
    return hashlib.sha256(json.dumps([vars(p) for p in prompts], ensure_ascii=False).encode()).hexdigest()


def prompt_table(prompt_sets):
    return pd.DataFrame([dict(item_set=s, format=f, item_id=p.item_id, condition=p.condition, text=p.text,
                              candidates=json.dumps(p.candidates, ensure_ascii=False))
                         for (s, f), prompts in prompt_sets.items() for p in prompts])


SCORERS = {
    "battery": lambda model, tok, prompts, chat, bs: mb.candidate_scores(model, tok, prompts, chat, bs),
    "family": lambda model, tok, prompts, chat, bs: ms.score_naming_prompts(model, tok, prompts, chat, bs),
    "family_cap": lambda model, tok, prompts, chat, bs: ms.score_naming_prompts(model, tok, prompts, chat, bs),
}


def score_state(model, tok, chat, prompt_sets, batches, name, comps=None, ablate=None):
    """Every cell of one lesion state, once per batch layout; (scores, seconds per cell)."""
    rows, seconds = [], {}
    with nullcontext() if comps is None else ablate(comps):
        for (item_set, fmt), prompts in prompt_sets.items():
            start = time.time()
            for bs in batches:
                rows.append(SCORERS[item_set](model, tok, prompts, chat, bs)
                            .assign(item_set=item_set, format=fmt, state=name, layout=bs))
            seconds[f"{item_set}/{fmt}"] = round(time.time() - start, 1)
    return pd.concat(rows, ignore_index=True), seconds


def counts(prompt_sets, batches, n_states, tokenizer=None, chat=True):
    """Prompts, candidate sequences and (with a tokenizer) tokens, per cell and in total."""
    out = {}
    for (s, f), prompts in prompt_sets.items():
        jobs = sum(len(p.candidates) for p in prompts)
        row = dict(prompts=len(prompts), sequences_per_layout=jobs, sequences=jobs * len(batches) * n_states)
        if tokenizer is not None:
            tokens = sum(len(a) + len(b) for p in prompts for w in p.candidates.values()
                         for a, b in [ms.encode_answer(tokenizer, p.text, w, chat)])
            row.update(tokens_per_layout=tokens, tokens=tokens * len(batches) * n_states)
        out[f"{s}/{f}"] = row
    total = {k: sum(r[k] for r in out.values()) for k in next(iter(out.values()))}
    return dict(cells=out, total=total, states=n_states, layouts=list(batches))


def dry_run(prompt_sets, batches, n_states, model_name, chat):
    try:
        tok = models.load_tokenizer(model_name)
    except Exception as e:                  # no tokenizer files: counts without tokens
        print(f"(no tokenizer: {e})")
        tok = None
    print(json.dumps(counts(prompt_sets, batches, n_states, tok, chat), indent=1))
    for (s, f), prompts in prompt_sets.items():
        print(f"\n--- {s}/{f}: {prompts[0].item_id}, candidates {prompts[0].candidates}\n{prompts[0].text}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(ap)
    ap.add_argument("--also-existing", action="store_true",
                    help="also score the existing formats (battery with demos, families without) the same way")
    ap.add_argument("--capitalized", action="store_true",
                    help="also score both family formats with capitalized candidates (cells family_cap/*)")
    ap.add_argument("--dry-run", action="store_true", help="count and show the prompts, load no model")
    ap.add_argument("--limit", type=int, default=0, help="concepts per item set, tiny runs only (default 4)")
    ap.add_argument("--out", type=Path, help="output folder (default outputs/e3_errors[_tiny]/<model>/demo_format)")
    args = ap.parse_args()
    if args.ablation_mode != "mean":
        ap.error("the plan fixes mean ablation")
    if args.limit and not args.tiny:
        ap.error("--limit is for tiny rehearsals only")
    limit = args.limit or (4 if args.tiny else 0)
    battery, family = battery_items(limit), family_items(limit)
    if not args.tiny:
        for path, digest in ((BATTERY_ITEMS, BATTERY_SHA256), (FAMILY_ITEMS, FAMILY_SHA256)):
            if runinfo.sha256(path) != digest:
                raise SystemExit(f"{path} differs from the items of the original runs")
        if (len(battery), len(family)) != (N_BATTERY, N_FAMILY):
            raise SystemExit(f"expected {N_BATTERY} and {N_FAMILY} concepts, found {len(battery)} and {len(family)}")
    sets = {k: tuple(v) for k, v in (TINY_SETS if args.tiny else component_sets(args.model)).items()}
    plan = [("intact", None)] + [(k, [Component.parse(c) for c in v]) for k, v in sets.items()]
    cells = NEW_CELLS + (EXISTING_CELLS if args.also_existing else ()) + (CAP_CELLS if args.capitalized else ())
    batches = TINY_BATCHES if args.tiny else BATCHES
    chat = models.uses_chat(args.model)
    prompt_sets = build_cells(cells, battery, family, chat)
    if args.dry_run:
        dry_run(prompt_sets, batches, len(plan), args.model, chat)
        return

    out = args.out or ROOT / ("outputs/e3_errors_tiny" if args.tiny else "outputs/e3_errors") / args.model / "demo_format"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    if not args.tiny and runinfo.sha256(args.means) != MEANS_SHA256:
        raise SystemExit(f"{args.means} is not the WikiText means file of experiment 3 and the family runs")
    commit = common.guard(args)
    fields = dict(stage="demo_format", model=args.model, revision=models.MODELS[args.model][1], tiny=args.tiny,
                  dtype=args.dtype, moe_forward=args.moe, means_path=str(args.means),
                  means_sha256=runinfo.sha256(args.means), ablation=dict(mode="mean", alpha=0.0),
                  sets={k: list(v) for k, v in sets.items()}, states=[name for name, _ in plan],
                  cells=[list(c) for c in cells], also_existing=args.also_existing, batches=list(batches), chat=chat,
                  battery_items_sha256=runinfo.sha256(BATTERY_ITEMS), family_items_sha256=runinfo.sha256(FAMILY_ITEMS),
                  n_concepts=dict(battery=len(battery), family=len(family)),
                  prompts_sha256={f"{s}/{f}": prompt_digest(p) for (s, f), p in prompt_sets.items()},
                  prompt_sources_sha256={p: runinfo.sha256(ROOT / p) for p in PROMPT_SOURCES},
                  resume_commit=commit)
    partial = out.with_name(out.name + ".partial")
    tables, timing, done = [], {}, []
    if partial.exists():
        old = json.loads((partial / "meta.json").read_text(encoding="utf-8"))
        changed = [k for k in fields if json.dumps(old.get(k), default=str) != json.dumps(fields[k], default=str)]
        if changed:
            raise SystemExit(f"{partial} was started with other {changed}; move it away to start again")
        if (partial / "scores.parquet").exists():
            tables = [pd.read_parquet(partial / "scores.parquet")]
        done, timing = list(old.get("done", [])), dict(old.get("timing", {}))
        print(f"resuming after {done}", flush=True)
    partial.mkdir(parents=True, exist_ok=True)
    prompt_table(prompt_sets).to_parquet(partial / "prompts.parquet", index=False)

    def save(status, **extra):
        if tables:
            tmp = partial / "scores.parquet.tmp"
            pd.concat(tables, ignore_index=True).to_parquet(tmp, index=False)
            tmp.replace(partial / "scores.parquet")
        return runinfo.write_meta(partial, status=status, done=done, timing=timing, **fields, **extra)

    save("running")
    t0 = time.time()
    model, tok, model_chat, _, ablate = common.load(args)
    if model_chat != chat:
        raise SystemExit("prompt format and model disagree on the chat template")
    load_seconds = round(time.time() - t0, 1)
    print(f"model loaded in {load_seconds} s; {counts(prompt_sets, batches, len(plan))['total']}", flush=True)
    for name, comps in plan:
        if name in done:
            continue
        start = time.time()
        table, seconds = score_state(model, tok, chat, prompt_sets, batches, name, comps, ablate)
        tables.append(table)
        done.append(name)
        timing[name] = dict(seconds=round(time.time() - start, 1), cells=seconds)
        save("running")
        print(f"{name}: {timing[name]}", flush=True)
    save("done", complete=True, load_seconds=load_seconds, seconds=round(time.time() - t0, 1),
         n_prompts=sum(len(p) for p in prompt_sets.values()), counts=counts(prompt_sets, batches, len(plan)))
    partial.rename(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

"""Setup shared by the MVP entry scripts: arguments, model, means and the ablation switch."""
import functools
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import fast_moe, models, prefix_scoring, runinfo, scoring  # noqa: E402

MATERIAL_FILES = ["items.csv", "split.csv", "demos.csv", "templates.yaml"]
# prefix: shared prefix once, float32 output layer; full32: each prompt in full, float32 output layer;
# full: the frozen G1 scorer (output layer in the model's precision).
# prefix_avg3: prefix scoring averaged over three batch compositions (ADR 0004; --batch-size is ignored).
SCORERS = {"prefix": prefix_scoring.score_label_prompts, "full32": prefix_scoring.score_label_prompts_full,
           "full": scoring.score_label_prompts,
           "prefix_avg3": prefix_scoring.averaged(prefix_scoring.score_label_prompts, (400, 128, 48)),
           # the same average over three batch compositions, smaller batches for a model that leaves less GPU
           # memory free (Phi-3.5-MoE: about 78 GiB of weights on a 96 GB card; second model, P6)
           "prefix_avg3_small": prefix_scoring.averaged(prefix_scoring.score_label_prompts, (96, 40, 16))}


def add_args(ap):
    ap.add_argument("--model", required=True, help="registered model name, the version chosen at G1")
    ap.add_argument("--means", required=True, help="means file from experiments/prep_means/run.py")
    ap.add_argument("--tiny", action="store_true", help="random 2-layer model with the same architecture (local check)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--dtype", default="bfloat16", choices=models.DTYPES,
                    help="model precision; must match the means file (ADR 0004: float16 for Qwen)")
    ap.add_argument("--ablation-mode", default="mean", choices=["mean", "zero"],
                    help="mean ablation (default, all frozen results) or zero ablation (robustness checks only)")
    ap.add_argument("--scorer", default="prefix", choices=list(SCORERS),
                    help="prefix: shared prompt prefix computed once (same scores as full, faster; ADR 0003)")
    ap.add_argument("--moe", default="hf", choices=("hf",) + fast_moe.MODES,
                    help="MoE forward (lesion.fast_moe): hf (default), sorted (bit-identical to hf, faster) or grouped "
                         "(fastest, equal to rounding; freezes all weights)")


def guard(args):
    """Real runs need a clean committed tree so outputs map to one commit; tiny local checks do not."""
    return runinfo.commit() if args.tiny else runinfo.require_clean()


def out_dir(args, stage):
    return ROOT / ("outputs/e1_screen_tiny" if args.tiny else "outputs/e1_screen") / args.model / stage


def load(args):
    """(model, tokenizer, chat, means, ablate); ablate(components) is a context manager (lesion.hooks, agent2)."""
    from lesion import hooks, means as means_mod

    if args.model in models.PROMPT_FORMATS:          # checked before the weights are loaded
        raise SystemExit(f"{args.model} is registered for the P12 measurement check only (experiments/e0_baseline/run.py); "
                         "the lesion pipeline does not support its prompt format or components yet")
    if args.tiny:
        model, tok = models.load_tiny(args.model)
    else:
        models.check_environment()
        model, tok = models.load_model(args.model, dtype=models.torch_dtype(args.dtype))
    m = means_mod.Means.load(args.means)
    if m.meta.get("model_name") != args.model or (not args.tiny and m.meta.get("revision") != models.MODELS[args.model][1]):
        raise SystemExit(f"{args.means} was collected on {m.meta.get('model_name')}@{m.meta.get('revision')}, not {args.model}")
    if not args.tiny and m.meta.get("dtype", "bfloat16") != args.dtype:
        raise SystemExit(f"{args.means} was collected in {m.meta.get('dtype', 'bfloat16')}, not {args.dtype}")
    moe = getattr(args, "moe", "hf")
    if moe == "grouped":                              # it needs frozen weights; the lesion pipeline trains none
        for p in model.parameters():
            p.requires_grad_(False)
    if moe != "hf":
        print(f"MoE forward: {moe} (lesion.fast_moe), {fast_moe.install(model, moe)} blocks", flush=True)
    ablate = functools.partial(hooks.ablate, model, means=m, alpha=0.0, mode=getattr(args, "ablation_mode", "mean"))
    return model, tok, models.uses_chat(args.model), m, ablate


def moe_fields(args):
    """The MoE forward among the fields a resumed or reused run must match: only "grouped", whose results differ from
    hf's by rounding ("sorted" is bit-identical, so earlier runs, which have no such field, stay compatible). The
    forward actually used goes into meta.json as moe_forward, which no check compares; compare runs with moe_mismatch,
    which also catches an earlier grouped run met by a current hf or sorted one."""
    return {"moe": "grouped"} if getattr(args, "moe", "hf") == "grouped" else {}


def moe_mismatch(old, fields):
    """True if exactly one of an earlier run's meta.json and the current fields used the grouped forward."""
    return (old.get("moe") == "grouped") != (fields.get("moe") == "grouped")


def run_fields(args, commit, stage, split, n_prompts, **params):
    """Everything that must match before a stopped run may be resumed into the same folder.

    params: the stage's own settings that change its results (e.g. kind; n_random, max_pct_gap, n_boot).
    Only batch_size may differ on resume, since it does not change the scores.
    """
    return dict(stage=stage, model=args.model, revision=models.MODELS[args.model][1], tiny=args.tiny, split=split,
                ablation=dict(mode=getattr(args, "ablation_mode", "mean"), alpha=0.0), means_path=str(args.means), means_sha256=runinfo.sha256(args.means),
                materials_sha256={f: runinfo.sha256(scoring.M1 / f) for f in MATERIAL_FILES},
                analysis_plan_sha256=runinfo.analysis_plan_sha256(), scorer=args.scorer, dtype=args.dtype, params=params,
                resume_commit=commit, n_prompts=n_prompts, batch_size=args.batch_size, moe_forward=getattr(args, "moe", "hf"),
                **moe_fields(args))


# Files whose changes alter scores, lesions, damage or the selection rule (agent2 2026-09-28): a later stage
# that reuses a finished selection at a newer commit (same_code=False) still requires these to be unchanged.
SCORE_FILES = ("src/lesion/hooks.py", "src/lesion/means.py", "src/lesion/scoring.py", "src/lesion/prefix_scoring.py",
               "src/lesion/components.py", "src/lesion/damage.py", "src/lesion/screen.py",
               "data/materials/m1/items.csv", "data/materials/m1/split.csv", "data/materials/m1/demos.csv",
               "data/materials/m1/templates.yaml", "data/neutral_text")


def changes_since(old, new):
    """Files changed between two commits (all paths), for the record."""
    if not old or old == new:
        return []
    return runinfo.git("diff", "--name-only", old, new).splitlines()


SAME_RUN = ("model", "revision", "tiny", "ablation", "means_sha256", "materials_sha256", "analysis_plan_sha256", "scorer", "dtype",
            "moe")


def check_same_run(meta, fields, where, keys=SAME_RUN, same_code=True, allow_score_change=False):
    """Stop unless an earlier stage (its meta.json) used the same model, ablation, means and materials, and a
    commit whose src/, data/ and analysis plan are identical to the current one (entry-script or other document
    edits are allowed). same_code=False, for a later stage that reuses a finished selection (its hash is checked
    by load_selection) at a newer commit, such as the held-out test: only SCORE_FILES must be unchanged; record
    changes_since() in the stage's metadata. allow_score_change: a change to SCORE_FILES alone does not stop the run
    (exploratory reruns at a newer commit whose score code was shown to give identical scores; the caller records
    changes_since())."""
    # "moe" is a field only for the grouped forward (moe_fields)
    diff = [k for k in keys
            if json.dumps(meta.get(k), default=str) != json.dumps(fields.get(k) if k == "moe" else fields[k], default=str)]
    old, new = meta.get("resume_commit"), fields["resume_commit"]
    if old != new:
        paths = ("src", "data", runinfo.ANALYSIS_PLAN) if same_code else SCORE_FILES
        if not old or not runinfo.git("cat-file", "-t", old):
            diff.append(f"unknown earlier commit {old}")
        elif runinfo.git("diff", "--name-only", old, new, "--", *paths) != "" and not (allow_score_change and not same_code):
            diff.append(f"{'src/, data/ or the analysis plan' if same_code else 'score-relevant files'} ({old} -> {new})")
    if diff:
        raise SystemExit(f"{where} was produced with different {diff}; rerun it with the current code and arguments")


def shortlist(candidates_csv, comps, top):
    """The components among the top `top` of each direction by estimated effect (attribution.csv), in model order."""
    est = pd.read_csv(candidates_csv)
    keep = set()
    for col in ("control_effect", "knowledge_effect"):
        keep |= set(est.nlargest(top, col).component)
    return [c for c in comps if c.key in keep]


def component_set_sha256(keys):
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def load_selection(screen_dir, screen_meta):
    """selection.json of a finished screen, checked against the hashes its meta.json recorded."""
    sel_file, csv_file = screen_dir / "selection.json", screen_dir / "screen.csv"
    if screen_meta.get("status") != "done":
        raise SystemExit(f"{screen_dir} is unfinished; run run_screen.py first")
    for f, key in ((sel_file, "selection_sha256"), (csv_file, "screen_csv_sha256")):
        if runinfo.sha256(f) != screen_meta.get(key):
            raise SystemExit(f"{f} does not match the hash in {screen_dir / 'meta.json'}; it was changed after screening")
    selection = json.loads(sel_file.read_text(encoding="utf-8"))
    screened = set(pd.read_csv(csv_file).component)
    if component_set_sha256(screened) != screen_meta.get("screened_components_sha256"):
        raise SystemExit(f"{csv_file} does not hold the screened component set recorded in meta.json")
    chosen = [k for d in ("control", "knowledge") for k in selection.get(d, [])]
    extra = selection.get("supplementary", {}).get("control_unfiltered", [])
    if not set(chosen) | set(extra) <= screened or len(chosen) != len(set(chosen)):
        raise SystemExit(f"{sel_file} names components that were not screened, or one twice")
    return selection


def check_resume(out, fields):
    meta = out / "meta.json"
    if not meta.exists():
        return
    old = json.loads(meta.read_text(encoding="utf-8"))
    # batch_size does not change the scores; moe_forward is a record (moe_mismatch compares the forwards)
    diff = [k for k, v in fields.items() if k not in ("batch_size", "moe_forward", "moe")
            and json.dumps(old.get(k), default=str) != json.dumps(v, default=str)]
    if moe_mismatch(old, fields):
        diff.append("moe")
    if diff:
        raise SystemExit(f"{out} holds a run with different {diff}; move it away or use another --model/--means")

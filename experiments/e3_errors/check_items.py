"""Intact-model validity check of the M3 and M3-cal items (materials-m2-m3.md §5.8), run once before any lesion.

    python experiments/e3_errors/check_items.py --model qwen3-30b-a3b-instruct --dtype float16
    python experiments/e3_errors/check_items.py --model qwen3-30b-a3b-instruct --tiny --device cpu   # smoke test

All items in data/materials/m3/items.csv, reserves included, held-out included (no lesion is applied, as for the
G1 check on all of M1):
- six-choice definition naming, 6 rotations each (rule 1);
- free naming without a hint for the items usable for free naming (rules 2 and 4);
- free naming with the cue and the miscue for calibration_b only (dev set): whether the hints move the intact
  model at all, report only.

Writes outputs/e3_errors/<model>/check/{choice_scores.csv, naming_scores.csv, validity.csv, hints.csv,
summary.json, meta.json} (check_first<n>/ with --limit). A real run never overwrites an earlier one: it stops
before loading the model if the folder exists, writes into <folder>.partial and renames it only after meta.json
(complete: true) is written (agent2 2026-10-07). Then, locally: python -m lesion.m3_items final <that folder> (run
from src/), which accepts only a complete run of the main model on all items.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from lesion import m3_items, m3_scoring as ms, models, runinfo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float16", choices=models.DTYPES)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="first n items only (smoke tests)")
    ap.add_argument("--naming-errors", action="store_true",
                    help="only the intact free-naming scores of the five candidate answers without a hint (H3a in "
                         "free naming, user decision Q13 2026-10-07); written to check_errors/, no rule applied")
    args = ap.parse_args()

    name = "check_errors" if args.naming_errors else "check"
    out = (ROOT / ("outputs/e3_errors_tiny" if args.tiny else "outputs/e3_errors") / args.model
           / (name if args.limit is None else f"{name}_first{args.limit}"))
    if out.exists() and not args.tiny:
        raise SystemExit(f"Refusing to run: {out} exists; the check is kept, not overwritten")
    commit = runinfo.commit() if args.tiny else runinfo.require_clean()
    path = m3_items.OUT / "items.csv"
    items = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if args.limit is not None:
        items = items.head(args.limit)
    if args.tiny:
        model, tok = models.load_tiny(args.model)
    else:
        model, tok = models.load_model(args.model, device=args.device, dtype=models.torch_dtype(args.dtype))
    chat = models.uses_chat(args.model)
    t0 = time.time()
    if args.naming_errors:
        scores = ms.averaged_naming(model, tok, ms.build_naming_prompts(items, ("none",), chat, roles=ms.ERROR_ROLES), chat)
        partial = out.with_name(out.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        scores.to_csv(partial / "naming_error_scores.csv", index=False)
        ms.naming_errors(scores).to_csv(partial / "naming_errors.csv", index=False)
        runinfo.write_meta(partial, model=args.model, revision=models.MODELS[args.model][1], dtype=args.dtype,
                           tiny=args.tiny, resume_commit=commit, device=args.device, items_sha256=runinfo.sha256(path),
                           limit=args.limit, n_items=len(items), naming_batches=ms.NAMING_BATCHES, demos=ms.DEMOS,
                           seconds=round(time.time() - t0, 1), complete=True)
        if out.exists():
            shutil.rmtree(out)
        partial.rename(out)
        print(f"written to {out}")
        return
    choice = ms.averaged_choice(model, tok, ms.build_choice_prompts(items, chat), chat)
    fn = items[items.free_naming_ok.astype(bool)]
    prompts = (ms.build_naming_prompts(fn, ("none",), chat)
               + ms.build_naming_prompts(fn[fn.role == "calibration_b"], ("cue", "miscue"), chat))
    naming = ms.averaged_naming(model, tok, prompts, chat)
    check = ms.validity(items, choice, naming)
    hints = ms.hint_effects(naming[naming.item_id.isin(fn.item_id[fn.role == "calibration_b"])])
    summary = ms.validity_summary(check, hints)
    summary["preview"] = m3_items.finalize(items, check)[1] if args.limit is None else None

    partial = out.with_name(out.name + ".partial")
    if partial.exists():                 # leftovers of an interrupted run
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    choice.to_csv(partial / "choice_scores.csv", index=False)
    naming.to_csv(partial / "naming_scores.csv", index=False)
    check.to_csv(partial / "validity.csv", index=False)
    hints.to_csv(partial / "hints.csv", index=False)
    (partial / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=int), encoding="utf-8")
    runinfo.write_meta(partial, model=args.model, revision=models.MODELS[args.model][1], dtype=args.dtype, tiny=args.tiny,
                       resume_commit=commit, device=args.device, items_sha256=runinfo.sha256(path), limit=args.limit,
                       n_items=len(items), choice_batches=ms.CHOICE_BATCHES, naming_batches=ms.NAMING_BATCHES,
                       demos=ms.DEMOS, seconds=round(time.time() - t0, 1), complete=True)
    if out.exists():                     # tiny runs only
        shutil.rmtree(out)
    partial.rename(out)
    print(json.dumps({k: v for k, v in summary.items() if k != "preview"}, indent=2, ensure_ascii=False, default=int))
    if summary["preview"]:
        p = summary["preview"]
        print(f"after replacement: {p['by_role']}; shortfall {p['shortfall'] or 'none'}; "
              f"{len(p['replaced'])} replaced, {len(p['unfilled'])} unfilled; reserves left {p['reserves_left']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()

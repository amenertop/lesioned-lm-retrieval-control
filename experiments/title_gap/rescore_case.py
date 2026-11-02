"""Exploratory correction (2026-11-02): rescore the semantic-family candidates with the first letter capitalized.

The family prompts ask for "one word" but not for lowercase, while the candidates scored were lowercase; the intact model
capitalizes most of its family answers, so a lesion that shifts the case shifts the lowercase score without any change
in naming. This script reuses a finished run's plan (items, prompts, states, batch layouts, settings), applies each state
in the same way, and scores every candidate word again with its first letter capitalized. The lowercase scores of the
states named by --check are scored again too, to confirm that the run reproduces the stored values. A case-insensitive
score is then log(P(word) + P(Word)), combined offline by analyze_case.py.

    python experiments/title_gap/rescore_case.py --model qwen3-30b-a3b-instruct --means <means> --dtype float16 \
        --moe sorted --run outputs/title_gap_execution_20261019/fine_final_core --out outputs/case_rescore/fine_final_core

Resumable: a state with complete.json is skipped. This is not a frozen test and does not pass through the freeze guard.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
import common  # noqa: E402
from lesion import m3_battery, m3_scoring, runinfo, title_gap_states  # noqa: E402


def capitalized(naming):
    out = []
    for p in naming:
        cands = {f"{role}_cap": w[:1].upper() + w[1:] for role, w in p["candidates"].items() if w[:1].islower()}
        if cands:
            out.append(m3_scoring.NamingPrompt(p["item_id"], p["condition"], p["text"], cands))
    return out


def score(model, tok, prompts, chat, batches, region):
    if region is None:
        return m3_scoring.averaged_naming(model, tok, prompts, chat, batches)
    tables = [m3_battery.candidate_scores(model, tok, prompts, chat, b, region) for b in batches]
    out = tables[0].copy()
    for col in ("logprob", "first_token_rank"):
        out[col] = np.mean([t[col].to_numpy() for t in tables], axis=0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(ap)
    ap.add_argument("--run", type=Path, required=True, help="finished run directory holding plan.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated state names (default: all states of the run)")
    ap.add_argument("--check", default="intact,WR8_a0,SC4_a0", help="states whose lowercase scores are recomputed")
    args = ap.parse_args()
    plan = json.loads((args.run / "plan.json").read_text(encoding="utf-8"))
    prov = plan["provenance"]
    if (args.model != prov["model"] or args.dtype != prov["dtype"] or args.moe != prov["moe"]
            or runinfo.sha256(args.means) != prov["means_sha256"]):
        ap.error("model, dtype, moe and means must match the run being rescored")
    model, tok, chat, means, _ = common.load(args)
    if chat != plan["chat"]:
        ap.error("chat template setting differs from the run")
    naming = plan["naming"]
    cap_prompts = capitalized(naming)
    low_prompts = [m3_scoring.NamingPrompt(p["item_id"], p["condition"], p["text"], p["candidates"]) for p in naming]
    batches = plan.get("naming_batches") or plan["batches"]
    states = plan["states"]
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        states = [s for s in states if s["name"] in wanted]
    check = {s.strip() for s in args.check.split(",") if s.strip()}
    title_gap_states.validate_states(states, model.config)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plan.json").write_text(json.dumps(dict(source_run=str(args.run), source_plan_sha256=runinfo.sha256(args.run / "plan.json"),
                                                        batches=batches, n_cap_prompts=len(cap_prompts), states=[s["name"] for s in states],
                                                        check=sorted(check), commit=runinfo.git_commit() if hasattr(runinfo, "git_commit") else None),
                                                   indent=1), encoding="utf-8")
    for state in states:
        d = args.out / state["name"]
        if (d / "complete.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with title_gap_states.apply_state(model, means, state):
            cap = score(model, tok, cap_prompts, chat, batches, state.get("region"))
            low = score(model, tok, low_prompts, chat, batches, state.get("region")) if state["name"] in check else None
        cap.to_parquet(d / "candidates_cap.parquet", index=False)
        files = {"candidates_cap.parquet": runinfo.sha256(d / "candidates_cap.parquet")}
        if low is not None:
            low.to_parquet(d / "candidates_lower_check.parquet", index=False)
            files["candidates_lower_check.parquet"] = runinfo.sha256(d / "candidates_lower_check.parquet")
        (d / "complete.json").write_text(json.dumps(dict(files=files, seconds=round(time.monotonic() - start, 1))), encoding="utf-8")
        print(f"{state['name']}: {time.monotonic() - start:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

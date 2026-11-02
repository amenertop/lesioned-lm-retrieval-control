"""Does the definition-naming test show a frequency effect in an intact model? (knowledge test, 2026-09-28)

    python experiments/e3_knowledge/check_items.py --model qwen3-30b-a3b-instruct --dtype float16
    python experiments/e3_knowledge/check_items.py --model olmo2-1b              # local proxy (LESION_MODEL_DIR)

Writes outputs/e3_knowledge/<model>/check/{scores.csv, summary.json, meta.json}. The items go to
human review only after the check passes on the main model.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from lesion import fast_moe, knowledge_items as ki, models, naming, runinfo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=models.DTYPES)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--average-batches", action="store_true", help="use the naming experiment's fixed 16,11,7 compositions")
    ap.add_argument("--moe", choices=("hf", "sorted"), default="hf")
    ap.add_argument("--device", default="cuda", help="cpu works for small proxy models (e.g. olmo2-1b with --dtype float32)")
    args = ap.parse_args()

    commit = runinfo.commit() if args.tiny else runinfo.require_clean()
    # Intact-model item check over all items, held-out included: no lesion is applied, as for G1 on all of M1.
    items, demos = pd.read_csv(ki.OUT / "items.csv"), pd.read_csv(ki.OUT / "demos.csv")
    if args.tiny:
        model, tok = models.load_tiny(args.model)
    else:
        model, tok = models.load_model(args.model, device=args.device, dtype=models.torch_dtype(args.dtype))
    if args.moe == "sorted":
        fast_moe.install(model, args.moe)
    chat = models.uses_chat(args.model)
    prompts = naming.build_prompts(items, demos, chat=chat)
    t0 = time.time()
    scores = (naming.averaged_logprobs(model, tok, prompts, chat, (16, 11, 7)) if args.average_batches
              else naming.answer_logprobs(model, tok, prompts, chat, args.batch_size))
    check = naming.frequency_check(scores, items, ki.ZIPF_BINS)
    out = ROOT / ("outputs/e3_knowledge_tiny" if args.tiny else "outputs/e3_knowledge") / args.model / "check"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    (out / "summary.json").write_text(json.dumps(check, indent=2, ensure_ascii=False), encoding="utf-8")
    runinfo.write_meta(out, model=args.model, revision=models.MODELS[args.model][1], dtype=args.dtype, tiny=args.tiny,
                       resume_commit=commit, device=args.device, items_sha256=runinfo.sha256(ki.OUT / "items.csv"),
                       batches=[16, 11, 7] if args.average_batches else [args.batch_size],
                       moe_forward=args.moe,
                       seconds=round(time.time() - t0, 1), passed=check["passed"])
    s = check["slope"]
    print(f"slope per Zipf unit {s['estimate']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]; "
          f"rarest-bin top-1 {check['rare_top1']:.2f}; passed={check['passed']}")
    for b in check["by_bin"]:
        print(f"  {b['bin']}: n={b['n']}, mean logprob {b['mean_logprob']:.2f}, top-1 {b['top1']:.2f}")


if __name__ == "__main__":
    main()

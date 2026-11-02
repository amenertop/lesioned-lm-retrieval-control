"""Collect neutral-text component means for a registered model.

Examples from the repository root::

    python experiments/prep_means/run.py --model olmoe-0924-base
    python experiments/prep_means/run.py --model olmoe-0924-base --tiny --output outputs/means/tiny.pt
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import fast_moe, means, models, neutral_text, runinfo  # noqa: E402


def default_output(model_name, revision, commit, tiny=False):
    suffix = "tiny" if tiny else revision[:7]
    return ROOT / "outputs" / "means" / f"{model_name}@{suffix}__{commit[:7]}.pt"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--tiny", action="store_true", help="use the random two-layer architecture for local checks")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--moe", choices=("hf", "sorted"), default="hf")
    ap.add_argument("--dtype", default="bfloat16", choices=models.DTYPES,
                    help="precision of the model's weights and activations (ADR 0004: float16 for Qwen)")
    args = ap.parse_args(argv)

    revision = models.MODELS[args.model][1]
    if args.tiny:
        model, tokenizer = models.load_tiny(args.model)
        commit = runinfo.commit()
    else:
        commit = runinfo.require_clean()
        model, tokenizer = models.load_model(args.model, dtype=models.torch_dtype(args.dtype))
    if args.moe == "sorted":
        print(f"MoE forward: sorted, {fast_moe.install(model, args.moe)} blocks", flush=True)
    texts = neutral_text.load()
    result = means.collect_means(model, tokenizer, texts, max_tokens=args.max_tokens,
                                 batch_size=args.batch_size, model_name=args.model, revision=revision)
    result.meta["git_commit"] = commit
    result.meta["dtype"] = args.dtype
    result.meta["moe_forward"] = args.moe
    output = args.output or default_output(args.model, revision, commit, args.tiny)
    result.save(output)
    print(f"saved {output} ({result.n_tokens} valid tokens)")
    if result.expert_count is not None:
        low = int((result.expert_count < 500).sum().item())
        print(f"experts with fewer than 500 routed tokens: {low}")


if __name__ == "__main__":
    main()

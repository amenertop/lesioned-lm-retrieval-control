"""Run one pinned census candidate after the user schedules a GPU session. Never downloads weights in preparation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import fast_moe, model_census as mc, models, runinfo, scoring


def evaluate_and_save(model, tok, args, commit, verification_sha256, verification_kind="local_files"):
    spec = mc.CANDIDATES[args.model]
    moe = getattr(args, "moe", "hf")
    if moe == "sorted":
        print(f"MoE forward: sorted, {fast_moe.install(model, moe)} blocks", flush=True)
    if spec["mode"] == "nonthinking":
        tok = mc.NonThinkingTokenizer(tok)
    items, demos, templates = scoring.load_materials()
    prompts = scoring.build_label_prompts(items, demos, templates, chat=True)
    sample = scoring.build_label_prompts(mc.demo_items(demos), demos, templates, chat=True)
    thinking = spec["mode"] == "native_thinking"
    check, result = mc.format_check(model, tok, sample, thinking, 2048 if thinking else 4)
    args.out.mkdir(parents=True, exist_ok=True)
    check.to_csv(args.out / "format.csv", index=False)
    result.update(model=args.model, specification=spec, commit=commit, dtype=args.dtype, moe_forward=moe,
                  verification_sha256=verification_sha256, verification_kind=verification_kind,
                  material_hashes={f: runinfo.sha256(scoring.M1 / f) for f in ("items.csv", "demos.csv", "templates.yaml")})
    if result["pass_format"] and not thinking:
        scores, gate = mc.measurement(model, tok, prompts, items, args.batch_size)
        scores.to_csv(args.out / "label_main.csv", index=False)
        result["g1"] = gate
        result["replication_eligible"] = bool(spec["replication"] and gate["g1_decision"] == "pass")
    else:
        result["g1"] = None
        result["replication_eligible"] = False
        result["reason"] = "native thinking is a format-only check in this plan" if thinking else "format gate failed"
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=mc.CANDIDATES)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", choices=models.DTYPES, default="float16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--moe", choices=("hf", "sorted"), default="hf")
    ap.add_argument("--weights", type=Path, help="verified local checkpoint, no download during measurement")
    args = ap.parse_args()
    commit = runinfo.require_clean()
    models.check_environment()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("use a fresh census output directory")
    spec = mc.CANDIDATES[args.model]
    weight_path = mc.verified_candidate_path(args.model, args.weights)
    if args.model in models.MODELS and args.weights is None:
        model, tok = models.load_model(args.model, dtype=models.torch_dtype(args.dtype))
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(weight_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(weight_path, local_files_only=True,
                                                    dtype=torch.bfloat16, device_map="cuda").to(models.torch_dtype(args.dtype)).eval()
    evaluate_and_save(model, tok, args, commit, runinfo.sha256(weight_path / "VERIFIED.json"))


if __name__ == "__main__":
    main()

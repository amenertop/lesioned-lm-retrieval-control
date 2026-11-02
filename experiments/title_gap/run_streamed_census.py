"""Prepare a verified census model in CPU memory and wait for an explicit scheduler turn on the GPU."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import model_census, models, runinfo, streaming_weights
from run_census import evaluate_and_save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=model_census.CANDIDATES)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--start-file", type=Path, required=True, help="scheduler creates this only when the GPU is free")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", choices=models.DTYPES, default="float16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--moe", choices=("hf", "sorted"), default="hf")
    args = ap.parse_args()
    commit = runinfo.require_clean()
    models.check_environment()
    if args.start_file.exists() or args.out.exists():
        ap.error("use fresh start-file and output paths")
    source = next(s for s in json.loads(args.manifest.read_text(encoding="utf-8")) if s["name"] == args.model)
    if any(source[k] != model_census.CANDIDATES[args.model][k] for k in ("repo", "revision")):
        ap.error("manifest differs from the fixed candidate")
    streaming_weights.metadata(source, args.metadata)
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    model = streaming_weights.load_cpu(AutoConfig.from_pretrained(args.metadata, local_files_only=True), source)
    tok = AutoTokenizer.from_pretrained(args.metadata, local_files_only=True)
    model.generation_config = GenerationConfig.from_pretrained(args.metadata, local_files_only=True)
    args.out.mkdir(parents=True)
    ready = args.out / "cpu-ready.json"
    ready.write_text(json.dumps(dict(complete=True, revision=source["revision"], commit=commit,
        storage="CPU memory only, no persisted weight checkpoint", verified_weights=source["weights"],
        manifest_sha256=runinfo.sha256(args.manifest)), indent=2)+"\n", encoding="utf-8", newline="\n")
    print("CPU model verified and ready, waiting for the scheduler", flush=True)
    while not args.start_file.exists():
        time.sleep(10)
    model = model.to(device="cuda", dtype=models.torch_dtype(args.dtype))
    evaluate_and_save(model, tok, args, commit, runinfo.sha256(ready), "verified_in_memory_shards")


if __name__ == "__main__":
    main()

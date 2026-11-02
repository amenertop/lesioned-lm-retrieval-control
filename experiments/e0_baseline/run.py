"""Experiment 0: score OLMoE-0924 base and Instruct on all M1 items (docs/plans/pilot-protocol.md §4; ticket pilot 07).

Run on the GPU server from the repo root:
    python experiments/e0_baseline/run.py            # real run; refuses unless HEAD is the `g1-frozen` tag
    python experiments/e0_baseline/run.py --smoke    # demo items only; allowed before the freeze

Writes outputs/e0/<version>/{label_<variant>.csv, cloze.csv, meta.json}.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import models, scoring  # noqa: E402

VERSIONS = ["olmoe-0924-base", "olmoe-0924-instruct"]
MATERIAL_FILES = ["items.csv", "split.csv", "demos.csv", "templates.yaml"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def check_frozen(tag_name="g1-frozen", require_tag=True):
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", "data/materials", "src", "experiments")
    if dirty:
        sys.exit("Refusing to run: uncommitted changes to data/materials, src or experiments (pilot protocol §8).")
    if require_tag and head != git("rev-parse", f"{tag_name}^{{commit}}"):
        sys.exit(f"Refusing to run real models on M1: HEAD must be the `{tag_name}` tag (pilot protocol §8). "
                 "Use --smoke before the freeze, or --exploratory for models outside the frozen plan.")
    return head


def smoke_items(demos):
    """Demo items reshaped as test items, for checking the pipeline and throughput before the freeze."""
    import pandas as pd

    rows = []
    for d in demos.to_dict("records"):
        is_syn = d["task"] == "synonym"
        rows.append(dict(item_id=f"demo_{d['task']}_{d['position']}", task=d["task"], probe=d["probe"],
                         option1="similar" if is_syn else d["option_a"], option2="different" if is_syn else d["option_b"],
                         correct_option="option1" if d["correct_label"] == "A" else "option2",
                         word1=d["word1"], word2=d["word2"]))
    return pd.DataFrame(rows)


FORMAT_SHARE = 0.9          # P12: share of demo prompts whose first greedy token must be a label (format gate)


def format_check(model, tok, prompts, chat, fmt, prefix, max_new_tokens=4):
    """P12 format gate (docs/plans/second-model-g1-p12.md): greedy continuation of each scoring prompt (the same
    prompt the labels are read from); passes when the first generated token is one of the two labels for at least
    FORMAT_SHARE of the prompts. Returns the record written to format_check.json."""
    import torch

    labels = scoring.label_token_ids(tok, chat, prefix=prefix)
    device = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for p in prompts:
            ids = torch.tensor([scoring.encode(tok, p.text, chat, fmt)], device=device)
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
            rows.append(dict(item_id=p.item_id, task=p.task, order=p.order, generated_ids=gen,
                             generated_text=tok.decode(gen), first_token=tok.convert_ids_to_tokens(gen[:1]),
                             first_is_label=bool(gen) and gen[0] in labels.values()))
    share = sum(r["first_is_label"] for r in rows) / len(rows)
    return dict(status="ok" if share >= FORMAT_SHARE else "format_failure", share_first_token_is_label=share,
                threshold=FORMAT_SHARE, prompt_format=fmt, label_prefix=prefix, label_ids=labels,
                max_new_tokens=max_new_tokens, n_prompts=len(rows), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=VERSIONS)
    ap.add_argument("--variant", default="main", choices=["main", "alt1", "alt2"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="score demo items only (allowed before the freeze)")
    ap.add_argument("--freeze-tag", default="g1-frozen", help="tag HEAD must match for a confirmatory run")
    ap.add_argument("--exploratory", action="store_true",
                    help="models outside the frozen plan: needs a clean committed tree, no tag; results go to outputs/e0_explore")
    ap.add_argument("--format-check", action="store_true",
                    help="P12: greedy check on the demo items that answers start with a label; writes format_check.json")
    ap.add_argument("--tiny", action="store_true", help="tiny random models (rehearsal on a machine without the weights)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.tiny:
        models.check_environment()
    if args.smoke:
        commit = git("rev-parse", "HEAD")
    else:
        commit = check_frozen(args.freeze_tag, require_tag=not args.exploratory)
    args.out = args.out or str(ROOT / ("outputs/e0_explore" if args.exploratory else "outputs/e0"))
    items, demos, templates = scoring.load_materials()
    if args.smoke or args.format_check:
        items = smoke_items(demos)
    out_root = Path(args.out) / ("smoke" if args.smoke else "")

    import torch

    materials = {f: sha256(scoring.M1 / f) for f in MATERIAL_FILES}
    for version in args.versions:
        fmt, prefix = models.prompt_format(version)
        chat = fmt != "plain"
        check_file = out_root / version / "format_check.json"
        if version in models.PROMPT_FORMATS and not (args.smoke or args.format_check):
            # P12: the G1 scoring of a model with its own prompt format runs only after its format gate passed
            record = json.loads(check_file.read_text(encoding="utf-8")) if check_file.exists() else {}
            # tiny random models fail the gate by nature; a rehearsal only needs the check to have run
            passed = record.get("status") == "ok" or (args.tiny and bool(record))
            # a record from a tiny rehearsal is never accepted by a real run, nor one from another commit or format
            same = (record.get("git_commit") == commit and record.get("tiny", False) == args.tiny
                    and (record.get("prompt_format"), record.get("label_prefix")) == (fmt, prefix)
                    and record.get("variant") == args.variant and record.get("materials_sha256") == materials)
            if not (passed and same):
                sys.exit(f"{version}: no passed format check from this commit at {check_file}; run --format-check first")
        model, tok = models.load_tiny(version) if args.tiny else models.load_model(version)
        prompts = scoring.build_label_prompts(items, demos, templates, variant=args.variant, chat=chat)
        if args.format_check:
            record = format_check(model, tok, prompts, chat, fmt, prefix)
            record.update(version=version, git_commit=commit, tiny=args.tiny, variant=args.variant,
                          materials_sha256=materials, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
            check_file.parent.mkdir(parents=True, exist_ok=True)
            check_file.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{version}: format check {record['status']} ({record['share_first_token_is_label']:.0%} start with a label)")
            del model
            continue
        t0 = time.time()
        labels = scoring.score_label_prompts(model, tok, prompts, chat=chat, batch_size=args.batch_size,
                                             fmt=fmt, label_prefix=prefix)
        label_secs = time.time() - t0
        cloze = None
        if args.variant == "main" and not args.smoke:
            cloze = scoring.score_cloze(model, tok, scoring.build_cloze_prompts(items, demos, templates), args.batch_size)

        out = out_root / version
        out.mkdir(parents=True, exist_ok=True)
        labels.to_csv(out / f"label_{args.variant}.csv", index=False)
        if cloze is not None:
            cloze.to_csv(out / "cloze.csv", index=False)
        repo, rev, _ = models.MODELS[version]
        meta = dict(version=version, repo=repo, revision=rev, chat=chat, prompt_format=fmt, label_prefix=prefix,
                    tiny=args.tiny, variant=args.variant, smoke=args.smoke,
                    exploratory=args.exploratory, freeze_tag=None if args.exploratory or args.smoke else args.freeze_tag,
                    weights_dir=str(models.verified_dir(version) or "hf-cache"),
                    git_commit=commit, n_prompts=len(prompts), label_seconds=round(label_secs, 2),
                    prompts_per_second=round(len(prompts) / label_secs, 1), batch_size=args.batch_size,
                    materials_sha256=materials,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    python=platform.python_version(), timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{version}: {len(prompts)} prompts in {label_secs:.1f}s -> {out}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Where does a forward pass spend its time? (GPU session 14; user's question 2026-10-15: the GPU ran at about 40% in
session 13 while the run's Python process held one CPU core at 100%.) Exploratory; it changes nothing in any run.

    python experiments/profile_forward.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16 \
        --scorer prefix_avg3 [--out outputs/profile/<model>.json]

On 64 M1 development prompts (no held-out item), batch sizes 16, 32 and 64, intact and with K8 mean-ablated:
wall time per batch (one warm-up, then the median of three repeats), CUDA time and CPU time from torch.profiler over
one repeat, the share of wall time the GPU was busy (CUDA time over wall time), peak GPU memory, and the ten most
expensive operators by CPU time. Also the container's peak resident memory (ru_maxrss) after loading.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))

import common  # noqa: E402
from lesion import materials, scoring  # noqa: E402
from lesion.components import Component  # noqa: E402

K8 = ("expert:46:61", "expert:46:101", "head:37:24", "expert:45:32", "expert:45:48", "expert:43:70", "expert:42:115",
      "expert:45:93")
BATCHES = (16, 32, 64)


CUDA = torch.cuda.is_available()
sync = torch.cuda.synchronize if CUDA else (lambda: None)       # a tiny rehearsal runs on the CPU


def timed(fn, repeats=3):
    fn()                                                  # warm-up
    sync()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "outputs/profile" / f"{args.model}.json"
    model, tok, chat, means, ablate = common.load(args)
    try:                                                  # Linux (the server); not available on Windows
        import resource
        load_rss_gb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    except ImportError:
        load_rss_gb = None
    dev = materials.load_split("dev")
    _, demos, templates = scoring.load_materials()
    prompts = scoring.build_label_prompts(dev, demos, templates, variant="main", chat=chat)[:64]
    seqs = [scoring.encode(tok, p.text, chat) for p in prompts]
    device = next(model.parameters()).device
    k8 = [Component.parse(c) for c in K8] if not args.tiny else []
    result = dict(model=args.model, dtype=args.dtype, tiny=args.tiny, load_peak_rss_gb=load_rss_gb,
                  n_prompts=len(seqs), runs=[])
    for bs in BATCHES:
        ids, mask, pos = scoring._pad_left(seqs[:bs], tok.eos_token_id or 0, device)

        def forward():
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1)

        for state, comps in (("intact", []), ("K8", k8)):
            if state == "K8" and not comps:
                continue
            with ablate(comps) if comps else torch.no_grad():
                if CUDA:
                    torch.cuda.reset_peak_memory_stats()
                wall = timed(forward)
                activities = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if CUDA else [])
                with torch.profiler.profile(activities=activities) as prof:
                    forward()
                    sync()
            events = prof.key_averages()
            cuda_us = sum(e.self_device_time_total for e in events)
            cpu_us = sum(e.self_cpu_time_total for e in events)
            top = sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:10]
            result["runs"].append(dict(
                batch_size=bs, state=state, wall_seconds=round(wall, 4), prompts_per_second=round(bs / wall, 2),
                cuda_seconds=round(cuda_us / 1e6, 4), cpu_seconds=round(cpu_us / 1e6, 4),
                gpu_busy_share=round(cuda_us / 1e6 / wall, 3),
                peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if CUDA else None,
                kernel_launches=int(sum(e.count for e in events if e.key.startswith("cuda") and "Launch" in e.key)),
                top_cpu_ops=[dict(op=e.key, cpu_ms=round(e.self_cpu_time_total / 1e3, 2), calls=e.count) for e in top]))
            print(json.dumps({k: v for k, v in result["runs"][-1].items() if k != "top_cpu_ops"}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    result["status"] = "done"                             # written last and atomically, so a cut-off run is not "done"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

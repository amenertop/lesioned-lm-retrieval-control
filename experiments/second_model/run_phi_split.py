"""GPU session 12, item 1 (docs/plans/gpu-session12.md), exploratory: Phi-3.5-MoE's knowledge set K8 taken apart
around its two candidate super experts (expert:3:7 and expert:5:10, mean output 74 and 303 times their layer's median;
reports/explore/2026-10-14-session11-joint-reading.md 3.1).

    python experiments/second_model/run_phi_split.py --model phi35-moe-instruct --means <fp16 means> --dtype float16

States (lesion.hooks.ablate, mean ablation unless named): intact; K8; each super expert alone; both; K8 without them;
both with zero ablation; both in a two-by-two of mean or zero ablation at each passage's first real token only or at
the other tokens only (lesion.probes.lm_loss_region; a lesion at the first token reaches every later prediction
through attention); each other K8 component alone; rerouting instead of ablation (lesion.hooks.mask_router: the router
skips the experts and picks its next choices, nothing is put in their place) for the two experts and for K8's experts,
next to mean ablation of K8's experts (a reviewer's question: does mean ablation itself do the damage?).
Measures: language-model loss on 40 neutral passages of 256 tokens (every state); naming on the 120 development
concepts, scores averaged over three batch compositions, giving naming damage and the frequency effect with
concept-resampled intervals (states without a position restriction, other than the single other components);
greedy answers to the first 40 concepts (a few states). Intact only: every output of the two experts on the 40
passages, its length and position (lesion.probes.expert_output_norms).
Writes outputs/e3_knowledge/<model>/split/{results.json, naming_scores.parquet, norms.parquet, meta.json}.
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))

import common  # noqa: E402
from lesion import damage, hooks, knowledge_items as ki, naming, neutral_text, probes, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402

SUPER = ["expert:3:7", "expert:5:10"]
BATCHES = (16, 7, 3)
N_TEXTS, N_TOKENS, N_GREEDY = 40, 256, 40
GREEDY_STATES = ("intact", "K8", "super_both", "K8_without_super", "super_both_zero", "super_both_reroute")


def states(k8, sup):
    """{name: (components, mode, region, naming)}."""
    rest = [c for c in k8 if c not in sup]
    out = {"intact": ([], "mean", None, True), "K8": (k8, "mean", None, True)}
    out.update({f"super_{c.split(':', 1)[1].replace(':', '_')}": ([c], "mean", None, True) for c in sup})
    out.update({"super_both": (sup, "mean", None, True), "K8_without_super": (rest, "mean", None, True),
                "super_both_zero": (sup, "zero", None, True)})
    # two by two: what is put in (the neutral-text mean, or nothing) x where (the first real token, or the others)
    out.update({f"super_both_{mode}_{region}": (sup, mode, region, False) for mode in ("mean", "zero")
                for region in ("first", "rest")})
    out.update({f"single_{c.replace(':', '_')}": ([c], "mean", None, False) for c in rest})
    # rerouting (agent4's review, user 2026-10-14): the router skips the experts and picks its next choices, so nothing
    # is put in their place; compared with mean ablation of the same experts (heads cannot be rerouted, so K8's
    # experts only)
    experts = [c for c in k8 if c.startswith("expert:")]
    out.update({"super_both_reroute": (sup, "reroute", None, True), "K8_experts_mean": (experts, "mean", None, True),
                "K8_experts_reroute": (experts, "reroute", None, True)})
    return out


def lesion_context(model, parsed, means, mode):
    if not parsed:
        return nullcontext()
    if mode == "reroute":
        return hooks.mask_router(model, parsed)
    return hooks.ablate(model, parsed, means, mode=mode)


def greedy(model, tok, prompts, chat, ends_max=8):
    out = []
    for p in prompts:
        ids, _ = naming.encode_pair(tok, p, chat)
        x = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=ends_max, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        out.append(dict(item_id=str(p.item_id), target=p.target, greedy=tok.decode(g[0, len(ids):], skip_special_tokens=False)))
    return out


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--k8", default=None, help="comma-separated K8 (default: the Phi dissociation test's main set)")
    ap.add_argument("--super", default=",".join(SUPER), help="comma-separated candidate super experts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    commit = common.guard(args)
    sets_file = ROOT / "outputs/e2_confirm" / args.model / "dissociation/sets.json"
    k8 = args.k8.split(",") if args.k8 else json.loads(sets_file.read_text(encoding="utf-8"))["main"]
    sup = args.super.split(",")
    if not set(sup) <= set(k8):
        raise SystemExit(f"{sup} are not all in K8 {k8}")
    out = Path(args.out) if args.out else ROOT / ("outputs/e3_knowledge_tiny" if args.tiny else "outputs/e3_knowledge") / args.model / "split"
    model, tok, chat, means, _ = common.load(args)
    plan = states(k8, sup)
    texts = neutral_text.load()[:N_TEXTS]
    seqs = [tok(t, add_special_tokens=False).input_ids[:N_TOKENS] for t in texts]
    dev = ki.load_items("dev")
    median = naming.zipf_median(pd.read_csv(ki.OUT / "items.csv")[["item_id", "zipf"]])
    prompts = naming.build_prompts(dev, pd.read_csv(ki.OUT / "demos.csv"), chat)
    fields = dict(stage="phi_split", model=args.model, tiny=args.tiny, means_path=str(args.means),
                  means_sha256=runinfo.sha256(args.means), k8=k8, super=sup, n_texts=len(seqs), neutral_tokens=N_TOKENS,
                  naming_batches=list(BATCHES), n_boot=args.n_boot, run_commit=commit)
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()
    result = dict(states={k: dict(components=v[0], mode=v[1], region=v[2]) for k, v in plan.items()}, lm_loss={},
                  naming={}, greedy={}, zipf_median=median)
    scores = []
    for name, (comps, mode, region, do_naming) in plan.items():
        parsed = [Component.parse(c) for c in comps]
        if mode == "reroute":
            with lesion_context(model, parsed, means, mode):
                result["lm_loss"][name] = probes.lm_loss_region(model, seqs, [], means, pad_id=tok.eos_token_id or 0)
        else:
            result["lm_loss"][name] = probes.lm_loss_region(model, seqs, parsed, means, region=region, mode=mode,
                                                            pad_id=tok.eos_token_id or 0)
        if do_naming:
            with lesion_context(model, parsed, means, mode):
                s = naming.averaged_logprobs(model, tok, prompts, chat, BATCHES).assign(state=name)
                if name in GREEDY_STATES:
                    result["greedy"][name] = greedy(model, tok, prompts[:N_GREEDY], chat)
            scores.append(s)
        print(f"{name}: lm_loss {result['lm_loss'][name]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    scores = pd.concat(scores, ignore_index=True)
    intact = scores[scores.state == "intact"]
    for name in scores.state.unique():
        if name == "intact":
            continue
        frame = naming.item_frame(naming.item_damage(intact, scores[scores.state == name]), dev, median)
        result["naming"][name] = damage.bootstrap(frame, naming.effects, args.n_boot)
    norms, firsts = probes.expert_output_norms(model, seqs, [Component.parse(c) for c in sup])
    result["norms"] = probes.norm_summary(norms, len(seqs))
    for key, first in firsts.items():
        mean = means.expert_mean[Component.parse(key).layer, Component.parse(key).index].float()
        result["norms"][key]["mean_output_norm"] = float(mean.norm())
        result["norms"][key]["cosine_first_to_mean"] = (
            float(torch.nn.functional.cosine_similarity(first, mean[None], dim=-1).mean()) if len(first) else None)
    scores.to_parquet(out / "naming_scores.parquet", index=False)
    norms.to_parquet(out / "norms.parquet", index=False)
    (out / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), **fields)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

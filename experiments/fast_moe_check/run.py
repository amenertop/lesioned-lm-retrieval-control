"""Check lesion.fast_moe on the real model before any run uses it (engineering check; user decision 2026-10-16;
agent2's review, reports/explore/2026-10-16-fast-moe/1-agent2-code-review.md).

    python experiments/fast_moe_check/run.py [--tiny] [--means <float16 means file>]

Qwen3-30B-A3B in float16 (P14's precision), one model load; hf, then sorted, then grouped are installed in turn.
Measured under each forward:
1. Neutral passages 0 to 15 (128 tokens, one batch): each next token's log-probability, the last position's logits,
   every layer's output, the raw router logits and the routed experts in rank order.
2. A P14-like gate step with gradient checkpointing: gates (lesion.gates, g = 0.95 as P14 starts) on the knowledge_p512
   pool; the loss is the mean naming log-probability of 16 screening words plus the mean M1 margin of 16 screening
   prompts (lesion.gate_objective, left-padded batches). Gradients of the gate logits, of the router weights and of the
   embedding output at the real tokens; under hf and sorted also of three experts' weights.
3. The production paths, on left-padded batches: M1 scores (prefix scorer with its key-value cache) intact, under mean
   ablation of LESION, and with the lesion at the last three positions only (position masks); naming scores intact,
   ablated and router-masked; attribution of 8 M1 prompts; expert output norms of 10 experts on 2 passages.
4. Seconds per forward and per gate step, peak GPU memory, and GPU memory around grouped's install.
The scale of rounding noise: hf with two other batch compositions (batch sizes 5 and 3 for passages, gate steps,
and production paths). This is the noise the project already lives with: the prefix_avg3 scorer
averages three batch compositions (ADR 0004).
Passes: sorted is bit-identical to hf in everything. grouped's difference from hf is, for every measure, at most
TOL_FACTOR times hf's own batch-composition difference plus a floor (mean and 99th percentile of absolute differences;
share of tokens routed differently; gradients' relative L2 difference), and its gradient cosines are at least the
batch-composition ones minus 1e-4. hf in bfloat16 is reported for scale only.
Writes outputs/fast_moe_check/<model>/result.json; exits 1 if sorted is not bit-identical (grouped is opt-in: reported).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import attribution, damage, fast_moe, gate_objective as go, gates, hooks, knowledge_items as ki, \
    materials, means as means_mod, models, naming, neutral_text, prefix_scoring, probes, runinfo, scoring  # noqa: E402
from lesion.components import Component, moe_block  # noqa: E402

NAME, N_TEXTS, N_TOKENS, REPS, POOL = "qwen3-30b-a3b-instruct", 16, 128, 3, "knowledge_p512"
MEANS = ROOT / "outputs/means/qwen3-30b-a3b-instruct@0d7cf23__42ae809__fp16.pt"
BATCH, NOISE_BATCHES = 8, (5, 3)
TOL_FACTOR = 1.5
EXPERT_GRADS = ("model.layers.0.mlp.experts.3.", "model.layers.20.mlp.experts.64.", "model.layers.47.mlp.experts.127.")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gb(x):
    return round(x / 2 ** 30, 3)


def memory():
    if not torch.cuda.is_available():
        return None
    return dict(allocated_gb=gb(torch.cuda.memory_allocated()), reserved_gb=gb(torch.cuda.memory_reserved()),
                peak_allocated_gb=gb(torch.cuda.max_memory_allocated()), peak_reserved_gb=gb(torch.cuda.max_memory_reserved()))


class Inputs:
    """Everything measured, encoded once."""

    def __init__(self, model, tok, chat, members):
        self.pad = tok.eos_token_id or 0
        self.device = next(model.parameters()).device
        texts = neutral_text.load()
        seqs = [s for s in (tok(t, add_special_tokens=False).input_ids[:N_TOKENS] for t in texts[:N_TEXTS])
                if len(s) == N_TOKENS]
        if len(seqs) != N_TEXTS:
            raise SystemExit(f"{len(seqs)} neutral passages of {N_TOKENS} tokens, expected {N_TEXTS}")
        self.passages = torch.tensor(seqs, device=self.device)
        self.norm_seqs = [s[:64] for s in seqs[:2]]
        demos = pd.read_csv(ki.OUT / "demos.csv")
        n_items = ki.load_items("screening").head(40)
        self.n_prompts = naming.build_prompts(n_items, demos, chat)
        self.n_pairs = [naming.encode_pair(tok, p, chat) for p in self.n_prompts[:16]]
        all_items, m1_demos, templates = scoring.load_materials()
        items = materials.load_split("screening").groupby(["task", "high_demand"]).head(5)
        self.m_prompts = scoring.build_label_prompts(items, m1_demos, templates, variant="main", chat=chat)
        lab = scoring.label_token_ids(tok, chat)
        first = self.m_prompts[:16]
        self.m_seqs = [scoring.encode(tok, p.text, chat) for p in first]
        self.m_correct = [lab[p.correct_label] for p in first]
        self.m_wrong = [lab["B" if p.correct_label == "A" else "A"] for p in first]
        pick = [p.item_id for p in self.m_prompts[:8]]
        self.a_prompts = self.m_prompts[:8]
        self.a_weights = attribution.statistic_weights(self.a_prompts, items[items.item_id.isin(pick)],
                                                       damage.frequency_median(all_items))
        self.members = members
        comps = [Component.parse(m) for m in members]
        self.lesion = comps[:8]
        self.experts = [c for c in comps if c.kind == "expert"][:10]
        self.weight = model.get_output_embeddings().weight.float()


def forward_pass(model, inp, split=0):
    with torch.no_grad():
        if not split:
            return model(input_ids=inp.passages, output_hidden_states=True, output_router_logits=True)
        parts = [model(input_ids=x, output_hidden_states=True, output_router_logits=True) for x in inp.passages.split(split)]
        joined = type(parts[0])(logits=torch.cat([p.logits for p in parts]),
                                hidden_states=tuple(torch.cat(h) for h in zip(*(p.hidden_states for p in parts))),
                                router_logits=tuple(torch.cat(r) for r in zip(*(p.router_logits for p in parts))))
        return joined


def summarise(model, inp, out):
    """The vocabulary is large: the full logits only at the last position; elsewhere each next token's log-probability."""
    k = model.config.num_experts_per_tok
    logp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1).gather(-1, inp.passages[:, 1:, None])[..., 0]
    return dict(next_token_logprob=logp.cpu(), last_logits=out.logits[:, -1].float().cpu(),
                top1=out.logits.argmax(-1).cpu(), hidden=[h.cpu() for h in out.hidden_states],
                router=[r.float().cpu() for r in out.router_logits],
                routed=torch.stack([torch.topk(r.float(), k, dim=-1).indices for r in out.router_logits]).cpu())


def gate_step(model, inp, means, split=0, expert_grads=False):
    """Gradients of the gate logits, the router weights, the embedding output (and three experts' weights)."""
    names = [n for n, _ in model.named_parameters()]
    train = [n for n in names if n.endswith("mlp.gate.weight")
             or (expert_grads and any(n.startswith(e) for e in EXPERT_GRADS))]
    params = dict(model.named_parameters())
    for n in train:
        params[n].requires_grad_(True)
        params[n].grad = None
    g = gates.Gates(inp.members, init=3.0).to(inp.device)
    embedded = []

    def embed_hook(_module, _inputs, output):
        output.requires_grad_(True)                      # a leaf (the embedding is frozen): its gradient is kept
        embedded.append(output)
        return output

    handle = model.model.embed_tokens.register_forward_hook(embed_hook)
    size = split or len(inp.n_pairs)
    halves = [(slice(i, i + size), slice(i, i + size)) for i in range(0, len(inp.n_pairs), size)]
    total, lengths = 0.0, []
    try:
        with gates.gated(model, g, means):
            for ns, ms in halves:
                logp = go.naming_logprobs(model, inp.n_pairs[ns], inp.pad, inp.weight)
                margins = go.label_margins(model, inp.m_seqs[ms], inp.m_correct[ms], inp.m_wrong[ms], inp.pad, inp.weight)
                lengths += [[len(a) + len(b) for a, b in inp.n_pairs[ns]], [len(s) for s in inp.m_seqs[ms]]]
                share = len(inp.n_pairs[ns]) / len(inp.n_pairs)
                loss = (logp.mean() + margins.mean()) * share
                loss.backward()
                total += float(loss.detach())
    finally:
        handle.remove()
    # the embedding gradient at the real tokens (left padding), naming prompts first, then M1, in item order: the same
    # rows whatever the batch composition
    rows = [e.grad[r, e.shape[1] - n:].float().cpu() for e, ns in zip(embedded, lengths) for r, n in enumerate(ns)]
    naming_rows = [x for i, x in enumerate(_split_rows(rows, lengths)) if i % 2 == 0]
    m1_rows = [x for i, x in enumerate(_split_rows(rows, lengths)) if i % 2 == 1]
    out = dict(loss=total, gate_grad=g.logits.grad.float().cpu(),
               router_grad=torch.cat([params[n].grad.float().flatten().cpu() for n in train if "experts" not in n]),
               embedding_grad=torch.cat([x.flatten() for part in naming_rows + m1_rows for x in part]))
    if expert_grads:
        out["expert_grad"] = torch.cat([params[n].grad.float().flatten().cpu() for n in train if "experts" in n])
    for n in train:
        params[n].requires_grad_(False)
        params[n].grad = None
    return out


def _split_rows(rows, lengths):
    """rows (one per sequence, in hook order) regrouped as one list per forward call."""
    out, i = [], 0
    for ns in lengths:
        out.append(rows[i:i + len(ns)])
        i += len(ns)
    return out


def production(model, tok, chat, inp, means, batch):
    out = {}
    lesion = (inp.lesion, means)
    out["m1"] = prefix_scoring.score_label_prompts(model, tok, inp.m_prompts, chat, batch)
    with hooks.ablate(model, inp.lesion, means):
        out["m1_ablated"] = prefix_scoring.score_label_prompts(model, tok, inp.m_prompts, chat, batch)
    out["m1_last3"] = probes.label_scores_in_positions(model, tok, inp.m_prompts, chat,
                                                       positions=[[-1, -2, -3]] * len(inp.m_prompts),
                                                       batch_size=batch, lesion=lesion)
    out["naming"] = naming.answer_logprobs(model, tok, inp.n_prompts, chat, batch_size=batch)
    with hooks.ablate(model, inp.lesion, means):
        out["naming_ablated"] = naming.answer_logprobs(model, tok, inp.n_prompts, chat, batch_size=batch)
    with hooks.mask_router(model, inp.experts):
        out["naming_masked"] = naming.answer_logprobs(model, tok, inp.n_prompts, chat, batch_size=batch)
    out["attribution"] = attribution.component_attribution(model, tok, inp.a_prompts, chat, means, inp.a_weights,
                                                           batch_size=batch)
    out["norms"] = probes.expert_output_norms(model, inp.norm_seqs, inp.experts)[0]
    return out


def numbers(x):
    """A result as one float32 vector (DataFrames: their numeric columns, in order)."""
    if isinstance(x, pd.DataFrame):
        return torch.tensor(x.select_dtypes("number").to_numpy(np.float64)).float().flatten()
    if isinstance(x, (list, tuple)):
        return torch.cat([numbers(v) for v in x])
    return (torch.tensor(x) if isinstance(x, np.ndarray) else x).float().flatten()


def identical(a, b):
    if isinstance(a, pd.DataFrame):
        return list(a.columns) == list(b.columns) and a.equals(b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))
    return a == b


def difference(a, b):
    """How far b is from a: absolute differences (mean, 99th percentile, max) and relative L2 difference."""
    x, y = numbers(a), numbers(b)
    if x.shape != y.shape:
        return dict(shape_differs=True)
    d = (x - y).abs()
    if not len(d):
        return dict(mean_abs=0.0, p99_abs=0.0, max_abs=0.0, rel_l2=0.0, cosine=1.0)
    sample = d[::max(1, len(d) // 4_000_000)]           # torch.quantile takes at most 2**24 values
    return dict(mean_abs=float(d.double().mean()), p99_abs=float(torch.quantile(sample, 0.99)), max_abs=float(d.max()),
                rel_l2=float((x - y).double().norm() / x.double().norm().clamp_min(1e-30)),
                cosine=float(torch.nn.functional.cosine_similarity(x.double(), y.double(), dim=0)))


def routed_differently(a, b):
    return [float((x.sort(-1).values != y.sort(-1).values).any(-1).float().mean()) for x, y in zip(a, b)]


def run_mode(model, tok, chat, inp, means, mode, noise=0):
    """Everything measured under one forward (noise: hf with the other batch compositions)."""
    sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    fwd = summarise(model, inp, forward_pass(model, inp, split=noise))
    sync()
    t_fwd = time.time() - t0
    model.train()                                       # no dropout in Qwen3-MoE; train() is what checkpointing needs
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        sync()
        t0 = time.time()
        step = gate_step(model, inp, means, split=noise, expert_grads=mode in ("hf", "sorted"))
        sync()
        t_step = time.time() - t0
    finally:
        model.gradient_checkpointing_disable()
        model.eval()
    prod = production(model, tok, chat, inp, means, noise or BATCH)
    times = None
    if not noise:                                         # timing: the mean of REPS repeats after the run above
        sync()
        t0 = time.time()
        for _ in range(REPS):
            forward_pass(model, inp)
        sync()
        t_fwd = (time.time() - t0) / REPS
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        try:
            t0 = time.time()
            for _ in range(REPS):
                gate_step(model, inp, means)
            sync()
            t_step = (time.time() - t0) / REPS
        finally:
            model.gradient_checkpointing_disable()
            model.eval()
        times = dict(seconds_forward=t_fwd, seconds_gate_step=t_step)
    print(f"{mode}{' (other batch compositions)' if noise else ''}: done; {times}", flush=True)
    return dict(fwd=fwd, step=step, prod=prod, times=times, memory=memory())


def compare(a, b):
    """b against a, measure by measure."""
    out = dict(next_token_logprob=difference(a["fwd"]["next_token_logprob"], b["fwd"]["next_token_logprob"]),
               last_logits=difference(a["fwd"]["last_logits"], b["fwd"]["last_logits"]),
               top1_differs=[float((a["fwd"]["top1"] != b["fwd"]["top1"]).float().mean())],
               hidden=difference(a["fwd"]["hidden"], b["fwd"]["hidden"]),
               router_logits=difference(a["fwd"]["router"], b["fwd"]["router"]),
               routed_differently=routed_differently(a["fwd"]["routed"], b["fwd"]["routed"]),
               rank_order_differs=[float((x != y).any(-1).float().mean()) for x, y in zip(a["fwd"]["routed"], b["fwd"]["routed"])])
    for k in ("gate_grad", "router_grad", "embedding_grad"):
        out[k] = difference(a["step"][k], b["step"][k])
    out["gate_step_loss_abs"] = abs(a["step"]["loss"] - b["step"]["loss"])
    for k in a["prod"]:
        out[k] = norm_difference(a["prod"][k], b["prod"][k]) if k == "norms" else difference(a["prod"][k], b["prod"][k])
    return out


def norm_difference(a, b):
    """Routing can change the rows. Compare values only after matching their identities."""
    keys = ["sequence", "token_id", "expert", "position", "rank"]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        return dict(shape_differs=True)
    x, y = a.set_index(keys).norm, b.set_index(keys).norm
    common = x.index.intersection(y.index)
    return dict(**difference(x.reindex(common).to_numpy(), y.reindex(common).to_numpy()),
                row_mismatch=1 - len(common) / max(len(x), len(y), 1), common_rows=len(common),
                rows_hf=len(x), rows_other=len(y))


def judge(grouped, noise):
    """Per layer and measure, using the envelope of two HF batch comparisons."""
    noise = [noise] if isinstance(noise, dict) else noise
    verdict = {}
    for k, g in grouped.items():
        ns = [n[k] for n in noise]
        if k in ("routed_differently", "rank_order_differs", "top1_differs"):
            verdict[k] = all(np.isfinite(v) for n in [g, *ns] for v in n) and all(len(g) == len(n) for n in ns) and all(
                v <= TOL_FACTOR * max(n[i] for n in ns) + 1e-3 for i, v in enumerate(g))
        elif k == "gate_step_loss_abs":
            verdict[k] = bool(np.isfinite(g) and all(np.isfinite(n) for n in ns) and g <= TOL_FACTOR * max(ns) + 1e-6)
        elif g.get("shape_differs") or any(n.get("shape_differs") for n in ns):
            verdict[k] = False
        else:
            metrics = ("mean_abs", "p99_abs", "max_abs", "rel_l2")
            finite = all(np.isfinite(x[m]) for x in [g, *ns] for m in (*metrics, "cosine"))
            verdict[k] = finite and all(g[m] <= TOL_FACTOR * max(n[m] for n in ns) + 1e-6 for m in metrics)
            verdict[k] = bool(verdict[k] and g["cosine"] >= min(n["cosine"] for n in ns) - 1e-4)
            if k == "norms":
                verdict[k] = verdict[k] and all(n["common_rows"] > 0 for n in [g, *ns]) and (
                    g["row_mismatch"] <= TOL_FACTOR * max(n["row_mismatch"] for n in ns) + 1e-3)
    return verdict


def capture_block_inputs(model, ids):
    """A small, fixed HF input for every MoE block, before any accelerated forward is installed."""
    inputs, handles = {}, []
    for i, layer in enumerate(model.model.layers):
        handles.append(moe_block(layer).register_forward_pre_hook(
            lambda _m, args, i=i: inputs.update({i: args[0].detach().cpu().clone()})))
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for handle in handles:
            handle.remove()
    return inputs


def same_input_check(model, inputs):
    """Compare HF and the installed forward on identical block inputs, without earlier-layer drift."""
    out = []
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            block = moe_block(layer)
            x = inputs[i].to(next(block.parameters()))
            hf, rh = type(block).forward(block, x)
            fast, rf = block(x)
            atol, rtol = {torch.float32: (1e-6, 1e-5), torch.float16: (2e-3, 1e-2),
                          torch.bfloat16: (2e-2, 2e-2)}[x.dtype]
            th = torch.topk(torch.softmax(rh.float(), -1), model.config.num_experts_per_tok, -1).indices
            tf = torch.topk(torch.softmax(rf.float(), -1), model.config.num_experts_per_tok, -1).indices
            router, routes = torch.equal(rh, rf), torch.equal(th, tf)
            close = torch.allclose(hf, fast, atol=atol, rtol=rtol)
            out.append(dict(layer=i, router_exact=router, topk_exact=routes, output_close=close,
                            max_abs=float((hf - fast).abs().max()), atol=atol, rtol=rtol,
                            passed=router and routes and close))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--means", default=None)
    args = ap.parse_args()
    commit = runinfo.commit() if args.tiny else runinfo.require_clean()
    p14 = ROOT / ("outputs/p14_tiny" if args.tiny else "outputs/p14") / NAME
    out_dir = ROOT / ("outputs/fast_moe_check_tiny" if args.tiny else "outputs/fast_moe_check") / NAME
    if args.tiny:
        model, tok = models.load_tiny(NAME, num_experts=16, num_experts_per_tok=4)
        model = model.to(device="cuda", dtype=torch.float16) if torch.cuda.is_available() else model
        means = means_mod.collect_means(model, tok, neutral_text.load()[:4], max_tokens=32, batch_size=4)
        members = json.loads((p14 / "pools.json").read_text(encoding="utf-8"))["pools"]["knowledge_p8"]["members"]
        global EXPERT_GRADS
        EXPERT_GRADS = ("model.layers.0.mlp.experts.3.", "model.layers.1.mlp.experts.9.")
    else:
        models.check_environment()
        model, tok = models.load_model(NAME, dtype=torch.float16)
        means = means_mod.Means.load(args.means or MEANS)
        if means.meta.get("dtype") != "float16" or means.meta.get("revision") != models.MODELS[NAME][1]:
            raise SystemExit(f"the means are not float16 means of {NAME}@{models.MODELS[NAME][1]}")
        members = json.loads((p14 / "pools.json").read_text(encoding="utf-8"))["pools"][POOL]["members"]
    for p in model.parameters():
        p.requires_grad_(False)
    chat = models.uses_chat(NAME)
    blocks = [moe_block(layer) for layer in model.model.layers]
    setup = dict(devices=sorted({str(next(b.parameters()).device) for b in blocks}),
                 accelerate_hooks=any(hasattr(e, "_hf_hook") for b in blocks for e in b.experts))
    print(f"MoE blocks on {setup['devices']}; accelerate hooks: {setup['accelerate_hooks']}", flush=True)
    inp = Inputs(model, tok, chat, members)
    block_inputs = capture_block_inputs(model, inp.passages[:2, :16])

    res = {"hf": run_mode(model, tok, chat, inp, means, "hf")}
    for batch in NOISE_BATCHES:
        res[f"hf_batch{batch}"] = run_mode(model, tok, chat, inp, means, "hf", noise=batch)
    fast_moe.install(model, "sorted")
    res["sorted"] = run_mode(model, tok, chat, inp, means, "sorted")
    fast_moe.uninstall(model)
    install = dict(before=memory())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fast_moe.install(model, "grouped")
    install["after"] = memory()
    structural = same_input_check(model, block_inputs)
    res["grouped"] = run_mode(model, tok, chat, inp, means, "grouped")
    fast_moe.uninstall(model)
    model.to(torch.bfloat16)
    res["hf_bfloat16"] = dict(fwd=summarise(model, inp, forward_pass(model, inp)))

    noise = [compare(res["hf"], res[f"hf_batch{batch}"]) for batch in NOISE_BATCHES]
    grouped = compare(res["hf"], res["grouped"])
    bf16 = dict(next_token_logprob=difference(res["hf"]["fwd"]["next_token_logprob"],
                                              res["hf_bfloat16"]["fwd"]["next_token_logprob"]),
                routed_differently=routed_differently(res["hf"]["fwd"]["routed"], res["hf_bfloat16"]["fwd"]["routed"]))
    sorted_same = {k: identical(res["hf"][k], res["sorted"][k]) for k in ("fwd", "step", "prod")}
    grouped_ok = judge(grouped, noise)
    routes = res["hf"]["fwd"]["routed"]
    route_counts = {name: [dict(layer=i, tokens=x.shape[0], rank_disagreements=int((x != y).any(-1).sum()),
                                set_disagreements=int((x.sort(-1).values != y.sort(-1).values).any(-1).sum()))
                          for i, (x, y) in enumerate(zip(routes, res[name]["fwd"]["routed"]))]
                    for name in ("grouped", *(f"hf_batch{b}" for b in NOISE_BATCHES))}
    passes = dict(sorted=all(sorted_same.values()), grouped=all(grouped_ok.values()) and all(x["passed"] for x in structural))
    speed = {m: dict(**res[m]["times"], peak_gb=res[m]["memory"] and res[m]["memory"]["peak_allocated_gb"],
                     speedup_forward=res["hf"]["times"]["seconds_forward"] / res[m]["times"]["seconds_forward"],
                     speedup_gate_step=res["hf"]["times"]["seconds_gate_step"] / res[m]["times"]["seconds_gate_step"])
             for m in ("hf", "sorted", "grouped")}
    result = dict(model=NAME, revision=models.MODELS[NAME][1], tiny=args.tiny, commit=commit, pool=POOL,
                  n_gates=len(members), tested=list(res), same_input_blocks=structural, noise_batches=NOISE_BATCHES,
                  tol_factor=TOL_FACTOR, setup=setup, passes=passes, routing_counts=route_counts, sorted_identical=sorted_same,
                  grouped_within_noise=grouped_ok, grouped_vs_hf=grouped, noise_vs_hf=noise,
                  sorted_vs_hf=compare(res["hf"], res["sorted"]), bfloat16_vs_hf_for_scale=bf16, speed=speed,
                  grouped_install_memory=install)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    runinfo.write_meta(out_dir, stage="fast_moe_check", model=NAME, tiny=args.tiny, commit=commit, passes=passes,
                       moe_forward_tested=result["tested"])
    print(json.dumps(dict(passes=passes, sorted_identical=sorted_same,
                          grouped_failed=[k for k, v in grouped_ok.items() if not v], speed=speed), indent=1))
    if not passes["sorted"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

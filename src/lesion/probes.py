"""Read-outs and targeted replacements for the GPU session 12 mechanism checks (docs/plans/gpu-session12.md).

Works on token-id sequences, so the tests run on tiny random models without a tokenizer. Forward passes only.
- head_outputs_last: what chosen attention heads write at the last position (the answer position), as the input slice
  of o_proj that lesion.hooks.ablate replaces, so a captured vector can be put back exactly there;
- patched_means: a copy of a Means object with some heads' means replaced by given vectors (for example a task's
  average head output), to be used with hooks.ablate(mode="mean") and a last-position mask;
- last_position_mask, first_token_mask: (batch, seq) masks for hooks.position_mask on left-padded batches;
- expert_output_norms: the length of each chosen expert's output on every token routed to it, with the token's
  position, in the row order the MoE block feeds the expert;
- lm_loss_region: language-model loss with a lesion restricted to the first real token or to all other tokens;
- last_logprobs, position_logprobs, label_scores_at_answer: next-token log-probabilities and A/B label logits that
  project only the positions needed onto the vocabulary (the last one, or chosen ones);
- token_zipf, vocab_shift: how a lesion moves the whole next-token distribution across word frequency.
GPU session 13 (docs/plans/gpu-session13.md):
- position_parts, label_scores_in_positions: an M1 prompt split into seven disjoint parts (context, probe, options,
  labels, start, end_template, answer), and A/B label logits with a lesion restricted to chosen positions;
- target_change_rank: where a target token's log-probability change falls in the whole vocabulary's changes.
"""
import copy
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

from . import hooks, scoring
from .components import moe_block


def _head_dim(model):
    config = model.config
    return int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)


def last_position_mask(attention_mask):
    """True at the last column of every row: with left padding, each row's last real token."""
    mask = torch.zeros(attention_mask.shape, dtype=torch.bool, device=attention_mask.device)
    mask[:, -1] = True
    return mask


def first_token_mask(attention_mask):
    """True at each row's first real token of a left-padded batch."""
    real = attention_mask.bool()
    first = real & ~torch.nn.functional.pad(real, (1, 0))[:, :-1]
    return first


def head_outputs_last(model, seqs, heads, batch_size=16, pad_id=0):
    """{head key: float32 tensor (len(seqs), head_dim)}: each head's slice of the o_proj input at the last position of
    each sequence (rows in the order of `seqs`)."""
    device = next(model.parameters()).device
    head_dim = _head_dim(model)
    out = {h.key: torch.zeros(len(seqs), head_dim) for h in heads}
    store = {}
    handles = []
    for h in heads:
        module = model.model.layers[h.layer].self_attn.o_proj
        lo, hi = h.index * head_dim, (h.index + 1) * head_dim

        def pre(_module, inputs, key=h.key, lo=lo, hi=hi):
            store[key] = inputs[0][:, -1, lo:hi].detach().float().cpu()

        handles.append(module.register_forward_pre_hook(pre))
    try:
        with torch.no_grad():
            for idx in scoring._batches(seqs, batch_size):
                ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], pad_id, device)
                model(input_ids=ids, attention_mask=mask, position_ids=pos)
                for key in out:
                    out[key][idx] = store[key]
    finally:
        for handle in handles:
            handle.remove()
    return out


def patched_means(means, vectors):
    """A copy of `means` whose head_mean[layer, head] is replaced by vectors[Component] (a head_dim tensor)."""
    patched = copy.copy(means)
    patched.head_mean = means.head_mean.clone()
    for component, vector in vectors.items():
        if component.kind != "head":
            raise ValueError(f"only heads can be patched: {component}")
        patched.head_mean[component.layer, component.index] = torch.as_tensor(vector, dtype=patched.head_mean.dtype)
    return patched


def expert_output_norms(model, seqs, experts):
    """One row per (sequence, routed token, expert): the L2 norm of the expert's output on that token (before the
    routing weight), the token's position in the sequence and its id. Sequences are run one at a time, unpadded, so
    a flattened row index is the position. Returns (DataFrame, {expert key: first-token outputs as a float32 tensor
    (n, hidden), one per sequence whose first token was routed to it})."""
    device = next(model.parameters()).device
    rows, first_outputs = [], {e.key: [] for e in experts}
    by_layer = {}
    for e in experts:
        by_layer.setdefault(e.layer, []).append(e)
    routed, handles, current = {}, [], {"rows": []}
    for layer, chosen in by_layer.items():
        block = moe_block(model.model.layers[layer])

        def gate_hook(_module, _inputs, output, layer=layer, block=block):
            logits = output[0] if isinstance(output, tuple) else output
            # the same routing reproduction lesion.hooks.ablate uses (tested against the models' own routers)
            routed[layer] = hooks._routed_experts(block, logits.reshape(-1, logits.shape[-1]), block.top_k)

        handles.append(block.gate.register_forward_hook(gate_hook))
        for e in chosen:
            def expert_hook(_module, _inputs, output, e=e):
                value = output[0] if isinstance(output, tuple) else output
                # HF feeds an expert torch.where(expert_mask[e]) rows: grouped by top-k rank, then by token
                rank, token = torch.where(routed[e.layer].T == e.index)
                if len(token) != value.shape[0]:
                    raise RuntimeError(f"routing does not match the rows fed to {e.key}")
                norms = value.detach().float().norm(dim=-1).cpu().numpy()
                for r, (t, n) in enumerate(zip(token.tolist(), norms)):
                    current["rows"].append(dict(expert=e.key, position=t, rank=int(rank[r]), norm=float(n)))
                    if t == 0:
                        first_outputs[e.key].append(value[r].detach().float().cpu())

            handles.append(block.experts[e.index].register_forward_hook(expert_hook))
    try:
        with torch.no_grad():
            for s, seq in enumerate(seqs):
                current["rows"] = []
                ids = torch.tensor([seq], dtype=torch.long, device=device)
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
                for row in current["rows"]:
                    rows.append(dict(sequence=s, token_id=int(seq[row["position"]]), **row))
    finally:
        for handle in handles:
            handle.remove()
    firsts = {k: (torch.stack(v) if v else torch.zeros(0)) for k, v in first_outputs.items()}
    return pd.DataFrame(rows), firsts


def outputs_agree(total, scale, output):
    """The check of gptoss_expert_norms. total: the routing-weighted sum of the recomputed expert outputs per token
    (fp32); scale: per token, the sum of the L2 norms of the weighted outputs added into it; output: the module's
    output. A bf16 or fp16 module rounds every weighted output and the sum (about 2^-8 relative each time), and on the
    GPU it adds all 32 experts at once (zero weights for the unrouted ones), so entries near 0 can miss any fixed
    entry-wise tolerance once the outputs are large (a tiny bf16 model on a GPU with outputs up to about 40: entry-wise
    allclose fails while the per-token distance is 0.3% of the output's norm). So the check is per token: the L2
    distance over the size of what was added (scale), at most 2e-2 (1e-4 in fp32); a wrong expert, weight or formula
    misses by the size of a whole expert's output. Its limit: an expert whose weighted output is under about 2% of
    what was added can be left out unnoticed, so it shows that the recomputation adds up to the module's output, not
    that every small contribution is right (the formulas themselves are checked in fp32, where the tolerance is 1e-4).
    Non-finite values never agree. Also returns the distance over the output's own L2 norm.
    Returns (agree, {"max_rel_l2": ..., "max_rel_scale": ...})."""
    o = output.float().reshape(total.shape)
    dist = (total - o).norm(dim=-1)
    rel_scale = float((dist / scale.clamp_min(1e-6)).max())
    rel_l2 = float((dist / o.norm(dim=-1).clamp_min(1e-6)).max())
    tol = 1e-4 if output.dtype == torch.float32 else 2e-2
    finite = bool(torch.isfinite(total).all() and torch.isfinite(o).all())
    return finite and rel_scale <= tol, dict(max_rel_l2=rel_l2, max_rel_scale=rel_scale)


def gptoss_expert_norms(model, seqs, check=None):
    """expert_output_norms for gpt-oss (P12, docs/plans/second-model-g1-p12.md), whose 32 experts per layer share one
    module (GptOssExperts), so they cannot be hooked one by one. For every layer and every routed (token, expert):
    the L2 norm of the expert's output before the routing weight, recomputed from the module's input and the router's
    choice with the module's own weights and formulas. Read-only: nothing is written back and nothing is kept for
    backward. Sequences run one at a time, unpadded, so a row index is the position. check (a dict) also verifies that
    the routing-weighted sum of the recomputed outputs equals the module's output (outputs_agree), raising if not, and
    records the largest differences in check (costs time): "max_abs" and "max_rel" entry by entry, "max_rel_l2" and
    "max_rel_scale" per token (outputs_agree).
    Returns (DataFrame: sequence, token_id, expert, position, rank, norm; {expert key: norm of its mean output over
    the tokens routed to it, and the count})."""
    device = next(model.parameters()).device
    parts, sums, counts, handles, current = [], {}, {}, [], {}

    def hook(module, args, kwargs, output, layer):
        h = (args[0] if args else kwargs["hidden_states"]).reshape(-1, module.hidden_size)
        idx, weights = kwargs["router_indices"], kwargs["routing_weights"]
        total = torch.zeros(h.shape, dtype=torch.float32, device=h.device) if check is not None else None
        scale = torch.zeros(h.shape[0], dtype=torch.float32, device=h.device) if check is not None else None
        for e in idx.unique().tolist():
            rank, token = torch.where(idx.T == e)
            gate_up = h[token] @ module.gate_up_proj[e] + module.gate_up_proj_bias[e]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=module.limit)
            up = up.clamp(min=-module.limit, max=module.limit)
            out = ((up + 1) * (gate * torch.sigmoid(gate * module.alpha))) @ module.down_proj[e] + module.down_proj_bias[e]
            out = out.float()
            key = f"expert:{layer}:{e}"
            parts.append(pd.DataFrame(dict(sequence=current["s"], expert=key, position=token.cpu().numpy(),
                                           rank=rank.cpu().numpy(), norm=out.norm(dim=-1).cpu().numpy())))
            sums[key] = sums.get(key, 0) + out.sum(0).cpu()
            counts[key] = counts.get(key, 0) + len(token)
            if check is not None:
                weighted = out * weights[token, e, None].float()
                total.index_add_(0, token, weighted)
                scale.index_add_(0, token, weighted.norm(dim=-1))
        if check is not None:
            diff = (total.reshape(output.shape) - output.float()).abs()
            check["max_abs"] = max(check.get("max_abs", 0.0), float(diff.max()))
            check["max_rel"] = max(check.get("max_rel", 0.0), float((diff / output.float().abs().clamp_min(1e-3)).max()))
            agree, stats = outputs_agree(total, scale, output)
            for key, value in stats.items():
                check[key] = max(check.get(key, 0.0), value)
            if not agree:
                check["failed_layer"] = layer
                raise RuntimeError(f"recomputed expert outputs of layer {layer} do not add up to the module's output: "
                                   f"{stats}")

    for layer, block in enumerate(model.model.layers):
        handles.append(block.mlp.experts.register_forward_hook(
            lambda m, a, k, o, layer=layer: hook(m, a, k, o, layer), with_kwargs=True))
    try:
        with torch.no_grad():
            for s, seq in enumerate(seqs):
                current["s"] = s
                ids = torch.tensor([seq], dtype=torch.long, device=device)
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        for handle in handles:
            handle.remove()
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["sequence", "expert", "position", "rank", "norm"])
    rows["token_id"] = [int(seqs[s][pos]) for s, pos in zip(rows.sequence, rows.position)]
    means = {k: dict(count=int(counts[k]), mean_norm=float((sums[k] / counts[k]).norm())) for k in sums}
    return rows, means


def lm_loss_region(model, seqs, components, means, region=None, mode="mean", batch_size=8, pad_id=0):
    """Mean next-token negative log-probability (nats per token) over `seqs` with `components` ablated, everywhere
    (region None), only at each sequence's first real token ("first"), or at every other real token ("rest")."""
    if region not in (None, "first", "rest"):
        raise ValueError(f"unknown region {region!r}")
    device = next(model.parameters()).device
    seqs = [s for s in seqs if len(s) > 1]
    total, count = 0.0, 0
    lesion = hooks.ablate(model, components, means, mode=mode) if components else nullcontext()
    with torch.no_grad(), lesion:
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], pad_id, device)
            restrict = nullcontext()
            if region is not None and components:
                first = first_token_mask(mask)
                restrict = hooks.position_mask(model, first if region == "first" else mask.bool() & ~first)
            with restrict:
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
            for r in range(len(idx)):
                keep = mask[r].bool()
                lg = logits[r, keep][:-1].float()
                tgt = ids[r, keep][1:]
                total += float(torch.nn.functional.cross_entropy(lg, tgt, reduction="sum"))
                count += int(tgt.numel())
    return total / count


def last_logprobs(model, seqs, batch_size=16, pad_id=0):
    """(len(seqs), vocab) float32 log-probabilities of the next token after each sequence (left-padded batches; only
    the last position is projected onto the vocabulary)."""
    device = next(model.parameters()).device
    out = [None] * len(seqs)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], pad_id, device)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1).logits[:, -1]
            lp = torch.log_softmax(logits.float(), -1).cpu()
            for r, i in enumerate(idx):
                out[i] = lp[r]
    return torch.stack(out)


def position_logprobs(model, ids, cols):
    """(batch, len(cols), vocab) float32 log-probabilities at the chosen positions of unpadded equal-length rows,
    projecting only those positions onto the vocabulary (all positions of a long batch would not fit in memory)."""
    hidden = model.model(input_ids=ids).last_hidden_state[:, cols]
    return torch.log_softmax(model.lm_head(hidden).float(), -1)


def label_scores_at_answer(model, tokenizer, prompts, chat, batch_size=16, lesion=None):
    """lesion.scoring.score_label_prompts' A/B logits, with an optional lesion (components, means) applied only at
    the answer position (each prompt's last token)."""
    return label_scores_in_positions(model, tokenizer, prompts, chat, [[-1]] * len(prompts), batch_size, lesion)


PARTS = ("context", "probe", "options", "labels", "start", "end_template", "answer")


def position_parts(tokenizer, prompt, chat):
    """(token ids, {part: sorted token indices}) for a two-choice M1 prompt, every token in exactly one part:
    context = the instruction and the demos; probe = the test block's first line ("Word: spinach"); options = the two
    option words; labels = the rest of the test block ("A:", "B:", "Answer:"); start = the first token and any
    chat-template tokens before the prompt text; end_template = the chat-template tokens after it; answer = the last
    token, where the A/B logits are read (taken out of whichever part it would otherwise fall in)."""
    from .attention import regions

    ids, reg = regions(tokenizer, prompt, chat)
    content = [j for r in ("instruction", "demos", "probe_line", "correct_option", "wrong_option", "cue") for j in reg[r]]
    first, last = min(content), len(ids) - 1
    parts = dict(context=reg["instruction"] + reg["demos"], probe=reg["probe_line"],
                 options=reg["correct_option"] + reg["wrong_option"], labels=reg["cue"],
                 start=reg["start"] + [j for j in reg["template"] if j < first],
                 end_template=[j for j in reg["template"] if j > first])
    parts = {k: sorted(j for j in v if j != last) for k, v in parts.items()}
    parts["answer"] = [last]
    return ids, parts


LABEL_PARTS = ("label_a", "label_b", "label_answer")


def split_labels(parts):
    """position_parts with the `labels` part split by where its tokens sit (session 14b): label_a before the first
    option word ("A:"), label_b between the two option words ("B:"), label_answer after the second ("Answer:")."""
    options = sorted(parts["options"])
    gaps = [k for k in range(1, len(options)) if options[k] != options[k - 1] + 1]
    if len(gaps) != 1:
        raise ValueError(f"expected two separate option words, got positions {options}")
    first_end, second_start = options[gaps[0] - 1], options[gaps[0]]
    labels = parts["labels"]
    out = {k: v for k, v in parts.items() if k != "labels"}
    out["label_a"] = [j for j in labels if j < options[0]]
    out["label_b"] = [j for j in labels if first_end < j < second_start]
    out["label_answer"] = [j for j in labels if j > options[-1]]
    if sorted(out["label_a"] + out["label_b"] + out["label_answer"]) != sorted(labels):
        raise ValueError("a label token sits inside an option word")
    return out


def label_scores_in_positions(model, tokenizer, prompts, chat, positions=None, batch_size=16, lesion=None):
    """A/B label logits as lesion.scoring.score_label_prompts gives them (one batch composition, no prefix cache),
    with an optional lesion (components, means). positions[i] lists the token indices of prompt i's own unpadded
    sequence where the lesion acts (negative indices count from its end; an empty list leaves the prompt intact);
    positions=None lesions every position."""
    device = next(model.parameters()).device
    lab = scoring.label_token_ids(tokenizer, chat)
    ids_ab = [lab[l] for l in scoring.LABELS]
    seqs = [scoring.encode(tokenizer, p.text, chat) for p in prompts]
    if positions is not None and len(positions) != len(prompts):
        raise ValueError("one list of positions per prompt")
    logits = [None] * len(prompts)
    ctx = hooks.ablate(model, lesion[0], lesion[1]) if lesion else nullcontext()
    with torch.no_grad(), ctx:
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
            restrict = nullcontext()
            if lesion and positions is not None:
                where = torch.zeros(mask.shape, dtype=torch.bool, device=mask.device)
                for r, i in enumerate(idx):
                    n = len(seqs[i])
                    cols = [j % n for j in positions[i]]
                    if any(not -n <= j < n for j in positions[i]):
                        raise IndexError(f"position out of range for prompt {i} of length {n}")
                    where[r, [mask.shape[1] - n + j for j in cols]] = True
                restrict = hooks.position_mask(model, where)
            with restrict:
                out = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                            logits_to_keep=1).logits[:, -1, ids_ab].float().cpu()
            for r, i in enumerate(idx):
                logits[i] = out[r].tolist()
    rows = []
    for p, (la, lb) in zip(prompts, logits):
        by = {"A": la, "B": lb}
        wrong = "B" if p.correct_label == "A" else "A"
        rows.append(dict(item_id=str(p.item_id), task=p.task, order=p.order, correct_label=p.correct_label, logit_A=la,
                         logit_B=lb, logit_correct=by[p.correct_label], logit_wrong=by[wrong]))
    return pd.DataFrame(rows)


def target_change_rank(intact_logp, lesioned_logp, targets, top=50):
    """One row per sequence: the target token's log-probability change (lesioned - intact); the share of the vocabulary
    whose change is at least as negative (a small share: the target lost more than almost every token); and the
    target's rank by change among the `top` tokens most probable when intact (1 = lost the most), or NaN when the
    target is not among them. intact_logp, lesioned_logp: (n, vocab) log-probabilities at the answer position."""
    if intact_logp.shape != lesioned_logp.shape or intact_logp.shape[0] != len(targets):
        raise ValueError("intact and lesioned log-probabilities must share a shape with one target per row")
    change = (lesioned_logp - intact_logp).float()
    rows = []
    for i, t in enumerate(targets):
        c = change[i]
        top_ids = torch.topk(intact_logp[i], min(top, c.numel())).indices
        in_top = (top_ids == t).nonzero()
        rank = float((c[top_ids] <= c[t]).sum()) if len(in_top) else float("nan")
        rows.append(dict(target_change=float(c[t]), share_at_least_as_negative=float((c <= c[t]).float().mean()),
                         rank_in_top=rank))
    return pd.DataFrame(rows)


def token_zipf(tokenizer, zipf_of, leading_space=True, size=None):
    """Float array over the vocabulary: zipf_of(word) for tokens that decode to one English word made of letters,
    with a leading space (" river": a word inside running text) or without one ("river": a word at the start of a
    chat reply, where the naming answers begin); NaN for every other token and where zipf_of returns 0 (unknown).
    size: the model's output vocabulary (often padded beyond the tokenizer's); ids past the tokenizer are NaN."""
    import re

    word = re.compile(r"^ [A-Za-z]+$" if leading_space else r"^[A-Za-z]+$")
    out = np.full(max(size or 0, len(tokenizer)), np.nan)
    for i in range(len(tokenizer)):
        text = tokenizer.decode([i])
        if word.match(text):
            z = zipf_of(text.strip().lower())
            if z > 0:
                out[i] = z
    return out


def vocab_shift(intact_logp, lesioned_logp, zipf, n_bins=5):
    """How a lesion moves the next-token distribution across word frequency, one row per position.

    intact_logp, lesioned_logp: (n, vocab) log-probabilities; zipf: (vocab,) with NaN for tokens left out. Per row,
    over the tokens with a zipf value: the Spearman correlation between the change (lesioned minus intact
    log-probability) and zipf (positive: rare tokens lose more); the change of the KL divergence from the word-frequency
    distribution (10**zipf, normalised over those tokens) to the model's distribution renormalised over them
    (negative: the lesioned model is closer to plain word frequency); the mean change in each zipf bin (equal-count bins
    over those tokens)."""
    if len(zipf) != intact_logp.shape[-1] or intact_logp.shape != lesioned_logp.shape:
        raise ValueError(f"zipf covers {len(zipf)} tokens, the log-probabilities {tuple(intact_logp.shape)} and "
                         f"{tuple(lesioned_logp.shape)}")
    valid = torch.as_tensor(~np.isnan(zipf))
    z = torch.as_tensor(np.nan_to_num(zipf))[valid].float()
    a, b = intact_logp[:, valid].float(), lesioned_logp[:, valid].float()
    delta = b - a
    from scipy.stats import rankdata

    rz = rankdata(z.numpy())
    rd = rankdata(delta.numpy(), axis=-1)                     # ties share their average rank
    rz, rd = rz - rz.mean(), rd - rd.mean(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (rd * rz).sum(-1) / (np.linalg.norm(rd, axis=-1) * np.linalg.norm(rz))
    log_u = z * np.log(10.0)
    log_u = log_u - torch.logsumexp(log_u, 0)

    def kl_to_frequency(logp):
        lp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
        return (log_u.exp()[None] * (log_u[None] - lp)).sum(-1)

    edges = torch.quantile(z, torch.linspace(0, 1, n_bins + 1))
    bins = torch.bucketize(z, edges[1:-1], right=True)
    out = pd.DataFrame(dict(spearman=rho, kl_change=(kl_to_frequency(b) - kl_to_frequency(a)).numpy()))
    for k in range(n_bins):
        out[f"bin{k}"] = delta[:, bins == k].mean(-1).numpy()
    return out


def norm_summary(norms, n_sequences):
    """Per expert: how its output length splits between the first token and the rest."""
    out = {}
    for key, g in norms.groupby("expert"):
        first, rest = g[g.position == 0], g[g.position > 0]
        sq = (g.norm ** 2).sum()
        out[key] = dict(n_routed=int(len(g)), share_of_first_tokens_routed=float(len(first) / n_sequences),
                        median_norm_first=float(first.norm.median()) if len(first) else np.nan,
                        median_norm_rest=float(rest.norm.median()) if len(rest) else np.nan,
                        p99_norm_rest=float(rest.norm.quantile(.99)) if len(rest) else np.nan,
                        share_of_squared_norm_on_first=float((first.norm ** 2).sum() / sq) if sq > 0 else np.nan)
    return out

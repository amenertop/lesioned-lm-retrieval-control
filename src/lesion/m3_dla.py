"""Direct logit attribution of lesioned components at the answer position (mechanism battery,
docs/plans/experiment3-protocol.md §5.5; added 2026-10-08 after agent2 noted that the logit lens reads the whole
residual stream, not the components' own contribution).

Mean ablation of a component removes, at each position, its output minus its mean:
- expert e of layer L: w_e * (expert_e(x) - mean_e), where w_e is the token's normalised top-k routing weight
  for e (0 when the token is not routed to e, and then the ablation does not touch it);
- head h of layer L: W_O[:, slice_h] @ (z_h - mean_h), z_h the head's input slice of o_proj.
The direct effect of that removed vector c on the output logits, holding the final RMSNorm scale fixed, is
(c * gamma / rms(r)) @ W_U[token], with r the final residual stream at that position. Read at the last prompt
position (the one that predicts the answer's first token) for chosen tokens: the concept's own target, the targets
of the other concepts (mean; the concept-specific part is own minus others), its associate, coordinate and
superordinate, and the vocabulary mean.
"""
import pandas as pd

from . import scoring
from .m3_scoring import encode_answer


def _first_tokens(tokenizer, prompts, role, chat):
    return [encode_answer(tokenizer, p.text, p.candidates[role], chat)[1][0] for p in prompts]


def routing_weight(block, x, index):
    """Normalised top-k routing weight of expert `index` for inputs x (n, hidden), float32, (n, 1); 0 where the
    expert is not among the top k. The same path as the Qwen3-MoE block: gate, float32 softmax, top-k, optional
    renormalisation."""
    import torch

    p = torch.softmax(block.gate(x).float(), dim=-1)
    w, sel = torch.topk(p, block.top_k, dim=-1)
    if getattr(block, "norm_topk_prob", False):
        w = w / w.sum(dim=-1, keepdim=True)
    return (w * (sel == index)).sum(dim=-1, keepdim=True)


def removed_vector(model, means, comp, x, model_dtype=False):
    """What mean ablation of `comp` removes from the residual stream, returned in float32 (n, hidden). x is the
    module input at the chosen positions (n, width): the mixture-of-experts block input for an expert, the o_proj
    input for a head. Shared by contributions and lesion.m3_optogenetics.implant, so both use one definition.

    model_dtype=False (default; the direct logit attribution of experiment 3, S4): computed in float32.
    model_dtype=True (the implant, version 2 as revised after agent2's review of 2026-10-12): the operations the
    model itself performs, in its own dtype, as in the Qwen3-MoE block and hooks.ablate: the routing weight cast to
    the activation dtype times the expert's output, minus the same weight times the expert's mean; for a head,
    (z - mean) times its o_proj columns. Only the result is cast to float32. What remains different from the
    model's path is the rounding of the block's sum over experts (and of o_proj's sum over heads)."""
    import torch

    if comp.kind not in ("expert", "head"):
        raise ValueError(f"removed vectors are defined for experts and heads, not {comp.kind}")
    layer = model.model.layers[comp.layer]
    if comp.kind == "expert":
        block = layer.mlp
        w = routing_weight(block, x, comp.index)
        if model_dtype:
            w, mean = w.to(x.dtype), means.expert_mean[comp.layer, comp.index].to(device=x.device, dtype=x.dtype)
            return (block.experts[comp.index](x) * w).float() - (mean * w).float()
        mean = means.expert_mean[comp.layer, comp.index].to(device=x.device, dtype=torch.float32)
        return w * (block.experts[comp.index](x).float() - mean)
    cfg = model.config
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    cols = slice(comp.index * head_dim, (comp.index + 1) * head_dim)
    W = layer.self_attn.o_proj.weight[:, cols]
    if model_dtype:
        mean = means.head_mean[comp.layer, comp.index].to(device=x.device, dtype=x.dtype)
        return ((x[:, cols] - mean) @ W.T).float()
    z = x[:, cols].float()
    mean = means.head_mean[comp.layer, comp.index].to(device=z.device, dtype=torch.float32)
    return (z - mean) @ W.float().T


def contributions(model, means, comps, prompts, tokenizer, chat=True, batch_size=8, model_dtype=False):
    """Removed vectors at the last prompt position: {component key: tensor (n_prompts, hidden) float32} and the
    final residual stream there (n_prompts, hidden). model_dtype: see removed_vector (True for the implant)."""
    import torch

    device = next(model.parameters()).device
    layers = model.model.layers
    seqs = [encode_answer(tokenizer, p.text, p.candidates["target"], chat)[0] for p in prompts]
    need_mlp = sorted({c.layer for c in comps if c.kind == "expert"})
    need_o = sorted({c.layer for c in comps if c.kind == "head"})
    store = {}
    handles = []
    for L in need_mlp:
        handles.append(layers[L].mlp.register_forward_pre_hook(lambda _m, inp, L=L: store.__setitem__(("mlp", L), inp[0][:, -1].detach())))
    for L in need_o:
        handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(lambda _m, inp, L=L: store.__setitem__(("o", L), inp[0][:, -1].detach())))
    handles.append(model.model.norm.register_forward_pre_hook(lambda _m, inp: store.__setitem__("final", inp[0][:, -1].detach())))
    out = {c.key: [None] * len(prompts) for c in comps}
    final = [None] * len(prompts)
    try:
        with torch.no_grad():
            for idx in scoring._batches(seqs, batch_size):
                ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
                store.clear()
                model.model(input_ids=ids, attention_mask=mask, position_ids=pos)
                for c in comps:
                    vec = removed_vector(model, means, c, store[("mlp" if c.kind == "expert" else "o", c.layer)], model_dtype)
                    for r, i in enumerate(idx):
                        out[c.key][i] = vec[r].cpu()
                for r, i in enumerate(idx):
                    final[i] = store["final"][r].float().cpu()
    finally:
        for h in handles:
            h.remove()
    return {k: torch.stack(v) for k, v in out.items()}, torch.stack(final)


def direct_effects(model, tokenizer, prompts, contrib, final, chat=True):
    """One row per prompt and component: direct logit effect of the removed vector on the target's first token,
    on the other concepts' targets (mean), on associate, coordinate and superordinate, and on the vocabulary mean;
    `specific` = own target minus other targets. Positive = the component pushed that token up."""
    import torch

    with torch.no_grad():
        return _direct_effects(model, tokenizer, prompts, contrib, final, chat)


def _direct_effects(model, tokenizer, prompts, contrib, final, chat):
    import torch

    norm = model.model.norm
    gamma = norm.weight.float().cpu()
    eps = float(getattr(norm, "variance_epsilon", 1e-6))
    W = model.get_output_embeddings().weight.float().cpu()
    rms = torch.sqrt(final.pow(2).mean(dim=-1, keepdim=True) + eps)
    tok = {r: _first_tokens(tokenizer, prompts, r, chat) for r in ("target", "associate", "coordinate", "superordinate")
           if all(r in p.candidates for p in prompts)}
    vocab_mean = W.mean(dim=0)
    rows = []
    for key, c in contrib.items():
        scaled = c * gamma / rms                             # (n, hidden)
        own = (scaled * W[tok["target"]]).sum(dim=-1)
        all_targets = scaled @ W[tok["target"]].T             # (n, n): column j = target of prompt j
        n = len(prompts)
        others = (all_targets.sum(dim=1) - all_targets.diagonal()) / max(n - 1, 1)
        for i, p in enumerate(prompts):
            row = dict(item_id=p.item_id, component=key, target=float(own[i]), other_targets=float(others[i]),
                       specific=float(own[i] - others[i]), vocab_mean=float(scaled[i] @ vocab_mean))
            for r in ("associate", "coordinate", "superordinate"):
                if r in tok:
                    row[r] = float(scaled[i] @ W[tok[r][i]])
            rows.append(row)
    return pd.DataFrame(rows)


def first_token_logits(model, tokenizer, prompts, chat=True, batch_size=8):
    """Float32 logits at the last prompt position for each prompt's own target first token and the mean over the
    other prompts' target first tokens (to compare the total effect of a lesion with its direct effect)."""
    import torch

    device = next(model.parameters()).device
    seqs = [encode_answer(tokenizer, p.text, p.candidates["target"], chat)[0] for p in prompts]
    targets = _first_tokens(tokenizer, prompts, "target", chat)
    W = model.get_output_embeddings().weight[targets].float()
    rows = [None] * len(prompts)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
            h = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state[:, -1].float()
            lg = h @ W.T                                      # (batch, n_prompts)
            for r, i in enumerate(idx):
                own = float(lg[r, i])
                rows[i] = dict(item_id=prompts[i].item_id, target_logit=own,
                               other_targets_logit=float((lg[r].sum() - lg[r, i]) / max(len(prompts) - 1, 1)))
    return pd.DataFrame(rows)

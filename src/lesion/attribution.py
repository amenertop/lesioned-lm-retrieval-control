"""First-order estimates of single-expert mean-ablation effects (attribution patching; ADR 0004).

Mean-ablating expert e replaces its output o_t (before the routing weight) by its mean m_e on every token t
routed to it. To first order, a statistic S of the answer margins changes by

    dS(e) ~= sum over routed tokens t of (m_e - o_t) . dS/do_t

and one backward pass gives dS/do_t for every expert at once. Routing choices are held fixed, so the
estimate is not affected by the routing flips that make single-expert effects in Qwen3-30B-A3B hard to
reproduce across batch compositions. Candidates are then checked with real ablations (screen.py).

Statistics are linear in the per-prompt margins (logit of the correct label minus the wrong one): S =
sum_p w_p margin_p, with the weights from statistic_weights(). They follow lesion.damage.effects, where
damage h = intact margin - lesioned margin, averaged over the two option orders.
"""
import numpy as np
import pandas as pd

from . import damage, scoring
from .components import Component, moe_block, moe_layers, n_experts
from .prefix_scoring import label_logits

STATISTICS = ("control_effect", "knowledge_effect", "semantic_damage", "string_damage")


def statistic_weights(prompts, items, freq_median):
    """{statistic: per-prompt weights v}: under a lesion, S = sum_p v_p margin_p changes by exactly that
    statistic of the damage h (lesion.damage.effects), since h = intact margin - lesioned margin."""
    meta = items.set_index("item_id")
    task = np.array([p.task for p in prompts])
    high = np.array([meta.at[p.item_id, "high_demand"] == "1" for p in prompts])
    rare = np.array([p.task == "synonym" and float(meta.at[p.item_id, "min_freq"]) < freq_median for p in prompts])
    n = len(prompts)
    out = {s: np.zeros(n) for s in STATISTICS}
    matching = [t for t in damage.MATCHING if (task == t).any()]
    for t in matching:
        # mean over items = mean over prompts, since every item has exactly two prompts
        for sel, sign in (((task == t) & high, 1.0), ((task == t) & ~high, -1.0)):
            if sel.any():
                out["control_effect"][sel] += sign / sel.sum() / len(matching)
    syn = task == "synonym"
    for sel, sign in ((syn & rare, 1.0), (syn & ~rare, -1.0)):
        if sel.any():
            out["knowledge_effect"][sel] += sign / sel.sum()
    sem = np.isin(task, damage.SEMANTIC)
    out["semantic_damage"][sem] = 1.0 / max(sem.sum(), 1)
    string = task == "string"
    out["string_damage"][string] = 1.0 / max(string.sum(), 1)
    return {s: -w for s, w in out.items()}


def expert_attribution(model, tokenizer, prompts, chat, means, weights, batch_size=4):
    """component_attribution for experts only."""
    return component_attribution(model, tokenizer, prompts, chat, means, weights, batch_size, kinds=("expert",))


def component_attribution(model, tokenizer, prompts, chat, means, weights, batch_size=4, kinds=("expert", "head")):
    """DataFrame with one row per component: estimated value of each statistic of the damage when that
    component alone is mean-ablated (same units and signs as lesion.damage.effects), for two-choice label
    prompts (the per-prompt score is the correct-minus-wrong label margin).

    Heads: mean ablation replaces the head's slice of the o_proj input by its mean (lesion.hooks), so the
    estimate is sum over tokens of (m_h - x_t,h) . dS/dx_t,h on that slice.
    """
    import torch

    lab_ids = scoring.label_token_ids(tokenizer, chat)
    token_ids = [lab_ids[l] for l in scoring.LABELS]
    seqs = [scoring.encode(tokenizer, p.text, chat) for p in prompts]
    correct_is_a = np.array([p.correct_label == "A" for p in prompts])

    def margin(ids, mask, pos, idx):
        logits = label_logits(model, token_ids, input_ids=ids, attention_mask=mask, position_ids=pos)
        sign = torch.tensor(np.where(correct_is_a[idx], 1.0, -1.0), device=ids.device, dtype=torch.float32)
        return sign * (logits[:, 0] - logits[:, 1])

    return attribute(model, seqs, tokenizer.eos_token_id or 0, margin, weights, means, batch_size, kinds)


def naming_attribution(model, tokenizer, prompts, chat, means, weights, batch_size=4, kinds=("expert", "head"), word_only=False):
    """component_attribution for the definition-naming test: the per-prompt score is the float32
    log-probability of the answer (lesion.naming.answer_logprobs); statistics follow lesion.naming.effects."""
    import torch

    from . import m3_scoring, naming

    pairs = [(m3_scoring.encode_answer(tokenizer, p.text, p.target, chat) if word_only
              else naming.encode_pair(tokenizer, p, chat)) for p in prompts]
    seqs = [a + b for a, b in pairs]
    n_answer = [len(b) for _, b in pairs]
    weight = model.get_output_embeddings().weight.detach().float()

    def answer_logprob(ids, mask, pos, idx):
        hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
        width = ids.shape[1]
        out = []
        for row, i in enumerate(idx):
            n = n_answer[i]
            logp = torch.log_softmax(hidden[row, width - n - 1:width - 1].float() @ weight.T, dim=-1)
            out.append(logp.gather(1, ids[row, width - n:, None]).sum())
        return torch.stack(out)

    return attribute(model, seqs, tokenizer.eos_token_id or 0, answer_logprob, weights, means, batch_size, kinds)


def attribute(model, seqs, pad_id, objective, weights, means, batch_size=4, kinds=("expert", "head")):
    """Core: one forward pass and one backward pass per statistic and batch.

    objective(ids, mask, positions, idx) returns the per-sequence score (with gradients) for the batch idx;
    each statistic is S = sum_i weights[s][i] * score_i.
    """
    import torch

    layers = model.model.layers
    expert_layers = moe_layers(model.config)
    n_exp = n_experts(model.config)
    names = list(weights)
    n_heads = model.config.num_attention_heads
    totals = {s: torch.zeros((len(layers), max(n_exp, 1)), dtype=torch.float64) for s in names}
    head_totals = {s: torch.zeros((len(layers), n_heads), dtype=torch.float64) for s in names}
    w = {s: torch.tensor(v, dtype=torch.float32) for s, v in weights.items()}
    device = next(model.parameters()).device
    use_experts, use_heads = "expert" in kinds and bool(expert_layers), "head" in kinds
    mean_of = {l: means.expert_mean[l].to(device=device, dtype=torch.float32) for l in expert_layers} if use_experts else {}
    head_mean = means.head_mean.to(device=device, dtype=torch.float32) if use_heads else None
    head_inputs = []

    for p in model.parameters():
        p.requires_grad_(False)
    captured = []

    def expert_hook(layer, e):
        def hook(_module, _inputs, output):
            if output.shape[0]:
                output.retain_grad()
                captured.append((layer, e, output))
        return hook

    def embed_hook(_module, _inputs, output):
        return output.requires_grad_(True)       # builds the graph through every later computation

    def head_hook(layer):
        def hook(_module, inputs):
            inputs[0].retain_grad()
            head_inputs.append((layer, inputs[0]))
        return hook

    handles = [model.model.embed_tokens.register_forward_hook(embed_hook)]
    if use_experts:
        for l in expert_layers:
            for e, module in enumerate(moe_block(layers[l]).experts):
                handles.append(module.register_forward_hook(expert_hook(l, e)))
    if use_heads:
        for l, layer in enumerate(layers):
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(head_hook(l)))
    try:
        for idx in scoring._batches(seqs, batch_size):
            captured.clear()
            head_inputs.clear()
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], pad_id, device)
            with torch.enable_grad():
                score = objective(ids, mask, pos, idx)
                for k, s in enumerate(names):
                    for c in captured:
                        c[2].grad = None
                    for _, x in head_inputs:
                        x.grad = None
                    stat = (w[s][idx].to(device) * score).sum()
                    stat.backward(retain_graph=k < len(names) - 1)
                    for layer, e, out in captured:
                        if out.grad is not None:
                            delta = mean_of[layer][e] - out.detach().float()
                            totals[s][layer, e] += float((delta * out.grad.float()).sum())
                    for layer, x in head_inputs:
                        if x.grad is not None:
                            xs = x.detach().float().reshape(*x.shape[:-1], n_heads, -1)
                            delta = head_mean[layer] - xs
                            per_head = (delta * x.grad.float().reshape(xs.shape)).sum(dim=(0, 1, 3))
                            head_totals[s][layer] += per_head.double().cpu()
            del score, captured[:], head_inputs[:]
    finally:
        for h in handles:
            h.remove()
    rows = []
    if use_experts:
        for l in expert_layers:
            for e in range(n_exp):
                rows.append(dict(component=Component("expert", l, e).key, kind="expert", layer=l, index=e,
                                 **{s: float(totals[s][l, e]) for s in names}))
    if use_heads:
        for l in range(len(layers)):
            for h in range(n_heads):
                rows.append(dict(component=Component("head", l, h).key, kind="head", layer=l, index=h,
                                 **{s: float(head_totals[s][l, h]) for s in names}))
    return pd.DataFrame(rows)

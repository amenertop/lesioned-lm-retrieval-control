"""Optogenetics-inspired interventions on the M3 battery (docs/plans/optogenetics-inspired-experiments.md §9;
user decision Q23, 2026-10-11).

- lm_loss: mean next-token negative log-probability on neutral text, to check that amplifying a component
  (hooks.ablate with alpha > 1) does not simply break the model;
- donor_pairs: for every recipient concept a near donor (same category, else same domain) and a far donor (another
  domain), fixed by a seed before any run;
- implant: at the last prompt position, swap each component's own removed vector (lesion.m3_dla.removed_vector:
  what mean ablation takes away there, computed from the module's input in this very forward pass) for a donor
  prompt's recorded one (lesion.m3_dla.contributions with model_dtype=True: the model's own dtype path, agent2
  review 2026-10-12). The swap is one float32 sum per module, cast back once, so a self-implant (donor =
  recipient, same batches) adds exactly zero and restores the intact model bit for bit.
  Version 2 (2026-10-12, after GPU session 9 failed the fidelity check with hooks.ablate in float16 followed by a
  float32 add-back; agent2 review the same day);
- implant_scores: one row per (recipient, donor) pair, the first-token log-probabilities of both names at the
  recipient's last prompt position;
- router_selection: how often given experts are among the router's top k on given prompts.
"""
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import m3_dla, scoring
from .m3_scoring import encode_answer


def lm_loss(model, tokenizer, texts, max_tokens=256, batch_size=8):
    """Mean next-token negative log-probability (nats per token) over the texts, each cut to max_tokens; every
    predicted real token counts once."""
    import torch

    device = next(model.parameters()).device
    seqs = [tokenizer(t, add_special_tokens=False).input_ids[:max_tokens] for t in texts]
    seqs = [s for s in seqs if len(s) > 1]
    total, count = 0.0, 0
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
            for r in range(len(idx)):
                keep = mask[r].bool()
                lg = logits[r, keep][:-1].float()
                tgt = ids[r, keep][1:]
                total += float(torch.nn.functional.cross_entropy(lg, tgt, reduction="sum"))
                count += int(tgt.numel())
    return total / count


def donor_pairs(items, seed=0):
    """One row per recipient: a near donor (same category if any, else same domain) and a far donor (another
    domain). items: rows with item_id, target, category, domain. Donors never share the recipient's target. A
    recipient alone in its domain gets no near donor (near = "", near_kind = "none") rather than a forced one."""
    rng = np.random.default_rng(seed)
    rows = items.reset_index(drop=True)
    out = []
    for _, b in rows.iterrows():
        others = rows[(rows.item_id != b.item_id) & (rows.target != b.target)]
        same_cat = others[others.category == b.category]
        near_pool, near_kind = (same_cat, "category") if len(same_cat) else (others[others.domain == b.domain], "domain")
        far_pool = others[others.domain != b.domain]
        if not len(far_pool):
            raise ValueError(f"no far donor for {b.item_id}")
        near = near_pool.item_id.iloc[rng.integers(len(near_pool))] if len(near_pool) else ""
        out.append(dict(recipient=b.item_id, near=near, near_kind=near_kind if near else "none",
                        far=far_pool.item_id.iloc[rng.integers(len(far_pool))]))
    return pd.DataFrame(out)


@contextmanager
def implant(model, means, comps, donor=None):
    """At the last position of every row, replace each component's own removed vector by donor[c.key] (batch,
    hidden), or by nothing when donor is None (mean ablation there, computed in float32). Hooks the output of the
    mixture-of-experts block (experts) or o_proj (heads); per module the change is
    output + sum over its components of (donor - own), summed in float32 in component order and cast back once.
    Only the first element of a tuple output is changed; routing is never altered."""
    import torch

    layers = model.model.layers
    by_module = {}
    for c in comps:
        if c.kind not in ("expert", "head"):
            raise ValueError(f"the implant is defined for experts and heads, not {c.kind}")
        module = layers[c.layer].mlp if c.kind == "expert" else layers[c.layer].self_attn.o_proj
        by_module.setdefault(module, []).append(c)
    handles = []

    def make_hook(module_comps):
        def hook(_module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            x = inputs[0]
            if value.ndim != 3 or x.ndim != 3 or value.shape[:2] != x.shape[:2]:
                raise RuntimeError(f"unexpected shapes: input {tuple(x.shape)}, output {tuple(value.shape)}")
            delta = torch.zeros(value.shape[0], value.shape[-1], dtype=torch.float32, device=value.device)
            for c in module_comps:
                delta = delta - m3_dla.removed_vector(model, means, c, x[:, -1], model_dtype=True)
                if donor is not None:
                    delta = delta + donor[c.key].to(device=value.device, dtype=torch.float32)
            changed = value.clone()
            changed[:, -1] = (value[:, -1].float() + delta).to(value.dtype)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        return hook

    try:
        for module, module_comps in by_module.items():
            handles.append(module.register_forward_hook(make_hook(module_comps)))
        yield
    finally:
        for h in handles:
            h.remove()


def implant_scores(model, tokenizer, means, comps, prompts, vectors, pairs, chat=True, batch_size=8, add=True):
    """First-token readout at the recipient's last prompt position with `comps`' own removed vectors swapped there
    for the donor's (implant).

    prompts: NamingPrompts indexed by item_id (dict); vectors: {component key: tensor (n, hidden)} recorded by
    lesion.m3_dla.contributions on `order` = list(prompts) (same row order); pairs: list of (recipient, donor)
    item ids (donor == recipient restores the intact model). add=False only ablates (the lesion-only control).
    Returns one row per pair: log-probabilities of the donor's and the recipient's first answer tokens, the donor
    token's rank and the top token.
    """
    import torch

    device = next(model.parameters()).device
    order = list(prompts)
    seqs = [scoring.encode(tokenizer, prompts[b].text, chat) for b, _ in pairs]
    first = lambda b, word: encode_answer(tokenizer, prompts[b].text, word, chat)[1][0]
    weight = model.get_output_embeddings().weight.float()
    rows = []
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
            batch_vec = {c.key: vectors[c.key][[order.index(pairs[i][1]) for i in idx]] for c in comps} if add else None
            with implant(model, means, comps, batch_vec):
                hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
            logp = torch.log_softmax(hidden[:, -1].float() @ weight.T, dim=-1).cpu()
            for r, i in enumerate(idx):
                b, a = pairs[i]
                ta = first(b, prompts[a].candidates["target"])
                tb = first(b, prompts[b].candidates["target"])
                rows.append(_row(tokenizer, logp[r], b, a, ta, tb))
    return pd.DataFrame(rows)


def _row(tokenizer, logp, recipient, donor, ta, tb, k=5):
    """One readout row: the donor's and the recipient's first-token log-probabilities, the donor token's rank, and
    the top-k tokens (decoded, joined by " | ") at the recipient's last prompt position."""
    top = logp.topk(k).indices.tolist()
    return dict(recipient=recipient, donor=donor, lp_donor=float(logp[ta]), lp_recipient=float(logp[tb]),
                rank_donor=int((logp > logp[ta]).sum()), same_first_token=ta == tb,
                top_token=tokenizer.decode([top[0]]), top5=" | ".join(tokenizer.decode([x]) for x in top))


def intact_first_tokens(model, tokenizer, prompts, pairs, chat=True, batch_size=8):
    """The same readout as implant_scores without any intervention (the reference for every implant condition)."""
    import torch

    device = next(model.parameters()).device
    seqs = [scoring.encode(tokenizer, prompts[b].text, chat) for b, _ in pairs]
    weight = model.get_output_embeddings().weight.float()
    rows = []
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
            hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
            logp = torch.log_softmax(hidden[:, -1].float() @ weight.T, dim=-1).cpu()
            for r, i in enumerate(idx):
                b, a = pairs[i]
                ta = encode_answer(tokenizer, prompts[b].text, prompts[a].candidates["target"], chat)[1][0]
                tb = encode_answer(tokenizer, prompts[b].text, prompts[b].candidates["target"], chat)[1][0]
                rows.append(_row(tokenizer, logp[r], b, a, ta, tb))
    return pd.DataFrame(rows)


def router_selection(model, tokenizer, prompts, experts, chat=True, batch_size=8):
    """One row per prompt and expert: whether the router puts the expert among its top k at the last prompt
    position, and at how many of the prompt's real positions it does (n_selected of n_positions). prompts: objects
    with item_id and text; experts: expert Components. The same routing path as lesion.m3_dla.routing_weight."""
    import torch

    device = next(model.parameters()).device
    layers = model.model.layers
    seqs = [scoring.encode(tokenizer, p.text, chat) for p in prompts]
    store = {}
    need = sorted({c.layer for c in experts})
    handles = [layers[L].mlp.register_forward_pre_hook(lambda _m, inp, L=L: store.__setitem__(L, inp[0].detach()))
               for L in need]
    rows = []
    try:
        with torch.no_grad():
            for idx in scoring._batches(seqs, batch_size):
                ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
                store.clear()
                model.model(input_ids=ids, attention_mask=mask, position_ids=pos)
                for c in experts:
                    x = store[c.layer]
                    flat = x.reshape(-1, x.shape[-1])
                    chosen = (m3_dla.routing_weight(layers[c.layer].mlp, flat, c.index) > 0).reshape(x.shape[:2])
                    for r, i in enumerate(idx):
                        real = mask[r].bool()
                        rows.append(dict(item_id=prompts[i].item_id, component=c.key, selected_last=bool(chosen[r, -1]),
                                         n_selected=int(chosen[r][real].sum()), n_positions=int(real.sum())))
    finally:
        for h in handles:
            h.remove()
    return pd.DataFrame(rows)

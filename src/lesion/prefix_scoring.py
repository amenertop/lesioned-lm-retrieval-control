"""Label scoring that computes each shared prompt prefix once (ADR 0003; same results as lesion.scoring).

All prompts of one task and option order share the instruction and the demos; only the test block at the
end differs (on the M1 screening split: 133-163 shared tokens, 18-24 own tokens with the Qwen tokenizer).
The shared prefix of every group is run once and cached; then the own tokens of all prompts are run in
batches on top of their group's cached prefix. With the same tokens, positions and attention, this gives
the same next-token logits as scoring each prompt in full, while doing about a seventh of the work.

Ablations work unchanged: the caller wraps the call in an ablation context, so the prefix and the own
tokens both pass through the lesioned model, exactly as in a full forward pass.

The two label logits are computed in float32 from the final hidden state (the rest of the model keeps its
own precision). In bf16 the output layer rounds logits to steps of 0.125-0.25 at the sizes Qwen produces
(median |logit| about 45), which would hide most single-component effects.

score_label_prompts(model, tokenizer, prompts, chat, batch_size) returns the same table as
lesion.scoring.score_label_prompts. This module is not used for G1; the frozen G1 runs use lesion.scoring.
"""
import pandas as pd

from . import scoring


def _common_prefix(seqs):
    first = seqs[0]
    n = min(len(s) for s in seqs)
    i = 0
    while i < n and all(s[i] == first[i] for s in seqs):
        i += 1
    return i


def split_prefixes(seqs, keys):
    """For each group key: (shared prefix tokens, [(index, own tokens), ...]).

    Every prompt keeps at least one own token, so its answer position is always computed in the suffix pass.
    """
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    out = {}
    for k, idx in groups.items():
        n = min(_common_prefix([seqs[i] for i in idx]), min(len(seqs[i]) for i in idx) - 1)
        out[k] = (seqs[idx[0]][:n], [(i, seqs[i][n:]) for i in idx])
    return out


def label_logits(model, token_ids, **inputs):
    """float32 logits of token_ids at the last position: the base model runs as usual, the output layer in float32."""
    hidden = model.model(**inputs).last_hidden_state[:, -1]
    weight = model.get_output_embeddings().weight[token_ids]
    return hidden.float() @ weight.float().T


def full_next_token_logits(model, seqs, token_ids, batch_size=32, pad_id=0):
    """Like lesion.scoring.next_token_logits (each prompt in full), with the float32 output layer."""
    import torch

    device = next(model.parameters()).device
    out = torch.empty((len(seqs), len(token_ids)), dtype=torch.float32)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], pad_id, device)
            out[idx] = label_logits(model, token_ids, input_ids=ids, attention_mask=mask, position_ids=pos).cpu()
    return out


def _prefix_cache(model, prefixes, pad_id, device):
    """Run all group prefixes in one left-padded batch; returns per-layer (keys, values), mask and lengths."""
    import torch

    ids, mask, pos = scoring._pad_left(prefixes, pad_id, device)
    with torch.no_grad():
        out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
    return kv, mask, torch.tensor([len(p) for p in prefixes], device=device)


def next_token_logits(model, seqs, keys, token_ids, batch_size=64, pad_id=0):
    """Logits of `token_ids` at the position after each sequence (n_seqs x n_tokens, float32), like
    lesion.scoring.next_token_logits, with the shared prefix of each key group computed once."""
    import torch
    from transformers import DynamicCache

    device = next(model.parameters()).device
    groups = split_prefixes(seqs, keys)
    names = list(groups)
    kv, prefix_mask, prefix_len = _prefix_cache(model, [groups[k][0] for k in names], pad_id, device)
    jobs = [(g, i, own) for g, k in enumerate(names) for i, own in groups[k][1]]
    jobs.sort(key=lambda j: len(j[2]))
    out = torch.empty((len(seqs), len(token_ids)), dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            rows = torch.tensor([g for g, _, _ in batch], device=device)
            own_ids, own_mask, _ = scoring._pad_left([own for _, _, own in batch], pad_id, device)
            width = own_ids.shape[1]
            # Own tokens are left-padded so every answer position is the last column; pads are masked out.
            lengths = own_mask.sum(-1, keepdim=True)
            offset = torch.arange(width, device=device) - (width - lengths)
            positions = (prefix_len[rows, None] + offset).clamp(min=0)
            cache = DynamicCache()
            for layer, (k, v) in enumerate(kv):
                cache.update(k.index_select(0, rows), v.index_select(0, rows), layer)
            mask = torch.cat([prefix_mask.index_select(0, rows), own_mask], dim=1)
            logits = label_logits(model, token_ids, input_ids=own_ids, attention_mask=mask, position_ids=positions,
                                  past_key_values=cache, use_cache=True)
            out[[i for _, i, _ in batch]] = logits.cpu()
    return out


def score_label_prompts(model, tokenizer, prompts, chat, batch_size=64):
    """Same output as lesion.scoring.score_label_prompts; prompts are grouped by (task, option order)."""
    lab_ids = scoring.label_token_ids(tokenizer, chat)
    seqs = [scoring.encode(tokenizer, p.text, chat) for p in prompts]
    logits = next_token_logits(model, seqs, [(p.task, p.order) for p in prompts],
                               [lab_ids[l] for l in scoring.LABELS], batch_size, tokenizer.eos_token_id or 0)
    return _rows(prompts, logits)


def score_label_prompts_full(model, tokenizer, prompts, chat, batch_size=32):
    """Each prompt in full, float32 output layer: the reference the prefix-shared scorer is checked against."""
    lab_ids = scoring.label_token_ids(tokenizer, chat)
    seqs = [scoring.encode(tokenizer, p.text, chat) for p in prompts]
    logits = full_next_token_logits(model, seqs, [lab_ids[l] for l in scoring.LABELS], batch_size,
                                    tokenizer.eos_token_id or 0)
    return _rows(prompts, logits)


def _rows(prompts, logits):
    rows = []
    for p, (la, lb) in zip(prompts, logits.tolist() if hasattr(logits, "tolist") else logits):
        wrong = "B" if p.correct_label == "A" else "A"
        by_label = {"A": la, "B": lb}
        rows.append(dict(item_id=p.item_id, task=p.task, order=p.order, correct_label=p.correct_label,
                         logit_A=la, logit_B=lb, logit_correct=by_label[p.correct_label], logit_wrong=by_label[wrong],
                         choice="A" if la > lb else "B", chosen_option=p.label_to_option["A" if la > lb else "B"],
                         option_A=p.label_to_option["A"]))
    return pd.DataFrame(rows)


def averaged(score_prompts, batch_sizes):
    """A scorer that runs score_prompts once per batch size and averages the two label logits.

    In Qwen3-30B-A3B the margins move by about 0.4-0.6 logit when only the batch composition changes
    (routing choices flip on rounding differences; ADR 0004). Averaging over several compositions shrinks
    that noise; intact and lesioned runs use the same compositions.
    """
    import numpy as np

    def score(model, tokenizer, prompts, chat, batch_size=None):
        tables = [score_prompts(model, tokenizer, prompts, chat, bs) for bs in batch_sizes]
        la = np.mean([t.logit_A.to_numpy() for t in tables], axis=0)
        lb = np.mean([t.logit_B.to_numpy() for t in tables], axis=0)
        return _rows(prompts, np.column_stack([la, lb]).tolist())
    return score

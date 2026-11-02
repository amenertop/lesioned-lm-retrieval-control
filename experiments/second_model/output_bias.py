"""Output-layer bias for Phi-3.5-MoE scoring.

Phi-3.5-MoE's output layer has a bias (config lm_head_bias: true; 32,064 values, mean 0.010, SD 0.019, range -0.065
to 0.115 logits). The shared scorers project the final hidden state with the output weight only, which is exact for
Qwen and OLMoE (no bias) but leaves Phi's bias out. The frozen scorer modules stay byte-identical (analysis protocols
and the figdata manifest record their hashes), so this module patches, inside a context, the three scorers the Phi
battery uses: lesion.prefix_scoring.label_logits (label logits), lesion.m3_battery.candidate_scores (candidate
log-probabilities) and lesion.naming.answer_logprobs (naming-test log-probabilities). The patched versions are the
originals with the bias added before the softmax; with a zero or absent bias they return the same numbers
(tests/test_output_bias.py). Greedy answers and the neutral-text loss run the model's own forward and already include
the bias.
"""
from contextlib import contextmanager

import pandas as pd

from lesion import m3_battery, naming, prefix_scoring, scoring


def _bias(model):
    b = getattr(model.get_output_embeddings(), "bias", None)
    return None if b is None else b.detach().float()


def candidate_scores(model, tokenizer, prompts, chat=True, batch_size=16, region=None):
    """lesion.m3_battery.candidate_scores with the output bias."""
    import torch

    from lesion.m3_scoring import encode_answer

    device = next(model.parameters()).device
    jobs = [(i, role, word, *encode_answer(tokenizer, p.text, word, chat))
            for i, p in enumerate(prompts) for role, word in p.candidates.items()]
    seqs = [pre + ans for *_, pre, ans in jobs]
    segments = ([m3_battery.prompt_segments(tokenizer, p.text, chat) for p in prompts]
                if m3_battery.is_segments(region) else None)
    weight = model.get_output_embeddings().weight.float()
    bias = _bias(model)
    out = [None] * len(jobs)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[j] for j in idx], tokenizer.eos_token_id or 0, device)
            width = ids.shape[1]
            bounds = [segments[jobs[j][0]] for j in idx] if m3_battery.is_segments(region) else None
            pm = None if region is None else m3_battery.answer_mask([len(seqs[j]) for j in idx],
                                                                     [len(jobs[j][4]) for j in idx], width, region, bounds)
            with m3_battery._region_context(model, pm):
                hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
            for row, j in enumerate(idx):
                i, role, word, _, ans = jobs[j]
                n = len(ans)
                logits = hidden[row, width - n - 1:width - 1].float() @ weight.T
                if bias is not None:
                    logits = logits + bias
                logp = torch.log_softmax(logits, dim=-1)
                answer = ids[row, width - n:]
                first = logp[0]
                out[j] = dict(item_id=prompts[i].item_id, condition=prompts[i].condition, role=role, word=word,
                              logprob=float(logp.gather(1, answer[:, None]).sum()), n_tokens=n,
                              first_token_rank=int((first > first[answer[0]]).sum()))
    return pd.DataFrame(out)


def answer_logprobs(model, tokenizer, prompts, chat, batch_size=16):
    """lesion.naming.answer_logprobs with the output bias."""
    import torch

    device = next(model.parameters()).device
    pairs = [naming.encode_pair(tokenizer, p, chat) for p in prompts]
    seqs = [p + a for p, a in pairs]
    weight = model.get_output_embeddings().weight.float()
    bias = _bias(model)
    out = [None] * len(prompts)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
            hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
            width = ids.shape[1]
            for row, i in enumerate(idx):
                n = len(pairs[i][1])
                logits = hidden[row, width - n - 1:width - 1].float() @ weight.T
                if bias is not None:
                    logits = logits + bias
                logp = torch.log_softmax(logits, dim=-1)
                answer = ids[row, width - n:]
                first = logp[0]
                out[i] = dict(item_id=prompts[i].item_id, target=prompts[i].target,
                              logprob=float(logp.gather(1, answer[:, None]).sum()), n_tokens=n - 1,
                              first_token_rank=int((first > first[answer[0]]).sum()))
    return pd.DataFrame(out)


@contextmanager
def with_output_bias(model):
    """Patch the three scorers to add the model's output bias; restored on exit. A model without a bias is left alone."""
    if _bias(model) is None:
        yield False
        return
    original = (prefix_scoring.label_logits, m3_battery.candidate_scores, naming.answer_logprobs)
    bias = _bias(model)

    def label_logits(model_, token_ids, **inputs):
        logits = original[0](model_, token_ids, **inputs)
        return logits + bias.to(logits.device)[list(token_ids)]

    prefix_scoring.label_logits = label_logits
    m3_battery.candidate_scores = candidate_scores
    naming.answer_logprobs = answer_logprobs
    try:
        yield True
    finally:
        prefix_scoring.label_logits, m3_battery.candidate_scores, naming.answer_logprobs = original

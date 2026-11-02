"""Scores that carry gradients, for training lesion.gates (P14, docs/plans/p14-learned-gates.md; agent1's module).

Each one computes what an existing no-gradient scorer computes, on one left-padded batch, so that a gated model can be
trained on it; with no gates applied they agree with those scorers (tests/test_gate_objective.py):
- naming_logprobs: the summed log-probability of the answer (word plus end marker), as lesion.naming.answer_logprobs;
- label_margins: the correct label's logit minus the other label's at the answer position, as the M1 scorers'
  logit_correct - logit_wrong (lesion.probes.label_scores_in_positions with no lesion);
- lm_loss: mean next-token negative log-probability over the real tokens, as lesion.probes.lm_loss_region.
Only the positions needed are projected onto the vocabulary (the vocabulary is large), in float32 as the scorers do.
"""
import torch

from . import scoring


def _device(model):
    return next(model.parameters()).device


def naming_logprobs(model, pairs, pad_id=0, weight=None):
    """(n,) float32 summed answer log-probabilities; pairs: [(prompt ids, answer ids)] (lesion.naming.encode_pair).
    weight: the output embedding as float32, passed in to avoid one copy per call."""
    seqs = [list(p) + list(a) for p, a in pairs]
    ids, mask, pos = scoring._pad_left(seqs, pad_id, _device(model))
    hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
    w = model.get_output_embeddings().weight.float() if weight is None else weight
    width = ids.shape[1]
    rows, cols, targets, owner = [], [], [], []
    for r, (_, a) in enumerate(pairs):
        n = len(a)
        rows += [r] * n
        cols += list(range(width - n - 1, width - 1))            # the positions that predict each answer token
        targets += list(range(width - n, width))
        owner += [r] * n
    rows_t = torch.tensor(rows, device=ids.device)
    h = hidden[rows_t, torch.tensor(cols, device=ids.device)].float()
    logp = torch.log_softmax(h @ w.T, dim=-1)
    picked = logp.gather(1, ids[rows_t, torch.tensor(targets, device=ids.device)][:, None])[:, 0]
    return torch.zeros(len(pairs), device=ids.device, dtype=picked.dtype).index_add(0, rows_t, picked)


def label_margins(model, seqs, correct_ids, wrong_ids, pad_id=0, weight=None):
    """(n,) float32: logit of correct_ids[i] minus logit of wrong_ids[i] after seqs[i]."""
    ids, mask, pos = scoring._pad_left(seqs, pad_id, _device(model))
    hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state[:, -1].float()
    w = model.get_output_embeddings().weight.float() if weight is None else weight
    c = torch.as_tensor(correct_ids, device=ids.device)
    o = torch.as_tensor(wrong_ids, device=ids.device)
    return (hidden * (w[c] - w[o])).sum(-1)


def lm_loss(model, seqs, pad_id=0):
    """Mean next-token negative log-probability (nats per token) over the real tokens of seqs, one batch."""
    seqs = [s for s in seqs if len(s) > 1]
    ids, mask, pos = scoring._pad_left(seqs, pad_id, _device(model))
    logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits
    keep = mask.bool()
    shift = keep[:, 1:] & keep[:, :-1]
    lg = logits[:, :-1][shift].float()
    tgt = ids[:, 1:][shift]
    return torch.nn.functional.cross_entropy(lg, tgt, reduction="mean")

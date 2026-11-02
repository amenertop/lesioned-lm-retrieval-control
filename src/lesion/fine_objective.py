"""A shared-concept hub objective. Reuses P14 gates and differentiable scoring, with no weight training.

Three aligned scores per concept: target-word production and two independently formatted comprehension tasks.
Frequency slopes, co-loss and residual cross-task correlation enter the search objective. Evaluation remains
on separate concepts, with all three scores saved, and is never inferred from this training objective.
"""
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import fine_semantics, gate_objective as go, gates as gates_mod, m3_scoring, scoring

TASKS = ("naming", "definition", "synonym")


def resident_means(model, means):
    """Copy fixed means once. The shared gate hooks then need no host-to-device copies per expert call."""
    parameter = next(model.parameters())
    return SimpleNamespace(expert_mean=means.expert_mean.to(device=parameter.device, dtype=parameter.dtype),
                           head_mean=means.head_mean.to(device=parameter.device, dtype=parameter.dtype))


def check_resident_means(model, means, data, components, batch=16, checkpointing=False):
    """One warmup and one paired score/gradient timing. No optimiser or model-weight updates."""
    device = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval()
    prepared = resident_means(model, means)
    results, values = {}, {}
    try:
        for name, fixed in [('warmup', means), ('host', means), ('resident', prepared)]:
            g = gates_mod.Gates(components).to(device)
            g.set_values(torch.linspace(.1, .9, len(components), device=device))
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with gates_mod.gated(model, g, fixed):
                scores = data.score(model, range(min(batch, len(data.ids))))
                scores.square().mean().backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            seconds = time.monotonic() - started
            results[name] = dict(seconds=seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else None)
            values[name] = (scores.detach().cpu(), g.logits.grad.detach().cpu())
            del scores, g
    finally:
        if checkpointing:
            model.gradient_checkpointing_disable()
        model.eval()
    same = [torch.equal(a, b) for a, b in zip(values['host'], values['resident'])]
    finite = all(torch.isfinite(x).all().item() for x in values['host']+values['resident'])
    return dict(scores_identical=same[0], gradients_identical=same[1], finite=finite,
                passed=all(same) and finite, measurements=results,
                speedup=results['host']['seconds']/results['resident']['seconds'],
                checkpointing=checkpointing, concepts=min(batch, len(data.ids)))


def hub_terms(damage, frequency, n_tokens, intact, mode="hub"):
    """damage and intact: (concept, task), in training-baseline SD units. All fits retain gradients in damage."""
    device, dtype = damage.device, damage.dtype
    frequency = torch.as_tensor(frequency, device=device, dtype=dtype)
    n_tokens = torch.as_tensor(n_tokens, device=device, dtype=dtype)
    z = lambda x: (x - x.mean(0)) / x.std(0, unbiased=False).clamp_min(0.1)
    x = torch.stack([torch.ones_like(frequency), -z(frequency), z(n_tokens)], dim=1)
    coef = torch.linalg.pinv(x) @ damage
    residuals = []
    for j in range(damage.shape[1]):
        nuisance = torch.cat([x, z(intact[:, j])[:, None]], dim=1)
        residuals.append(damage[:, j] - nuisance @ (torch.linalg.pinv(nuisance) @ damage[:, j]))
    residual = torch.stack(residuals, dim=1)
    correlations = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        u, v = residual[:, a], residual[:, b]
        correlations.append((u * v).mean() / torch.sqrt(u.square().mean() * v.square().mean() + 0.01))
    if mode == "hub":
        gradient = coef[1].mean()
        shared = damage.min(dim=1).values.mean()
        consistency = torch.stack(correlations).mean()
    elif mode == "comprehension":
        gradient = coef[1, 1:].mean()
        shared = damage[:, 1:].min(dim=1).values.mean()
        consistency = correlations[-1]
    else:
        raise ValueError("mode must be hub or comprehension")
    return dict(frequency=gradient, shared=shared, consistency=consistency, naming_damage=damage[:, 0].mean())


class FineData:
    """Encode once, align all tasks by item ID, and fix scales from intact screening scores."""

    def __init__(self, items, tokenizer, chat=True, naming_readout="target"):
        if naming_readout not in ("target", "accepted_vs_coordinate"):
            raise ValueError("unknown naming objective")
        self.naming_readout = naming_readout
        self.ids = items.item_id.tolist()
        self.pad = tokenizer.eos_token_id or 0
        prompts = fine_semantics.naming_prompts(items)
        self.pairs = [m3_scoring.encode_answer(tokenizer, p.text, p.candidates["target"], chat)
                      for p in prompts]
        self.alternatives = {}
        if naming_readout == "accepted_vs_coordinate":
            self.alternatives = {role: [m3_scoring.encode_answer(tokenizer, p.text, p.candidates[role], chat)
                                       for p in prompts] for role in ("synonym", "coordinate")}
        self.copy_pairs = [m3_scoring.encode_answer(tokenizer, p.text, p.candidates["target"], chat)
                           for p in fine_semantics.naming_prompts(items, copy=True)]
        self.n_tokens = np.array([len(a) for _, a in self.pairs])
        self.frequency = items.frequency.to_numpy(float)
        labels = [p for p in fine_semantics.label_prompts(items) if p.fmt in TASKS]
        label_ids = scoring.label_token_ids(tokenizer, chat)
        self.labels = {}
        for item in self.ids:
            for task in TASKS[1:]:
                ps = [p for p in labels if p.item_id == item and p.fmt == task]
                self.labels[item, task] = [(scoring.encode(tokenizer, p.text, chat), label_ids[p.correct],
                                            label_ids["B" if p.correct == "A" else "A"]) for p in ps]

    def score(self, model, indices):
        indices = list(indices)
        naming = go.naming_logprobs(model, [self.pairs[i] for i in indices], self.pad)
        if self.alternatives:
            synonym, coordinate = [go.naming_logprobs(model, [pairs[i] for i in indices], self.pad)
                                   for pairs in self.alternatives.values()]
            naming = torch.logaddexp(naming, synonym) - coordinate
        result = [naming]
        for task in TASKS[1:]:
            rows = [p for i in indices for p in self.labels[self.ids[i], task]]
            seq, correct, wrong = zip(*rows)
            margin = go.label_margins(model, seq, correct, wrong, self.pad).reshape(len(indices), 2).mean(1)
            result.append(margin)
        return torch.stack(result, dim=1)

    def baseline(self, model, batch=8):
        with torch.no_grad():
            self.intact = torch.cat([self.score(model, range(i, min(i + batch, len(self.ids)))).detach().cpu()
                                     for i in range(0, len(self.ids), batch)])
            self.copy = torch.cat([go.naming_logprobs(model, self.copy_pairs[i:i + batch], self.pad).cpu()
                                   for i in range(0, len(self.ids), batch)])
        self.scales = self.intact.std(0, unbiased=False).clamp_min(0.25)


def train(model, means, data, components, neutral_sequences, steps=200, batch=16, seed=0, mode="hub",
          sparsity=0.03, learning_rate=0.03, penalty=10.0, consistency_weight=0.25, shared_weight=0.5,
          checkpointing=False):
    if not neutral_sequences or min(map(len, neutral_sequences)) < 2:
        raise ValueError("training needs nonempty neutral passages")
    if batch < 6 or batch > len(data.ids):
        raise ValueError("use 6 or more concepts per batch, no more than the screening split")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    device = next(model.parameters()).device
    data.baseline(model, batch)
    with torch.no_grad():
        lm_base = [float(go.lm_loss(model, [s], data.pad)) for s in neutral_sequences]
    if checkpointing:
        if getattr(model.config, "attention_dropout", 0.0) != 0:
            raise ValueError("checkpointed training requires zero attention dropout")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()  # Qwen checkpoints only in training mode, as in the existing P14 runner.
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    gates = gates_mod.Gates(components).to(device)
    with torch.no_grad():
        gates.logits.add_(torch.randn_like(gates.logits) * 0.1)
    opt = torch.optim.Adam([gates.logits], lr=learning_rate)
    loss_scale = 128.0 if next(model.parameters()).dtype == torch.float16 else 1.0
    records = []
    started = time.monotonic()
    for step in range(steps):
        ix = np.sort(rng.choice(len(data.ids), batch, replace=False))
        neutral = int(rng.integers(len(neutral_sequences)))
        base = data.intact[ix].to(device)
        scale = data.scales.to(device)
        opt.zero_grad()
        gates.recording = True
        with gates_mod.gated(model, gates, means):
            damage = (base - data.score(model, ix)) / scale
            terms = hub_terms(damage, data.frequency[ix], data.n_tokens[ix], base / scale, mode)
            lm_change = go.lm_loss(model, [neutral_sequences[neutral]], data.pad) / lm_base[neutral] - 1
            copy_damage = (data.copy[ix].to(device) - go.naming_logprobs(model, [data.copy_pairs[i] for i in ix], data.pad)).mean()
            constraints = torch.relu(lm_change - 0.1).square() + torch.relu(copy_damage - 0.25).square()
            if mode == "comprehension":
                constraints = constraints + torch.relu(terms["naming_damage"] - 0.1).square()
            gain = terms["frequency"] + shared_weight * terms["shared"] + consistency_weight * terms["consistency"]
            loss = -gain + penalty * constraints + sparsity * (1 - gates.values()).sum()
            gates.recording = False
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite objective at step {step}")
            (loss * loss_scale).backward()
        if gates.logits.grad is not None:
            gates.logits.grad.div_(loss_scale)
        if gates.logits.grad is None or not torch.isfinite(gates.logits.grad).all():
            raise FloatingPointError(f"missing/non-finite gate gradient at step {step}")
        torch.nn.utils.clip_grad_norm_([gates.logits], 1.0)
        opt.step()
        values = gates.values().detach()
        records.append(dict(step=step, loss=float(loss.detach()), lm_change=float(lm_change.detach()),
                            copy_damage=float(copy_damage.detach()), seconds=time.monotonic() - started,
                            off_mass=float((1 - values).sum()), min_retained=float(values.min()),
                            n_below_half=int((values < 0.5).sum()),
                            **{k: float(v.detach()) for k, v in terms.items()}))
        if step % 10 == 0 or step == steps - 1:
            row = records[-1]
            print(f"step {step}: loss {row['loss']:.3f}, shared {row['shared']:.3f}, "
                  f"off {row['off_mass']:.1f}, below .5 {row['n_below_half']}, {row['seconds']:.0f}s", flush=True)
    values = gates.values().detach().cpu().tolist()
    return pd.DataFrame(dict(component=gates.keys(), retained=values, routed_tokens=gates.routed.tolist())), pd.DataFrame(records)

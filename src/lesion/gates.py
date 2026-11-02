"""Learnable per-component gates for P14 (docs/plans/p14-learned-gates.md; agent1's module).

Each chosen component's output becomes mean + g * (output - mean), with g = sigmoid(theta) in (0, 1), the same formula
as lesion.hooks.ablate's mean ablation at alpha = g: g = 0 is the mean ablation, g = 1 the intact model. Unlike
hooks.ablate, every component has its own g, and g carries gradients, so the gates can be trained while the model's own
weights stay frozen.
- attention heads: the head's slice of the o_proj input (all gated heads of a layer in one vectorised step);
- experts: the expert's output on the tokens routed to it, before the routing weight (as hooks.ablate).
Means come from a lesion.means.Means object (expert_mean, head_mean), as for hooks.ablate.
"""
from contextlib import contextmanager

import torch

from .components import Component, moe_block


def _head_dim(model):
    config = model.config
    return int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)


class Gates(torch.nn.Module):
    """One trainable logit per component; g = sigmoid(logit). init: the starting logit (3.0 gives g = 0.95)."""

    def __init__(self, components, init=3.0):
        super().__init__()
        self.components = [c if isinstance(c, Component) else Component.parse(c) for c in components]
        if len({c.key for c in self.components}) != len(self.components):
            raise ValueError("components must be distinct")
        if any(c.kind not in ("head", "expert") for c in self.components):
            raise ValueError("only heads and experts can be gated")
        self.logits = torch.nn.Parameter(torch.full((len(self.components),), float(init)))
        # tokens each gated expert processed while `recording` is on (heads see every token, so they are not counted);
        # a trainer turns it off for the backward pass, so that recomputation under gradient checkpointing is not counted
        self.routed = torch.zeros(len(self.components), dtype=torch.long)
        self.recording = False

    def values(self):
        return torch.sigmoid(self.logits)

    def keys(self):
        return [c.key for c in self.components]

    def set_values(self, values):
        """Set g directly (exactly 0 and 1 are allowed: their logits are -inf and +inf)."""
        v = torch.as_tensor(values, dtype=self.logits.dtype)
        with torch.no_grad():
            self.logits.copy_(torch.logit(v.clamp(0, 1)))


@contextmanager
def gated(model, gates, means):
    """Apply the gates for this context; all hooks are removed on exit (also after an error)."""
    layers = model.model.layers
    head_dim = _head_dim(model)
    by_layer_heads, experts = {}, []
    for i, c in enumerate(gates.components):
        if c.kind == "head":
            by_layer_heads.setdefault(c.layer, []).append((i, c.index))
        else:
            experts.append((i, c))
    handles = []
    try:
        for layer, members in by_layer_heads.items():
            rows = torch.tensor([i for i, _ in members])
            heads = torch.tensor([h for _, h in members])
            mean = means.head_mean[layer, heads]                           # (n_gated, head_dim)

            def head_hook(_module, inputs, rows=rows, heads=heads, mean=mean):
                x = inputs[0]
                shape = x.shape
                x = x.reshape(*shape[:-1], shape[-1] // head_dim, head_dim)
                g = gates.values()[rows].to(device=x.device, dtype=x.dtype)  # (n_gated,)
                m = mean.to(device=x.device, dtype=x.dtype)
                picked = x[..., heads.to(x.device), :]                       # (..., n_gated, head_dim)
                new = m + g[:, None] * (picked - m)
                x = x.index_copy(-2, heads.to(x.device), new)               # out of place: keeps the graph
                return (x.reshape(shape), *inputs[1:])

            handles.append(layers[layer].self_attn.o_proj.register_forward_pre_hook(head_hook))
        for i, c in experts:
            mean = means.expert_mean[c.layer, c.index]

            def expert_hook(_module, _inputs, output, i=i, mean=mean):
                value = output[0] if isinstance(output, tuple) else output
                g = gates.values()[i].to(device=value.device, dtype=value.dtype)
                m = mean.to(device=value.device, dtype=value.dtype)
                changed = m + g * (value - m)
                if gates.recording:
                    gates.routed[i] += value.reshape(-1, value.shape[-1]).shape[0]
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            handles.append(moe_block(layers[c.layer]).experts[c.index].register_forward_hook(expert_hook))
        yield gates
    finally:
        for handle in reversed(handles):
            handle.remove()

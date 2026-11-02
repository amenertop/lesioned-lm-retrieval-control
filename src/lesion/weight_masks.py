"""Temporary Bernoulli weight lesions (Roll et al. 2026, section 3.1).

Retained weights are not rescaled. Each matrix has its own reproducible CPU random stream, so the same seed
nests severities and gives the same mask when a projection is tested alone or within a family. All experts
in the selected layer are included. Routers and biases are untouched. No scoring or item selection here.
"""
from contextlib import contextmanager
import hashlib
import math

import torch

from .components import moe_block

FAMILIES = {"attention": ("q", "k", "v", "o"), "ffn": ("gate", "up", "down")}
PROJECTIONS = (*FAMILIES["attention"], *FAMILIES["ffn"])


def validate(config, layer, family, severity, seed):
    if type(layer) is not int or not 0 <= layer < config.num_hidden_layers:
        raise ValueError("layer must be a zero-based model layer")
    if family not in (*FAMILIES, *PROJECTIONS):
        raise ValueError("unknown weight family or projection")
    if not math.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("severity must lie in [0, 1]")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an integer in [0, 2**63)")


def matrices(model, layer, family):
    """Stable names and the existing Parameters. Never replace Parameters (grouped MoE keeps weight views)."""
    block = model.model.layers[layer]
    sparse = moe_block(block)
    experts = list(enumerate(sparse.experts)) if sparse is not None else [(None, block.mlp)]
    out = []
    for projection in FAMILIES.get(family, (family,)):
        targets = [("self_attn", block.self_attn)] if projection in FAMILIES["attention"] else [
            (f"mlp.experts.{i}" if i is not None else "mlp", expert) for i, expert in experts]
        for prefix, target in targets:
            name = f"layers.{layer}.{prefix}.{projection}_proj.weight"
            linear = getattr(target, f"{projection}_proj", None)
            if not isinstance(linear, torch.nn.Linear) or linear.weight.device.type == "meta":
                raise ValueError(f"unsupported or unloaded projection: {name}")
            out.append((name, linear.weight))
    return out


@contextmanager
def masked(model, layer, family, severity, seed):
    """Apply in place and restore even if scoring raises. CPU backups keep the GPU overhead to one matrix.

    For inference only: finish any backward pass before entering, and do not train inside this context.
    """
    validate(model.config, layer, family, severity, seed)
    targets = matrices(model, layer, family)  # validate every target before touching the first weight
    saved = []
    try:
        if severity:
            with torch.no_grad():
                for name, weight in targets:
                    original = weight.detach().to("cpu", copy=True)
                    saved.append((weight, original))
                    if severity == 1:
                        weight.zero_()
                    else:
                        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
                        rng = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little"))
                        keep = torch.rand(weight.shape, generator=rng) >= severity
                        weight.mul_(keep.to(weight.device))
        yield
    finally:
        with torch.no_grad():
            for weight, original in saved:
                weight.copy_(original)


def grid(config, layers=(4, 12, 20, 28, 36, 44), families=("attention", "ffn"),
         severities=(0.25, 0.5, 0.75, 1.0), seeds=(1910, 1911)):
    """The planned first-stage grid. Full zeroing has no mask variance, so it is scored once per cell."""
    if not all((layers, families, severities, seeds)):
        raise ValueError("grid axes must not be empty")
    states = []
    for layer in layers:
        for family in families:
            for severity in severities:
                for seed in seeds[:1] if severity in (0, 1) else seeds:
                    validate(config, layer, family, severity, seed)
                    states.append(dict(name=f"weight_{family}_L{layer}_s{severity:g}_seed{seed}", kind="weight_mask",
                                       gates={}, weight_mask=dict(layer=layer, family=family, severity=severity, seed=seed)))
    if len({s["name"] for s in states}) != len(states):
        raise ValueError("duplicate grid cells")
    return states

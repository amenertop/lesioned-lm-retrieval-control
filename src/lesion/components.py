"""Names of the ablatable components of a model (docs/plans/code-architecture.md §3.1).

Shared between the ablation side (hooks, means; agent2) and the analysis side (screen, baselines; agent1).
Changes need both sides' agreement.
"""
from dataclasses import dataclass

KINDS = ("expert", "head", "mlp")


@dataclass(frozen=True, order=True)
class Component:
    kind: str   # "expert" | "head" | "mlp"
    layer: int  # 0-based
    index: int  # expert or head number; always 0 for "mlp"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown component kind {self.kind!r}; expected one of {KINDS}")
        if self.layer < 0 or self.index < 0 or (self.kind == "mlp" and self.index != 0):
            raise ValueError(f"invalid component {self}")

    @property
    def key(self):
        """Stable text form used in CSV columns and file names, e.g. "expert:3:17"."""
        return f"{self.kind}:{self.layer}:{self.index}"

    @classmethod
    def parse(cls, key):
        kind, layer, index = key.split(":")
        return cls(kind, int(layer), int(index))

    def __str__(self):
        return self.key


def n_experts(config):
    """Experts per sparse layer: num_experts (OLMoE, Qwen3-MoE) or num_local_experts (Phi-3.5-MoE, Mixtral style)."""
    return int(getattr(config, "num_experts", 0) or getattr(config, "num_local_experts", 0) or 0)


def moe_block(layer):
    """A decoder layer's sparse MoE block, or None for a dense layer: layer.mlp in OLMoE and Qwen3-MoE,
    layer.block_sparse_moe in Phi-3.5-MoE. The block has .gate (router) and .experts (one module per expert)."""
    for name in ("mlp", "block_sparse_moe"):
        block = getattr(layer, name, None)
        if block is not None and hasattr(block, "experts"):
            return block
    return None


def moe_layers(config):
    """Layers whose MLP is a sparse MoE block (all layers for OLMoE; Qwen3-MoE can keep some layers dense)."""
    if n_experts(config) == 0:
        return []
    skip = set(getattr(config, "mlp_only_layers", []) or [])
    step = getattr(config, "decoder_sparse_step", 1) or 1
    return [l for l in range(config.num_hidden_layers) if l not in skip and (l + 1) % step == 0]


def all_components(config, kind):
    """Every component of one kind in a model with this config, sorted by (layer, index)."""
    if kind == "expert":
        return [Component("expert", l, e) for l in moe_layers(config) for e in range(n_experts(config))]
    if kind == "head":
        return [Component("head", l, h) for l in range(config.num_hidden_layers) for h in range(config.num_attention_heads)]
    if kind == "mlp":
        sparse = set(moe_layers(config))
        return [Component("mlp", l, 0) for l in range(config.num_hidden_layers) if l not in sparse]
    raise ValueError(f"unknown component kind {kind!r}; expected one of {KINDS}")

"""Forward-only ablation hooks for MoE experts, attention heads and dense MLPs."""

from contextlib import contextmanager

import torch

from .components import Component, moe_block, moe_layers


def _layers(model):
    try:
        return model.model.layers
    except AttributeError as exc:  # pragma: no cover - unsupported model wrapper
        raise TypeError("model must expose model.layers") from exc


def _mode_value(value, mean, alpha, mode):
    if mode == "zero":
        return value * alpha
    return mean + alpha * (value - mean)


def _check_mode(mode):
    if mode not in {"mean", "zero"}:
        raise ValueError("mode must be 'mean' or 'zero'")


def _mean_for(means, component, value):
    if means is None:
        raise ValueError("mean ablation requires a Means object")
    try:
        if component.kind == "expert":
            mean = means.expert_mean[component.layer, component.index]
        elif component.kind == "head":
            mean = means.head_mean[component.layer, component.index]
        else:
            mean = means.mlp_mean[component.layer]
    except (AttributeError, TypeError, IndexError) as exc:
        raise ValueError(f"means has no value for {component}") from exc
    if mean is None:
        raise ValueError(f"means has no value for {component}")
    return mean.to(device=value.device, dtype=value.dtype)


def _validate_component(model, component):
    layers = _layers(model)
    if component.layer >= len(layers):
        raise ValueError(f"layer out of range: {component}")
    layer = layers[component.layer]
    if component.kind == "expert":
        experts = getattr(moe_block(layer), "experts", None)
        if experts is None:
            raise ValueError(f"model has no expert at {component}")
        if component.index >= len(experts):
            raise ValueError(f"expert out of range: {component}")
    elif component.kind == "head":
        if component.index >= int(model.config.num_attention_heads):
            raise ValueError(f"head out of range: {component}")
    elif component.kind == "mlp":
        if component.index != 0 or moe_block(layer) is not None:
            raise ValueError(f"model has no dense MLP at {component}")
    else:  # Component validates this normally; keep a defensive check for duck types.
        raise ValueError(f"unknown component kind: {component.kind}")


def _routed_experts(block, logits, k):
    """The experts each token goes to, in the rank order the block's forward uses: Qwen3-MoE and OLMoE take the
    top-k of a float32 softmax (pinned HF code); Phi-3.5-MoE's sparsemixer (recognised by its router_jitter_noise)
    takes, in eval mode, the largest logit and then the largest of the rest, with torch.max, so exact ties resolve
    as there (topk does not promise the same order, and a softmax can tie two nearly equal logits)."""
    if hasattr(block, "router_jitter_noise"):
        first = logits.max(dim=-1, keepdim=True).indices
        second = logits.scatter(-1, first, float("-inf")).max(dim=-1, keepdim=True).indices
        return torch.cat((first, second), dim=-1)
    return torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float32), k, dim=-1).indices


def _active(model, name):
    return bool(getattr(model, f"_lesion_{name}_active", False))


@contextmanager
def position_mask(model, mask):
    """Restrict ablation to True entries of a bool (batch, seq) mask for this forward.

    Positions refer to the current, possibly left-padded input, not position_ids or
    the full sequence in a KV cache. Call inside ablate and update for each batch.
    A nested context restores its outer mask, including when a forward raises.
    """
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("position mask must be a bool tensor")
    if mask.ndim != 2:
        raise ValueError("position mask must have shape (batch, seq)")
    previous = getattr(model, "_lesion_position_mask", None)
    model._lesion_position_mask = mask
    try:
        yield
    finally:
        model._lesion_position_mask = previous


def _positions_for(model, value):
    """Validate an unflattened activation against the mask of the current forward."""
    mask = getattr(model, "_lesion_position_mask", None)
    if mask is not None:
        if value.ndim != 3 or mask.shape != value.shape[:2]:
            raise ValueError(f"position mask shape {tuple(mask.shape)} does not match "
                             f"activation shape {tuple(value.shape)}")
        mask = mask.to(device=value.device)
    return mask


def _restrict(value, changed, mask):
    return changed if mask is None else torch.where(mask.unsqueeze(-1), changed, value)


@contextmanager
def ablate(model, components, means=None, alpha=0.0, mode="mean"):
    """Apply mean or zero ablation, optionally restricted by position_mask.

    Expert outputs are replaced before multiplying by unchanged router weights.
    All hooks are removed on exit. With no position mask the original path is used.
    """
    _check_mode(mode)
    if _active(model, "router"):
        raise RuntimeError("router masking and ablation cannot be active together")
    components = list(components)
    for component in components:
        if not isinstance(component, Component):
            raise TypeError("components must contain Component objects")
        _validate_component(model, component)
    if mode == "mean" and means is None:
        raise ValueError("mode='mean' requires a Means object")
    if _active(model, "ablate"):
        raise RuntimeError("nested ablation contexts are not supported")

    handles = []
    setattr(model, "_lesion_ablate_active", True)
    try:
        layers = _layers(model)
        expert_masks = {}
        by_layer = {}
        for component in components:
            if component.kind == "expert":
                by_layer.setdefault(component.layer, set()).add(component.index)
        for layer_index, indices in by_layer.items():
            block = moe_block(layers[layer_index])

            def block_hook(_module, inputs, li=layer_index, experts=tuple(indices)):
                # Validate batch/sequence dimensions before the gate flattens them.
                _positions_for(model, inputs[0])
                for e in experts:
                    expert_masks.pop((li, e), None)

            def gate_hook(_module, _inputs, output, li=layer_index, experts=tuple(indices), k=block.top_k, block=block):
                mask = getattr(model, "_lesion_position_mask", None)
                if mask is None:
                    return
                logits = output[0] if isinstance(output, tuple) else output
                if logits.ndim != 2 or logits.shape[0] != mask.numel():
                    raise ValueError("router logits shape does not match the position mask")
                chosen = _routed_experts(block, logits, k)
                flat = mask.to(device=logits.device).reshape(-1)
                for e in experts:
                    # HF's expert_mask[e] is (top_k, tokens). torch.where therefore
                    # groups by top-k rank, NOT globally increasing token index.
                    token_rows = torch.where(chosen.transpose(0, 1) == e)[1]
                    expert_masks[(li, e)] = flat[token_rows]

            handles.append(block.register_forward_pre_hook(block_hook))
            handles.append(block.gate.register_forward_hook(gate_hook))

        for component in components:
            layer = layers[component.layer]
            module = (moe_block(layer).experts[component.index] if component.kind == "expert"
                      else layer.self_attn.o_proj if component.kind == "head"
                      else layer.mlp)
            if component.kind == "head":
                head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
                start = component.index * head_dim
                end = start + head_dim

                def head_hook(_module, inputs, c=component, lo=start, hi=end):
                    value = inputs[0]
                    if value is None:
                        return inputs
                    changed = value.clone()
                    piece = changed[..., lo:hi]
                    mean = _mean_for(means, c, piece) if mode == "mean" else torch.zeros((), device=piece.device, dtype=piece.dtype)
                    changed[..., lo:hi] = _restrict(piece, _mode_value(piece, mean, alpha, mode), _positions_for(model, value))
                    return (changed, *inputs[1:])

                handles.append(module.register_forward_pre_hook(head_hook))
            else:
                def output_hook(_module, _inputs, output, c=component):
                    value = output[0] if isinstance(output, tuple) else output
                    mean = torch.zeros((), device=value.device, dtype=value.dtype)
                    if mode == "mean":
                        mean = _mean_for(means, c, value)
                    changed = _mode_value(value, mean, alpha, mode)
                    if getattr(model, "_lesion_position_mask", None) is not None:
                        if c.kind == "expert":
                            mask = expert_masks.get((c.layer, c.index))
                            if mask is None or value.ndim != 2 or len(mask) != len(value):
                                raise RuntimeError(f"missing or mismatched token routing for {c}")
                            mask = mask.to(device=value.device)
                        else:
                            mask = _positions_for(model, value)
                        changed = _restrict(value, changed, mask)
                    if isinstance(output, tuple):
                        return (changed, *output[1:])
                    return changed

                handles.append(module.register_forward_hook(output_hook))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        setattr(model, "_lesion_ablate_active", False)


@contextmanager
def mask_router(model, experts):
    """Temporarily remove the selected experts from each sparse router's candidates."""
    if _active(model, "ablate"):
        raise RuntimeError("router masking and ablation cannot be active together")
    experts = list(experts)
    for component in experts:
        if not isinstance(component, Component) or component.kind != "expert":
            raise ValueError("mask_router accepts expert Components only")
        _validate_component(model, component)
    if _active(model, "router"):
        raise RuntimeError("nested router masks are not supported")

    by_layer = {}
    for component in experts:
        by_layer.setdefault(component.layer, set()).add(component.index)
    handles = []
    setattr(model, "_lesion_router_active", True)
    try:
        layers = _layers(model)
        for layer_index, indices in by_layer.items():
            gate = moe_block(layers[layer_index]).gate

            def gate_hook(_module, _inputs, output, blocked=tuple(sorted(indices))):
                logits = output[0] if isinstance(output, tuple) else output
                changed = logits.clone()
                changed[..., list(blocked)] = torch.finfo(changed.dtype).min
                if isinstance(output, tuple):
                    return (changed, *output[1:])
                return changed

            handles.append(gate.register_forward_hook(gate_hook))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        setattr(model, "_lesion_router_active", False)

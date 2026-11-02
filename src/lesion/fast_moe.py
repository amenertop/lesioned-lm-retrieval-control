"""Faster forwards for the sparse MoE blocks of Qwen3-MoE and OLMoE (agent1's module; user decision 2026-10-16).

Why. The HF forward (transformers 4.57.6) loops over the experts and finds each expert's rows with torch.where on a
one-hot mask, and Qwen3-MoE also indexes its expert list with a GPU tensor. Each of these makes the CPU wait for the GPU
(a host-device sync), about two per expert, 128 experts x 48 layers. The GPU then idles while one CPU core queues the
next expert's small kernels (session 15: GPU busy about 35% of the time, 190 of 600 W, one CPU core at 100%).

Two modes, installed per model with install(model, mode):
- "sorted": the HF forward, except that one stable sort and one count per layer give every expert's rows (one sync per
  layer). Each expert is still called as a module, on the same rows in the same order (by top-k rank, then by token),
  and on the same set of experts (Qwen3-MoE: the routed ones; OLMoE: all, unrouted ones on zero rows), so every expert
  hook sees what it sees under HF. Tested bit-identical to HF (tests/test_fast_moe.py, CPU float32, GPU float16 and
  bfloat16): logits, router logits, every layer's outputs, the gradients of the input embeddings, of every weight
  (router and experts included) and of the P14 gates; and the production paths (lesion.hooks, gates, means,
  attribution, probes, naming and M1 scores). Expert weights may be trained (the Wang-style comparison). Check it on
  the real model (experiments/fast_moe_check/run.py) before use.
- "grouped": a layer's experts in a few batched matrix products (consecutive experts padded to the group's largest
  count), from the expert weights stacked into one tensor per projection. The expert modules keep their weights as views
  of it, so direct calls and the state dict still work. A batched product rounds differently from HF's one product per
  expert, so results agree to rounding only; the per-token sum keeps HF's order (by expert index, in the model's
  precision), and no backward adds into repeated indices (repeated runs were identical in the tests). Frozen expert
  weights only (gates, lesions, evaluation); install it before creating any optimiser over the model's weights.
  Hooks: only plain forward hooks on the experts, as the lesion modules register them (they read the expert's input and
  output and may return a replacement output). They are called on each expert's rows, in HF's row order and in
  increasing expert order, but after all of the layer's experts have been computed (HF interleaves computing and
  hooks); a hook must not depend on that interleaving. Refused: every other kind of hook on the experts or their
  submodules (pre-hooks, hooks with kwargs or always_call, backward hooks, global hooks, accelerate's), and a hook that
  modifies its input in place or returns an output of another shape.
"""
import functools
import types

import torch
import torch.nn.functional as F

from .components import moe_block

TRANSFORMERS = "4.57.6"      # the HF forward this module reproduces
MODES = ("sorted", "grouped")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# grouped: consecutive experts share one batched product while the group's padded rows stay within PAD_FACTOR times its
# routed rows, or within MIN_ROWS rows (a small group's padding costs little)
PAD_FACTOR = 3
MIN_ROWS = 8192


def _supported():
    """MoE block class -> whether its HF forward skips experts no token was routed to."""
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    return {Qwen3MoeSparseMoeBlock: True, OlmoeSparseMoeBlock: False}


def _route(self, hidden_states):
    """HF's routing, line by line; then the routed slots sorted by expert. HF's torch.where(expert_mask[e]) walks a
    (top_k, tokens) mask row by row, slot j = rank * tokens + token, and a stable sort by expert keeps that order within
    each expert."""
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    n_tokens = hidden_states.shape[0]
    flat = selected_experts.t().reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=self.num_experts)
    token = order % n_tokens
    # every routed slot's weight, in sorted order: HF's routing_weights[top_x, idx], expert by expert
    weight = routing_weights.reshape(-1).index_select(0, token * self.top_k + order // n_tokens)
    return router_logits, flat, order, counts, token, weight


def sorted_forward(self, hidden_states, skip_empty):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits, _, _, counts, token, weight = _route(self, hidden_states)
    counts = counts.tolist()                                   # this layer's one host-device sync
    tokens, weights = token.split(counts), weight.split(counts)
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )
    # HF's expert loop, except that rows are gathered with index_select: it copies the same values, and its backward
    # writes each gradient row once into zeros, as the backward of HF's indexing does (without its sort)
    for expert_idx, count in enumerate(counts):
        if skip_empty and count == 0:
            continue
        expert_layer = self.experts[expert_idx]
        top_x = tokens[expert_idx]
        current_state = hidden_states.index_select(0, top_x)
        current_hidden_states = expert_layer(current_state) * weights[expert_idx][:, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


class _Stack:
    """A block's expert weights stacked per projection, (experts, out, in); each expert's Linear keeps a view."""

    def __init__(self, block):
        self.weights, self.expected = {}, []
        for name in PROJECTIONS:
            linears = [getattr(e, name) for e in block.experts]
            if any(lin.bias is not None for lin in linears):
                raise TypeError("the grouped forward assumes expert projections without bias")
            stacked = torch.stack([lin.weight.detach() for lin in linears])
            for i, lin in enumerate(linears):
                lin.weight = torch.nn.Parameter(stacked[i], requires_grad=False)
                self.expected.append((i, name, lin, lin.weight, self._layout(lin.weight)))
            self.weights[name] = stacked

    @staticmethod
    def _layout(w):
        return w.data_ptr(), w.dtype, w.device, tuple(w.shape), w.stride(), w.storage_offset()

    def check(self, block):
        for i, name, lin, param, layout in self.expected:
            if lin.weight is not param or self._layout(param) != layout or param.requires_grad:
                raise RuntimeError(f"expert {i}'s {name} weight changed after fast_moe.install (moved, converted, "
                                   "replaced or set to require gradients); install again, or use the 'sorted' mode")
        if any(w.requires_grad for w in self.weights.values()):
            raise RuntimeError("the stacked expert weights require gradients")
        g = torch.nn.modules.module
        if any(getattr(g, name, None) for name in ("_global_forward_hooks", "_global_forward_pre_hooks",
                                                   "_global_backward_hooks", "_global_backward_pre_hooks",
                                                   "_global_forward_hooks_always_called",
                                                   "_global_forward_hooks_with_kwargs")):
            raise RuntimeError("global module hooks are not supported by the grouped forward")
        for expert in block.experts:
            if (expert._forward_pre_hooks or expert._forward_hooks_with_kwargs or expert._forward_hooks_always_called
                    or expert._backward_hooks or expert._backward_pre_hooks or "forward" in vars(expert)
                    or hasattr(expert, "_hf_hook")):
                raise RuntimeError("the grouped forward supports only plain forward hooks on the experts")
            for sub in expert.children():
                if (sub._forward_hooks or sub._forward_pre_hooks or sub._backward_hooks or sub._backward_pre_hooks
                        or "forward" in vars(sub) or hasattr(sub, "_hf_hook")):
                    raise RuntimeError("the grouped forward does not call the experts' submodules; hooks on them are "
                                       "not supported")


def _groups(counts):
    """Consecutive experts that share one padded product: [(first, end, width)]."""
    groups, first, n = [], 0, len(counts)
    while first < n:
        end, width, rows = first + 1, counts[first], counts[first]
        while end < n:
            w, r = max(width, counts[end]), rows + counts[end]
            if (end + 1 - first) * w > max(PAD_FACTOR * r, MIN_ROWS):
                break
            end, width, rows = end + 1, w, r
        groups.append((first, end, max(width, 1)))
        first = end
    return groups


def _expert_hooks(self, x_sorted, y_sorted, counts, skip_empty):
    """Call the experts' forward hooks on their rows, as nn.Module would after HF's call of the expert."""
    hooked = [i for i, expert in enumerate(self.experts) if expert._forward_hooks]
    if not hooked:
        return y_sorted
    xs, ys = x_sorted.split(counts), list(y_sorted.split(counts))
    version = x_sorted._version
    for i in hooked:
        if skip_empty and counts[i] == 0:
            continue
        expert, out = self.experts[i], ys[i]
        for hook in tuple(expert._forward_hooks.values()):
            result = hook(expert, (xs[i],), out)
            if result is not None:
                if not isinstance(result, torch.Tensor) or result.shape != out.shape:
                    raise RuntimeError(f"a hook on expert {i} returned something other than a tensor of its output's "
                                       f"shape")
                out = result
        ys[i] = out
    if x_sorted._version != version:
        raise RuntimeError("a hook modified an expert's input in place; the grouped forward passes views of one tensor "
                           "where HF passes each expert its own copy")
    return torch.cat(ys)


def grouped_forward(self, hidden_states, skip_empty, stack):
    stack.check(self)
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits, flat, order, counts_t, token, weight = _route(self, hidden_states)
    counts = counts_t.tolist()                                 # this layer's one host-device sync
    n_tokens, k = hidden_states.shape[0], self.top_k
    expert = flat[order]
    within = torch.arange(len(order), device=order.device) - (torch.cumsum(counts_t, 0) - counts_t)[expert]
    # the inputs by slot (each token's row once per rank), then sorted: every backward below is then a permutation or a
    # sum, so it is deterministic, unlike gathering a token's row k times
    x_slot = hidden_states.unsqueeze(0).expand(k, n_tokens, hidden_dim).reshape(k * n_tokens, hidden_dim)
    x_sorted = x_slot.index_select(0, order)
    groups = _groups(counts)
    sizes = [sum(counts[a:b]) for a, b, _ in groups]
    w, act = stack.weights, self.experts[0].act_fn
    outputs = []
    for (a, b, width), x, e, pos in zip(groups, x_sorted.split(sizes), expert.split(sizes), within.split(sizes)):
        if len(x) == 0:                                        # a run of unrouted experts
            outputs.append(x)
            continue
        place = (e - a) * width + pos
        x_pad = x.new_zeros(((b - a) * width, hidden_dim)).index_copy(0, place, x).view(b - a, width, hidden_dim)
        h = act(torch.bmm(x_pad, w["gate_proj"][a:b].transpose(1, 2))) * torch.bmm(x_pad, w["up_proj"][a:b].transpose(1, 2))
        y_pad = torch.bmm(h, w["down_proj"][a:b].transpose(1, 2))
        outputs.append(y_pad.reshape((b - a) * width, hidden_dim).index_select(0, place))
    y_sorted = _expert_hooks(self, x_sorted, torch.cat(outputs), counts, skip_empty)
    y_sorted = y_sorted * weight[:, None]
    # HF adds a token's expert outputs in increasing expert order, in the model's precision; the sorted rows are grouped
    # by expert, so a stable sort by token lists each token's rows in that order
    by_token = torch.argsort(token, stable=True).view(n_tokens, k)
    final_hidden_states = torch.zeros((n_tokens, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    for r in range(k):
        final_hidden_states = final_hidden_states + y_sorted.index_select(0, by_token[:, r])
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def _blocks(model):
    return [b for b in (moe_block(layer) for layer in model.model.layers) if b is not None]


def install(model, mode="sorted"):
    """Give every MoE block of the model the faster forward of this mode; returns how many blocks. Refuses a model with
    an MoE block of another kind (Phi-3.5-MoE, gpt-oss, DeepSeek-V2), a transformers version other than the one
    reproduced, and, for "grouped", expert weights that require gradients. Install after any dtype or device change."""
    import transformers

    if transformers.__version__ != TRANSFORMERS:
        raise RuntimeError(f"fast_moe reproduces transformers {TRANSFORMERS}, not {transformers.__version__}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    supported = _supported()
    blocks = _blocks(model)
    for block in blocks:
        if type(block) not in supported:
            raise TypeError(f"no {mode} forward for {type(block).__name__}")
        if mode == "grouped" and any(p.requires_grad for p in block.experts.parameters()):
            raise ValueError("the grouped forward needs frozen expert weights; use the 'sorted' mode to train them")
    uninstall(model)
    for block in blocks:
        skip = supported[type(block)]
        if mode == "sorted":
            forward = functools.partial(sorted_forward, skip_empty=skip)
        else:
            forward = functools.partial(grouped_forward, skip_empty=skip, stack=_Stack(block))
        block.forward = types.MethodType(forward, block)
        block._lesion_moe_mode = mode
    return len(blocks)


def uninstall(model):
    """Back to the HF forward (after "grouped", the expert weights stay views of the stacked tensors)."""
    for block in _blocks(model):
        if "forward" in vars(block):
            del block.forward
        vars(block).pop("_lesion_moe_mode", None)


def impl(model):
    """The mode every MoE block uses, or 'hf' if none uses one (also for a model without MoE blocks)."""
    modes = {vars(b).get("_lesion_moe_mode", "hf") for b in _blocks(model)}
    if len(modes) > 1:
        raise RuntimeError(f"the MoE blocks use different forwards: {sorted(modes)}")
    return modes.pop() if modes else "hf"

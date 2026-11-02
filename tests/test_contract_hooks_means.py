"""Acceptance tests for lesion.hooks and lesion.means (docs/plans/code-architecture.md §3.2-3.3).

Written by agent1 from the interface, before the implementation; agent2's code must pass them. They run on
tiny random models of four architectures (OLMoE, Qwen3-MoE, OLMo-2, Phi-3.5-MoE) and are skipped until both modules exist.
agent2's own tests (test_hooks.py, test_means.py) cover the internals.
"""
from contextlib import contextmanager

import numpy as np
import pytest
import torch

hooks = pytest.importorskip("lesion.hooks")
means_mod = pytest.importorskip("lesion.means")

from lesion import models, neutral_text, scoring  # noqa: E402
from lesion.components import Component, moe_block, moe_layers, n_experts  # noqa: E402

ARCHS = ["olmoe-0924-base", "qwen3-30b-a3b-instruct", "olmo2-1b", "phi35-moe-instruct"]
MAX_TOKENS = 48


def as_np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=float)


@pytest.fixture(scope="module", params=ARCHS)
def tiny(request):
    name = request.param
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    # Texts of different lengths, so a batch needs padding.
    texts = [t[: 60 + 40 * i] for i, t in enumerate(neutral_text.load()[:5])]
    m = means_mod.collect_means(model, tok, texts, max_tokens=MAX_TOKENS, batch_size=5)
    return name, model, tok, texts, m


def is_moe(model):
    return bool(n_experts(model.config))


def targets(model, m):
    """A few components of every kind the model has: busiest expert in the first MoE layer, one head, one MLP."""
    out = [Component("head", model.config.num_hidden_layers - 1, 1)]
    if is_moe(model):
        layer = moe_layers(model.config)[0]
        out.append(Component("expert", layer, int(np.argmax(as_np(m.expert_count)[layer]))))
    else:
        out.append(Component("mlp", 0, 0))
    return out


def batch(tok, texts):
    seqs = [tok.encode(t, add_special_tokens=False)[:30] for t in texts[:4]]
    return scoring._pad_left(seqs, tok.pad_token_id or 0, "cpu")


def logits(model, ids, mask, pos):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[mask.bool()]


# --- means (§3.2) --------------------------------------------------------------------------------

def test_means_fields(tiny):
    name, model, tok, texts, m = tiny
    cfg = model.config
    L, H, d = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", None) or d // H
    assert m.meta["model_name"] == name and m.meta["revision"] == models.MODELS[name][1]
    assert {"git_commit", "text_sha256", "max_tokens", "n_texts"} <= set(m.meta)
    lengths = [min(len(tok.encode(t, add_special_tokens=False)), MAX_TOKENS) for t in texts]
    assert sum(lengths) <= m.n_tokens <= sum(lengths) + len(texts)   # a document-start token may be added
    assert as_np(m.head_mean).shape == (L, H, head_dim)
    if is_moe(model):
        E, k = n_experts(cfg), cfg.num_experts_per_tok
        assert as_np(m.expert_mean).shape == (L, E, d) and as_np(m.expert_count).shape == (L, E)
        for layer in moe_layers(cfg):   # every valid token is routed to exactly k experts per MoE layer
            assert as_np(m.expert_count)[layer].sum() == k * m.n_tokens
        assert np.allclose(np.asarray(m.routing_frequency()), as_np(m.expert_count) / m.n_tokens)
    else:
        assert m.expert_mean is None and as_np(m.mlp_mean).shape == (L, d)


def test_means_ignore_padding(tiny):
    """Batches with padding give the same means as one text at a time (no padding at all)."""
    name, model, tok, texts, m = tiny
    single = means_mod.collect_means(model, tok, texts, max_tokens=MAX_TOKENS, batch_size=1)
    assert single.n_tokens == m.n_tokens
    assert np.allclose(as_np(single.head_mean), as_np(m.head_mean), atol=1e-4)
    if is_moe(model):
        assert (as_np(single.expert_count) == as_np(m.expert_count)).all()
        assert np.allclose(as_np(single.expert_mean), as_np(m.expert_mean), atol=1e-4)
    else:
        assert np.allclose(as_np(single.mlp_mean), as_np(m.mlp_mean), atol=1e-4)


def test_means_save_load(tiny, tmp_path):
    name, model, tok, texts, m = tiny
    m.save(tmp_path / "m.pt")
    back = means_mod.Means.load(tmp_path / "m.pt")
    assert back.meta == m.meta and back.n_tokens == m.n_tokens
    assert np.array_equal(as_np(back.head_mean), as_np(m.head_mean))


# --- ablation switch (§3.3) --------------------------------------------------------------------------

def test_alpha_one_is_identity_and_alpha_zero_is_not(tiny):
    name, model, tok, texts, m = tiny
    ids, mask, pos = batch(tok, texts)
    base = logits(model, ids, mask, pos)
    comps = targets(model, m)
    with hooks.ablate(model, comps, m, alpha=1.0):
        assert (logits(model, ids, mask, pos) - base).abs().max() < 1e-5
    with hooks.ablate(model, comps, m, alpha=0.0):
        assert (logits(model, ids, mask, pos) - base).abs().max() > 1e-4
    with hooks.ablate(model, comps, None, alpha=0.0, mode="zero"):
        zero = logits(model, ids, mask, pos)
    with hooks.ablate(model, comps, m, alpha=0.0):
        assert (zero - logits(model, ids, mask, pos)).abs().max() > 1e-6
    assert torch.equal(logits(model, ids, mask, pos), base)          # restored on exit


def test_restored_after_error(tiny):
    name, model, tok, texts, m = tiny
    ids, mask, pos = batch(tok, texts)
    base = logits(model, ids, mask, pos)
    with pytest.raises(KeyError):
        with hooks.ablate(model, targets(model, m), m, alpha=0.0):
            raise KeyError("boom")
    assert torch.equal(logits(model, ids, mask, pos), base)


@contextmanager
def capture(module, pre=False):
    """Record a module's output (or, with pre=True, its first input) on every call."""
    seen = []
    if pre:
        h = module.register_forward_pre_hook(lambda mod, args: seen.append(args[0].detach().clone()))
    else:
        h = module.register_forward_hook(lambda mod, args, out: seen.append((out[0] if isinstance(out, tuple) else out).detach().clone()))
    try:
        yield seen
    finally:
        h.remove()


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0])   # 2.0: amplification (optogenetics plan O1)
def test_mean_ablation_formula(tiny, alpha):
    """Ablated output = mean + alpha * (output - mean), for the expert itself, the head's o_proj slice, or the MLP."""
    name, model, tok, texts, m = tiny
    ids, mask, pos = batch(tok, texts)
    valid = mask.bool()
    for c in targets(model, m):
        layer = model.model.layers[c.layer]
        if c.kind == "expert":
            module, pre, mean = moe_block(layer).experts[c.index], False, as_np(m.expert_mean)[c.layer, c.index]
        elif c.kind == "head":
            module, pre = layer.self_attn.o_proj, True
            hd = as_np(m.head_mean).shape[-1]
            mean = as_np(m.head_mean)[c.layer, c.index]
        else:
            module, pre, mean = layer.mlp, False, as_np(m.mlp_mean)[c.layer]
        with capture(module, pre) as before:            # registered first: sees the original value
            with hooks.ablate(model, [c], m, alpha=alpha):
                with capture(module, pre) as after:     # registered after the ablation hook
                    logits(model, ids, mask, pos)
        orig, new = as_np(torch.cat([x.reshape(-1, x.shape[-1]) for x in before])), as_np(torch.cat([x.reshape(-1, x.shape[-1]) for x in after]))
        if c.kind != "expert":                           # [batch, seq, ...]: compare valid positions only
            keep = valid.reshape(-1).numpy()
            orig, new = orig[keep], new[keep]
        if c.kind == "head":
            sl = slice(c.index * hd, (c.index + 1) * hd)
            assert np.allclose(new[:, sl], mean + alpha * (orig[:, sl] - mean), atol=1e-5)
            other = np.ones(orig.shape[1], bool)
            other[sl] = False
            assert np.allclose(new[:, other], orig[:, other], atol=1e-6)   # other heads untouched
        else:
            assert np.allclose(new, mean + alpha * (orig - mean), atol=1e-5)


def test_router_mask_and_errors(tiny):
    name, model, tok, texts, m = tiny
    comps = targets(model, m)
    if not is_moe(model):
        with pytest.raises(ValueError):
            with hooks.ablate(model, [Component("expert", 0, 0)], m):
                pass
        return
    expert = [c for c in comps if c.kind == "expert"]
    ids, mask, pos = batch(tok, texts)
    base = logits(model, ids, mask, pos)
    with hooks.mask_router(model, expert):
        assert (logits(model, ids, mask, pos) - base).abs().max() > 1e-6
        with pytest.raises(RuntimeError):
            with hooks.ablate(model, comps, m):
                pass
    with hooks.ablate(model, comps, m):
        with pytest.raises(RuntimeError):
            with hooks.mask_router(model, expert):
                pass
    with pytest.raises(ValueError):
        with hooks.ablate(model, comps, None, mode="mean"):
            pass
    assert torch.equal(logits(model, ids, mask, pos), base)

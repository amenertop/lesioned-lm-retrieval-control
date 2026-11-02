"""lesion.fast_moe. The sorted forward computes bit for bit what the HF forward computes: logits, router logits, every
layer's output, the gradients of the input embeddings and of every weight, under the lesion hooks, the P14 gates and
gradient checkpointing, and on the production paths (means, attribution, expert norms, naming and M1 scores, position
masks, router masking); it calls the same experts on the same rows. The grouped forward agrees to rounding (float32
here) on the same checks (its expert weights frozen), calls the experts' hooks on the same rows, repeats exactly, is as
accurate as HF in float16/bfloat16 on the GPU, and refuses what it cannot reproduce."""
import warnings
from types import SimpleNamespace

import pytest
import torch
from transformers import (OlmoeConfig, OlmoeForCausalLM, PhimoeConfig, PhimoeForCausalLM, Qwen3MoeConfig,
                          Qwen3MoeForCausalLM)

from lesion import fast_moe, gates, hooks
from lesion.components import Component, moe_block

N_EXPERTS, TOP_K, HIDDEN, LAYERS = 16, 4, 32, 2
IDS = torch.tensor([[2, 7, 3, 9, 5, 11, 4, 8, 1], [6, 4, 8, 1, 3, 5, 7, 2, 2], [9, 9, 1, 0, 4, 3, 12, 6, 5]])
MASK = torch.tensor([[1] * 9, [1] * 9, [0, 0, 1, 1, 1, 1, 1, 1, 1]])          # left padding in the last row
COMPS = [Component("head", 0, 1), Component("expert", 0, 2), Component("expert", 0, 9), Component("expert", 1, 0),
         Component("expert", 1, 15)]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU: the GPU kernels (GEMM, index_add_) are "
                                                                "the ones the real runs use")


def build(kind, dtype=torch.float32, device="cpu"):
    torch.manual_seed(3)
    config = dict(vocab_size=43, hidden_size=HIDDEN, intermediate_size=48, num_hidden_layers=LAYERS,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, attention_dropout=0.0,
                  use_cache=False, num_experts_per_tok=TOP_K)
    if kind == "qwen":
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**config, num_experts=N_EXPERTS, head_dim=8, moe_intermediate_size=24,
                                                   norm_topk_prob=True))
    else:
        model = OlmoeForCausalLM(OlmoeConfig(**config, num_experts=N_EXPERTS))
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def means_for(model):
    g = torch.Generator().manual_seed(1)
    head_dim = HIDDEN // model.config.num_attention_heads
    return SimpleNamespace(expert_mean=torch.randn(LAYERS, N_EXPERTS, HIDDEN, generator=g) * .3,
                           head_mean=torch.randn(LAYERS, 4, head_dim, generator=g) * .3)


def run(model, mode, fn):
    if mode is not None:
        assert fast_moe.install(model, mode) == LAYERS
    try:
        return fn(model)
    finally:
        fast_moe.uninstall(model)


def both(kind, fn, mode="sorted", **kw):
    """fn's result under the HF forward and under the mode's forward, on the same model."""
    model = build(kind, **kw)
    out_hf = run(model, None, fn)
    out_mode = run(model, mode, fn)
    assert fast_moe.impl(model) == "hf"
    return out_hf, out_mode


def forward(model, ids=IDS, mask=MASK):
    with torch.no_grad():
        out = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), output_router_logits=True)
    return out.logits, out.router_logits


def _plain(a):
    """numpy arrays as tensors; numbers and strings as they are."""
    import numpy as np

    return torch.from_numpy(np.asarray(a)) if isinstance(a, np.ndarray) else a


def assert_identical(a, b):
    a, b = _plain(a), _plain(b)
    if isinstance(a, (int, float, str)) or a is None:
        assert a == b
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_identical(a[k], b[k])
    elif hasattr(a, "columns"):                        # a DataFrame
        assert list(a.columns) == list(b.columns) and a.equals(b)
    else:
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_identical(x, y)


def assert_near(a, b):
    """Equal to float32 rounding (the grouped forward's batched products round differently)."""
    a, b = _plain(a), _plain(b)
    if isinstance(a, str) or a is None:
        assert a == b
    elif isinstance(a, (int, float)):
        assert abs(a - b) <= 1e-5 + 1e-5 * abs(b)
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_near(a[k], b[k])
    elif hasattr(a, "columns"):
        assert list(a.columns) == list(b.columns)
        num = a.select_dtypes("number").columns
        assert a.drop(columns=num).equals(b.drop(columns=num))
        torch.testing.assert_close(torch.tensor(a[num].to_numpy(float)), torch.tensor(b[num].to_numpy(float)),
                                   rtol=1e-4, atol=1e-5)
    else:
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_near(x, y)


@pytest.fixture(params=["qwen", "olmoe"])
def kind(request):
    return request.param


@pytest.fixture(params=["sorted", "grouped", "grouped-many"])
def mode(request, monkeypatch):
    """grouped-many: every expert in its own padded product, or nearly (the groups' boundaries are then exercised)."""
    if request.param == "grouped-many":
        monkeypatch.setattr(fast_moe, "MIN_ROWS", 1)
        monkeypatch.setattr(fast_moe, "PAD_FACTOR", 1)
        return "grouped"
    return request.param


def check(mode):
    return assert_identical if mode == "sorted" else assert_near


def test_logits_and_router_logits_agree(kind, mode):
    check(mode)(*both(kind, forward, mode))


def test_the_same_experts_are_called_on_the_same_rows_in_the_same_order(kind, mode):
    def calls(model):
        seen, handles = [], []
        for l, layer in enumerate(model.model.layers):
            for e, expert in enumerate(moe_block(layer).experts):
                handles.append(expert.register_forward_hook(
                    lambda _m, inputs, output, l=l, e=e: seen.append((l, e, inputs[0].clone(), output.clone()))))
        try:
            forward(model)
        finally:
            for h in handles:
                h.remove()
        return seen

    hf, fast = both(kind, calls, mode)
    assert [(l, e) for l, e, *_ in hf] == [(l, e) for l, e, *_ in fast]
    for (_, _, x, y), (_, _, x2, y2) in zip(hf, fast):
        check(mode)((x, y), (x2, y2))
    empty = [c for c in hf if c[2].shape[0] == 0]
    if kind == "qwen":           # Qwen3-MoE calls only the routed experts, OLMoE all of them
        assert not empty and len(hf) < LAYERS * N_EXPERTS
    else:
        assert len(hf) == LAYERS * N_EXPERTS


@pytest.mark.parametrize("masked", [False, True])
def test_mean_ablation_with_and_without_a_position_mask_agrees(kind, mode, masked):
    def ablated(model):
        with hooks.ablate(model, COMPS, means_for(model), alpha=0.0):
            if not masked:
                return forward(model)
            with hooks.position_mask(model, torch.tensor([[0, 1, 0, 1, 1, 0, 0, 1, 1]] * 3, dtype=torch.bool)):
                return forward(model)

    hf, fast = both(kind, ablated, mode)
    check(mode)(hf, fast)
    assert not torch.allclose(hf[0], both(kind, forward)[0][0])        # the lesion changed something


def test_router_masking_agrees(kind, mode):
    def masked(model):
        with hooks.mask_router(model, [c for c in COMPS if c.kind == "expert"]):
            return forward(model)

    check(mode)(*both(kind, masked, mode))


def gate_step(model, checkpointing=False, ids=IDS, mask=MASK):
    g = gates.Gates(COMPS, init=0.5).to(model.device)
    if checkpointing:
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        with gates.gated(model, g, means_for(model)):
            g.recording = True
            logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits
            g.recording = False
            loss = logits.float().log_softmax(-1)[..., 5].mean()
            loss.backward()
    finally:
        if checkpointing:
            model.gradient_checkpointing_disable()
            model.eval()
    return logits.detach(), g.logits.grad.clone(), g.routed.clone()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_gate_gradients_agree_also_under_gradient_checkpointing(kind, mode, checkpointing):
    hf, fast = both(kind, lambda m: gate_step(m, checkpointing), mode)
    check(mode)(hf, fast)
    assert hf[1].abs().sum() > 0


def full_backward(model, trainable, ids=IDS, mask=MASK):
    """Logits, every layer's output, router logits, and the gradients of the input embeddings and of every weight that
    trainable(name) selects."""
    for name, p in model.named_parameters():
        p.requires_grad_(trainable(name))
        p.grad = None
    emb = model.model.embed_tokens(ids.to(model.device)).detach().requires_grad_()
    out = model(inputs_embeds=emb, attention_mask=mask.to(model.device), output_hidden_states=True,
                output_router_logits=True)
    out.logits.float().log_softmax(-1)[..., 5].mean().backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    return dict(logits=out.logits.detach(), hidden=[h.detach() for h in out.hidden_states],
                router=[r.detach() for r in out.router_logits], embedding_grad=emb.grad.clone(), grads=grads)


def test_every_gradient_and_every_layer_output_agrees(kind, mode):
    # the grouped forward needs frozen expert weights: there, every other weight is trained
    trainable = (lambda name: True) if mode == "sorted" else (lambda name: ".experts." not in name)
    hf, fast = both(kind, lambda m: full_backward(m, trainable), mode)
    check(mode)(hf, fast)
    assert any(".experts." in k for k in hf["grads"]) == (mode == "sorted")
    assert any(".gate.weight" in k for k in hf["grads"])


PRODUCTION_EXPERTS = [Component("expert", 0, 1), Component("expert", 1, 5), Component("expert", 1, 11)]


def production(model, tok):
    """The lesion modules' own paths on the tiny Qwen: means, attribution, expert norms, naming and M1 scores (intact,
    mean-ablated, ablated at chosen positions, router-masked)."""
    from lesion import attribution, damage, materials, means as means_mod, naming, neutral_text, probes, \
        prefix_scoring, scoring
    from lesion import knowledge_items as ki
    import pandas as pd

    chat = True
    texts = neutral_text.load()[:3]
    m = means_mod.collect_means(model, tok, texts, max_tokens=32, batch_size=3)
    all_items, demos, templates = scoring.load_materials()
    items = materials.load_split("screening").groupby("task").head(3)
    prompts = scoring.build_label_prompts(items, demos, templates, chat=chat)
    weights = attribution.statistic_weights(prompts, items, damage.frequency_median(all_items))
    comps = PRODUCTION_EXPERTS + [Component("head", 0, 2)]
    out = dict(means=(m.expert_mean, m.expert_count, m.head_mean),
               attribution=attribution.component_attribution(model, tok, prompts, chat, m, weights, batch_size=3))
    seqs = [tok(x, add_special_tokens=False).input_ids[:24] for x in texts[:2]]
    out["norms"] = probes.expert_output_norms(model, seqs, PRODUCTION_EXPERTS)
    out["m1"] = prefix_scoring.score_label_prompts(model, tok, prompts, chat, 4)
    with hooks.ablate(model, comps, m):
        out["m1_ablated"] = prefix_scoring.score_label_prompts(model, tok, prompts, chat, 4)
    out["m1_positions"] = probes.label_scores_in_positions(model, tok, prompts, chat, positions=[[-1, -2]] * len(prompts),
                                                           batch_size=4, lesion=(comps, m))
    n_items = ki.load_items("screening").head(6)
    n_prompts = naming.build_prompts(n_items, pd.read_csv(ki.OUT / "demos.csv"), chat)
    out["naming"] = naming.answer_logprobs(model, tok, n_prompts, chat, batch_size=4)
    with hooks.mask_router(model, PRODUCTION_EXPERTS):
        out["naming_masked"] = naming.answer_logprobs(model, tok, n_prompts, chat, batch_size=4)
    return out


def test_the_lesion_modules_production_paths_agree(mode):
    from lesion import models

    if not models.local_dir("qwen3-30b-a3b-instruct").exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny("qwen3-30b-a3b-instruct", num_experts=N_EXPERTS, num_experts_per_tok=TOP_K)
    for p in model.parameters():
        p.requires_grad_(False)
    hf = run(model, None, lambda m: production(m, tok))
    fast = run(model, mode, lambda m: production(m, tok))
    check(mode)(hf, fast)
    assert not hf["m1"].equals(hf["m1_ablated"]) and not hf["naming"].equals(hf["naming_masked"])


def test_install_refuses_other_moe_blocks_and_impl_reports_the_state():
    config = dict(vocab_size=43, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
                  rope_scaling=None)
    for mode in fast_moe.MODES:
        with pytest.raises(TypeError, match=f"no {mode} forward"):
            fast_moe.install(PhimoeForCausalLM(PhimoeConfig(**config)), mode)
    model = build("qwen")
    assert fast_moe.impl(model) == "hf"
    with pytest.raises(ValueError, match="mode"):
        fast_moe.install(model, "fused")
    for mode in fast_moe.MODES:
        fast_moe.install(model, mode)
        assert fast_moe.impl(model) == mode
    moe_block(model.model.layers[0])._lesion_moe_mode = "sorted"
    with pytest.raises(RuntimeError, match="different forwards"):
        fast_moe.impl(model)
    fast_moe.uninstall(model)
    assert fast_moe.impl(model) == "hf" and "forward" not in vars(moe_block(model.model.layers[0]))


def test_grouped_keeps_the_expert_modules_and_state_dict_working_on_views_of_the_stacked_weights():
    model = build("qwen")
    expert = moe_block(model.model.layers[1]).experts[5]
    x = torch.randn(3, HIDDEN)
    with torch.no_grad():
        before, state = expert(x), {k: v.clone() for k, v in model.state_dict().items()}
    n_params = sum(p.numel() for p in model.parameters())
    fast_moe.install(model, "grouped")
    with torch.no_grad():
        assert_identical(expert(x), before)
    assert sum(p.numel() for p in model.parameters()) == n_params
    after = model.state_dict()
    assert after.keys() == state.keys() and all(torch.equal(after[k], state[k]) for k in state)
    stacked = moe_block(model.model.layers[1]).forward.__func__.keywords["stack"].weights["down_proj"]
    assert expert.down_proj.weight.data_ptr() == stacked[5].data_ptr()


def test_grouped_is_deterministic_and_refuses_what_it_cannot_reproduce(kind):
    model = build(kind)
    fast_moe.install(model, "grouped")
    assert_identical(gate_step(model), gate_step(model))
    block, expert = moe_block(model.model.layers[0]), moe_block(model.model.layers[0]).experts[3]

    def refuses(match):
        with pytest.raises(RuntimeError, match=match):
            forward(model)

    h = expert.register_forward_pre_hook(lambda *_: None)
    refuses("only plain forward hooks")
    h.remove()
    h = expert.down_proj.register_forward_hook(lambda *_: None)
    refuses("submodules")
    h.remove()
    h = expert.register_forward_hook(lambda *_: None, always_call=True)
    refuses("only plain forward hooks")
    h.remove()
    h = expert.register_forward_hook(lambda *_: None, with_kwargs=True)
    refuses("only plain forward hooks")
    h.remove()

    def in_place(_m, inputs, _out):
        inputs[0].mul_(2)

    handles = [e.register_forward_hook(in_place) for e in block.experts]
    refuses("in place")
    for h in handles:
        h.remove()
    handles = [e.register_forward_hook(lambda _m, _i, out: out[:, :1]) for e in block.experts]
    refuses("tensor of its output's shape")
    for h in handles:
        h.remove()
    forward(model)                                                  # nothing left over
    expert.up_proj.weight.requires_grad_(True)
    refuses("weight changed")
    expert.up_proj.weight.requires_grad_(False)
    model.double()                                                  # converts each view separately
    refuses("weight changed")
    fast_moe.install(model, "grouped")
    forward(model)
    model.half()
    with pytest.raises(RuntimeError, match="weight changed"):
        model(input_ids=IDS)
    fast_moe.uninstall(model)
    model.float()
    for p in block.experts.parameters():
        p.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        fast_moe.install(model, "grouped")
    fast_moe.install(model, "sorted")                               # training the experts goes through 'sorted'


def test_groups_cover_every_expert_once_and_bound_the_padding(monkeypatch):
    counts = [0, 5, 7, 0, 0, 30, 2, 2, 9, 1, 0, 64, 3]
    for min_rows, factor in ((1, 1), (1, 3), (16, 2), (8192, 3)):
        monkeypatch.setattr(fast_moe, "MIN_ROWS", min_rows)
        monkeypatch.setattr(fast_moe, "PAD_FACTOR", factor)
        groups = fast_moe._groups(counts)
        assert [a for a, _, _ in groups] == [0] + [b for _, b, _ in groups[:-1]] and groups[-1][1] == len(counts)
        for a, b, width in groups:
            assert width == max(max(counts[a:b]), 1)          # a run of unrouted experts gets width 1 (it is skipped)
            assert b - a == 1 or (b - a) * max(counts[a:b]) <= max(factor * sum(counts[a:b]), min_rows)
    assert len(fast_moe._groups(counts)) == 1


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("model_kind", ["qwen", "olmoe"])
def test_on_the_gpu_sorted_is_bit_identical_and_grouped_repeats_and_is_as_accurate_as_hf(dtype, model_kind):
    torch.manual_seed(0)
    ids = torch.randint(0, 43, (16, 40))
    mask = torch.ones_like(ids)
    mask[3, :7] = 0

    def scaled(model):
        with torch.no_grad():
            for block in (moe_block(layer) for layer in model.model.layers):
                for expert in block.experts:          # outputs of a realistic size, as for the real model
                    expert.down_proj.weight.mul_(30)
        return model

    def go(model):
        return gate_step(model, ids=ids, mask=mask)

    def everything(model):
        return full_backward(model, lambda name: True, ids=ids, mask=mask)

    def no_experts(model):
        return full_backward(model, lambda name: ".experts." not in name, ids=ids, mask=mask)

    reference = run(scaled(build(model_kind, device="cuda")), None, go)          # float32
    hf = run(scaled(build(model_kind, dtype=dtype, device="cuda")), None, go)
    assert_identical(hf, run(scaled(build(model_kind, dtype=dtype, device="cuda")), "sorted", go))
    assert_identical(run(scaled(build(model_kind, dtype=dtype, device="cuda")), None, everything),
                     run(scaled(build(model_kind, dtype=dtype, device="cuda")), "sorted", everything))
    model = scaled(build(model_kind, dtype=dtype, device="cuda"))
    fast_moe.install(model, "grouped")
    grouped = go(model)
    for _ in range(2):
        assert_identical(grouped, go(model))
    first = no_experts(model)
    assert_identical(first, no_experts(model))
    for i in (0, 1):                                  # logits and gate gradients: errors against float32
        err_hf = (hf[i].float() - reference[i]).abs().mean()
        err_grouped = (grouped[i].float() - reference[i]).abs().mean()
        assert err_grouped <= 1.2 * err_hf + 1e-6
    model.to("cpu")
    with pytest.raises(RuntimeError, match="weight changed"):
        model(input_ids=ids)


@CUDA
def test_on_the_gpu_the_grouped_block_uses_no_nondeterministic_operation(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    model = build("qwen", dtype=torch.float16, device="cuda")
    fast_moe.install(model, "grouped")
    block = moe_block(model.model.layers[0])
    x = torch.randn(4, 30, HIDDEN, device="cuda", dtype=torch.float16, requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            out, _ = block(x)
            out.float().pow(2).sum().backward()
        finally:
            torch.use_deterministic_algorithms(False)
    alerts = [str(w.message) for w in caught if "deterministic" in str(w.message) and "CuBLAS" not in str(w.message)]
    assert not alerts and x.grad is not None

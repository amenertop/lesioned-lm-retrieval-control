"""lesion.probes on tiny random Qwen3-MoE and Phi-MoE models (no tokenizer needed)."""
from types import SimpleNamespace

import pytest
import torch
from transformers import PhimoeConfig, PhimoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from lesion import hooks, probes
from lesion.components import Component, moe_block

SEQS = [[2, 7, 3, 9, 5, 11], [6, 4, 8], [1, 3, 5, 7], [9, 9, 2, 4, 6]]


@pytest.fixture(scope="module", params=["qwen", "phi"])
def tiny(request):
    torch.manual_seed(5)
    config = dict(vocab_size=43, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
                  max_position_embeddings=64, attention_dropout=0.0, use_cache=False)
    if request.param == "qwen":
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**config, head_dim=8, moe_intermediate_size=24))
    else:
        config["num_local_experts"] = config.pop("num_experts")
        model = PhimoeForCausalLM(PhimoeConfig(**config, rope_scaling=None))
    model.eval()
    model.config._attn_implementation = "eager"
    head_dim = probes._head_dim(model)
    means = SimpleNamespace(expert_mean=torch.randn(2, 4, 32) * .3, head_mean=torch.randn(2, 4, head_dim) * .3)
    return model, means


def test_masks_mark_last_and_first_real_tokens():
    attention = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]])
    assert probes.last_position_mask(attention)[:, -1].all() and probes.last_position_mask(attention).sum() == 3
    assert probes.first_token_mask(attention).tolist() == [[False, False, True, False], [True, False, False, False],
                                                           [False, True, False, False]]


def test_head_outputs_do_not_depend_on_batching(tiny):
    model, _ = tiny
    heads = [Component("head", 0, 1), Component("head", 1, 3)]
    one = probes.head_outputs_last(model, SEQS, heads, batch_size=1)
    many = probes.head_outputs_last(model, SEQS, heads, batch_size=4)
    for h in heads:
        assert one[h.key].shape == (len(SEQS), probes._head_dim(model))
        assert torch.allclose(one[h.key], many[h.key], atol=1e-5)


def test_patching_a_head_with_its_own_last_output_changes_nothing(tiny):
    model, means = tiny
    head = Component("head", 1, 2)
    seq = [SEQS[0]]
    own = probes.head_outputs_last(model, seq, [head])[head.key][0]
    patched = probes.patched_means(means, {head: own})
    assert torch.equal(means.head_mean, means.head_mean)  # the original is untouched
    assert not torch.equal(patched.head_mean, means.head_mean)
    ids = torch.tensor(seq)
    attention = torch.ones_like(ids)
    with torch.no_grad():
        intact = model(input_ids=ids, attention_mask=attention).logits
        with hooks.ablate(model, [head], patched), hooks.position_mask(model, probes.last_position_mask(attention)):
            same = model(input_ids=ids, attention_mask=attention).logits
        with hooks.ablate(model, [head], means), hooks.position_mask(model, probes.last_position_mask(attention)):
            other = model(input_ids=ids, attention_mask=attention).logits
    assert torch.allclose(intact, same, atol=1e-5)
    assert torch.allclose(intact[:, :-1], other[:, :-1], atol=1e-6)      # only the last position may change
    assert not torch.allclose(intact[:, -1], other[:, -1], atol=1e-6)


def test_expert_output_norms_match_a_direct_recomputation(tiny):
    model, _ = tiny
    expert = Component("expert", 0, 1)
    block = moe_block(model.model.layers[0])
    seen = []
    handle = block.register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[0].detach().clone()))
    try:
        norms, firsts = probes.expert_output_norms(model, SEQS, [expert])
    finally:
        handle.remove()
    with torch.no_grad():
        for s, seq in enumerate(SEQS):
            hidden = seen[s].reshape(-1, seen[s].shape[-1])
            logits = block.gate(hidden)
            chosen = hooks._routed_experts(block, logits, block.top_k)
            tokens = sorted(torch.where((chosen == expert.index).any(-1))[0].tolist())
            got = norms[norms.sequence == s].sort_values("position")
            assert got.position.tolist() == tokens
            direct = block.experts[expert.index](hidden[tokens]).float().norm(dim=-1)
            assert torch.allclose(torch.tensor(got.norm.to_numpy(), dtype=torch.float32), direct, atol=1e-5)
    n_first = int((norms.position == 0).sum())
    assert firsts[expert.key].shape[0] == n_first
    summary = probes.norm_summary(norms, len(SEQS))[expert.key]
    assert summary["n_routed"] == len(norms) and 0 <= summary["share_of_first_tokens_routed"] <= 1


def test_lm_loss_regions(tiny):
    model, means = tiny
    comps = [Component("expert", 0, 1), Component("head", 1, 0)]
    plain = probes.lm_loss_region(model, SEQS, [], means)
    everywhere = probes.lm_loss_region(model, SEQS, comps, means)
    first = probes.lm_loss_region(model, SEQS, comps, means, region="first")
    rest = probes.lm_loss_region(model, SEQS, comps, means, region="rest")
    # everywhere equals a hand-run ablation without a mask
    total, count = 0.0, 0
    with torch.no_grad(), hooks.ablate(model, comps, means):
        for seq in SEQS:
            ids = torch.tensor([seq])
            lg = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, :-1].float()
            total += float(torch.nn.functional.cross_entropy(lg, ids[0, 1:], reduction="sum"))
            count += len(seq) - 1
    assert everywhere == pytest.approx(total / count, abs=1e-5)
    assert len({round(x, 6) for x in (plain, everywhere, first, rest)}) == 4
    with pytest.raises(ValueError):
        probes.lm_loss_region(model, SEQS, comps, means, region="middle")


def test_token_zipf_keeps_only_whole_words_with_a_known_frequency():
    vocab = [" river", "river", " New", " x2", " zzqx", ".", " the"]
    tok = type("Tok", (), {"decode": lambda self, ids: vocab[ids[0]], "__len__": lambda self: len(vocab)})()
    freq = {"river": 4.5, "new": 6.0, "the": 7.7}
    z = probes.token_zipf(tok, lambda w: freq.get(w, 0.0))
    assert z[0] == 4.5 and z[2] == 6.0 and z[6] == 7.7
    assert all(map(lambda v: v != v, z[[1, 3, 4, 5]]))  # NaN
    bare = probes.token_zipf(tok, lambda w: freq.get(w, 0.0), leading_space=False, size=10)
    assert len(bare) == 10 and bare[1] == 4.5 and all(map(lambda v: v != v, bare[[0, 2, 6, 7, 8, 9]]))


def test_vocab_shift_reads_a_tilt_towards_frequent_words():
    torch.manual_seed(0)
    zipf = torch.linspace(1, 7, 200).numpy()
    zipf[::10] = float("nan")
    intact = torch.log_softmax(torch.randn(6, 200), dim=-1)
    tilt = torch.nan_to_num(torch.as_tensor(zipf), nan=0.0).float()
    lesioned = torch.log_softmax(intact + 0.3 * tilt, dim=-1)       # frequent tokens gain, rare ones lose
    out = probes.vocab_shift(intact, lesioned, zipf)
    assert (out.spearman > 0.99).all()
    assert (out.kl_change < 0).all()                                 # closer to the word-frequency distribution
    assert (out.bin0 < out.bin4).all()
    with pytest.raises(ValueError):
        probes.vocab_shift(intact, lesioned, zipf[:-1])
    same = probes.vocab_shift(intact, intact, zipf)
    assert abs(same.kl_change).max() < 1e-5 and same.spearman.isna().all()  # no change, no correlation


def test_partial_projections_match_the_full_forward(tiny):
    model, _ = tiny
    ids = torch.tensor([SEQS[0] + SEQS[2], SEQS[3] + SEQS[1] + [3, 3]])
    with torch.no_grad():
        full = torch.log_softmax(model(input_ids=ids).logits.float(), -1)
        assert torch.allclose(probes.position_logprobs(model, ids, [2, 5, 9]), full[:, [2, 5, 9]], atol=1e-5)
        last = probes.last_logprobs(model, SEQS, batch_size=3)
        for i, seq in enumerate(SEQS):
            one = torch.log_softmax(model(input_ids=torch.tensor([seq])).logits[0, -1].float(), -1)
            assert torch.allclose(last[i], one, atol=1e-5)


class _Tok:
    """Enough of a tokenizer for scoring.encode (base-model path) and label ids."""
    eos_token_id, unk_token_id = 0, None

    def encode(self, text, add_special_tokens=False):
        return [1 + (ord(c) % 40) for c in text][-12:]

    def convert_tokens_to_ids(self, tokens):
        return [1 for _ in tokens]


def test_label_scores_at_answer_match_score_label_prompts_and_lesion_only_the_answer(tiny, monkeypatch):
    from lesion import scoring

    model, means = tiny
    monkeypatch.setattr(scoring, "label_token_ids", lambda tokenizer, chat, labels=scoring.LABELS, **kw: {"A": 41, "B": 42})
    tok = _Tok()
    prompts = [scoring.LabelPrompt(str(i), "colour", i % 2, "abcdefgh"[: 3 + i], "A" if i % 2 else "B",
                                   {"A": "option1", "B": "option2"}) for i in range(5)]
    a = scoring.score_label_prompts(model, tok, prompts, False, 2)
    b = probes.label_scores_at_answer(model, tok, prompts, False, 3)
    head = Component("head", 1, 1)
    c = probes.label_scores_at_answer(model, tok, prompts, False, 3, ([head], means))
    expected = []
    with torch.no_grad():
        for p in prompts:
            ids = torch.tensor([scoring.encode(tok, p.text, False)])
            with hooks.ablate(model, [head], means), hooks.position_mask(model, probes.last_position_mask(torch.ones_like(ids))):
                expected.append(model(input_ids=ids).logits[0, -1, [41, 42]].tolist())
    ab = lambda d: torch.tensor(d[["logit_A", "logit_B"]].to_numpy(), dtype=torch.float32)  # noqa: E731
    assert torch.allclose(ab(a), ab(b), atol=1e-5)
    assert torch.allclose(ab(c), torch.tensor(expected), atol=1e-5)
    assert not torch.allclose(ab(b), ab(c), atol=1e-6)


def test_rerouting_sends_nothing_to_the_masked_experts(tiny):
    """Item 7 (session 12): under hooks.mask_router the router skips the expert in both architectures (Phi's
    sparsemixer included), so it gets no rows, and the language-model loss stays finite."""
    model, means = tiny
    expert = Component("expert", 0, 1)
    seen = []
    handle = moe_block(model.model.layers[0]).experts[1].register_forward_hook(lambda _m, _i, out: seen.append(out.shape[0]))
    try:
        plain = probes.lm_loss_region(model, SEQS, [], means)
        seen.clear()
        with hooks.mask_router(model, [expert]):
            rerouted = probes.lm_loss_region(model, SEQS, [], means)
    finally:
        handle.remove()
    assert sum(seen) == 0
    assert rerouted == rerouted and abs(rerouted - plain) < 5  # finite, and a different routing


def test_rerouting_two_experts_in_a_float16_phi():
    """As the session 12 run does it: Phi in float16 (sparsemixer), two experts of one layer and one of another
    skipped together; none of them gets a row and the loss stays finite."""
    torch.manual_seed(7)
    config = dict(vocab_size=43, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
                  attention_dropout=0.0, use_cache=False)
    model = PhimoeForCausalLM(PhimoeConfig(**config, rope_scaling=None)).eval().to(torch.float16)
    model.config._attn_implementation = "eager"
    skipped = [Component("expert", 0, 1), Component("expert", 0, 2), Component("expert", 1, 0)]
    seen = []
    handles = [moe_block(model.model.layers[e.layer]).experts[e.index].register_forward_hook(
        lambda _m, _i, out: seen.append(out.shape[0])) for e in skipped]
    try:
        with hooks.mask_router(model, skipped):
            loss = probes.lm_loss_region(model, SEQS, [], None)
    finally:
        for h in handles:
            h.remove()
    assert sum(seen) == 0
    assert torch.isfinite(torch.tensor(loss))


def test_label_scores_in_positions_cover_all_none_and_a_mix(tiny, monkeypatch):
    from lesion import scoring

    model, means = tiny
    monkeypatch.setattr(scoring, "label_token_ids", lambda tokenizer, chat, labels=scoring.LABELS, **kw: {"A": 41, "B": 42})
    tok = _Tok()
    prompts = [scoring.LabelPrompt(str(i), "size", i % 2, "abcdefghij"[: 3 + i], "A" if i % 2 else "B",
                                   {"A": "option1", "B": "option2"}) for i in range(5)]
    lesion = ([Component("head", 0, 2), Component("head", 1, 1)], means)
    lengths = [len(scoring.encode(tok, p.text, False)) for p in prompts]
    ab = lambda d: torch.tensor(d[["logit_A", "logit_B"]].to_numpy(), dtype=torch.float32)  # noqa: E731
    intact = ab(probes.label_scores_in_positions(model, tok, prompts, False, None, 3))
    everywhere = ab(probes.label_scores_in_positions(model, tok, prompts, False, None, 3, lesion))
    listed = ab(probes.label_scores_in_positions(model, tok, prompts, False, [list(range(n)) for n in lengths], 2, lesion))
    nowhere = ab(probes.label_scores_in_positions(model, tok, prompts, False, [[] for _ in prompts], 3, lesion))
    assert torch.allclose(everywhere, listed, atol=1e-5)
    assert torch.allclose(nowhere, intact, atol=1e-5)
    assert not torch.allclose(everywhere, intact, atol=1e-4)
    # an arbitrary subset, padded inside a batch, against one prompt at a time without padding
    chosen = [[0, n - 2] for n in lengths]
    mixed = ab(probes.label_scores_in_positions(model, tok, prompts, False, chosen, 4, lesion))
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = torch.tensor([scoring.encode(tok, p.text, False)])
            where = torch.zeros(ids.shape, dtype=torch.bool)
            where[0, chosen[i]] = True
            with hooks.ablate(model, *lesion), hooks.position_mask(model, where):
                assert torch.allclose(mixed[i], model(input_ids=ids).logits[0, -1, [41, 42]].float(), atol=1e-5)
    with pytest.raises(IndexError):
        probes.label_scores_in_positions(model, tok, prompts, False, [[99]] * len(prompts), 3, lesion)


@pytest.mark.parametrize("name,chat", [("olmoe-0924-base", False), ("qwen3-30b-a3b-instruct", True)])
def test_position_parts_partition_an_m1_prompt(name, chat):
    from lesion import models, scoring

    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    tok = models.load_tokenizer(name)
    items, demos, templates = scoring.load_materials()
    for p in scoring.build_label_prompts(items[items.task == "colour"].head(2), demos, templates, chat=chat):
        ids, parts = probes.position_parts(tok, p, chat)
        assert set(parts) == set(probes.PARTS)
        flat = sorted(j for v in parts.values() for j in v)
        assert flat == list(range(len(ids)))                    # disjoint and complete
        assert parts["answer"] == [len(ids) - 1] and parts["start"][0] == 0
        words = sorted(tok.decode([ids[j]]).strip() for j in parts["options"])
        assert "".join(words) and tok.decode([ids[j] for j in parts["probe"]]).strip().startswith("Word:")
        assert bool(parts["end_template"]) == chat
        assert max(parts["context"]) < min(parts["probe"]) < min(parts["options"])


def test_target_change_rank():
    intact = torch.log_softmax(torch.tensor([[4.0, 3.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, 3.0, -2.0]]), -1)
    lesioned = intact.clone()
    lesioned[0, 0] -= 2.0                     # the target loses the most
    lesioned[1, 2] -= 0.5
    lesioned[1, 3] -= 3.0                     # a competitor loses more than the target
    out = probes.target_change_rank(intact, lesioned, [0, 2], top=3)
    assert out.target_change.tolist() == pytest.approx([-2.0, -0.5])
    assert out.share_at_least_as_negative.tolist() == pytest.approx([0.2, 0.4])
    assert out.rank_in_top.tolist() == pytest.approx([1.0, 2.0])
    assert probes.target_change_rank(intact, lesioned, [4, 4], top=2).rank_in_top.isna().all()
    with pytest.raises(ValueError):
        probes.target_change_rank(intact, lesioned[:1], [0, 2])


def test_gptoss_expert_norms_are_read_only_and_add_up():
    from lesion import models

    if not models.local_dir("gpt-oss-20b").exists():
        pytest.skip("config not downloaded")
    model, _ = models.load_tiny("gpt-oss-20b")
    seqs = [[5, 9, 2, 7, 11, 3], [4, 4, 8]]
    with torch.no_grad():
        before = [model(input_ids=torch.tensor([s])).logits for s in seqs]
    check = {}
    rows, means = probes.gptoss_expert_norms(model, seqs, check=check)      # raises if the outputs do not add up
    assert check["max_abs"] < 1e-4 and check["max_rel_scale"] < 1e-4 and check["max_rel_l2"] < 1e-3
    with torch.no_grad():
        after = [model(input_ids=torch.tensor([s])).logits for s in seqs]
    assert all(torch.equal(a, b) for a, b in zip(before, after))            # no hook left behind, nothing changed
    k = model.config.num_experts_per_tok
    assert len(rows) == model.config.num_hidden_layers * k * sum(len(s) for s in seqs)
    assert (rows.norm > 0).all() and set(rows.token_id) <= {t for s in seqs for t in s}
    assert sum(v["count"] for v in means.values()) == len(rows)


def test_outputs_agree_tolerates_bf16_rounding_of_large_outputs_but_not_a_missing_expert():
    g = torch.Generator().manual_seed(0)
    parts = torch.randn(4, 50, 64, generator=g) * 40                 # four weighted expert outputs per token
    total, scale = parts.sum(0), parts.norm(dim=-1).sum(0)
    rounded = parts.bfloat16().float().sum(0).bfloat16()             # the module: rounded terms, a rounded sum
    assert not torch.allclose(total, rounded.float(), atol=1e-2, rtol=2e-2)     # the old entry-wise check fails
    agree, stats = probes.outputs_agree(total, scale, rounded)
    assert agree and stats["max_rel_scale"] < 5e-3
    agree, stats = probes.outputs_agree(total, scale, parts[:3].sum(0).bfloat16())   # one expert left out
    assert not agree and stats["max_rel_scale"] > 0.1
    assert probes.outputs_agree(total, scale, total)[0]                 # fp32: a strict tolerance
    assert not probes.outputs_agree(total, scale, total * 1.001)[0]
    bad = rounded.clone()
    bad[3, 5] = float("nan")
    assert not probes.outputs_agree(total, scale, bad)[0]
    # the documented limit: in bf16 an expert adding about 1% of what was added can be left out unnoticed
    small = parts.clone()
    small[3] = small[3] / small[3].norm(dim=-1, keepdim=True) * 0.01 * small[:3].norm(dim=-1).sum(0)[..., None]
    total, scale = small.sum(0), small.norm(dim=-1).sum(0)
    assert probes.outputs_agree(total, scale, small[:3].sum(0).bfloat16())[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU: gpt-oss adds all experts at once only there")
def test_gptoss_check_passes_on_the_gpu_path_in_bf16_with_large_outputs():
    from lesion import models

    if not models.local_dir("gpt-oss-20b").exists():
        pytest.skip("config not downloaded")
    model, _ = models.load_tiny("gpt-oss-20b", dtype=torch.bfloat16)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.experts.down_proj.mul_(3000)                   # outputs up to tens, as in session 14's failure
    model = model.cuda()
    check = {}
    probes.gptoss_expert_norms(model, [[5, 9, 2, 7, 11, 3, 8, 1, 4, 6] * 3], check=check)
    assert check["max_abs"] > 1e-2 and check["max_rel_scale"] < 2e-2


@pytest.mark.parametrize("name,chat", [("olmoe-0924-base", False), ("qwen3-30b-a3b-instruct", True)])
def test_split_labels_puts_a_b_and_answer_around_the_two_option_words(name, chat):
    from lesion import models, scoring

    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    tok = models.load_tokenizer(name)
    items, demos, templates = scoring.load_materials()
    for p in scoring.build_label_prompts(items[items.task == "size"].head(2), demos, templates, chat=chat):
        ids, parts = probes.position_parts(tok, p, chat)
        split = probes.split_labels(parts)
        assert set(split) == (set(parts) - {"labels"}) | set(probes.LABEL_PARTS)
        assert sorted(j for v in split.values() for j in v) == list(range(len(ids)))
        text = {k: tok.decode([ids[j] for j in split[k]]) for k in probes.LABEL_PARTS}
        # in base-model prompts the colon of "Answer:" is the answer position itself, so it is not a label token
        assert "A:" in text["label_a"] and "B:" in text["label_b"] and "Answer" in text["label_answer"], text
        assert max(split["label_a"]) < min(split["options"]) and max(split["options"]) < min(split["label_answer"])

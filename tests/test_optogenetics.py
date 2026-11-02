"""Optogenetics-inspired interventions (lesion.m3_optogenetics, segment regions in lesion.m3_battery): segments
partition every prompt, segment lesions compose to the whole lesion, alpha = 1 is the intact model, a self-implant
restores the intact model exactly, donor pairs follow their rules."""
import numpy as np
import pandas as pd
import pytest
import torch

from lesion import hooks, m3_battery as mb, m3_dla, m3_items, m3_optogenetics as mo, models, scoring
from lesion.components import Component

QWEN = "qwen3-30b-a3b-instruct"


@pytest.fixture(scope="module")
def final():
    return pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)


@pytest.fixture(scope="module")
def pool(final):
    return final[final.role.isin(mb.POOL_ROLES)].reset_index(drop=True)


@pytest.fixture(scope="module")
def qwen():
    if not models.local_dir(QWEN).exists():
        pytest.skip("tokenizer files not downloaded")
    return models.load_tiny(QWEN)


@pytest.fixture(scope="module")
def means(qwen):
    from lesion import means as means_mod, neutral_text

    model, tok = qwen
    return means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3)


def test_segments_partition_every_format(qwen, pool):
    _, tok = qwen
    prompts = mb.scored_prompts(pool.head(2)) + mb.label_items(pool.head(2), pool)
    for p in prompts:
        ids = scoring.encode(tok, p.text, True)
        c, q = mb.prompt_segments(tok, p.text, True)
        assert 0 < c and 0 < q and c + q < len(ids) - 1
        probe = tok.decode(ids[c:c + q])
        assert probe.startswith(("Description: ", "Word: ")) and "\n" not in probe
        assert "Answer:" in tok.decode(ids[c + q:len(ids) - 1])


def test_segment_masks_compose_to_the_old_regions():
    lengths, answers, width, bounds = [9, 7], [2, 0], 10, [(3, 2), (2, 2)]
    m = lambda r: mb.answer_mask(lengths, answers, width, r, bounds)
    assert torch.equal(m(mb.SEGMENTS), mb.answer_mask(lengths, answers, width, None))
    assert torch.equal(m(("context", "probe", "middle")), mb.answer_mask(lengths, answers, width, "prompt"))
    assert torch.equal(m(("answer",)), mb.answer_mask(lengths, answers, width, "answer"))
    parts = [m((s,)) for s in mb.SEGMENTS]
    assert int(sum(p.int() for p in parts).max()) == 1          # the four segments never overlap
    with pytest.raises(ValueError):
        mb.answer_mask(lengths, answers, width, ("probe",))       # bounds are required
    with pytest.raises(ValueError):
        m(("hint",))


def test_segment_lesion_of_all_segments_equals_the_plain_lesion(qwen, pool, means):
    model, tok = qwen
    comps = [Component("expert", 0, 3), Component("head", 1, 2)]
    prompts = mb.scored_prompts(pool.head(2))[:4]
    labels = mb.label_items(pool.head(2), pool)[:3]
    with hooks.ablate(model, comps, means=means):
        plain = mb.candidate_scores(model, tok, prompts, batch_size=4)
        segs = mb.candidate_scores(model, tok, prompts, batch_size=4, region=mb.SEGMENTS)
        plain_l = mb.label_scores(model, tok, labels, batch_size=4)
        segs_l = mb.label_scores(model, tok, labels, batch_size=4, region=mb.SEGMENTS)
        probe = mb.candidate_scores(model, tok, prompts, batch_size=4, region=("probe",))
    intact = mb.candidate_scores(model, tok, prompts, batch_size=4)
    assert np.allclose(plain.logprob, segs.logprob, atol=1e-4)
    assert np.allclose(plain_l.margin, segs_l.margin, atol=1e-4)
    assert not np.allclose(probe.logprob, intact.logprob, atol=1e-6, rtol=0)    # layer 0 expert: the probe change propagates


def test_alpha_one_is_intact_and_two_amplifies(qwen, pool, means):
    model, tok = qwen
    comps = [Component("expert", 1, 3), Component("head", 1, 2)]
    prompts = mb.scored_prompts(pool.head(1))[:3]
    intact = mb.candidate_scores(model, tok, prompts, batch_size=4)
    with hooks.ablate(model, comps, means=means, alpha=1.0):
        same = mb.candidate_scores(model, tok, prompts, batch_size=4)
    with hooks.ablate(model, comps, means=means, alpha=2.0):
        amp = mb.candidate_scores(model, tok, prompts, batch_size=4)
    with hooks.ablate(model, comps, means=means, alpha=0.0):
        abl = mb.candidate_scores(model, tok, prompts, batch_size=4)
    assert np.allclose(intact.logprob, same.logprob, atol=1e-5)
    assert not np.allclose(intact.logprob, amp.logprob, atol=1e-4)
    # mean + 2 (x - mean) moves the output as far beyond the intact model as ablation moves it back
    assert not np.allclose(amp.logprob, abl.logprob, atol=1e-4)


def test_lm_loss_is_finite_and_unchanged_at_alpha_one(qwen, means):
    from lesion import neutral_text

    model, tok = qwen
    texts = neutral_text.load()[:3]
    base = mo.lm_loss(model, tok, texts, max_tokens=32, batch_size=2)
    with hooks.ablate(model, [Component("head", 0, 1)], means=means, alpha=1.0):
        same = mo.lm_loss(model, tok, texts, max_tokens=32, batch_size=2)
    assert np.isfinite(base) and base > 0 and abs(base - same) < 1e-5


def test_self_implant_restores_the_intact_model(qwen, pool, means):
    model, tok = qwen
    comps = [Component("expert", 0, 3), Component("expert", 1, 5), Component("head", 1, 2)]
    plist = [p for p in mb.scored_prompts(pool.head(3)) if p.condition == "name_free"]
    prompts = {p.item_id: p for p in plist}
    vectors, _ = m3_dla.contributions(model, means, comps, plist, tok, batch_size=2, model_dtype=True)
    self_pairs = [(b, b) for b in prompts]
    intact = mo.intact_first_tokens(model, tok, prompts, self_pairs, batch_size=2)
    restored = mo.implant_scores(model, tok, means, comps, prompts, vectors, self_pairs, batch_size=2)
    lesion = mo.implant_scores(model, tok, means, comps, prompts, vectors, self_pairs, batch_size=2, add=False)
    assert np.array_equal(intact.lp_recipient, restored.lp_recipient)          # exact, not within a tolerance
    assert not np.allclose(intact.lp_recipient, lesion.lp_recipient, atol=1e-5)
    ids = list(prompts)
    donor = mo.implant_scores(model, tok, means, comps, prompts, vectors, [(ids[0], ids[1])], batch_size=2)
    assert not np.isclose(donor.lp_recipient.iloc[0], restored.lp_recipient.iloc[0], atol=1e-5)


def test_donor_pairs_follow_their_rules(final):
    items = final[final.role.isin(("calibration_a", "calibration_b"))].reset_index(drop=True)
    a, b = mo.donor_pairs(items, seed=0), mo.donor_pairs(items, seed=0)
    assert a.equals(b) and len(a) == len(items)
    rows = items.set_index("item_id")
    for r in a.itertuples():
        assert r.far != r.recipient and rows.at[r.far, "domain"] != rows.at[r.recipient, "domain"]
        if r.near_kind == "none":                     # alone in its domain: no forced near donor
            assert r.near == "" and (rows.domain == rows.at[r.recipient, "domain"]).sum() == 1
            continue
        assert r.near != r.recipient and rows.at[r.near, "target"] != rows.at[r.recipient, "target"]
        key = "category" if r.near_kind == "category" else "domain"
        assert rows.at[r.near, key] == rows.at[r.recipient, key]
    assert (a.near_kind == "category").sum() > len(a) / 2


def test_amplification_under_a_position_mask_leaves_other_positions_bitwise(qwen, pool, means):
    """alpha = 2 with a last-position mask: the expert block's output changes only at the masked position."""
    model, tok = qwen
    comps = [Component("expert", 1, c) for c in range(8)]          # every expert of the layer: some is routed at the last position
    p = [x for x in mb.scored_prompts(pool.head(1)) if x.condition == "name_free"][0]
    ids = torch.tensor([scoring.encode(tok, p.text, True)])
    block = model.model.layers[1].mlp
    seen = []
    h = block.register_forward_hook(lambda _m, _i, out: seen.append((out[0] if isinstance(out, tuple) else out).detach().clone()))
    try:
        with torch.no_grad():
            model.model(input_ids=ids)
            last = torch.zeros_like(ids, dtype=torch.bool)
            last[:, -1] = True
            with hooks.ablate(model, comps, means=means, alpha=2.0), hooks.position_mask(model, last):
                model.model(input_ids=ids)
    finally:
        h.remove()
    before, after = seen
    assert torch.equal(before[:, :-1], after[:, :-1])
    assert not torch.allclose(before[:, -1], after[:, -1])


def _layer_outputs(model, run):
    """Every decoder layer's output hidden states during run()."""
    seen = []
    hs = [layer.register_forward_hook(lambda _m, _i, out: seen.append((out[0] if isinstance(out, tuple) else out).detach().clone()))
          for layer in model.model.layers]
    try:
        with torch.no_grad():
            run()
    finally:
        for h in hs:
            h.remove()
    return seen


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_self_implant_is_bitwise_at_every_layer(qwen, pool, means, dtype):
    """Implant v2 (2026-10-12): swapping each component's own removed vector for itself changes nothing, layer by
    layer, also in a half-precision model (GPU session 9 failed this with ablate-then-add in float16). The expert
    block's output is a tuple (hidden states, router logits): only the hidden states may be touched. The batch must
    be built as lesion.m3_dla.contributions builds it (rows sorted by length): a row's float32 result can depend on
    its place in the batch, which is why the runner records and self-implants in the same order."""
    base_model, tok = qwen
    model = base_model.to(dtype)
    try:
        comps = [Component("expert", 0, 3), Component("expert", 0, 6), Component("expert", 1, 5), Component("head", 1, 2)]
        plist = [x for x in mb.scored_prompts(pool.head(2)) if x.condition == "name_free"]
        seqs = [scoring.encode(tok, x.text, True) for x in plist]
        order = next(scoring._batches(seqs, len(seqs)))
        ids, mask, pos = scoring._pad_left([seqs[i] for i in order], tok.eos_token_id or 0, "cpu")
        rec, _ = m3_dla.contributions(model, means, comps, plist, tok, batch_size=len(plist), model_dtype=True)
        own = {k: v[order] for k, v in rec.items()}
        run = lambda: model.model(input_ids=ids, attention_mask=mask, position_ids=pos)
        intact = _layer_outputs(model, run)
        with mo.implant(model, means, comps, own):
            implanted = _layer_outputs(model, run)
        assert all(torch.equal(a, b) for a, b in zip(intact, implanted, strict=True))
        router = []
        every = [Component("expert", 0, e) for e in range(8)]          # some expert is routed at the last position
        block = model.model.layers[0].mlp
        capture = lambda: block.register_forward_hook(lambda _m, _i, out: router.append(out))   # runs after the implant hook
        with torch.no_grad():
            h = capture()
            run()
            h.remove()
            with mo.implant(model, means, every):
                h = capture()
                run()
                h.remove()
        assert isinstance(router[0], tuple) and len(router[0]) == len(router[1])
        assert all(torch.equal(a, b) for a, b in zip(router[0][1:], router[1][1:]))      # router logits untouched
        assert not torch.equal(router[0][0][:, -1], router[1][0][:, -1])                 # the hidden states did change
        assert torch.equal(router[0][0][:, :-1], router[1][0][:, :-1])                   # only at the last position
    finally:
        base_model.to(torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["expert", "head"])
def test_lesion_only_implant_is_last_position_mean_ablation(qwen, pool, means, dtype, kind):
    """donor=None removes each component's own vector at the last position: layer by layer the same lesion as
    hooks.ablate under a last-position mask. The removed vectors follow the model's own dtype path (agent2 review
    2026-10-12), so what is left is the rounding of the block's sum over experts or o_proj's sum over heads: under
    2% of the lesion's own effect in float16, the dtype of every Qwen run (10% in bfloat16, with three fewer
    mantissa bits); with the float32 path it was of the order of the effect itself."""
    base_model, tok = qwen
    model = base_model.to(dtype)
    try:
        comps = ([Component("expert", 1, e) for e in range(8)] if kind == "expert"
                 else [Component("head", 1, 2), Component("head", 0, 1)])
        plist = [x for x in mb.scored_prompts(pool.head(4)) if x.condition == "name_free"]
        seqs = [scoring.encode(tok, x.text, True) for x in plist]
        ids, mask, pos = scoring._pad_left(seqs, tok.eos_token_id or 0, "cpu")
        last = torch.zeros_like(ids, dtype=torch.bool)
        last[:, -1] = True
        run = lambda: model.model(input_ids=ids, attention_mask=mask, position_ids=pos)
        with hooks.ablate(model, comps, means=means), hooks.position_mask(model, last):
            ablated = _layer_outputs(model, run)
        with mo.implant(model, means, comps):
            implanted = _layer_outputs(model, run)
        intact = _layer_outputs(model, run)
        for a, b, c in zip(ablated, implanted, intact, strict=True):
            effect = float((a.float() - c.float()).abs().max())
            gap = float((a.float() - b.float()).abs().max())
            assert gap <= max((0.10 if dtype == torch.bfloat16 else 0.02) * effect, 1e-6), (gap, effect)
        assert max(float((a.float() - c.float()).abs().max()) for a, c in zip(ablated, intact)) > 0
    finally:
        base_model.to(torch.float32)


def test_removed_vectors_reject_other_component_kinds(qwen, means):
    model, _ = qwen
    with pytest.raises(ValueError):
        m3_dla.removed_vector(model, means, Component("mlp", 0, 0), torch.zeros(1, model.config.hidden_size))
    with pytest.raises(ValueError):
        with mo.implant(model, means, [Component("mlp", 0, 0)]):
            pass


def test_router_selection_matches_the_routing_weights(qwen, pool):
    model, tok = qwen
    experts = [Component("expert", 1, e) for e in range(8)]
    plist = [x for x in mb.scored_prompts(pool.head(3)) if x.condition == "name_free"]
    df = mo.router_selection(model, tok, plist, experts, batch_size=2)
    assert len(df) == len(plist) * len(experts)
    top_k = model.model.layers[1].mlp.top_k
    per_prompt = df.groupby("item_id")
    assert (per_prompt.selected_last.sum() == top_k).all()                 # exactly k experts at the last position
    assert (per_prompt.n_selected.sum() == top_k * per_prompt.n_positions.first()).all()
    assert (df.n_selected <= df.n_positions).all()

"""Direct logit attribution (lesion.m3_dla): the extracted vectors are exactly what a last-position ablation removes."""
import pandas as pd
import pytest
import torch

from lesion import hooks, m3_dla, m3_items, m3_scoring as ms, means as means_mod, models, neutral_text, scoring
from lesion.components import Component

QWEN = "qwen3-30b-a3b-instruct"


@pytest.fixture(scope="module")
def setup():
    if not models.local_dir(QWEN).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(QWEN)
    m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3)
    f = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    prompts = ms.build_naming_prompts(f[f.role == "calibration_b"].head(3), ("none",), True, roles=ms.ERROR_ROLES)
    return model, tok, m, prompts


def final_residual(model, tok, prompts, ablate_comp=None, means=None):
    """Final residual stream at the last prompt position, optionally with one component ablated there only."""
    store = []
    h = model.model.norm.register_forward_pre_hook(lambda _m, inp: store.append(inp[0][:, -1].detach().float()))
    seqs = [ms.encode_answer(tok, p.text, p.candidates["target"], True)[0] for p in prompts]
    out = []
    try:
        with torch.no_grad():
            for s in seqs:
                ids = torch.tensor([s])
                if ablate_comp is None:
                    model.model(input_ids=ids)
                else:
                    mask = torch.zeros_like(ids, dtype=torch.bool)
                    mask[:, -1] = True
                    with hooks.ablate(model, [ablate_comp], means=means), hooks.position_mask(model, mask):
                        model.model(input_ids=ids)
                out.append(store.pop()[0])
    finally:
        h.remove()
    return torch.stack(out)


def test_removed_vectors_match_a_last_position_ablation(setup):
    model, tok, m, prompts = setup
    last = len(model.model.layers) - 1
    # an expert of the last layer that the first prompt's last token is routed to
    seq = ms.encode_answer(tok, prompts[0].text, prompts[0].candidates["target"], True)[0]
    store = {}
    hk = model.model.layers[last].mlp.register_forward_pre_hook(lambda _m, inp: store.__setitem__("x", inp[0][:, -1]))
    with torch.no_grad():
        model.model(input_ids=torch.tensor([seq]))
    hk.remove()
    block = model.model.layers[last].mlp
    routed = int(torch.topk(torch.softmax(block.gate(store["x"]).float(), -1), block.top_k, -1).indices[0, 0])
    comps = [Component("head", last, 2), Component("expert", last, routed)]
    contrib, final = m3_dla.contributions(model, m, comps, prompts, tok, batch_size=2)
    intact = final_residual(model, tok, prompts)
    assert torch.allclose(final, intact, atol=1e-4)
    # the expert is the last thing added to the residual stream: its removal changes it by exactly the vector
    expert = comps[1]
    ablated = final_residual(model, tok, prompts, expert, m)
    assert torch.allclose(intact - ablated, contrib[expert.key], atol=1e-4)
    # a head's removal also changes the same layer's MLP input; its own (direct) part is the change of o_proj's output
    head = comps[0]
    o = model.model.layers[last].self_attn.o_proj
    got = []
    for ab in (False, True):
        store = []
        hk = o.register_forward_hook(lambda _m, _i, out: store.append(out[:, -1].detach().float()))
        seqs = [ms.encode_answer(tok, p.text, p.candidates["target"], True)[0] for p in prompts]
        with torch.no_grad():
            for s in seqs:
                ids = torch.tensor([s])
                if ab:
                    mask = torch.zeros_like(ids, dtype=torch.bool)
                    mask[:, -1] = True
                    with hooks.ablate(model, [head], means=m), hooks.position_mask(model, mask):
                        model.model(input_ids=ids)
                else:
                    model.model(input_ids=ids)
        hk.remove()
        got.append(torch.cat(store))
    assert torch.allclose(got[0] - got[1], contrib[head.key], atol=1e-4)


def test_direct_effects_and_first_token_logits(setup):
    model, tok, m, prompts = setup
    comps = [Component("head", 1, 2), Component("expert", 1, 3)]
    contrib, final = m3_dla.contributions(model, m, comps, prompts, tok, batch_size=2)
    d = m3_dla.direct_effects(model, tok, prompts, contrib, final)
    assert len(d) == len(prompts) * len(comps)
    assert ((d.specific - (d.target - d.other_targets)).abs() < 1e-5).all()
    logits = m3_dla.first_token_logits(model, tok, prompts, batch_size=2)
    with torch.no_grad():
        for p, row in zip(prompts, logits.itertuples()):
            pre, ans = ms.encode_answer(tok, p.text, p.candidates["target"], True)
            full = model(input_ids=torch.tensor([pre])).logits[0, -1].float()
            assert abs(row.target_logit - float(full[ans[0]])) < 1e-3

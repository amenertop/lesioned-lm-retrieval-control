"""Cross-format battery (lesion.m3_battery), mechanism readouts (lesion.m3_mechanism) and the evidence-weighting
model (lesion.e3_analysis.evidence_weights)."""
import numpy as np
import pandas as pd
import pytest
import torch

from lesion import e3_analysis as ea, hooks, m3_battery as mb, m3_items, m3_mechanism as mm, m3_scoring as ms, models, scoring
from lesion.components import Component

QWEN = "qwen3-30b-a3b-instruct"


@pytest.fixture(scope="module")
def pool():
    f = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    return f[f.role.isin(mb.POOL_ROLES)].reset_index(drop=True)


@pytest.fixture(scope="module")
def qwen():
    if not models.local_dir(QWEN).exists():
        pytest.skip("tokenizer files not downloaded")
    return models.load_tiny(QWEN)


def test_foils_prefer_the_same_category_and_never_name_the_target(pool):
    fo = mb.foils(pool)
    p = pool.set_index("item_id")
    assert set(fo) == set(p.index) and all(k != v for k, v in fo.items())
    same_cat = sum(p.at[k, "category"] == p.at[v, "category"] for k, v in fo.items())
    assert same_cat >= 0.75 * len(fo)
    lone = {k for k in fo if (p.domain == p.at[k, "domain"]).sum() == 1}
    assert all(p.at[k, "domain"] == p.at[v, "domain"] for k, v in fo.items()
               if p.at[k, "category"] != p.at[v, "category"] and k not in lone)
    from lesion import knowledge_items as ki
    assert not any(ki.leaks(p.at[v, "definition"], p.at[k, "target"]) for k, v in fo.items())


def test_prompt_builders(pool):
    items = pool[pool.role == "calibration_b"].head(3)
    w2d = mb.word_to_definition_items(items, pool)
    assert len(w2d) == 6 and [w.correct for w in w2d[:2]] == ["A", "B"]
    row = items.iloc[0]
    assert f"A. {row.definition}" in w2d[0].text and f"B. {row.definition}" in w2d[1].text
    assert w2d[0].text.rstrip().endswith("Answer:") and f'Word: {row.target}' in w2d[0].text
    ow = mb.options_word_prompts(items)
    assert set(ow[0].candidates) == set(ms.ROLES) and ow[0].candidates["cannot_tell"] == ms.CANNOT_TELL
    assert ow[0].text.count("Answer:") == 4 and "Reply with the chosen word only" in ow[0].text
    hd = mb.name_hinted_demos_prompts(items)[0]
    assert hd.text.count('Hint: the word starts with "') == 3 and hd.text.rstrip().endswith("Answer:")
    cp = mb.copy_prompts(items)[0]
    assert cp.text.rstrip().endswith(f"Word: {row.target}\nAnswer:") and cp.candidates == dict(target=row.target)
    sp = mb.scored_prompts(items)
    assert {p.condition for p in sp} == {"name_free", "none_hinted_demos", "cue", "miscue", "options_word", "copy"}


def test_answer_mask_regions():
    m = {r: mb.answer_mask([5, 3], [2, 1], 6, r) for r in ("answer", "prompt", None)}
    assert m["answer"][0].tolist() == [False, False, False, True, True, True]
    assert m["prompt"][0].tolist() == [False, True, True, False, False, False]
    assert m[None][1].tolist() == [False, False, False, True, True, True]
    assert m["prompt"][1].tolist() == [False, False, False, True, False, False]
    assert not (m["answer"] & m["prompt"]).any() and ((m["answer"] | m["prompt"]) == m[None]).all()


def test_scores_match_the_reference_scorers(qwen, pool):
    model, tok = qwen
    items = pool[pool.role == "calibration_b"].head(2)
    prompts = mb.scored_prompts(items)
    got = mb.candidate_scores(model, tok, prompts, batch_size=5)
    ref = ms.score_naming_prompts(model, tok, prompts, batch_size=3)
    assert np.allclose(got.logprob, ref.logprob, atol=1e-3)
    li = mb.label_items(items, pool)
    lab = mb.label_scores(model, tok, li, batch_size=5)
    ids = scoring.label_token_ids(tok, True, ms.LETTERS)
    with torch.no_grad():
        for it, row in zip(li[:4], lab.head(4).itertuples()):
            logits = model(input_ids=torch.tensor([scoring.encode(tok, it.text, True)])).logits[0, -1].float()
            assert abs(getattr(row, f"logit_{it.correct}") - float(logits[ids[it.correct]])) < 1e-3


def test_greedy_and_classify(qwen, pool):
    model, tok = qwen
    items = pool.head(2)
    g = mb.greedy_answers(model, tok, mb.scored_prompts(items)[:2], batch_size=2, max_new_tokens=3)
    assert len(g) == 2 and g.answer.map(lambda s: isinstance(s, str)).all()
    row = dict(target="camel", coordinate="llama", superordinate="animal", associate="desert", unrelated="violin")
    assert mb.classify_answer("Camels.", row) == "target" and mb.classify_answer("animal", row) == "superordinate"
    assert mb.classify_answer("a horse", row) == "other" and mb.classify_answer("", row) == "empty"


def test_logit_lens_last_layer_is_the_model_output(qwen, pool):
    model, tok = qwen
    prompts = ms.build_naming_prompts(pool.head(2), ("none",), True, roles=ms.ERROR_ROLES)
    lens = mm.logit_lens(model, tok, prompts, batch_size=2)
    assert len(lens) == 2 * len(model.model.layers)
    with torch.no_grad():
        for p in prompts:
            pre, ans = ms.encode_answer(tok, p.text, p.candidates["target"], True)
            logp = torch.log_softmax(model(input_ids=torch.tensor([pre])).logits[0, -1].float(), -1)
            last = lens[(lens.item_id == p.item_id) & (lens.layer == len(model.model.layers) - 1)].iloc[0]
            assert abs(last.logprob - float(logp[ans[0]])) < 1e-3


def test_hint_spans_and_attention(qwen, pool):
    model, tok = qwen
    prompts = ms.build_naming_prompts(pool.head(2), ("cue", "miscue"), True)
    for p in prompts:
        ids, hint, definition = mm.hint_spans(tok, p.text)
        assert tok.decode([ids[j] for j in hint]).strip().strip('"') in p.text and definition
    model.config._attn_implementation = "eager"
    out = mm.hint_attention(model, tok, prompts, [Component("head", 1, 2)], batch_size=2)
    assert len(out) == len(prompts) and ((out.to_hint >= 0) & (out.to_hint <= 1)).all()


def test_battery_state_runs_end_to_end(qwen, pool):
    import sys
    from contextlib import nullcontext
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e3_errors"))
    import run_battery as rb

    model, tok = qwen
    items = pool[pool.role.isin(("calibration_a", "calibration_b"))].groupby("role").head(2).reset_index(drop=True)
    res = rb.run_state(model, tok, True, items, pool, ["x"], lambda comps: nullcontext(), None, "K8",
                       heads=[Component("head", 1, 2)])
    assert set(res["candidates"].condition) == {"name_free", "none_hinted_demos", "cue", "miscue", "options_word", "copy"}
    assert set(res["labels"].fmt) == {"options_letter", "word_to_definition"} and len(res["greedy"]) == len(items)
    assert res["lens"].layer.nunique() == len(model.model.layers) and len(res["hint_attention"]) > 0
    assert (res["candidates"].state == "K8").all()
    assert [s[0] for s in rb.states({k: [] for k in ("K8", "C4", "K30", "C16")}, None, None, False)] == ["intact", "K8", "C4", "K30", "C16"]


def test_battery_analysis_runs_on_a_small_battery(qwen, pool, tmp_path):
    """Two states on four concepts through run_state, then analyze_battery end to end."""
    import json
    import sys
    from contextlib import nullcontext
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e3_errors"))
    import analyze_battery
    import run_battery as rb

    model, tok = qwen
    items = pool[pool.role.isin(("calibration_a", "calibration_b"))].groupby("role").head(3).reset_index(drop=True)
    items = items[items.free_naming_ok.astype(bool)].reset_index(drop=True)
    tables = {}
    for name in ("intact", "K8"):
        res = rb.run_state(model, tok, True, items, pool, None if name == "intact" else ["x"], lambda c: nullcontext(), None,
                           name, heads=[Component("head", 1, 2)])
        for t, df in res.items():
            if df is not None:
                tables.setdefault(t, []).append(df)
    for t, dfs in tables.items():
        pd.concat(dfs, ignore_index=True).to_parquet(tmp_path / f"{t}.parquet", index=False)
    (tmp_path / "meta.json").write_text(json.dumps(dict(states=["intact", "K8"], positional=False)), encoding="utf-8")
    analyze_battery.main(["--dir", str(tmp_path), "--out", str(tmp_path / "out.json")])
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    k8 = out["states"]["K8"]
    assert abs(k8["change"]["name_target_logprob"]["mean"]) < 0.05       # the fake lesion changes nothing
    assert "evidence_weights" in k8["hints_vs_none_hinted_demos"] and "lens_mean_target_logprob" in out


def test_evidence_weights_recover_known_weights():
    rng = np.random.default_rng(0)
    di = np.column_stack([rng.normal(20, 5, 60), rng.normal(26, 5, 60), rng.normal(5, 5, 60)])
    a, bc, bm = 0.8, 1.3, 1.6
    dl = np.column_stack([a * di[:, 0], a * di[:, 0] + bc * (di[:, 1] - di[:, 0]), a * di[:, 0] + bm * (di[:, 2] - di[:, 0])])
    assert np.allclose(ea.evidence_weights(di, dl), (a, bc, bm))
    w = ea.evidence_weights_bootstrap(di, dl + rng.normal(0, 0.1, dl.shape), n_boot=200)
    assert w["b_miscue"]["ci"][0] < bm < w["b_miscue"]["ci"][1] + 0.05


@pytest.mark.parametrize("layer", [0, 1])
def test_position_restricted_lesion_regions(qwen, pool, layer):
    """The whole region equals a plain lesion. Earlier prompt-position changes can
    propagate to answers, but the last layer has no later attention to do so."""
    from lesion import means as means_mod, neutral_text

    model, tok = qwen
    m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3)
    comps = [Component("expert", layer, 3), Component("head", layer, 2)]
    prompts = mb.scored_prompts(pool.head(2))[:3]
    intact = mb.candidate_scores(model, tok, prompts, batch_size=4)
    with hooks.ablate(model, comps, means=m):
        plain = mb.candidate_scores(model, tok, prompts, batch_size=4)
        whole = mb.candidate_scores(model, tok, prompts, batch_size=4, region="all")
        answer = mb.candidate_scores(model, tok, prompts, batch_size=4, region="answer")
        prompt = mb.candidate_scores(model, tok, prompts, batch_size=4, region="prompt")
    assert np.allclose(plain.logprob, whole.logprob, atol=1e-4)
    assert not np.allclose(answer.logprob, intact.logprob, atol=1e-6)
    if layer == 0:
        assert not np.allclose(prompt.logprob, intact.logprob, atol=1e-6, rtol=0)
    else:
        assert np.allclose(prompt.logprob, intact.logprob, atol=1e-6, rtol=0)

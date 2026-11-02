"""Experiment 3 prompts and scoring (lesion.m3_scoring) and the replacement step after the validity check
(lesion.m3_items.finalize)."""
import re

import numpy as np
import pandas as pd
import pytest
import torch

from lesion import knowledge_items as ki, m3_items as m3, m3_scoring as ms, models, scoring

QWEN = "qwen3-30b-a3b-instruct"
ITEMS = m3.OUT / "items.csv"


def item(i, **kw):
    row = dict(item_id=f"m{i:03d}", role="test", zipf_bin=1, target=f"camel{i}", definition=f"a humped beast number {i}",
               coordinate="llama", superordinate="animal", associate="desert", unrelated="violin",
               free_naming_ok=True, target_more_common=False, order=i)
    row.update(kw)
    return row


def material_words():
    """Every word in the option and target columns of M1, the naming test and M3 (definitions excluded)."""
    words = set()
    for path in (scoring.M1 / "items.csv", scoring.M1 / "demos.csv", ki.OUT / "items.csv", ki.OUT / "demos.csv", ITEMS):
        t = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in t.columns.difference(["definition", "wordnet_gloss", "note", "synset", "category", "domain"]):
            for v in t[col]:
                words |= set(re.findall("[a-z]+", v.lower()))
    return words


def test_demos_are_unused_and_well_formed():
    used = material_words()
    letters = set()
    for d in ms.DEMOS:
        words = [w for w in d["options"] if w != ms.CANNOT_TELL]
        assert len(d["options"]) == 6 and ms.CANNOT_TELL in d["options"] and d["target"] in words
        assert not set(words) & used, set(words) & used
        assert m3.definition_problems(d["definition"], words) == []
        letters.add(d["options"].index(d["target"]))
    assert len(letters) == len(ms.DEMOS)


def test_rotations_put_every_option_at_every_letter():
    items = pd.DataFrame([item(0), item(1)])
    prompts = ms.build_choice_prompts(items, chat=True)
    assert len(prompts) == 12 and ms.base_order("m000") == ms.base_order("m000")
    for item_id in ("m000", "m001"):
        mine = [p for p in prompts if p.item_id == item_id]
        for letter in ms.LETTERS:
            assert sorted(p.letter_to_role[letter] for p in mine) == sorted(ms.ROLES)
        p = mine[3]
        block = p.text.split("\n\n")[-1].splitlines()
        row = items.set_index("item_id").loc[item_id]
        for line, letter in zip(block[1:7], ms.LETTERS):
            role = p.letter_to_role[letter]
            assert line == f"{letter}. " + (ms.CANNOT_TELL if role == "cannot_tell" else row[role])
        assert block[-1] == "Answer:" and p.text.count("Description:") == 4
        assert p.text.startswith(ms.CHOICE_INSTRUCTION.format(ct=ms.CANNOT_TELL) + ms.CHOICE_REPLY_RULE)


def test_naming_prompts_hints_and_demos():
    items = pd.DataFrame([item(0, target="camel", associate="desert")])
    by_cond = {p.condition: p for p in ms.build_naming_prompts(items, chat=False)}
    assert 'Hint: the word starts with "ca".\nAnswer:' in by_cond["cue"].text
    assert 'Hint: the word starts with "de".\nAnswer:' in by_cond["miscue"].text
    assert "Hint" not in by_cond["none"].text
    for cond in ("cue", "miscue"):    # demo hints are always correct
        assert by_cond[cond].text.count('Hint: the word starts with "') == 4
        for d in ms.DEMOS:
            assert f'Hint: the word starts with "{d["target"][:2]}".\nAnswer: {d["target"]}' in by_cond[cond].text
    assert by_cond["none"].candidates == dict(target="camel", associate="desert")


def test_choice_probabilities_average_softmax_over_rotations():
    rows = []
    for r, (a, b) in enumerate([(2.0, 0.0), (0.0, 0.0)]):
        logits = dict(target=a, coordinate=b, superordinate=-50, associate=-50, unrelated=-50, cannot_tell=-50)
        rows.append(dict(item_id="m000", rotation=r, **{f"logit_{k}": v for k, v in logits.items()}))
    p = ms.choice_probabilities(pd.DataFrame(rows)).iloc[0]
    want = (np.exp(2) / (np.exp(2) + 1) + 0.5) / 2
    assert abs(p.p_target - want) < 1e-9 and abs(p.p_target + p.p_coordinate - 1) < 1e-9


def naming_rows(item_id, cond, target_lp, assoc_lp, rank=0, n=(1, 1)):
    return [dict(item_id=item_id, condition=cond, role="target", word="t", logprob=target_lp, n_tokens=n[0], first_token_rank=rank),
            dict(item_id=item_id, condition=cond, role="associate", word="a", logprob=assoc_lp, n_tokens=n[1], first_token_rank=3)]


def test_validity_flags_and_hint_effects():
    items = pd.DataFrame([item(0), item(1), item(2, free_naming_ok=False)])
    choice = []
    for item_id, target in (("m000", 5.0), ("m001", 0.0), ("m002", 5.0)):
        for r in range(6):
            choice.append(dict(item_id=item_id, rotation=r, logit_target=target, logit_coordinate=0.0, logit_superordinate=0.0,
                               logit_associate=0.0, logit_unrelated=0.0, logit_cannot_tell=0.0))
    naming = pd.DataFrame(naming_rows("m000", "none", -1.0, -3.0, rank=2) + naming_rows("m001", "none", -3.0, -2.5, rank=0)
                          + naming_rows("m000", "cue", -0.5, -4.0) + naming_rows("m000", "miscue", -2.0, -1.0, n=(2, 1)))
    v = ms.validity(items, pd.DataFrame(choice), naming).set_index("item_id")
    assert v.choice_ok.tolist() == [True, False, True]
    assert v.naming_ok.tolist() == [True, False, False] and np.isnan(v.at["m002", "delta"])
    assert v.sensitivity_ok.tolist() == [True, False, False]
    assert abs(v.at["m000", "delta"] - 2.0) < 1e-12
    h = ms.hint_effects(naming[naming.item_id == "m000"]).iloc[0]
    assert (h.cue_effect, h.miscue_effect, h.R) == (1.5, -3.0, 4.5)
    d = ms.naming_deltas(naming).set_index(["item_id", "condition"])
    assert abs(d.loc[("m000", "miscue"), "delta_per_token"] - (-2.0 / 2 + 1.0)) < 1e-12
    s = ms.validity_summary(v.reset_index(), ms.hint_effects(naming))
    assert s["choice_pass"]["test, bin 1"] == dict(n=3, passed=2, failed=1)
    assert s["naming_pass"]["test, bin 1"] == dict(n=2, passed=1, failed=1)


def test_finalize_replaces_within_bin_and_direction_and_reports_gaps():
    rows = [item(0, role="test", zipf_bin=0), item(1, role="test", zipf_bin=1), item(2, role="calibration_a", zipf_bin=1),
            item(3, role="calibration_b", target_more_common=True), item(4, role="calibration_b", target_more_common=False),
            item(5, role="test", zipf_bin=1),
            item(6, role="reserve", zipf_bin=1, free_naming_ok=False), item(7, role="reserve", zipf_bin=0),
            item(8, role="reserve", zipf_bin=1, target_more_common=False), item(9, role="reserve", zipf_bin=1)]
    items = pd.DataFrame(rows)
    ok = dict(choice_ok=[True, False, False, False, True, True, True, False, True, True],
              naming_ok=[True, True, True, True, False, False, False, True, True, True])
    check = pd.DataFrame(dict(item_id=items.item_id, **ok))
    final, log = m3.finalize(items, check, n_free=5, seed=0)
    role = final.set_index("item_id").role
    # m001 (test, bin 1) takes the first passing bin-1 reserve, m006; m002 (calibration_a, bin 1) the next, m008;
    # m003 (calibration_b, more common) has no reserve of its direction; m004 (rarer) gets m009, the last one left
    assert role["m001"] == role["m002"] == role["m003"] == role["m004"] == "excluded"
    assert (role["m006"], role["m008"], role["m009"]) == ("test", "calibration_a", "calibration_b")
    assert role["m007"] == "reserve"     # bin 0 reserve failed rule 1 and was never needed
    assert final.set_index("item_id").replaces["m006"] == "m001"
    assert [u["failed"] for u in log["unfilled"]] == ["m003"]
    assert log["shortfall"]["calibration_b, target more common"] == 1
    # free naming: all calibration_b, plus test concepts usable for free naming that pass rule 2 (m005 fails it)
    fn = set(final.item_id[final.free_naming])
    assert fn == {"m000", "m009"} and log["free_naming_test_pool"] == 1
    assert log["shortfall"]["free naming, test"] == 4
    # m007 is still unused but fails rule 1, so nothing is left to replace with (agent2 2026-10-07)
    assert log["reserves_left"] == dict(unused=1, passing_rule_1_by_bin={0: 0, 1: 0},
                                        passing_rules_1_and_2_for_calibration_b={"target more common": 0, "target rarer": 0})


def test_finalize_needs_every_item_checked_once():
    items = pd.DataFrame([item(i) for i in range(3)])
    for ids in (["m000", "m001"], ["m000", "m001", "m002", "m002"], ["m000", "m001", "m009"]):
        with pytest.raises(ValueError, match="does not cover"):
            m3.finalize(items, pd.DataFrame(dict(item_id=ids, choice_ok=True, naming_ok=True)))


def good_meta(sha):
    return dict(model=m3.MAIN_MODEL, revision=models.MODELS[m3.MAIN_MODEL][1], dtype=m3.MAIN_DTYPE, tiny=False,
                limit=None, complete=True, clean_tree=True, items_sha256=sha)


def test_only_a_complete_main_model_run_sets_the_final_items(tmp_path):
    import hashlib
    import json

    items = pd.DataFrame([item(i) for i in range(3)])
    items.to_csv(tmp_path / "items.csv", index=False)
    sha = hashlib.sha256((tmp_path / "items.csv").read_bytes()).hexdigest()
    assert m3.check_run_problems(good_meta(sha), sha) == []
    for bad in (dict(tiny=True), dict(limit=1), dict(model="olmoe-0924-instruct"), dict(revision="main"),
                dict(dtype="bfloat16"), dict(complete=None), dict(clean_tree=False), dict(items_sha256="0" * 64)):
        assert len(m3.check_run_problems({**good_meta(sha), **bad}, sha)) == 1, bad
    run = tmp_path / "check"
    run.mkdir()
    pd.DataFrame(dict(item_id=items.item_id, choice_ok=True, naming_ok=True)).to_csv(run / "validity.csv", index=False)
    (run / "meta.json").write_text(json.dumps({**good_meta(sha), "tiny": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="tiny"):
        m3.write_final(run, out=tmp_path)
    assert not (tmp_path / "final_items.csv").exists()
    (run / "meta.json").write_text(json.dumps(good_meta(sha)), encoding="utf-8")
    final, log = m3.write_final(run, out=tmp_path)
    assert (tmp_path / "final_items.csv").exists() and log["check_meta"]["tiny"] is False


def test_finalize_draws_free_naming_with_a_fixed_seed():
    items = pd.DataFrame([item(i) for i in range(30)])
    check = pd.DataFrame(dict(item_id=items.item_id, choice_ok=True, naming_ok=True))
    a, _ = m3.finalize(items, check, n_free=10, seed=0)
    b, _ = m3.finalize(items.sample(frac=1, random_state=1), check, n_free=10, seed=0)
    assert a.free_naming.sum() == 10 and a.item_id[a.free_naming].tolist() == b.item_id[b.free_naming].tolist()


@pytest.fixture(scope="module")
def qwen_tokenizer():
    if not models.local_dir(QWEN).exists():
        pytest.skip("tokenizer files not downloaded")
    return models.load_tokenizer(QWEN)


def test_chat_template_encodes_answers_as_the_materials_check(qwen_tokenizer):
    items = pd.read_csv(ITEMS, keep_default_na=False)
    for p in ms.build_naming_prompts(items, ("cue",), chat=True):
        for word in p.candidates.values():
            prompt, answer = ms.encode_answer(qwen_tokenizer, p.text, word, chat=True)
            assert prompt == scoring.encode(qwen_tokenizer, p.text, chat=True)
            assert len(answer) == m3.token_counts([word], qwen_tokenizer)[0]


@pytest.mark.parametrize("name,chat", [(QWEN, True), ("olmoe-0924-base", False)])
def test_scores_match_a_full_forward_pass(name, chat):
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    items = pd.read_csv(ITEMS, keep_default_na=False).head(3)
    choice = ms.build_choice_prompts(items, chat)
    got = ms.score_choice_prompts(model, tok, choice, chat, batch_size=5)
    ids = scoring.label_token_ids(tok, chat, ms.LETTERS)
    with torch.no_grad():
        for p, row in zip(choice, got.itertuples()):
            logits = model(input_ids=torch.tensor([scoring.encode(tok, p.text, chat)])).logits[0, -1].float()
            for letter in ms.LETTERS:
                want = float(logits[ids[letter]])
                assert abs(getattr(row, f"logit_{letter}") - want) < 1e-3
                assert getattr(row, f"logit_{p.letter_to_role[letter]}") == getattr(row, f"logit_{letter}")
    naming = ms.build_naming_prompts(items, chat=chat)
    got = ms.score_naming_prompts(model, tok, naming, chat, batch_size=4)
    assert len(got) == 2 * len(naming)
    with torch.no_grad():
        for row in got.itertuples():
            p = next(q for q in naming if q.item_id == row.item_id and q.condition == row.condition)
            prompt, answer = ms.encode_answer(tok, p.text, row.word, chat)
            logp = torch.log_softmax(model(input_ids=torch.tensor([prompt + answer])).logits[0].float(), -1)
            want = sum(float(logp[len(prompt) - 1 + j, answer[j]]) for j in range(len(answer)))
            assert abs(row.logprob - want) < 1e-3 and row.n_tokens == len(answer)
            first, a = logp[len(prompt) - 1], logp[len(prompt) - 1, answer[0]]
            # a random tiny model gives near-uniform logits, so allow rounding differences at the boundary
            assert int((first > a + 1e-4).sum()) <= row.first_token_rank <= int((first > a - 1e-4).sum())

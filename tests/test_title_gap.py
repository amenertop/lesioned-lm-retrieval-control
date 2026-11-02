"""New battery contracts, paired statistics and real forward/backward rehearsals on tiny random models."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import (OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast,
                          Qwen3MoeConfig, Qwen3MoeForCausalLM)

from lesion import (clinical_alignment as ca, fine_objective as fo, fine_semantics as fs, homonym_items as hi,
                    model_census as mc, semantic_diversity as sd, title_gap_analysis as ta,
                    title_gap_scoring as scoring, title_gap_states as states)
from lesion.components import Component
from lesion.hooks import ablate

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_battery_real_scores_resume_and_file_guard(tiny, fine, tmp_path):
    from lesion import legacy_battery, naming, prefix_scoring, scoring as m1

    model, tok, means = tiny
    words = [naming.NamingPrompt("old_word", fine.iloc[0].target, "Name the thing: " + fine.iloc[0].definition)]
    controls = [m1.LabelPrompt("old_control", "colour", 0, "Choose: A. red B. blue Answer:", "A",
                               {"A": "option1", "B": "option2"})]
    labels = fs.label_prompts(fine.head(1))
    plan = [dict(name="intact", gates={}), dict(name="lesion", gates={Component("expert", 0, 0).key: 0.0})]
    args = (model, tok, means, words, controls, labels, plan, tmp_path, {"tiny": True})
    legacy_battery.run(*args, prefix_scoring.score_label_prompts_full, batches=(2, 1))
    intact = pd.read_parquet(tmp_path / "intact/naming.parquet")
    assert len(intact) == 1 and (tmp_path / "lesion/control.parquet").exists()

    def no_recompute(*args, **kwargs):
        pytest.fail("a complete run must not spend another forward pass")

    legacy_battery.run(*args, no_recompute, batches=(2, 1))
    (tmp_path / "lesion/control.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        legacy_battery.run(*args, no_recompute, batches=(2, 1))


def test_generation_region_tracks_prefill_cache_and_restores(tiny):
    from lesion import regional_generation as rg

    model, tok, _ = tiny
    ids = torch.tensor([[0, 2, 3, 4], [2, 3, 4, 5]])
    attention = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    observed = []
    with rg.generation_region(model, 4, "answer"):
        handle = model.register_forward_pre_hook(
            lambda module, args, kwargs: observed.append(module._lesion_position_mask.clone()), with_kwargs=True)
        try:
            model.generate(input_ids=ids, attention_mask=attention, use_cache=True, do_sample=False,
                           min_new_tokens=3, max_new_tokens=4, pad_token_id=0)
        finally:
            handle.remove()
    assert observed[0].tolist() == [[False, False, False, True]] * 2
    assert all(m.shape == (2, 1) and m.all() for m in observed[1:])
    assert getattr(model, "_lesion_position_mask", None) is None


def test_regional_generation_all_equals_original_and_errors_restore(tiny, fine):
    from lesion import m3_battery, regional_generation as rg

    model, tok, means = tiny
    prompts = fs.naming_prompts(fine.head(2))
    model.generation_config.use_cache = True
    with ablate(model, [Component("expert", 0, 0)], means):
        expected = m3_battery.greedy_answers(model, tok, prompts, max_new_tokens=3)
        actual = rg.greedy_answers(model, tok, prompts, max_new_tokens=3, region="all")
    pd.testing.assert_frame_equal(expected, actual)
    with pytest.raises(RuntimeError, match="interrupted"):
        with rg.generation_region(model, 3, "prompt"):
            model(input_ids=torch.tensor([[2, 3, 4]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
            raise RuntimeError("interrupted")
    assert getattr(model, "_lesion_position_mask", None) is None


@pytest.fixture
def fine():
    t = pd.read_csv(ROOT / "data/materials/title_gap/fine/items.csv", keep_default_na=False)
    t = t[t.review != "excluded"].head(8).reset_index(drop=True).copy()
    t["split"] = "screening"
    t["review"] = "draft"
    t["property_true"] = ""
    t["property_false"] = ""
    t["frequency"] = np.linspace(1.0, 5.0, len(t))
    return t


@pytest.fixture
def homonyms():
    return pd.read_csv(ROOT / "data/materials/title_gap/homonyms/items.csv", keep_default_na=False).head(4)


def test_materials_are_complete_separate_and_normed():
    f = pd.read_csv(ROOT / "data/materials/title_gap/fine/items.csv", keep_default_na=False)
    manifest = json.loads((ROOT / "data/materials/title_gap/fine/manifest.json").read_text())
    assert len(f[f.review == "excluded"]) == manifest["excluded_items"]
    f = f[f.review != "excluded"]
    h = pd.read_csv(ROOT / "data/materials/title_gap/homonyms/items.csv", keep_default_na=False)
    fs.validate(f)
    hi.validate(h)
    assert f.groupby("split").size().to_dict() == manifest["split_counts"]
    assert h.groupby("split").family.nunique().to_dict() == {"dev": 40, "test": 40}
    assert h.norm_source.str.contains("2022").all()


def test_material_guards_reject_missing_semantics_and_wrong_context_counts(fine, homonyms):
    with pytest.raises(ValueError, match="missing"):
        fs.validate(fine.drop(columns="accepted_answers"))
    with pytest.raises(ValueError, match="bin"):
        fs.validate(fine.assign(semd_bin="unknown"))
    with pytest.raises(ValueError, match="missing"):
        hi.validate(homonyms.drop(columns="other_sense"))
    with pytest.raises(ValueError, match="word counts"):
        hi.validate(homonyms.assign(context_low_words=1))


def test_fine_semd_keeps_damage_direction_and_missing_property_subset():
    rows = pd.DataFrame([dict(item_id=i, family=i, state=s, naming=d+i, definition=d-i,
        property=d if i == 0 else np.nan, intact_naming=8., intact_definition=4., intact_property=3.,
        semd=1.5, frequency=3., n_tokens=1) for s, d in (("WR8_a0", 1.), ("SC4_a0", 2.)) for i in (0, 1)])
    frames = sd.fine_frames(rows)
    assert (frames["naming"].h_sc - frames["naming"].h_wr).tolist() == [1., 1.]
    assert frames["property"].item_id.tolist() == [0]


def test_clinical_cueing_retains_early_rescue_and_does_not_hide_reversals():
    rows = pd.DataFrame([dict(state="s", item_id=i, letters=j, n_letters=4,
        category="correct" if hit else "unrelated") for i, hits in enumerate(((0,1,0,0), (1,0,0,0), (0,0,0,1)))
        for j, hit in enumerate(hits)])
    result = ca.cue_profiles(rows, n_boot=20)["s"]
    assert result["accuracy_two"]["estimate"] == 2/3
    assert result["rescue_two"]["estimate"] == .5
    assert result["rescue_final"]["estimate"] == 1
    assert result["initially_correct_lost_at_two"]["estimate"] == 1
    rows.loc[0, "category"] = "unclassified"
    assert ca.cue_profiles(rows, n_boot=0)["s"]["status"] == "needs_blind_response_review"


def test_locked_test_and_unreviewed_real_materials(tmp_path, fine):
    p = tmp_path / "items.csv"
    fine.to_csv(p, index=False)
    with pytest.raises(PermissionError):
        fs.load_items(p, "test", rehearsal=True)
    with pytest.raises(ValueError, match="approved"):
        fs.load_items(p, "screening")
    assert len(fs.load_items(p, "screening", rehearsal=True)) == len(fine)


def test_lexical_overlap_and_homonym_split_leakage_fail(fine, homonyms):
    fine.loc[1, "split"] = "test"
    fine.loc[1, "synonym"] = fine.loc[0, "synonym"]
    with pytest.raises(ValueError, match="crosses"):
        fs.validate(fine)
    homonyms.loc[homonyms.index[1], "split"] = "test"
    homonyms.loc[homonyms.index[0], "split"] = "dev"
    with pytest.raises(ValueError, match="cross"):
        hi.validate(homonyms)


def test_meaning_cue_and_context_are_independent(homonyms):
    naming, labels = hi.prompts(homonyms)
    assert len(naming) == len(homonyms) * 6 and len(labels) == len(homonyms) * 12
    first = homonyms.iloc[0]
    by = {p.condition: p for p in naming if p.item_id == first.item_id}
    assert 'Hint' not in by['low_none'].text
    assert f'"{first.target[:2]}"' in by['low_correct'].text
    assert f'"{first.competitor[:2]}"' in by['low_misleading'].text
    assert by['low_none'].candidates == by['high_correct'].candidates
    for condition in hi.CONTEXTS:
        p = [p for p in labels if p.item_id == first.item_id and p.fmt == condition + '_none']
        assert [x.correct for x in p] == ['A', 'B']


def test_generated_unknown_is_not_assumed_to_be_an_error(homonyms):
    r = homonyms.iloc[0].to_dict()
    lex = {(r['word'], 'novel'): [int(r['sense'])], (r['word'], r['competitor']): [int(r['other_sense'])]}
    assert hi.classify(r['target'], r, lex) == 'target'
    assert hi.classify('novel', r, lex) == 'same_meaning'
    assert hi.classify(r['competitor'], r, lex) == 'other_meaning'
    assert hi.classify('unknown', r, lex) == 'unclassified'
    assert hi.classify(r['word'], r, lex) == 'ambiguous_response'
    assert hi.classify('the answer is novel', r, lex) == 'format_invalid'


def test_homonym_review_cannot_override_known_or_ambiguous_answers(homonyms):
    generated = pd.DataFrame([dict(state='intact', item_id=r.item_id, condition=f'{context}_{cue}',
                                  answer='wibble' if cue == 'none' else r.word if cue == 'misleading' else r.target)
                              for r in homonyms.itertuples() for context in hi.CONTEXTS
                              for cue in ('none', 'correct', 'misleading')])
    _, rows = ta.homonym_recovery(generated, homonyms, {}, 0)
    unknown = rows[rows.initial_category == 'unclassified'].response_id.iloc[0]
    review = pd.DataFrame([dict(response_id=unknown, category='unrelated')])
    _, resolved = ta.homonym_recovery(generated, homonyms, {}, 0, review)
    assert resolved.loc[resolved.response_id == unknown, 'category'].eq('unrelated').all()
    assert resolved.exact_correct.equals(rows.exact_correct)
    with pytest.raises(ValueError, match='invalid'):
        ta.homonym_recovery(generated, homonyms, {}, 0, review.assign(category='target'))
    for category in ('target', 'ambiguous_response'):
        known = rows[rows.initial_category == category].response_id.iloc[0]
        review = pd.DataFrame([dict(response_id=known, category='unrelated')])
        with pytest.raises(ValueError, match='initially unclassified'):
            ta.homonym_recovery(generated, homonyms, {}, 0, review)
    alternate = generated.assign(state='alternate', answer=generated.answer.str.upper() + '.')
    _, normalized = ta.homonym_recovery(pd.concat([generated, alternate]), homonyms, {}, 0)
    identities = normalized.pivot(index=['item_id', 'condition'], columns='state', values='response_id')
    assert identities.intact.equals(identities.alternate)


@pytest.mark.parametrize('changed,message', [('bootstrap', 'bootstrap count'), ('comparisons', 'comparisons'),
                                             ('runs', 'every frozen'), ('lexicon', 'lexicon differs')])
def test_final_analysis_rejects_changed_frozen_inputs(tmp_path, monkeypatch, capsys, changed, message):
    import importlib.util
    import sys
    from lesion import runinfo

    spec = importlib.util.spec_from_file_location('analyze_battery_test', ROOT / 'experiments/title_gap/analyze_battery.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    items, lexicon = tmp_path / 'items.csv', tmp_path / 'lexicon.json'
    items.write_text('items', encoding='utf-8')
    lexicon.write_text('[]', encoding='utf-8')
    protocol = dict(states_sha256={'core': 'core', 'random': 'random'}, n_boot=10000,
                    comparisons=['first:second'], lexicon_sha256=runinfo.sha256(lexicon), sources_sha256={})
    path = tmp_path / 'docs/analysis-plan/title-gap-homonyms.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(protocol), encoding='utf-8')
    plan = dict(provenance=dict(split='test', tiny=False, battery='homonyms', freeze_tag='analysis-frozen-test',
                               items_sha256=runinfo.sha256(items), protocol_sha256=runinfo.sha256(path)),
                source_runs=[dict(provenance=dict(states_sha256=s)) for s in ('core', 'random')])
    argv = ['analyze', '--run', str(tmp_path), '--items', str(items), '--lexicon', str(lexicon), '--out', str(tmp_path / 'analysis')]
    if changed == 'bootstrap':
        argv += ['--n-boot', '0']
    elif changed == 'comparisons':
        argv += ['--compare', 'third:second']
    elif changed == 'runs':
        plan['source_runs'].pop()
    else:
        lexicon.write_text('[1]', encoding='utf-8')
    monkeypatch.setattr(entry.ta, 'read_runs', lambda paths: ({}, plan))
    monkeypatch.setattr(entry.runinfo, 'sha256_at', lambda tag, relative: runinfo.sha256(path))
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2 and message in capsys.readouterr().err


def test_semd_join_does_not_invent_missing_word_forms():
    items = pd.DataFrame(dict(item_id=['a', 'b', 'c'], target=[' BANK ', 'banks', 'unknown']))
    norms = pd.DataFrame(dict(word=['bank'], semd=[1.7], log_bnc_freq=[3.0]))
    summary, joined = sd.coverage(items, norms)
    assert summary['matched'] == 1 and joined.semd.isna().sum() == 2


def test_paired_regression_recovers_direct_interaction():
    rng = np.random.default_rng(4)
    n = 120
    semd, freq, length, intact = rng.normal(size=(4, n))
    frame = pd.DataFrame(dict(item_id=[str(i) for i in range(n)], family=[str(i // 2) for i in range(n)],
                              semd=semd, frequency=freq, n_tokens=length, intact=intact,
                              h_wr=-freq + rng.normal(size=n) * 0.1,
                              h_sc=2 * semd - freq + rng.normal(size=n) * 0.1))
    result = sd.paired_regression(frame, n_boot=100)
    row = next(c for c in result['coefficients'] if c['state'] == 'SC_minus_WR' and c['predictor'] == 'semd')
    assert row['ci_low'] > 1.5 and row['estimate'] == pytest.approx(2 * semd.std(), abs=0.05)
    broken = pd.concat([frame, frame.iloc[:1]])
    with pytest.raises(ValueError, match='one paired row'):
        sd.paired_regression(broken, n_boot=20)


def test_clinical_held_out_markers_not_rescaled_per_marker():
    rng = np.random.default_rng(5)
    x = rng.uniform(0.1, 1.0, (18, 2))
    frame = pd.DataFrame(dict(marker=[str(i) for i in range(18)], family=[str(i // 3) for i in range(18)],
                              observed=x @ [0.2, 0.8], retrieval=x[:, 0], control=x[:, 1]))
    result = ca.compare_profiles(frame)
    assert result.dual_error.max() < 1e-20 and result.single_error.mean() > 0.005
    with pytest.raises(ValueError):
        ca.relative_loss(0.5, 0.5, 0.5)


def test_clinical_missingness_and_error_denominators_are_preserved():
    table = pd.read_csv(ROOT / 'data/literature/jefferies2006_individual.csv', keep_default_na=False)
    result = ca.patient_tasks(table)
    absent = result.query("patient_id == 'KA' and task == 'concrete_synonyms'").iloc[0]
    assert absent.missing_reason == 'testing_abandoned' and pd.isna(absent.value)
    corrected = result.query("patient_id == 'JP' and measure == 'coordinate'").iloc[0]
    assert corrected.printed_value == '100' and corrected.value == 1.0
    assert corrected.denominator == 'single_word_semantic_errors'
    assert result.query("task == 'picture_naming_64'").relative_loss.isna().all()
    chance = result.query("patient_id == 'SC' and task == 'picture_cct'").iloc[0]
    assert chance.relative_loss == pytest.approx((58.9 - 46) / (58.9 - 16))


def test_source_model_revision_and_tiny_status_must_match():
    states.check_source(dict(model='qwen', revision='abc'), 'qwen', 'abc')
    for payload in (dict(model='other', revision='abc'), dict(model='qwen', revision='def'),
                    dict(model='qwen', revision='abc', tiny=True)):
        with pytest.raises(ValueError):
            states.check_source(payload, 'qwen', 'abc')


def test_missing_paired_metadata_is_not_silently_dropped():
    scores = pd.DataFrame(dict(state=['intact', 'K8', 'C4'], item_id=['a'] * 3, score=[1., .5, .7]))
    with pytest.raises(ValueError, match='metadata'):
        sd.paired_frame(scores, pd.DataFrame(dict(item_id=['b'], family=['b'])))


def test_native_thinking_is_not_forced_to_a_next_token_label():
    assert mc.format_answer('A') == 'A'
    assert mc.format_answer('Answer: A') is None
    assert mc.format_answer('thinking A', True) is None
    assert mc.format_answer('thinking\n</think>\n B', True) == 'B'
    class TokenizerFlag:
        def apply_chat_template(self, *a, **kw):
            return kw
    wrapped = mc.NonThinkingTokenizer(TokenizerFlag())
    assert wrapped.apply_chat_template([], add_generation_prompt=True)['enable_thinking'] is False


@pytest.fixture(params=['qwen', 'olmoe'])
def tiny(request, fine, homonyms):
    prompts = fs.naming_prompts(fine) + fs.word_answer_prompts(fine) + hi.prompts(homonyms)[0]
    vocab_text = [p.text for p in prompts] + [w for p in prompts for w in p.candidates.values()] + ['A B user assistant']
    raw = Tokenizer(WordLevel(unk_token='<unk>'))
    raw.pre_tokenizer = Whitespace()
    raw.train_from_iterator(vocab_text, WordLevelTrainer(special_tokens=['<pad>', '<eos>', '<unk>']))
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
    tok.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    torch.manual_seed(7)
    cfg = dict(vocab_size=len(tok), hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
               num_key_value_heads=1, num_experts=4, num_experts_per_tok=2, max_position_embeddings=512, use_cache=False,
               pad_token_id=0, eos_token_id=1, attention_dropout=0.0)
    model = (Qwen3MoeForCausalLM(Qwen3MoeConfig(**cfg, head_dim=8, moe_intermediate_size=12)) if request.param == 'qwen'
             else OlmoeForCausalLM(OlmoeConfig(**cfg)))
    model.eval()
    means = SimpleNamespace(expert_mean=torch.randn(2, 4, 16) * 0.05, head_mean=torch.randn(2, 2, 8) * 0.05,
                            routing_frequency=lambda: np.ones((2, 4)))
    return model, tok, means


def test_broad_masks_are_nested_and_dose_one_restores(tiny):
    model, tok, means = tiny
    broad = states.broad_states(model.config, seeds=(0,), alphas=(0.5,))
    for band in states.layer_bands(model.config):
        selected = [s for s in broad if s['band'] == band]
        assert set(selected[0]['gates']) <= set(selected[1]['gates']) <= set(selected[2]['gates'])
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        base = model(ids).logits
        state = dict(name='all_retained', gates={k: 1.0 for k in broad[0]['gates']})
        with states.apply_state(model, means, state):
            assert torch.allclose(model(ids).logits, base, atol=1e-6)
        with pytest.raises(RuntimeError), states.apply_state(model, means, broad[0]):
            raise RuntimeError('deliberate exit')
        assert torch.equal(model(ids).logits, base)


def test_factorial_states_are_real_union_and_reject_overlap():
    result = states.combination_states(['head:0:0'], ['head:1:1'], 0.25, 0.75)
    assert result[-1]['gates'] == {'head:0:0': 0.25, 'head:1:1': 0.75}
    with pytest.raises(ValueError, match='disjoint'):
        states.combination_states(['head:0:0'], ['head:0:0'])


def test_dense_soft_controls_shuffle_doses_within_layers():
    state = dict(name='soft', kind='learned', gates={'head:0:0': .1, 'head:0:1': .9, 'head:1:0': .3, 'head:1:1': .8})
    randoms = states.shuffled_gates(state, {}, n=20)
    assert any(r['gates'] != state['gates'] for r in randoms)
    for r in randoms:
        for layer in (0, 1):
            assert sorted(v for k, v in r['gates'].items() if k.startswith(f'head:{layer}:')) == sorted(
                v for k, v in state['gates'].items() if k.startswith(f'head:{layer}:'))


def test_all_tasks_score_resume_and_corruption_is_rejected(tiny, fine, tmp_path):
    model, tok, means = tiny
    names, labels = fs.naming_prompts(fine.head(2)), fs.label_prompts(fine.head(2))
    plan = states.reference_states({'lesion': ['head:0:0']}, alphas=(0.5,))
    scoring.run_states(model, tok, means, names, labels, plan, tmp_path, {'tiny': True}, batches=(2, 1), max_new_tokens=2)
    tables, saved = ta.read_run(tmp_path)
    assert len(tables['candidates']) == 12 and len(tables['labels']) == 16
    assert tables['language'].state.tolist() == ['intact', 'lesion_a0.5']
    combined, union = ta.read_runs([tmp_path, tmp_path])
    assert union['kind'] == 'analysis_union' and len(union['source_runs']) == 2
    for name in tables:
        pd.testing.assert_frame_equal(combined[name], tables[name], check_exact=True)
    before = (tmp_path / 'lesion_a0.5/complete.json').read_bytes()
    scoring.run_states(model, tok, means, names, labels, plan, tmp_path, {'tiny': True}, batches=(2, 1), max_new_tokens=2)
    assert (tmp_path / 'lesion_a0.5/complete.json').read_bytes() == before
    separate = tmp_path / 'candidate_batches'
    scoring.run_states(model, tok, means, names, labels, plan, separate, {'tiny': True}, batches=(2, 1),
                       max_new_tokens=2, naming_batches=(4, 3))
    alternative, recorded = ta.read_run(separate)
    assert recorded['naming_batches'] == [4, 3] and recorded['batches'] == [2, 1]
    for key in ('labels', 'generated', 'language'):
        pd.testing.assert_frame_equal(alternative[key], tables[key], check_exact=True)
    for runs in ([tmp_path, separate], [separate, tmp_path]):
        with pytest.raises(ValueError, match='scoring settings'):
            ta.read_runs(runs)
    with pytest.raises(ValueError, match='different inputs or settings'):
        scoring.run_states(model, tok, means, names, labels, plan, separate, {'tiny': True}, batches=(2, 1),
                           max_new_tokens=2, naming_batches=(4, 2))
    f = tmp_path / 'lesion_a0.5/candidates.parquet'
    t = pd.read_parquet(f)
    t.iloc[:-1].to_parquet(f, index=False)
    with pytest.raises(ValueError, match='changed'):
        ta.read_run(tmp_path)


def test_homonym_evidence_keeps_cue_gain_and_miscue_cost_separate(homonyms):
    candidates, labels = [], []
    for row in homonyms.itertuples():
        for state in ('intact', 'lesion'):
            for cue, advantage in (('none', 2.0), ('correct', 5.0), ('misleading', -2.0)):
                condition = 'low_' + cue
                for role, value in (('target', advantage - 6), ('other_meaning', -6)):
                    candidates.append(dict(state=state, item_id=row.item_id, condition=condition, role=role, logprob=value))
                for variant in (0, 1):
                    labels.append(dict(state=state, item_id=row.item_id, fmt=condition, variant=variant, margin=advantage * 2))
    result, frame = ta.homonym_scores(dict(candidates=pd.DataFrame(candidates), labels=pd.DataFrame(labels)), homonyms, 0)
    for entry in result:
        scale = 2 if entry['readout'] == 'choice_logit' else 1
        assert entry['correct_gain']['estimate'] == 3 * scale
        assert entry['misleading_cost']['estimate'] == 4 * scale
    assert set(frame.readout) == {'candidate_log_odds', 'choice_logit'}


def test_property_subset_is_analyzed_without_inventing_missing_items(fine):
    labels, candidates = [], []
    for i, row in enumerate(fine.itertuples()):
        for state in ('intact', 'lesion'):
            for task in ('definition', 'synonym', 'property'):
                if task == 'property' and i >= 5:
                    continue
                labels.append(dict(state=state, item_id=row.item_id, fmt=task, margin=2.0 - (0.5 if state == 'lesion' else 0)))
            candidates.append(dict(state=state, item_id=row.item_id, condition='naming', role='target',
                                   logprob=-3.0 - (0.1 * i if state == 'lesion' else 0), n_tokens=1 + i % 3))
    result, frame = ta.fine_profiles(dict(labels=pd.DataFrame(labels), candidates=pd.DataFrame(candidates)), fine, 0)
    assert result['lesion']['definition']['estimate'] == 0.5
    assert result['lesion']['synonym']['estimate'] == 0.5
    assert result['lesion']['property']['estimate'] == 0.5
    assert result['lesion']['property']['n_families'] == 5
    assert frame.query("state == 'lesion'").property.isna().sum() == 3
    contrast = ta.fine_contrast(frame, 'lesion', 'intact', 0)
    assert contrast['property_damage_difference']['estimate'] == 0.5


@pytest.mark.parametrize('readout', ['target', 'accepted_vs_coordinate'])
def test_hub_training_updates_only_gates_and_scores_both_comprehension_tasks(tiny, fine, readout):
    model, tok, means = tiny
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    data = fo.FineData(fine, tok, naming_readout=readout)
    result, log = fo.train(model, means, data, ['head:0:0', 'head:1:1', 'expert:0:0'], [[3, 4, 5, 6]],
                           steps=2, batch=8, seed=0)
    assert len(log) == 2 and np.isfinite(log.loss).all()
    assert data.intact.shape == (8, 3) and result.retained.between(0, 1).all()
    assert all(torch.equal(p, before[k]) for k, p in model.named_parameters())
    assert all(p.grad is None for p in model.parameters())


def test_relative_training_readout_matches_saved_candidate_scores(tiny, fine):
    from lesion import m3_scoring

    model, tok, _ = tiny
    prompts = fs.naming_prompts(fine)
    raw = m3_scoring.score_naming_prompts(model, tok, prompts, batch_size=8)
    tables = dict(candidates=pd.concat([raw.assign(state='intact'), raw.assign(state='lesion')]))
    original = tables['candidates'].copy()
    adjusted = ta.fine_readout_tables(tables, fine, 'accepted_vs_coordinate')['candidates']
    expected = adjusted.query("state == 'intact' and role == 'target'").set_index('item_id').logprob
    data = fo.FineData(fine, tok, naming_readout='accepted_vs_coordinate')
    actual = data.score(model, range(len(fine)))
    np.testing.assert_allclose(actual[:, 0].detach(), expected.reindex(data.ids), atol=2e-5)
    ordinary = fo.FineData(fine, tok).score(model, range(len(fine)))
    torch.testing.assert_close(actual[:, 1:], ordinary[:, 1:])
    pd.testing.assert_frame_equal(tables['candidates'], original)


def test_checkpointed_hub_matches_normal_training(tiny, fine):
    model, tok, means = tiny
    data = fo.FineData(fine, tok)
    args = (model, means, data, ['head:0:0', 'head:1:1', 'expert:0:0'], [[3, 4, 5, 6]])
    plain, log1 = fo.train(*args, steps=2, batch=8, seed=3)
    checked, log2 = fo.train(*args, steps=2, batch=8, seed=3, checkpointing=True)
    np.testing.assert_allclose(plain.retained, checked.retained, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(log1.loss, log2.loss, rtol=1e-5, atol=1e-6)


def test_hub_objective_penalises_unshared_damage():
    freq = torch.arange(10, dtype=torch.float32)
    intact = torch.randn(10, 3)
    shared = torch.ones(10, 3)
    naming_only = shared.clone()
    naming_only[:, 1:] = 0
    a = fo.hub_terms(shared, freq, freq * 0 + 1, intact)
    b = fo.hub_terms(naming_only, freq, freq * 0 + 1, intact)
    assert a['shared'] > b['shared']


def test_homonym_contrast_uses_paired_difference_of_differences(homonyms):
    rows = []
    for state in ['WR', 'SC']:
        for family in ['a', 'b', 'c', 'd']:
            for dom in ['dominant', 'subordinate']:
                for cue in ['none', 'correct', 'misleading']:
                    correct = not (state == 'SC' and dom == 'subordinate' and cue != 'correct')
                    rows.append(dict(state=state, family=family, dominance=dom, cue=cue, context='low', semantic_correct=correct))
    result = ta.homonym_contrast(pd.DataFrame(rows), 'SC', 'WR', n_boot=20)
    assert result['low']['subordinate_extra']['estimate'] == 1.0
    assert result['low']['cue_gain']['estimate'] == 1.0


def test_unresolved_response_bounds_keep_repeated_judgements_together():
    rows = pd.DataFrame(dict(response_id=['unknown', 'unknown', 'known'],
                             semantic_correct=pd.array([None, None, True], dtype='boolean')))
    bounds = ta.binary_response_bounds(rows, [1., -1., .25])
    assert bounds['lower'] == bounds['upper'] == .25
    rows = pd.DataFrame([
        dict(item_id='a', cue=c, semantic_correct=v, response_id=r)
        for c, v, r in [('none', None, 'repeat'), ('correct', None, 'repeat'), ('misleading', False, 'wrong')]
    ])
    rows['semantic_correct'] = rows.semantic_correct.astype('boolean')
    bounds = ta.homonym_rescue_bounds(rows)
    assert bounds['denominator_min'] == 0 and bounds['denominator_max'] == 1
    assert bounds['lower'] == bounds['upper'] == 0

    sample = pd.DataFrame(dict(family=['d', 'a', 'b', 'c'], value=[1., -2., 3., 4.]))
    assert ta.bootstrap_family_mean(sample, 'value', n_boot=100) == ta.bootstrap(sample, lambda t: t.value.mean(), n_boot=100)


def test_fine_actual_answers_keep_unclassified_synonyms_pending(fine):
    item = fine.head(1)
    generated = pd.DataFrame([dict(state='intact', item_id=item.iloc[0].item_id, condition='naming', answer='unlisted')])
    labels = pd.DataFrame([dict(state='intact', item_id=item.iloc[0].item_id, fmt=fmt, margin=1.)
                           for fmt in ('definition', 'synonym')])
    tables = dict(generated=generated, labels=labels)
    result, rows = ta.fine_responses(tables, item, n_boot=10)
    assert result['intact']['naming_accuracy_status'] == 'needs_blind_response_review'
    assert rows.correct_response.isna().all()
    review = rows[['response_id']].assign(category='correct')
    result, rows = ta.fine_responses(tables, item, n_boot=10, review=review)
    assert result['intact']['naming_accuracy_status'] == 'fully_classified' and rows.correct_response.all()


def test_frozen_naming_alias_does_not_make_copy_correct(fine):
    item = fine.head(1).assign(accepted_answers='alias')
    generated = pd.DataFrame([dict(state='intact', item_id=item.iloc[0].item_id, condition=c, answer='alias')
                              for c in ('naming', 'copy')])
    labels = pd.DataFrame([dict(state='intact', item_id=item.iloc[0].item_id, fmt=f, margin=1.)
                           for f in ('definition', 'synonym')])
    _, rows = ta.fine_responses(dict(generated=generated, labels=labels), item, n_boot=0)
    assert rows.set_index('condition').category.to_dict() == {'naming': 'correct', 'copy': 'incorrect'}


def test_property_accuracy_does_not_reward_always_true():
    labels = pd.DataFrame([dict(state='intact', item_id='a', fmt='property', variant=v,
                                margin=10. if v >= 2 else -1.) for v in range(4)])
    result = ta.fine_label_accuracy(labels)
    assert result.correct_response.item() == .5


def test_semantic_error_profile_uses_semantic_error_denominator():
    named = pd.DataFrame(dict(category=['correct', 'coordinate', 'coordinate', 'associate', 'unrelated']))
    result = ta.fine_error_profile(named)
    assert result['semantic_share_of_all_errors'] == .75
    assert result['single_word_semantic_proportions']['associate'] == pytest.approx(1 / 3)
    unresolved = pd.concat([named, pd.DataFrame(dict(category=['unclassified']))])
    result = ta.fine_error_profile(unresolved)
    assert result['semantic_share_of_all_errors'] is None
    assert result['single_word_semantic_proportions']['associate'] is None


def test_census_rejects_another_revision(tmp_path):
    from lesion import model_census
    spec = model_census.CANDIDATES['qwen3-32b']
    marker = tmp_path / 'VERIFIED.json'
    marker.write_text(json.dumps(dict(all_safetensors_match_hf=True, hf_revision='different')))
    with pytest.raises(ValueError, match='revision'):
        model_census.verified_candidate_path('qwen3-32b', tmp_path)
    marker.write_text(json.dumps(dict(all_safetensors_match_hf=True, hf_revision=spec['revision'])))
    assert model_census.verified_candidate_path('qwen3-32b', tmp_path) == tmp_path


def test_patient_bootstrap_preserves_missing_and_group_direction():
    from lesion import clinical_alignment
    rows = []
    for group, shift in [('SD', 0.), ('SA', 10.)]:
        for i in range(4):
            for task in ['a','b']:
                rows.append(dict(group=group, patient_id=group+str(i), task=task, measure='correct',
                    value=np.nan if task=='b' and i==0 else i+shift, unit='count', denominator='test_items'))
    result = clinical_alignment.patient_group_profiles(pd.DataFrame(rows), n_boot=200)
    assert result.query("group == 'SA' and task == 'b'").n.item() == 3
    assert result.query("group == 'SA_minus_SD' and task == 'a'").estimate.item() == 10.
    assert result.query("group == 'SA_minus_SD' and task == 'a'").ci_low.item() > 0


def test_clinical_measured_states_hold_out_whole_families():
    from lesion import clinical_alignment as ca
    profile = lambda r, c: dict(frequency=r, cue=c, errors=.5 + .5 * (c-r))
    patients = pd.DataFrame([dict(group=g, marker=m, family=m, value=v)
        for g, r, c in [('SD', 1., 0.), ('SA', 0., 1.)] for m, v in profile(r, c).items()])
    predictions = pd.DataFrame([dict(model=model, curve='fixed', state=f'{r}_{c}', marker=m, value=v)
        for model in ['single', 'dual'] for r in [0., .5, 1.] for c in [0., .5, 1.]
        if model == 'dual' or r == c for m, v in profile(r, c).items()])
    result = ca.compare_measured_profiles(patients, predictions)
    assert result.query("model == 'dual'").error.sum() == 0
    assert result.query("model == 'single'").error.sum() > 0
    changed = patients.copy()
    changed.loc[changed.family == 'cue', 'value'] = .37
    check = ca.compare_measured_profiles(changed, predictions)
    a = result.query("family == 'cue'").reset_index(drop=True)
    b = check.query("family == 'cue'").reset_index(drop=True)
    pd.testing.assert_frame_equal(a[['state', 'predicted']], b[['state', 'predicted']])
    new_items = predictions.assign(value=predictions.value + 1.)
    transferred = ca.predict_frozen_profiles(patients, new_items, result)
    assert transferred.state.tolist() == result.state.tolist()
    np.testing.assert_allclose(transferred.predicted, result.predicted + 1.)
    missing = new_items.loc[~((new_items.model == 'dual') & (new_items.marker == 'cue'))]
    incomplete = ca.predict_frozen_profiles(patients, missing, result)
    assert incomplete.state.tolist() == result.state.tolist()
    assert incomplete.query("model == 'dual' and marker == 'cue'").status.eq('missing_held_out_marker').all()
    assert incomplete.query("model == 'dual' and marker == 'cue'").error.isna().all()
    unavailable = predictions.copy()
    unavailable.loc[unavailable.marker == 'cue', 'value'] = np.nan
    selected = ca.compare_measured_profiles(patients, unavailable).query("family == 'cue'").reset_index(drop=True)
    pd.testing.assert_frame_equal(a[['state', 'curve']], selected[['state', 'curve']])
    assert selected.status.eq('missing_held_out_marker').all()
    with pytest.raises(ValueError, match='every model'):
        ca.predict_frozen_profiles(patients, new_items, result.iloc[:-1])
    changed_curve = result.copy()
    changed_curve.loc[0, 'curve'] = 'different'
    with pytest.raises(ValueError, match='share each curve'):
        ca.predict_frozen_profiles(patients, new_items, changed_curve)


def test_published_marker_denominators_and_figure_precision():
    markers = ca.published_marker_profiles(ROOT / 'data/literature', n_boot=100)
    assert markers.family.nunique() == 3 and not markers.duplicated(['group', 'marker']).any()
    lookup = markers.set_index(['group', 'marker'])
    assert lookup.loc[('SA', 'rescue_two'), 'value'] == pytest.approx((.767-.456)/(1-.456))
    assert lookup.loc[('SD', 'frequency_gap'), 'value'] == pytest.approx(.233/(.95-1/3))
    figures = markers[markers.family != 'semantic_errors']
    assert figures.ci_low.isna().all() and figures.sensitivity_low.notna().all()
    assert lookup.loc[('SA', 'frequency_gap'), 'sensitivity_low'] < 0
    errors = markers[markers.family == 'semantic_errors']
    assert errors.ci_low.notna().all() and errors.sensitivity_low.isna().all()
    assert errors.query("group == 'SD'").value.sum() == pytest.approx(1.001)
    assert lookup.loc[('SA', 'error_associate'), 'n'] == 8
    assert errors.denominator_status.eq('counts_not_reported_patient_proportions_available').all()
    scenarios = list(ca.figure_scenarios(markers))
    assert len(scenarios) == 256
    for _, profile, endpoints in scenarios:
        values = profile.set_index(['group', 'marker']).value
        for row in endpoints:
            for endpoint in ('two', 'final'):
                assert values.loc[(row['group'], 'rescue_'+endpoint)] == pytest.approx(
                    (row[endpoint+'_accuracy']-row['uncued_accuracy'])/(1-row['uncued_accuracy']))


def test_measured_clinical_profiles_keep_recovery_and_error_denominators():
    rows = pd.DataFrame([dict(state='intact', item_id=item, letters=j, n_letters=3, category=category)
        for item, responses in [('a', ['correct', 'unrelated', 'unrelated']),
                                ('b', ['coordinate', 'correct', 'correct']),
                                ('c', ['associate', 'associate', 'associate'])]
        for j, category in enumerate(responses)])
    items = pd.DataFrame(dict(item_id=['a', 'b', 'c', 'd'], frequency_bin=['high', 'high', 'low', 'low']))
    labels = items[['item_id']].assign(state='intact', fmt='synonym', margin=[1., 1., 1., -1.])
    markers = ca.measured_marker_profiles(rows, labels, items)
    values = markers.set_index('marker')
    assert values.loc['rescue_two', 'value'] == .5
    assert values.loc['rescue_two', 'raw_value'] == values.loc['rescue_two', 'rescued_count'] == 1
    assert values.loc['rescue_two', 'initial_error_count'] == 2
    assert values.loc['rescue_two', 'uncued_accuracy'] == pytest.approx(1/3)
    assert values.loc['rescue_two', 'cued_accuracy'] == pytest.approx(2/3)
    assert values.loc['error_associate', 'value'] == .5
    assert values.loc['rescue_two', 'denominator'] == values.loc['error_associate', 'denominator'] == 2
    assert values.loc['frequency_gap', 'value'] == 2.
    curves = pd.DataFrame(dict(model=['single', 'single'], curve=['one', 'one'], state=['intact', 'not_run']))
    complete, coverage = ca.complete_curve_profiles(markers, curves)
    assert set(complete.state) == {'intact', 'not_run'}
    assert complete.query("state == 'not_run'").value.isna().all()
    assert coverage.complete.tolist() == [True, False]
    assert coverage.iloc[1].reason == 'state_not_measured'


def test_production_context_cancels_word_bias_and_never_splits_meanings(homonyms):
    from lesion import production_control as pc

    items = pd.read_csv(ROOT / 'data/materials/title_gap/homonyms/items.csv').query("split == 'dev'")
    split = pc.split_families(items)
    assert len(split['screening']) == len(split['dev']) == 20
    assert not set(split['screening']) & set(split['dev'])
    with pytest.raises(ValueError, match='development'):
        pc.split_families(items.assign(split='test'))
    jobs, weights = pc.attribution_prompts(homonyms)
    word_bias = {p.target: i * .7 for i, p in enumerate(jobs)}
    assert np.dot(weights['context_damage'], [word_bias[p.target] for p in jobs]) == pytest.approx(0., abs=1e-12)


def test_production_word_only_attribution_matches_actual_small_lesion(tiny, homonyms):
    from lesion import attribution, m3_scoring, production_control as pc

    model, tok, means = tiny
    items = homonyms.head(2)
    jobs, weights = pc.attribution_prompts(items)
    predicted = attribution.naming_attribution(model, tok, jobs, True, means, weights, batch_size=2, word_only=True)
    prompts = [p for p in hi.prompts(items)[0] if p.condition.endswith('_none')]
    def statistic():
        scores = m3_scoring.score_naming_prompts(model, tok, prompts, batch_size=2).assign(state='now')
        return -pc.context_scores(scores, items).margin.mean()
    baseline = statistic()
    component = Component('head', 0, 0)
    with ablate(model, [component], means, alpha=.999):
        actual = (statistic() - baseline) / .001
    estimate = predicted.set_index('component').at[component.key, 'context_damage']
    assert actual == pytest.approx(estimate, rel=.05, abs=.0002)


def test_candidate_specificity_separates_word_gain_from_neighbour_gain():
    items = pd.DataFrame(dict(item_id=['a', 'b', 'c', 'd'], family=['a', 'b', 'c', 'd']))
    rows = []
    for state in ('intact', 'lesion'):
        for item in items.item_id:
            for role, word, value in [('target', 'cat', -3.), ('synonym', 'feline', -4.), ('coordinate', 'dog', -5.)]:
                gain = (1. if role != 'coordinate' else 3.) if state == 'lesion' else 0.
                rows.append(dict(state=state, item_id=item, condition='naming', role=role, word=word, logprob=value+gain))
    candidates = pd.DataFrame(rows)
    result, paired = ta.fine_candidate_specificity(candidates, items, 100)
    for metric in ('target_vs_coordinate_damage', 'accepted_vs_coordinate_damage'):
        assert result['lesion'][metric]['estimate'] == pytest.approx(2.)
        assert result['lesion'][metric]['ci_low'] == pytest.approx(2.)
        assert result['intact'][metric]['estimate'] == 0
    assert len(paired) == 16
    broken = candidates.copy()
    broken.loc[broken.role == 'synonym', 'word'] = 'cat'
    with pytest.raises(ValueError, match='distinct'):
        ta.fine_candidate_specificity(broken, items, 0)


def test_property_accuracy_can_fall_while_truth_order_is_preserved():
    items = pd.DataFrame(dict(item_id=['a', 'b', 'c', 'd'], family=['a', 'b', 'c', 'd']))
    rows = []
    for state, offset in [('intact', 0.), ('endorse', 3.)]:
        for item in items.item_id:
            for variant in range(4):
                true = variant >= 2
                score = (2. if true else -2.) + offset
                rows.append(dict(state=state, item_id=item, fmt='property', variant=variant,
                                 margin=score if true else -score))
    labels = pd.DataFrame(rows)
    result, _ = ta.property_response_diagnostics(labels, items, 100)
    assert result['states']['intact']['accuracy']['estimate'] == 1.
    assert result['states']['endorse']['accuracy']['estimate'] == .5
    assert result['states']['endorse']['auc']['estimate'] == 1.
    assert result['lesion_minus_intact']['endorse']['score_separation']['estimate'] == 0.
    assert result['lesion_minus_intact']['endorse']['true_score_offset']['estimate'] == 3.
    with pytest.raises(ValueError, match='both truths'):
        ta.property_response_diagnostics(labels[labels.variant != 3], items, 0)


def test_clinical_sampling_keeps_missing_folds_and_paired_concepts():
    from lesion import clinical_resampling as cr

    fit = pd.DataFrame([dict(model=model, group=group, family=family, error=.25)
        for model in ('single', 'dual') for group in ('SD', 'SA') for family in ('f1', 'f2', 'f3')])
    fit.loc[0, 'error'] = np.nan
    errors = cr.balanced_errors(fit).set_index(['model', 'family']).error
    assert np.isnan(errors.loc['single', 'overall'])
    assert np.isnan(errors.loc['dual_minus_single', 'overall'])
    assert errors.loc['dual', 'overall'] == .25
    clinical = pd.DataFrame([dict(state=state, item_id=str(i), letters=j, n_letters=3, category=category)
        for state in ('intact', 'same')
        for i, responses in enumerate([['correct']*3, ['coordinate', 'correct', 'correct'],
                                        ['associate']*3, ['superordinate']*3])
        for j, category in enumerate(responses)])
    items = pd.DataFrame(dict(item_id=['0', '1', '2', '3'], frequency_bin=['high', 'high', 'low', 'low']))
    labels = pd.concat([items[['item_id']].assign(state=state, fmt='synonym', margin=1.)
                        for state in ('intact', 'same')], ignore_index=True)
    patient = ca.published_marker_profiles(ROOT / 'data/literature', n_boot=20)
    source = pd.read_csv(ROOT / 'data/literature/jefferies2006_individual.csv', keep_default_na=False)
    for _, sampled_patient, model in cr.marker_samples(clinical, labels, items, patient, source, n_boot=20):
        values = model.pivot(index='marker', columns='state', values='value')
        np.testing.assert_allclose(values.intact, values.same, equal_nan=True)
        np.testing.assert_allclose(sampled_patient.query("family != 'semantic_errors'").value,
                                   patient.query("family != 'semantic_errors'").value)


def test_cue_content_keeps_original_cues_and_scores_without_choice_items(tiny, tmp_path):
    from lesion import cue_content, m3_cued

    items = pd.read_csv(ROOT / 'data/materials/m3/final_items.csv', keep_default_na=False)
    items = items[(items.role == 'calibration_a') & items.free_naming_ok].head(2)
    prompts = cue_content.prompts(items)
    original = {(p.item_id, p.condition): p.text for p in m3_cued.prompts(items)}
    for p in prompts:
        word = p.candidates['target']
        length = p.condition.split('_')[1]
        letters = {'none': 0, 'two': 2, 'final': len(word)-1}[length]
        if p.condition.startswith('definition'):
            assert p.text == original[p.item_id, f'cue{letters}']
        else:
            assert f'"{word[:letters]}"' in p.text
            assert 'Description:' not in p.text.rsplit('\n\n', 1)[-1]
            assert p.text.endswith('Answer:')
    model, tok, means = tiny
    scoring.run_states(model, tok, means, prompts, [], [dict(name='intact', gates={})], tmp_path,
                       {'tiny': True}, batches=(2, 1), max_new_tokens=2)
    tables, _ = ta.read_run(tmp_path)
    assert tables['labels'].empty and len(tables['generated']) == 14


def test_cue_content_locks_semantic_errors_and_clusters_repeated_prefixes():
    from lesion import cue_content

    items = pd.read_csv(ROOT / 'data/materials/m3/final_items.csv', keep_default_na=False)
    items = items[items.target.isin(['bagel', 'bagpipes', 'drugstore', 'axe'])]
    records, scores = [], []
    for prompt in cue_content.prompts(items):
        target = prompt.candidates['target']
        answer = target
        if prompt.condition == 'definition_none':
            answer = {'bagel': 'pretzel', 'drugstore': 'pharmacy'}.get(target, target)
        if prompt.condition == 'removed_two' and target.startswith('ba'):
            answer = 'bagel'
        if prompt.condition == 'removed_final' and target == 'drugstore':
            answer = 'clock\nkite\nelevator\n'
        if target == 'axe':
            answer = ' AXE! '
        records.append(dict(state='intact', item_id=prompt.item_id, condition=prompt.condition, answer=answer))
        scores.append(dict(state='intact', item_id=prompt.item_id, condition=prompt.condition,
                           role='target', logprob=-2. if prompt.condition.startswith('removed') else -5., n_tokens=2))
    rows, queue = cue_content.classify(pd.DataFrame(records), items)
    assert len(queue) == 1 and queue.target.iloc[0] == 'drugstore'
    waiting = cue_content.summarize(rows, pd.DataFrame(scores), 100)
    assert waiting['meta']['error_sets']['intact']['status'] == 'needs_blind_response_review'
    review = queue[['response_id']].assign(category='correct')
    rows, queue = cue_content.classify(pd.DataFrame(records), items, review)
    assert queue.empty
    pharmacy = rows[rows.answer == 'pharmacy'].iloc[0]
    assert pharmacy.semantic_correct and not pharmacy.target_exact
    assert rows.query("target == 'bagpipes' and condition == 'removed_two'").category.iloc[0] == 'other_completion'
    assert rows[rows.target == 'axe'].target_exact.all()
    result = cue_content.summarize(rows, pd.DataFrame(scores), 100)
    assert result['meta']['error_sets']['intact']['item_ids'] == items.loc[items.target == 'bagel', 'item_id'].tolist()
    assert result['summaries']['intact/initial_errors']['definition_final']['target_exact']['n'] == 1
    assert result['summaries']['intact/length_gt3']['definition_final']['target_exact']['n'] == 3
    prompts = pd.DataFrame(result['meta']['prompts']).set_index('condition')
    assert prompts.loc['removed_two', 'n_concepts'] == 4
    assert prompts.loc['removed_two', 'n_unique_prompts'] == 3
    difference = result['contrasts']['intact/all']['removed_minus_definition_two/logprob']
    assert difference['estimate'] == difference['ci_low'] == difference['ci_high'] == 3.
    diagnostics = result['summaries']['intact/all']['removed_final']
    assert diagnostics['demo_sequence_replay']['estimate'] == .25
    assert diagnostics['format_invalid']['estimate'] == .25
    assert result['summaries']['intact/all']['definition_none']['demo_sequence_replay']['estimate'] == 0


def test_resident_gate_means_preserve_scores_gradients_and_source(tiny, fine):
    model, tok, means = tiny
    original = means.expert_mean.clone()
    source_device = means.expert_mean.device
    data = fo.FineData(fine.assign(frequency=1.), tok)
    components = [Component('expert', 0, 0).key, Component('head', 1, 1).key]
    report = fo.check_resident_means(model, means, data, components, batch=2, checkpointing=True)
    assert report['passed'] and report['scores_identical'] and report['gradients_identical']
    assert torch.equal(means.expert_mean, original) and means.expert_mean.device == source_device
    assert not model.training and not model.is_gradient_checkpointing


def test_precision_plan_uses_noise_not_selected_effect_and_requires_dev():
    from lesion import title_gap_precision as precision

    values = np.tile([-1., 1.], 10)
    a = precision.mean_precision(values, 40)
    shifted = precision.mean_precision(values + 100, 40)
    assert a == shifted
    assert precision.mean_precision(values, 80)['estimate'] < a['estimate']
    assert precision.mean_precision(np.zeros(20), 40)['estimate'] == 0
    with pytest.raises(ValueError, match='complete'):
        precision.mean_precision([1, 2, np.nan, 4], 40)
    with pytest.raises(ValueError, match='development'):
        precision.homonym_precision(pd.DataFrame({'readout': ['choice_logit'], 'split': ['test']}), pd.DataFrame())


def test_uncertain_response_sampling_keeps_label_identity_across_states():
    rows = pd.DataFrame([dict(family=i, response_id=f'same_{i}', semantic_correct=pd.NA)
                         for i in range(8) for _ in range(2)])
    result = ta.binary_response_bounds(rows, np.tile([1/8, -1/8], 8), n_boot=100)
    assert result['lower'] == result['upper'] == 0
    assert result['sampling_envelope']['lower'] == result['sampling_envelope']['upper'] == 0
    rows['response_id'] = [f'new_{i}' for i in range(len(rows))]
    separate = ta.binary_response_bounds(rows, np.tile([1/8, -1/8], 8), n_boot=100)
    assert separate['lower'] == separate['sampling_envelope']['lower'] == -1
    assert separate['upper'] == separate['sampling_envelope']['upper'] == 1

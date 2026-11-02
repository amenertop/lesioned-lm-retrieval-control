"""The exploratory lesion probe on the M3 calibration concepts (experiments/e3_errors/probe_lesions.py)."""
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import probe_lesions as pl  # noqa: E402
from lesion import models  # noqa: E402


def test_probe_reads_only_calibration_concepts():
    items = pl.calibration_items()
    assert set(items.role) == set(pl.CALIBRATION) and len(items) == 80


def test_dose_sets_follow_the_rankings_and_skip_domain_general_components():
    import pandas as pd

    control = pd.DataFrame(dict(component=[f"c{i}" for i in range(20)], control_effect=[20 - i for i in range(20)],
                                string_damage=[0.0] * 2 + [99.0] + [0.0] * 17))
    knowledge = pd.DataFrame(dict(component=[f"k{i}" for i in range(32)], knowledge_effect=[31 - i - 0.5 for i in range(32)]))
    s = pl.dose_sets(control, knowledge)
    assert s["C4"] == ["c0", "c1", "c3", "c4"] and len(s["C16"]) == 16 and "c2" not in s["C16"]
    assert s["K8"] == [f"k{i}" for i in range(8)] and len(s["K30"]) == 30 and s["K16"] == s["K30"][:16]


def test_logit_gaps():
    import pandas as pd

    row = dict(set="intact", item_id="m1", logit_target=10.0, logit_coordinate=4.0, logit_superordinate=2.0,
               logit_associate=1.0, logit_unrelated=0.0, logit_cannot_tell=0.0)
    g = pl.logit_gaps(pd.DataFrame([row, {**row, "logit_associate": 3.0}])).loc[("intact", "m1")]
    assert g.associate == -8.0 and g.S_log == -8.0 - (-6.0 + -8.0) / 2


def test_naming_errors_gaps_and_renormalised_probabilities():
    import numpy as np
    import pandas as pd

    from lesion import m3_scoring as ms

    lp = dict(target=-1.0, coordinate=-3.0, superordinate=-5.0, associate=-2.0, unrelated=-9.0)
    nt = dict(target=1, coordinate=2, superordinate=1, associate=1, unrelated=1)
    rows = [dict(item_id="m1", condition="none", role=r, word=r, logprob=v, n_tokens=nt[r], first_token_rank=0)
            for r, v in lp.items()]
    e = ms.naming_errors(pd.DataFrame(rows)).iloc[0]
    z = np.exp(np.array(list(lp.values())))
    assert abs(e.p_target - z[0] / z.sum()) < 1e-12 and abs(e.gap_coordinate - (-2.0)) < 1e-12
    assert abs(e.S_log - (-1.0 - (-2.0 + -4.0) / 2)) < 1e-12
    assert abs(e.gap_pt_coordinate - (-1.5 + 1.0)) < 1e-12 and e.target_logprob == -1.0


def test_error_probe_runs_on_a_tiny_model():
    name = "qwen3-30b-a3b-instruct"
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    items = pl.calibration_items().head(3)
    naming = pl.run_errors(model, tok, True, items, {"fake": ["x"]}, lambda comps: nullcontext())
    assert len(naming) == 2 * 3 * 5
    s = pl.summarize_errors(naming)
    assert abs(sum(s["intact"]["mean_probability"].values()) - 1) < 1e-6 and "change" in s["fake"]


def test_all_m3_candidate_words_encode_as_in_the_materials_check():
    """Every option word scored as a free-naming answer: the chat template and the materials check agree."""
    import pandas as pd

    from lesion import m3_items, m3_scoring as ms

    name = "qwen3-30b-a3b-instruct"
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    tok = models.load_tokenizer(name)
    items = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    for p in ms.build_naming_prompts(items, ("none",), True, roles=ms.ERROR_ROLES):
        for word in p.candidates.values():
            ms.encode_answer(tok, p.text, word, chat=True)


def test_probe_runs_and_summarizes_on_a_tiny_model():
    name = "qwen3-30b-a3b-instruct"
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    items = pl.calibration_items().groupby("role").head(2)
    choice, naming = pl.run_probe(model, tok, True, items, {"fake": ["x"]}, lambda comps: nullcontext())
    assert set(choice.set) == set(naming.set) == {"intact", "fake"} and len(choice) == 2 * 6 * len(items)
    s = pl.summarize(choice, naming, items)
    assert set(s) == {"intact", "fake"} and abs(sum(s["intact"]["mean_probability"].values()) - 1) < 1e-3
    # the fake lesion changes nothing, up to batch-composition noise
    assert abs(s["fake"]["S_change"]["mean"]) < 1e-3 and abs(s["fake"]["R_change"]["mean"]) < 1e-2

"""Offline regressions for the five prewriting repairs (2026-10-19)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, OlmoeConfig, OlmoeForCausalLM

from lesion import e3_sources, fast_moe, p14_analysis as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))
import run_control_test as control  # noqa: E402
import run_experiment3 as e3  # noqa: E402
import run_dissociation_test as dissociation  # noqa: E402

spec = importlib.util.spec_from_file_location("fast_check", ROOT / "experiments/fast_moe_check/run.py")
fast_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast_check)


def test_p14_minimality_scope_and_signed_changes():
    group = dict(necessary=0, redundant=["a"], saturated_share=0.2, pairs=[])
    assert "necessary" not in pa.minimality_fields("hub", group)
    assert pa.minimality_fields("hub", group)["naming_only_necessary"] == 0
    data = pd.DataFrame(dict(objective=["control"] * 3, state=["minus_one"] * 2 + ["minus_two"],
                             change=[-1, 1, -2], ci_high=[1, 2, -1], necessary=[False, False, True]))
    result = pa.control_minimality(data)
    assert result["singles_point_estimate_down"] == result["singles_point_estimate_up"] == 1
    assert result["singles_necessary"] == 0 and result["pairs_necessary"] == 1


@pytest.mark.parametrize("primary,association,expected", [
    ("supported", "not_falsified", "supported"), ("supported", "falsified", "falsified"),
    ("falsified", "not_falsified", "falsified"), ("supported", "not_estimable", "insufficient"),
    ("insufficient", "not_falsified", "insufficient")])
def test_control_conjunction(primary, association, expected):
    assert control.core_claim(primary, association) == expected


def test_e3_refuses_bad_rows_and_confirmatory_dry_run_output():
    data = pd.DataFrame(dict(item=["a", "b"], score=[1., 2.]))
    e3_sources.check_rows(data, ["item"], [("a",), ("b",)], "toy", ["score"])
    for bad in (data.iloc[:1], pd.concat([data, data]), data.assign(score=[1, np.nan])):
        with pytest.raises(ValueError):
            e3_sources.check_rows(bad, ["item"], [("a",), ("b",)], "toy", ["score"])
    with pytest.raises(ValueError, match="outside"):
        e3_sources.check_output(ROOT / "reports/confirm/x.json", True, None)
    with pytest.raises(ValueError, match="external DLA"):
        e3_sources.check_output(ROOT / "reports/confirm/x.json", False, "elsewhere.parquet")


def test_e3_completed_source_and_tampered_metadata(monkeypatch, tmp_path):
    source = ROOT / "outputs/e3_errors/qwen3-30b-a3b-instruct/confirm"
    if not source.exists():
        pytest.skip("completed source not available")
    sets = dict(K8=dissociation.K8, C4=json.loads((ROOT / "reports/confirm/control/results.json").read_text())["control"],
                K30=e3.K30, C16=e3.C16)
    final = ROOT / "data/materials/m3/final_items.csv"
    result = e3_sources.check_source(source, final, sets)
    assert result["source_freeze_tag"] == "analysis-frozen-experiment3"
    meta = json.loads((source / "meta.json").read_text())
    for field, value in (("complete", False), ("resume_commit", "wrong"), ("analysis_plan_sha256", "wrong"),
                         ("final_items_sha256", "wrong"), ("states", [])):
        (tmp_path / "meta.json").write_text(json.dumps(meta | {field: value}))
        with pytest.raises(ValueError):
            e3_sources.check_source(tmp_path, final, sets)


def test_fast_judge_is_layerwise_and_shape_and_max_errors_fail():
    assert not fast_check.judge({"routed_differently": [0.01, 0.05]}, {"routed_differently": [0.1, 0.]})["routed_differently"]
    zero = fast_check.difference(np.ones(3), np.ones(3))
    assert not fast_check.judge({"hidden": zero}, {"hidden": dict(shape_differs=True)})["hidden"]
    assert not fast_check.judge({"hidden": zero | dict(max_abs=1.)}, {"hidden": zero})["hidden"]
    assert fast_check.judge({"hidden": zero}, [{"hidden": zero}, {"hidden": zero}])["hidden"]


def test_norm_rows_are_aligned_and_values_still_checked():
    a = pd.DataFrame(dict(sequence=[0, 0], token_id=[2, 3], expert=["e", "e"], position=[0, 1], rank=[0, 0], norm=[1., 2.]))
    zero = fast_check.norm_difference(a, a.iloc[::-1])
    assert zero["max_abs"] == 0 and zero["row_mismatch"] == 0
    changed = fast_check.norm_difference(a, a.assign(norm=[1., 3.]))
    assert not fast_check.judge({"norms": changed}, {"norms": zero})["norms"]
    assert fast_check.norm_difference(a, a.iloc[:1])["row_mismatch"] == 0.5


@pytest.mark.parametrize("kind", ["qwen", "olmoe"])
def test_grouped_same_input_structure_on_tiny_models(kind):
    torch.manual_seed(7)
    args = dict(vocab_size=31, hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=2, num_experts=4, num_experts_per_tok=2, use_cache=False)
    model = (Qwen3MoeForCausalLM(Qwen3MoeConfig(**args, head_dim=8, moe_intermediate_size=24)) if kind == "qwen"
             else OlmoeForCausalLM(OlmoeConfig(**args))).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    inputs = fast_check.capture_block_inputs(model, torch.randint(0, 31, (2, 8)))
    fast_moe.install(model, "grouped")
    try:
        assert all(r["passed"] for r in fast_check.same_input_check(model, inputs))
    finally:
        fast_moe.uninstall(model)

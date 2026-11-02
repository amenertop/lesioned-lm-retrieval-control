"""Checks the M1 formatting and split (ticket pilot 01). Run: pytest tests/test_materials.py"""
from pathlib import Path

import pandas as pd
import pytest

from lesion import materials

RAW_MISSING = not materials.RAW.exists()
pytestmark = pytest.mark.skipif(RAW_MISSING, reason="raw M1 data not downloaded (see data/raw/SOURCES.md)")


@pytest.fixture(scope="module")
def built():
    items = materials.format_items(materials.load_raw())
    return items, materials.split_groups(items)


def test_counts(built):
    items, split = built
    assert len(items) == 480
    assert items.task.value_counts().to_dict() == {"string": 160, "synonym": 160, "colour": 80, "size": 80}
    matching = items[items.task.isin(["colour", "size"])]
    assert (matching.high_demand == "1").sum() == 80
    # Median split equals TDS < 0 in this item set (pilot protocol §3).
    assert ((matching.tds < 0) == (matching.high_demand == "1")).all()


def test_correct_option_is_target(built):
    items, _ = built
    for r in items[items.task != "synonym"].itertuples():
        chosen = r.option1 if r.correct_option == "option1" else r.option2
        assert chosen == r.target
    syn = items[items.task == "synonym"]
    assert syn.synonym_type.value_counts().to_dict() == {"similar": 80, "different": 80}
    assert ((syn.correct_option == "option1") == (syn.synonym_type == "similar")).all()


def test_shared_probes_form_one_group(built):
    items, split = built
    shared = split[split.tasks == "colour+size"]
    assert len(shared) == 8
    for g in shared.concept_group:
        assert set(items.loc[items.concept_group == g, "task"]) == {"colour", "size"}


def test_every_group_on_one_side_and_balanced(built):
    items, split = built
    assert split.concept_group.is_unique
    assert set(split.concept_group) == set(items.concept_group)
    assert set(split.split) == {"screening", "dev", "test"}
    # About 40/20/40 overall, and within each task
    share = split.split.value_counts(normalize=True)
    assert abs(share["screening"] - 0.4) < 0.03 and abs(share["test"] - 0.4) < 0.03 and abs(share["dev"] - 0.2) < 0.03
    for task, t in split.groupby("tasks"):
        if len(t) >= 20:
            s = t.split.value_counts(normalize=True)
            assert abs(s.get("dev", 0) - 0.2) < 0.06, task
            assert abs(s.get("screening", 0) - s.get("test", 0)) < 0.06, task


def test_deterministic(built, tmp_path):
    items, split = built
    again_items = materials.format_items(materials.load_raw())
    again_split = materials.split_groups(again_items)
    pd.testing.assert_frame_equal(items, again_items)
    pd.testing.assert_frame_equal(split, again_split)


def test_written_files_match(built):
    items, split = built
    out = materials.OUT
    if not (out / "items.csv").exists():
        pytest.skip("run `python -m lesion.materials` first")
    on_disk = pd.read_csv(out / "split.csv", dtype=str)
    assert on_disk.concept_group.tolist() == split.concept_group.tolist()
    assert on_disk.split.tolist() == split.split.tolist()

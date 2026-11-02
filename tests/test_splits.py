"""Reading one split of M1, with the held-out test split locked (code-architecture.md §4 rule 4)."""
import pytest

from lesion import materials, runinfo


@pytest.fixture
def frozen_head(monkeypatch):
    """Pretend HEAD is a clean checkout of the analysis-plan freeze tag `analysis-frozen-v1`."""
    def fake_git(*args):
        if args[0] == "rev-parse":
            return "c" * 40
        if args[0] == "ls-tree":
            return "docs/analysis-plan/analysis-plan.md"
        return ""   # status --porcelain: clean
    monkeypatch.setattr(runinfo, "git", fake_git)


def test_splits_partition_m1(frozen_head):
    parts = {s: materials.load_split(s, freeze_tag="analysis-frozen-v1") for s in materials.SPLITS}
    ids = [set(p.item_id) for p in parts.values()]
    assert sum(len(i) for i in ids) == 480 and len(set.union(*ids)) == 480
    for s, p in parts.items():
        assert (p.split == s).all()
    groups = [set(p.concept_group) for p in parts.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


@pytest.mark.parametrize("tag", [None, "g1-frozen", "analysis-frozen-does-not-exist"])
def test_test_split_locked_without_the_freeze_tag(tag):
    with pytest.raises(PermissionError):
        materials.load_split("test", freeze_tag=tag)
    with pytest.raises(ValueError):
        materials.load_split("train")


def test_test_split_locked_when_head_is_not_the_tag_or_plan_missing(monkeypatch):
    answers = {"HEAD": "a" * 40, "analysis-frozen-v1^{commit}": "b" * 40}
    monkeypatch.setattr(runinfo, "git", lambda *a: answers.get(a[-1], "") if a[0] == "rev-parse" else "")
    with pytest.raises(PermissionError, match="HEAD must be"):
        materials.load_split("test", freeze_tag="analysis-frozen-v1")
    answers["HEAD"] = "b" * 40      # right commit, but the tagged tree has no analysis plan
    with pytest.raises(PermissionError, match="no analysis plan"):
        materials.load_split("test", freeze_tag="analysis-frozen-v1")

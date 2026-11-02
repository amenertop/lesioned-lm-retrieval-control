"""Stage-consistency and resume checks of the MVP entry scripts (experiments/e1_screen/common.py)."""
import json
import sys
from pathlib import Path

import pytest

from lesion import runinfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e1_screen"))
import common  # noqa: E402

HEAD = runinfo.commit()
FIELDS = dict(stage="joint", model="m", revision="r", tiny=False, split="dev", ablation=dict(mode="mean", alpha=0.0),
              means_path="x.pt", means_sha256="aa", materials_sha256={"items.csv": "bb"}, analysis_plan_sha256=None, scorer="prefix", dtype="bfloat16",
              params=dict(n_random=20, max_pct_gap=0.25, n_boot=10_000), resume_commit=HEAD, n_prompts=10, batch_size=8)


def test_same_run_accepts_matching_screen_meta():
    common.check_same_run(dict(FIELDS, stage="screen", split="screening", n_prompts=99), FIELDS, "screen")


@pytest.mark.parametrize("change", [dict(means_sha256="zz"), dict(materials_sha256={"items.csv": "cc"}),
                                    dict(ablation=dict(mode="zero", alpha=0.0)), dict(model="other"),
                                    dict(analysis_plan_sha256="dd"), dict(scorer="full"), dict(dtype="float16"), dict(resume_commit="0" * 40)])
def test_same_run_rejects_other_screen(change):
    with pytest.raises(SystemExit):
        common.check_same_run(dict(FIELDS, **change), FIELDS, "screen")


def test_resume_only_into_identical_run(tmp_path):
    common.check_resume(tmp_path, FIELDS)                       # empty folder: fine
    (tmp_path / "meta.json").write_text(json.dumps(dict(FIELDS, batch_size=64)), encoding="utf-8")
    common.check_resume(tmp_path, FIELDS)                       # only the batch size differs: fine
    for change in (dict(means_sha256="zz"), dict(params=dict(FIELDS["params"], n_random=5)),
                   dict(params=dict(FIELDS["params"], max_pct_gap=1.0)), dict(params=dict(FIELDS["params"], n_boot=500))):
        (tmp_path / "meta.json").write_text(json.dumps(dict(FIELDS, **change)), encoding="utf-8")
        with pytest.raises(SystemExit):
            common.check_resume(tmp_path, FIELDS)


def screened(tmp_path, selection, components=("expert:0:1", "expert:0:2", "expert:1:3")):
    """A finished screen folder whose meta.json records the hashes of its outputs."""
    (tmp_path / "screen.csv").write_text("component\n" + "\n".join(components) + "\n", encoding="utf-8")
    (tmp_path / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    return dict(status="done", selection_sha256=runinfo.sha256(tmp_path / "selection.json"),
                screen_csv_sha256=runinfo.sha256(tmp_path / "screen.csv"),
                screened_components_sha256=common.component_set_sha256(components))


def test_load_selection_accepts_untouched_outputs(tmp_path):
    sel = dict(control=["expert:0:1"], knowledge=["expert:1:3"], status=dict(control="ok", knowledge="ok"))
    assert common.load_selection(tmp_path, screened(tmp_path, sel)) == sel


def test_load_selection_rejects_edited_or_foreign_selection(tmp_path):
    sel = dict(control=["expert:0:1"], knowledge=["expert:1:3"], status=dict(control="ok", knowledge="ok"))
    meta = screened(tmp_path, sel)
    (tmp_path / "selection.json").write_text(json.dumps(dict(sel, control=["expert:0:2"])), encoding="utf-8")
    with pytest.raises(SystemExit, match="changed after screening"):
        common.load_selection(tmp_path, meta)
    foreign = dict(sel, control=["expert:5:5"])                 # hashes recorded, but not a screened component
    with pytest.raises(SystemExit, match="not screened"):
        common.load_selection(tmp_path, screened(tmp_path, foreign))
    with pytest.raises(SystemExit, match="unfinished"):
        common.load_selection(tmp_path, dict(meta, status="running"))


def test_reuse_at_newer_commit_checks_only_score_files(monkeypatch):
    """same_code=False: other files may change, score-relevant files may not."""
    def fake_git(*args):
        if args[0] == "cat-file":
            return "commit"
        if args[0] == "diff":
            return "src/lesion/hooks.py" if "src/lesion/hooks.py" in args else ""
        return ""
    monkeypatch.setattr(runinfo, "git", fake_git)
    old = dict(FIELDS, resume_commit="a" * 40)
    with pytest.raises(SystemExit, match="score-relevant"):
        common.check_same_run(old, FIELDS, "screen", same_code=False)
    monkeypatch.setattr(runinfo, "git", lambda *a: "commit" if a[0] == "cat-file" else "")
    common.check_same_run(old, FIELDS, "screen", same_code=False)


def test_only_the_grouped_moe_forward_is_a_field_that_runs_must_share(tmp_path):
    from types import SimpleNamespace

    assert common.moe_fields(SimpleNamespace()) == {} and common.moe_fields(SimpleNamespace(moe="sorted")) == {}
    forward = {m: dict(FIELDS, moe_forward=m, **common.moe_fields(SimpleNamespace(moe=m))) for m in ("hf", "sorted", "grouped")}
    assert forward["grouped"]["moe"] == "grouped"
    old_meta = dict(FIELDS)                                           # written before the MoE forward was recorded
    for old in (old_meta, forward["hf"], forward["sorted"], forward["grouped"]):
        for new in forward.values():
            compatible = (old.get("moe") == "grouped") == (new.get("moe") == "grouped")
            assert common.moe_mismatch(old, new) == (not compatible)
            if compatible:
                common.check_same_run(old, new, "screen")
            else:
                with pytest.raises(SystemExit, match="moe"):
                    common.check_same_run(old, new, "screen")
            (tmp_path / "meta.json").write_text(json.dumps(old), encoding="utf-8")
            if compatible:                                            # hf and sorted resume each other's runs
                common.check_resume(tmp_path, new)
            else:                                                     # grouped in either direction is refused
                with pytest.raises(SystemExit, match="moe"):
                    common.check_resume(tmp_path, new)


def test_load_installs_the_chosen_moe_forward_and_run_fields_record_it():
    import argparse

    from lesion import fast_moe, models

    if not models.local_dir("qwen3-30b-a3b-instruct").exists():
        pytest.skip("tokenizer files not downloaded")
    means = Path(__file__).resolve().parents[1] / "outputs/means_tiny/qwen3-30b-a3b-instruct.pt"
    if not means.exists():
        pytest.skip("tiny means not built")
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    for moe in ("hf", "sorted", "grouped"):
        args = ap.parse_args(["--model", "qwen3-30b-a3b-instruct", "--means", str(means), "--tiny", "--moe", moe])
        model = common.load(args)[0]
        assert fast_moe.impl(model) == moe
        fields = common.run_fields(args, HEAD, "x", "dev", 1)
        assert fields["moe_forward"] == moe and ("moe" in fields) == (moe == "grouped")
        assert not any(p.requires_grad for p in model.parameters()) or moe != "grouped"

"""A changed completion marker or ranked screen cannot authorize Coder's held-out run."""
import json
import sys

import pandas as pd
import pytest

from lesion import coder_sources as cs, runinfo


def test_stage_reuses_only_unchanged_request_and_outputs(tmp_path):
    out, record = tmp_path / "run", tmp_path / "receipt.json"
    command = [sys.executable, "-c", "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); (p/'scores.csv').write_text('original')", str(out)]
    assert cs.guarded_stage(out, record, {"commit": "one"}, command, ["scores.csv"])
    assert not cs.guarded_stage(out, record, {"commit": "one"}, command, ["scores.csv"])
    with pytest.raises(ValueError, match="inputs changed"):
        cs.guarded_stage(out, record, {"commit": "two"}, command, ["scores.csv"])
    (out / "scores.csv").write_text("changed")
    with pytest.raises(ValueError, match="output changed"):
        cs.guarded_stage(out, record, {"commit": "one"}, command, ["scores.csv"])


def test_extension_uses_the_locked_ranking_and_actual_development_set(tmp_path):
    keys = [f"head:0:{i}" for i in range(8)]
    pd.DataFrame(dict(component=keys, knowledge_effect=range(8, 0, -1))).to_csv(tmp_path / "screen.csv", index=False)
    (tmp_path / "selection.json").write_text(json.dumps(dict(knowledge=keys[:4], status={"knowledge": "ok"})))
    meta = dict(status="done", selection_sha256=runinfo.sha256(tmp_path / "selection.json"),
                screen_csv_sha256=runinfo.sha256(tmp_path / "screen.csv"), model=cs.MODEL)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    cs.knowledge_screen(tmp_path, require_lock=True)
    expansion = tmp_path / "expansion"
    expansion.mkdir()
    (expansion / "results.json").write_text(json.dumps(dict(sets={"k8": {"components": keys}})))
    emeta = dict(meta, params={"knowledge_screen_sha256": meta["screen_csv_sha256"]},
                 results_sha256=runinfo.sha256(expansion / "results.json"))
    (expansion / "meta.json").write_text(json.dumps(emeta))
    cs.check_expansion(expansion, meta, keys)
    with pytest.raises(ValueError, match="development extension"):
        cs.check_expansion(expansion, meta, keys[::-1])
    with (tmp_path / "screen.csv").open("a") as f:
        f.write("head:0:9,9\n")
    with pytest.raises(ValueError, match="ranking differs"):
        cs.knowledge_screen(tmp_path, require_lock=True)

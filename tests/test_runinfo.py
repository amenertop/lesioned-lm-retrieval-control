"""Run records (lesion.runinfo)."""
import json

from lesion import runinfo


def test_write_meta(tmp_path):
    meta = runinfo.write_meta(tmp_path / "run", model="x", split="screening")
    saved = json.loads((tmp_path / "run" / "meta.json").read_text(encoding="utf-8"))
    assert saved["model"] == "x" and saved["split"] == "screening"
    assert len(saved["git_commit"]) == 40 and isinstance(saved["clean_tree"], bool)
    assert meta == saved


def test_sha256(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"abc")
    assert runinfo.sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

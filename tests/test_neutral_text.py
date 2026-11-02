"""Neutral text sample (ticket pilot 13)."""
import pytest

from lesion import models, neutral_text as nt

TOKENIZERS = ["olmoe-0924-base", "olmo2-1b", "qwen3-30b-a3b-instruct"]


def test_sample_shape():
    recs = nt.load_records()
    assert len(recs) == nt.N_PASSAGES == 200
    assert len({r["id"] for r in recs}) == 200
    assert all(r["n_words"] >= nt.MIN_WORDS and r["text"].strip() for r in recs)
    assert [r["id"] for r in recs] == sorted(r["id"] for r in recs)


@pytest.mark.skipif(not nt.SOURCE.exists(), reason="WikiText-103 file not downloaded (see data/raw/SOURCES.md)")
def test_rebuild_is_identical(tmp_path):
    out = tmp_path / "rebuilt.jsonl"
    nt.build(out=out)
    assert out.read_bytes() == nt.OUT.read_bytes()


def test_detokenize():
    assert nt.detokenize(" a 3 @-@ year @,@ old , ( small ) \" cat \" 's toy . ") == 'a 3-year,old, (small) "cat"\'s toy.'


@pytest.mark.parametrize("name", TOKENIZERS)
def test_every_passage_long_enough(name):
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    tok = models.load_tokenizer(name)
    n = [len(tok.encode(t, add_special_tokens=False)) for t in nt.load()]
    assert min(n) >= nt.MIN_TOKENS

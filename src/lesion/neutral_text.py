"""Neutral text for expert/head means and LM-loss matching (spec.md, 模型、消融方法与基线; ticket pilot 13).

200 passages sampled with seed 0 from the WikiText-103 validation split (raw, pinned revision in
data/raw/SOURCES.md). A passage is a run of consecutive paragraphs from one article, long enough that
every registered tokenizer gives at least MIN_TOKENS tokens; users take the first 512.

Build once:  python -m lesion.neutral_text   (writes data/neutral_text/, which is committed)
Use:         lesion.neutral_text.load()     -> list of 200 strings
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/wikitext103/validation-00000-of-00001.parquet"
SOURCE_SHA256 = "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"
OUT = ROOT / "data/neutral_text/wikitext103_val_seed0.jsonl"
N_PASSAGES, SEED = 200, 0
MIN_WORDS = 480    # about 600 tokens with the OLMo, OLMo-2 and Qwen tokenizers
MIN_TOKENS = 520   # checked per tokenizer in tests/test_neutral_text.py

_HEADING = re.compile(r"^ = [^=].* = $")        # article title, e.g. " = Homarus gammarus = "
_SUBHEADING = re.compile(r"^ (= )+[^=].*( =)+ $")


def detokenize(s):
    """Undo WikiText's spaced punctuation (after the lm-evaluation-harness wikitext detokenizer)."""
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = s.replace(" 's", "'s").replace("s ' ", "s' ")
    for p in (":", ";", ".", "!", "?", ","):
        s = s.replace(f" {p} ", f"{p} ")
    s = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", s)
    s = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", s)
    s = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', s)
    s = s.replace(" " + chr(176) + " ", chr(176))
    s = re.sub(r" ([.,;:!?])$", r"\1", s.strip())
    return s


def articles(lines):
    """[(title, [paragraph, ...]), ...] in file order; section headings are dropped."""
    out = []
    for line in lines:
        line = line.rstrip("\n")
        if _HEADING.match(line):
            out.append((line.strip(" ="), []))
        elif line.strip() and not _SUBHEADING.match(line) and out:
            out[-1][1].append(detokenize(line))
    return out


def passages(lines, min_words=MIN_WORDS):
    """Non-overlapping runs of consecutive paragraphs with at least min_words words; article tails are dropped."""
    out = []
    for a, (title, paras) in enumerate(articles(lines)):
        start, buf = 0, []
        for p, para in enumerate(paras):
            buf.append(para)
            if sum(len(x.split()) for x in buf) >= min_words:
                text = "\n".join(buf)
                out.append(dict(id=f"wt103v_{a:02d}_{start:03d}", article=title, first_paragraph=start,
                                n_paragraphs=len(buf), n_words=len(text.split()), text=text))
                start, buf = p + 1, []
    return out


def sample(candidates, n=N_PASSAGES, seed=SEED):
    """n passages drawn without replacement, returned in file order."""
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} candidate passages, need {n}")
    pick = np.sort(np.random.default_rng(seed).choice(len(candidates), size=n, replace=False))
    return [candidates[i] for i in pick]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source=SOURCE, out=OUT):
    if sha256(source) != SOURCE_SHA256:
        raise RuntimeError(f"{source} differs from the pinned WikiText-103 file; see data/raw/SOURCES.md")
    lines = pd.read_parquet(source).text.tolist()
    records = sample(passages(lines))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def load_records(path=OUT):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load(path=OUT):
    """The 200 passage texts, in file order."""
    return [r["text"] for r in load_records(path)]


if __name__ == "__main__":
    recs = build()
    print(f"{len(recs)} passages from {len({r['article'] for r in recs})} articles -> {OUT} (sha256 {sha256(OUT)[:12]})")

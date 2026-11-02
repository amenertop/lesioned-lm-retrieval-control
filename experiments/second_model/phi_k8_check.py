"""Exploratory check after GPU session 11 (Phi-3.5-MoE): is the knowledge set's huge naming damage a general breakdown?
1. Language-model loss on the 40 neutral passages (256 tokens each, as in session 10) for intact, K8, K4, the control
   set and five random sets matched to K8 (from the dissociation test's sets.json).
2. Naming on the first 40 dev concepts, intact vs K8 vs control: the answer's log-probability split into the word
   tokens and the end-of-turn token, and the greedy answer (up to 8 new tokens).
Run from the main-s11b worktree with PYTHONPATH=src, one GPU. Writes one JSON."""
import json
import sys
import time

import pandas as pd
import torch

from lesion import hooks, knowledge_items as ki, models, naming, neutral_text
from lesion.components import Component
from lesion.m3_optogenetics import lm_loss
from lesion.means import Means

MODEL = "phi35-moe-instruct"
OUTDIR = "/root/autodl-tmp/main-master/outputs"
SETS = f"{OUTDIR}/e2_confirm/{MODEL}/dissociation/sets.json"
MEANS = f"{OUTDIR}/means/{MODEL}@4368845__fp16.pt"
out_path = sys.argv[1]

t0 = time.time()
model, tok = models.load_model(MODEL, dtype=torch.float16)
means = Means.load(MEANS)
sets = json.load(open(SETS))
states = {"intact": [], "K8": sets["main"], "K4": sets["reported"], "control": sets["control"]}
states.update({f"random_{i:03d}": sets[f"random_{i:03d}"] for i in range(5)})
chat = models.uses_chat(MODEL)
texts = neutral_text.load()[:40]
items = ki.load_items("dev").head(40)
prompts = naming.build_prompts(items, pd.read_csv(ki.OUT / "demos.csv"), chat)
ends = naming._end_ids(tok)


def naming_detail():
    rows = []
    for p in prompts:
        ids, ans = naming.encode_pair(tok, p, chat)
        x = torch.tensor([ids + ans], device=model.device)
        with torch.no_grad():
            logp = torch.log_softmax(model(x).logits[0, len(ids) - 1:-1].float(), dim=-1)
        lp = logp[torch.arange(len(ans)), torch.tensor(ans, device=model.device)].tolist()
        word = sum(v for v, t in zip(lp, ans) if t not in ends)
        end = sum(v for v, t in zip(lp, ans) if t in ends)
        with torch.no_grad():
            g = model.generate(torch.tensor([ids], device=model.device), max_new_tokens=8, do_sample=False)
        rows.append(dict(item_id=str(p.item_id), target=p.target, word_logprob=word, end_logprob=end,
                         greedy=tok.decode(g[0, len(ids):], skip_special_tokens=False)))
    return rows


result = dict(model=MODEL, means=MEANS, sets=SETS, n_texts=len(texts), neutral_tokens=256, n_naming=len(prompts),
              lm_loss={}, naming={})
for name, comps in states.items():
    ctx = hooks.ablate(model, [Component.parse(c) for c in comps], means) if comps else None
    if ctx:
        ctx.__enter__()
    try:
        result["lm_loss"][name] = lm_loss(model, tok, texts, max_tokens=256)
        if name in ("intact", "K8", "control"):
            result["naming"][name] = naming_detail()
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
    print(f"{name}: lm_loss {result['lm_loss'][name]:.4f} ({time.time() - t0:.0f} s)", flush=True)
json.dump(result, open(out_path, "w"), indent=1, ensure_ascii=False)
print("written", out_path)

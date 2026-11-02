"""Exploratory check during GPU session 11 (main model): language-model loss on the 40 neutral passages (256 tokens
each, as in session 10) for intact, K8, C4, the new store set S8, the semantic-only control set C4s, and the first five
random sets of the new dissociation test (matched to S8). The held-out tests do not measure it.
Run from the main-s11 worktree with PYTHONPATH=src, one GPU. Writes one JSON."""
import json
import sys
import time

import torch

from lesion import hooks, models, neutral_text
from lesion.components import Component
from lesion.m3_optogenetics import lm_loss
from lesion.means import Means

MODEL = "qwen3-30b-a3b-instruct"
OUTDIR = "/root/autodl-tmp/main-master/outputs"
MEANS = f"{OUTDIR}/means/{MODEL}@0d7cf23__42ae809__fp16.pt"
NEW = f"{OUTDIR}/e2_confirm/{MODEL}/dissociation_new/sets.json"
K8 = ["expert:46:61", "expert:46:101", "head:37:24", "expert:45:32", "expert:45:48", "expert:43:70", "expert:42:115",
      "expert:45:93"]
out_path = sys.argv[1]

t0 = time.time()
new = json.load(open(NEW))
c4 = json.load(open(f"{OUTDIR}/e1_screen/{MODEL}/screen/selection.json"))["control"]
states = {"intact": [], "K8": K8, "C4": c4, "S8": new["main"], "C4s": new["control"]}
randoms = sorted(k for k in new if k.startswith("random_"))[:5]
states.update({k: new[k] for k in randoms})
model, tok = models.load_model(MODEL, dtype=torch.float16)
means = Means.load(MEANS)
texts = neutral_text.load()[:40]
result = dict(model=MODEL, means=MEANS, sets=states, n_texts=len(texts), neutral_tokens=256, lm_loss={})
for name, comps in states.items():
    if comps:
        with hooks.ablate(model, [Component.parse(c) for c in comps], means):
            result["lm_loss"][name] = lm_loss(model, tok, texts, max_tokens=256)
    else:
        result["lm_loss"][name] = lm_loss(model, tok, texts, max_tokens=256)
    print(f"{name}: lm_loss {result['lm_loss'][name]:.4f} ({time.time() - t0:.0f} s)", flush=True)
json.dump(result, open(out_path, "w"), indent=1)
print("written", out_path)

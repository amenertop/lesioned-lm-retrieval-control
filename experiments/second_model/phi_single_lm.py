"""Exploratory diagnosis after GPU session 11 (Phi-3.5-MoE): language-model loss on the 40 neutral passages (256 tokens
each) with each component of the knowledge set K8 ablated alone, and K8 without expert:3:7 (the one the control screen
listed as domain-general). Run from the main-s11b worktree with PYTHONPATH=src, one GPU. Writes one JSON."""
import json
import sys
import time

import torch

from lesion import hooks, models, neutral_text
from lesion.components import Component
from lesion.m3_optogenetics import lm_loss
from lesion.means import Means

MODEL = "phi35-moe-instruct"
OUTDIR = "/root/autodl-tmp/main-master/outputs"
MEANS = f"{OUTDIR}/means/{MODEL}@4368845__fp16.pt"
k8 = json.load(open(f"{OUTDIR}/e2_confirm/{MODEL}/dissociation/sets.json"))["main"]
states = {"intact": []}
states.update({c: [c] for c in k8})
states["K8 without expert:3:7"] = [c for c in k8 if c != "expert:3:7"]
t0 = time.time()
model, tok = models.load_model(MODEL, dtype=torch.float16)
means = Means.load(MEANS)
texts = neutral_text.load()[:40]
result = dict(model=MODEL, means=MEANS, k8=k8, n_texts=len(texts), neutral_tokens=256, lm_loss={})
for name, comps in states.items():
    if comps:
        with hooks.ablate(model, [Component.parse(c) for c in comps], means):
            result["lm_loss"][name] = lm_loss(model, tok, texts, max_tokens=256)
    else:
        result["lm_loss"][name] = lm_loss(model, tok, texts, max_tokens=256)
    print(f"{name}: lm_loss {result['lm_loss'][name]:.4f} ({time.time() - t0:.0f} s)", flush=True)
json.dump(result, open(sys.argv[1], "w"), indent=1)
print("written", sys.argv[1])

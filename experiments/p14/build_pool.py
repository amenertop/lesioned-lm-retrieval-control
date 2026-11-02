"""P14 step 0 (docs/plans/p14-learned-gates.md), no GPU: the candidate pools the gates are trained on.

    python experiments/p14/build_pool.py --model qwen3-30b-a3b-instruct --means <fp16 means> [--tiny]

Knowledge side: the top components by attributed knowledge effect (outputs/e3_knowledge/<model>/screen/attribution.csv),
every component a knowledge screen measured alone (the knowledge, store and K8b screens) and random sentinels;
two sizes of the attribution cut, 128 and 512 (agent2's review: the cut is a computing path, and a group that changes
when the cut grows was limited by it). Control side: the top 512 by attributed control effect
(outputs/e1_screen/<model>/attribution/attribution.csv), every component the control screen measured, C4 and C4s,
and sentinels. Excluded from both: candidate super experts (mean output over 20 times the layer median); C4 is
excluded from the knowledge pools. Writes outputs/p14/<model>/pools.json.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import means as means_mod, p14, rescreen, runinfo  # noqa: E402

SIZES = {"knowledge": (128, 512), "control": (512,)}
N_SENTINELS = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--means", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--sizes", default=None, help="override, e.g. knowledge=4,8;control=8 (tiny runs)")
    ap.add_argument("--sentinels", type=int, default=N_SENTINELS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suf = "_tiny" if args.tiny else ""
    kroot = ROOT / f"outputs/e3_knowledge{suf}" / args.model
    sroot = ROOT / f"outputs/e1_screen{suf}" / args.model
    croot = ROOT / f"outputs/e2_confirm{suf}" / args.model
    sizes = dict(SIZES)
    if args.sizes:
        sizes = {k: tuple(int(x) for x in v.split(",")) for k, v in (p.split("=") for p in args.sizes.split(";"))}
    m = means_mod.Means.load(args.means)
    supers = sorted(f"expert:{l}:{e}" for l, e in rescreen.super_expert_flags(m.expert_mean)) if m.expert_mean is not None else []
    c4 = json.loads((sroot / "screen/selection.json").read_text(encoding="utf-8"))["control"]
    c4s = json.loads((croot / "control_semantic/sets.json").read_text(encoding="utf-8"))["control"]
    kscreens = [kroot / "screen/screen.csv", kroot / "store_screen/screen.csv", kroot / "k8b/screen.csv"]
    kscreened = sorted(set().union(*(set(pd.read_csv(f).component) for f in kscreens if f.exists())))
    cscreened = sorted(set(pd.read_csv(sroot / "screen/screen.csv").component) | set(c4) | set(c4s))
    katt = pd.read_csv(kroot / "screen/attribution.csv")
    catt = pd.read_csv(sroot / "attribution/attribution.csv")
    pools, inputs = {}, {}
    for top in sizes["knowledge"]:
        members, sources = p14.build_pool(katt, "knowledge_effect", top, kscreened, exclude=set(supers) | set(c4),
                                          n_sentinels=args.sentinels)
        pools[f"knowledge_p{top}"] = dict(side="knowledge", column="knowledge_effect", top=top, members=members, sources=sources)
    for top in sizes["control"]:
        members, sources = p14.build_pool(catt, "control_effect", top, cscreened, exclude=set(supers),
                                          n_sentinels=args.sentinels)
        pools[f"control_p{top}"] = dict(side="control", column="control_effect", top=top, members=members, sources=sources)
    for f in kscreens + [kroot / "screen/attribution.csv", sroot / "attribution/attribution.csv", sroot / "screen/screen.csv",
                         sroot / "screen/selection.json", croot / "control_semantic/sets.json"]:
        if f.exists():
            inputs[str(f.relative_to(ROOT))] = runinfo.sha256(f)
    out = Path(args.out) if args.out else ROOT / f"outputs/p14{suf}" / args.model / "pools.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # the content depends only on the inputs (no commit or time in it), so a rebuild gives the same file and hash;
    # the training runs record that hash and refuse to resume on a different pool
    record = dict(model=args.model, tiny=args.tiny, means_sha256=runinfo.sha256(args.means), super_experts_excluded=supers,
                  c4=c4, c4s=c4s, n_sentinels=args.sentinels, inputs_sha256=inputs, pools=pools)
    text = json.dumps(record, indent=1)
    if out.exists() and out.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{out} exists with different content (inputs changed?); move it away first")
    tmp = out.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)                    # atomic: a half-written pools.json never exists
    meta = out.parent / "pools_meta.json"
    meta.with_suffix(".tmp").write_text(json.dumps(dict(commit=runinfo.commit(), pools_sha256=runinfo.sha256(out)), indent=1),
                                        encoding="utf-8")
    meta.with_suffix(".tmp").replace(meta)
    for name, p in pools.items():
        kinds = pd.Series([k.split(":")[0] for k in p["members"]]).value_counts().to_dict()
        print(f"{name}: {len(p['members'])} members {kinds}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()

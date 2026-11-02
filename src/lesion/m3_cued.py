"""Letter-cued naming, a text version of the progressive phonemic cueing of Jefferies, Patterson & Lambon Ralph
(2008) (docs/plans/gpu-session11.md, part C): the hint gives the target's first j letters, j = 0 (no hint) up to all
but the last letter, the demos' hints the same length; the model's greedy answer is classified as in experiment 3, so
accuracy can be counted at each length, as for the patients."""
import numpy as np

from . import m3_battery as mb, m3_scoring as ms


def prompts(items, chat=True):
    """One naming prompt per concept and hint length j = 0 .. len(target) - 1 (condition "cue<j>")."""
    instruction = ms.NAMING_INSTRUCTION + (ms.NAMING_REPLY_RULE if chat else "")
    out = []
    for row in items.to_dict("records"):
        word = row["target"]
        for j in range(len(word)):
            demos = [ms._naming_block(d["definition"], d["target"][:min(j, len(d["target"]) - 1)] or None, d["target"])
                     for d in ms.DEMOS]
            text = "\n\n".join([instruction, *demos, ms._naming_block(row["definition"], word[:j] or None)])
            out.append(ms.NamingPrompt(row["item_id"], f"cue{j}", text, {"target": word}))
    return out


def label(answers, cue_prompts, items):
    """The greedy answers (mb.greedy_answers, in prompt order) with the hint length, the word length and the answer's
    kind (target, coordinate, superordinate, associate, unrelated or other)."""
    rows = items.set_index("item_id")
    out = answers.copy()
    out["letters"] = [int(p.condition[3:]) for p in cue_prompts]
    out["n_letters"] = [len(p.candidates["target"]) for p in cue_prompts]
    out["kind"] = [mb.classify_answer(a, rows.loc[i]) for i, a in zip(out.item_id, out.answer)]
    return out


def summarize(cued, n_boot=10_000, seed=0):
    """Per state: accuracy with no hint, 1 letter, 2 letters and the final hint (all but the last letter); the benefit
    of 2 letters and of the final hint over none; and, for concepts not named without a hint but named at some length,
    the mean share of the word's letters given at the first length that worked. 95% intervals resample concepts."""
    d = cued.assign(correct=cued.kind == "target")
    ids = sorted(d.item_id.unique())
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(ids), len(ids)) for _ in range(n_boot)]
    ci = lambda boots: [round(float(q), 4) for q in np.nanquantile(np.asarray(boots, dtype=float), (0.025, 0.975))]
    out = {}
    for state in dict.fromkeys(d.state):
        s = d[d.state == state]
        t = s.pivot_table(index="item_id", columns="letters", values="correct", aggfunc="first").reindex(ids)
        n = s.groupby("item_id").n_letters.first()
        at = lambda j: t[j].to_numpy(dtype=float) if j in t else np.full(len(ids), np.nan)
        c = dict(none=at(0), one=at(1), two=at(2), final=np.array([t.loc[i, n[i] - 1] for i in ids], dtype=float))
        needed = []
        for i in ids:
            hits = [j for j in range(1, n[i]) if t.loc[i, j] == 1]
            needed.append(hits[0] / n[i] if t.loc[i, 0] != 1 and hits else np.nan)
        needed = np.array(needed)
        res = {f"accuracy_{k}": dict(estimate=round(float(np.nanmean(x)), 4), ci=ci([np.nanmean(x[i]) for i in draws]))
               for k, x in c.items()}
        for k in ("two", "final"):
            x = c[k] - c["none"]
            res[f"benefit_{k}"] = dict(estimate=round(float(np.nanmean(x)), 4), ci=ci([np.nanmean(x[i]) for i in draws]))
        res["share_of_letters_needed"] = dict(estimate=round(float(np.nanmean(needed)), 4) if np.isfinite(needed).any() else None,
                                              n=int(np.isfinite(needed).sum()))
        out[state] = res
    return out

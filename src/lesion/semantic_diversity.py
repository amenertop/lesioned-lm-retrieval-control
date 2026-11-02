"""Hoffman et al. (2013) SemD: exact-word joins and paired, within-task regressions.

No model loading and no source download. The publisher's XLSX is read with the standard library.
Coefficient intervals resample concept families, keeping the lesions paired.
"""
import zipfile
import json
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def load_norms(path):
    with zipfile.ZipFile(path) as book:
        strings = ["".join(t.itertext()) for t in ET.fromstring(book.read("xl/sharedStrings.xml"))]
        rows = []
        for row in ET.fromstring(book.read("xl/worksheets/sheet1.xml")).findall(".//s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                value = cell.find("s:v", NS)
                if value is not None:
                    col = "".join(c for c in cell.attrib["r"] if c.isalpha())
                    values[col] = strings[int(value.text)] if cell.get("t") == "s" else value.text
            rows.append(values)
    header = next(i for i, row in enumerate(rows) if "!term" in row.values())
    names = rows[header]
    table = pd.DataFrame([{names[k]: v for k, v in r.items() if k in names} for r in rows[header + 1:]])
    table = table.rename(columns={"!term": "word", "SemD": "semd", "lg_BNC_freq": "log_bnc_freq"})
    table = table[["word", "semd", "log_bnc_freq"]].dropna(subset=["word"])
    table["word"] = table.word.str.strip().str.lower()
    for col in ("semd", "log_bnc_freq"):
        table[col] = pd.to_numeric(table[col], errors="raise")
    if table.word.duplicated().any():
        raise ValueError("duplicate norm words")
    return table


def match_words(items, norms, word="target"):
    """Only strip whitespace and lowercase. No stemming, guessing or nearest-word substitution."""
    out = items.copy()
    out["norm_word"] = out[word].astype(str).str.strip().str.lower()
    return out.merge(norms.rename(columns={"word": "norm_word"}), on="norm_word", how="left", validate="many_to_one")


def coverage(items, norms, word="target"):
    joined = match_words(items, norms, word)
    return dict(n=int(len(joined)), matched=int(joined.semd.notna().sum()),
                missing=sorted(joined.loc[joined.semd.isna(), "norm_word"].unique().tolist()),
                rule="lowercase and strip only"), joined


def paired_regression(frame, covariates=("frequency", "n_tokens", "intact"), n_boot=2000, seed=1910):
    """One row per item, columns h_wr, h_sc, semd, family, and covariates, one task and scale.

    h is intact minus lesion score. Thus a positive SemD coefficient means greater damage.
    The interaction is the direct regression of h_sc - h_wr, not a comparison of significance.
    All fits use exactly the same complete-case items. Missingness is reported.
    """
    if n_boot < 0:
        raise ValueError("bootstrap count must be nonnegative")
    names = [*covariates, "semd"]
    needed = ["h_wr", "h_sc", "family", *names]
    if frame.duplicated("item_id").any():
        raise ValueError("expected one paired row per item")
    data = frame.dropna(subset=needed).copy()
    numeric = data[["h_wr", "h_sc", *names]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite regression input")
    if len(data) <= len(names) + 2 or data.family.nunique() < len(names) + 2:
        return dict(status="insufficient", n=int(len(data)), missing=int(len(frame) - len(data)))
    raw = data[names].to_numpy(float)
    scales = raw.std(axis=0)
    if (scales == 0).any():
        raise ValueError("constant predictor: " + ", ".join(np.array(names)[scales == 0]))
    x = np.column_stack([np.ones(len(data)), (raw - raw.mean(axis=0)) / scales])
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("collinear predictors")
    y = np.column_stack([data.h_wr, data.h_sc, data.h_sc - data.h_wr])
    full = np.linalg.lstsq(x, y, rcond=None)[0]
    # Frequency without SemD is fitted on the same items, so missingness cannot explain the change.
    reduced = np.linalg.lstsq(x[:, :-1], y, rcond=None)[0]
    groups = [np.flatnonzero(data.family.to_numpy() == f) for f in data.family.unique()]
    rng = np.random.default_rng(seed)
    draws, reduced_draws = [], []
    for _ in range(n_boot):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if np.linalg.matrix_rank(x[ix]) == x.shape[1]:
            draws.append(np.linalg.lstsq(x[ix], y[ix], rcond=None)[0])
            reduced_draws.append(np.linalg.lstsq(x[ix, :-1], y[ix], rcond=None)[0])
    if len(draws) < 0.9 * n_boot:
        raise ValueError("too many singular bootstrap resamples")
    if n_boot:
        low, high = np.percentile(draws, [2.5, 97.5], axis=0)
        rlow, rhigh = np.percentile(reduced_draws, [2.5, 97.5], axis=0)
        shift = np.asarray(draws)[:, 1, :] - np.asarray(reduced_draws)[:, 1, :]
        slow, shigh = np.percentile(shift, [2.5, 97.5], axis=0)
    coefficients = []
    for j, state in enumerate(("WR", "SC", "SC_minus_WR")):
        for i, name in enumerate(["intercept", *names]):
            coefficients.append(dict(state=state, predictor=name, estimate=float(full[i, j]),
                                     ci_low=float(low[i, j]) if n_boot else None,
                                     ci_high=float(high[i, j]) if n_boot else None))
    return dict(status="estimated" if n_boot else "point_only", n=int(len(data)), n_families=len(groups), missing=int(len(frame) - len(data)),
                n_boot=n_boot, valid_boot=len(draws), seed=seed, coefficients=coefficients,
                without_semd={s: dict(zip(["intercept", *covariates], reduced[:, j].tolist()))
                              for j, s in enumerate(("WR", "SC", "SC_minus_WR"))},
                frequency_without_semd={s: dict(estimate=float(reduced[1, j]), ci_low=float(rlow[1, j]) if n_boot else None,
                                               ci_high=float(rhigh[1, j]) if n_boot else None)
                                       for j, s in enumerate(("WR", "SC", "SC_minus_WR"))},
                frequency_shift={s: dict(estimate=float(full[1, j] - reduced[1, j]), ci_low=float(slow[j]) if n_boot else None,
                                        ci_high=float(shigh[j]) if n_boot else None)
                                 for j, s in enumerate(("WR", "SC", "SC_minus_WR"))},
                predictor_sd=dict(zip(names, scales.tolist())), interpretation="damage per predictor SD")


def paired_frame(scores, items, wr="K8", sc="C4"):
    """Scores have state/item_id/score. Each state's exact item set must match the intact set."""
    scores = scores[scores.state.isin(["intact", wr, sc])]
    if scores.duplicated(["state", "item_id"]).any():
        raise ValueError("scores must first be averaged within item and state")
    wide = scores.pivot(index="item_id", columns="state", values="score")
    if set(wide) != {"intact", wr, sc} or wide.isna().any().any():
        raise ValueError("unpaired or missing lesion observations")
    table = pd.DataFrame(dict(item_id=wide.index, intact=wide.intact.to_numpy(),
                               h_wr=(wide.intact - wide[wr]).to_numpy(), h_sc=(wide.intact - wide[sc]).to_numpy()))
    out = table.merge(items, on="item_id", how="left", validate="one_to_one", indicator=True)
    if (out._merge != "both").any():
        raise ValueError("scored items missing from metadata")
    return out.drop(columns="_merge")


def fine_frames(frame, wr="WR8_a0", sc="SC4_a0"):
    """Reuse the paired fine-battery loss table, preserving the same task and complete-case concepts."""
    frames = {}
    for task in ("naming", "definition", "synonym", "property"):
        if task not in frame:
            continue
        rows = frame[frame.state.isin([wr, sc])].dropna(subset=[task])
        losses = rows.pivot(index="item_id", columns="state", values=task)
        if set(losses) != {wr, sc} or losses.isna().any().any():
            raise ValueError("unpaired fine-battery lesions")
        info = rows.drop_duplicates("item_id").set_index("item_id").loc[losses.index]
        frames[task] = pd.DataFrame(dict(item_id=losses.index, family=info.family,
            h_wr=losses[wr], h_sc=losses[sc], semd=info.semd, frequency=info.frequency,
            n_tokens=info.n_tokens, intact=info["intact_" + task])).reset_index(drop=True)
    return frames


def existing_frames(root, norms_path, tokenizer):
    """Build paired M1 and M3 frames from existing local scores, without loading any model."""
    from . import runinfo
    norms = load_norms(norms_path)
    tok = tokenizer
    cov, frames, sources = {}, {}, {"norms": runinfo.sha256(norms_path)}
    # M3 target-word comprehension and production use exactly the same concepts.
    items = pd.read_csv(root / "data/materials/m3/final_items.csv")
    cov["m3_all"], info = coverage(items, norms)
    info = info.assign(family=info.item_id, frequency=info.zipf)
    info["n_tokens"] = [len(tok.encode(w, add_special_tokens=False)) for w in info.target]
    m3_dir = root / "outputs/e3_errors/qwen3-30b-a3b-instruct/confirm"
    meta = json.loads((m3_dir / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("complete"):
        raise ValueError("M3 source is incomplete")
    sources["m3_labels"] = runinfo.sha256(m3_dir / "labels.parquet")
    sources["m3_candidates"] = runinfo.sha256(m3_dir / "candidates.parquet")
    labels = pd.read_parquet(m3_dir / "labels.parquet")
    for task in ("word_to_definition", "options_letter"):
        table = labels[labels.fmt == task].groupby(["state", "item_id"], as_index=False).margin.mean().rename(columns={"margin": "score"})
        frames["m3_" + task] = paired_frame(table, info[["item_id", "family", "frequency", "semd", "n_tokens"]])
    candidates = pd.read_parquet(m3_dir / "candidates.parquet")
    table = candidates.query("condition == 'name_free' and role == 'target'")[["state", "item_id", "logprob"]].rename(columns={"logprob": "score"})
    frames["m3_naming"] = paired_frame(table, info[["item_id", "family", "frequency", "semd", "n_tokens"]])
    # M1 synonyms are symmetric word pairs. Both norms must exist, and their mean is fixed before looking at effects.
    m1 = pd.read_csv(root / "data/materials/m1/items.csv", dtype={"item_id": str})
    syn = m1[m1.task == "synonym"].copy()
    norm = norms.set_index("word").semd
    syn["semd1"], syn["semd2"] = syn.word1.map(norm), syn.word2.map(norm)
    syn["semd"] = (syn.semd1 + syn.semd2) / 2
    syn["frequency"] = pd.to_numeric(syn.min_freq)
    syn["family"] = syn.concept_group
    syn["n_tokens"] = [len(tok.encode(a, add_special_tokens=False)) + len(tok.encode(b, add_special_tokens=False))
                         for a, b in zip(syn.word1, syn.word2)]
    syn["different_pair"] = (syn.synonym_type == "different").astype(int)
    cov["m1_synonym"] = dict(n=len(syn), matched=int(syn.semd.notna().sum()), rule="mean of both words, both required",
                             missing=sorted(set(syn.loc[syn.semd1.isna(), "word1"]) | set(syn.loc[syn.semd2.isna(), "word2"])))
    p = root / "outputs/server/e2_confirm/qwen3-30b-a3b-instruct/dissociation/m1_scores.parquet"
    m1_meta = json.loads((p.parent / "meta.json").read_text(encoding="utf-8"))
    sets = json.loads((p.parent / "sets.json").read_text(encoding="utf-8"))
    if m1_meta.get("status") != "done" or any(set(sets[a]) != set(meta["sets"][b]) for a, b in (("main", "K8"), ("control", "C4"))):
        raise ValueError("M1/M3 sources are incomplete or use different lesion groups")
    source = pd.read_parquet(p)
    source["score"] = source.logit_correct - source.logit_wrong
    source["item_id"] = source.item_id.astype(str)
    source["state"] = source["set"].map({"intact": "intact", "main": "K8", "control": "C4"})
    table = source[source.task == "synonym"].groupby(["state", "item_id"], as_index=False).score.mean()
    frames["m1_synonym"] = paired_frame(table, syn[["item_id", "family", "frequency", "semd", "n_tokens", "different_pair"]])
    sources["m1_scores"] = runinfo.sha256(p)
    sources.update({str(p.relative_to(root)): runinfo.sha256(p) for p in
                    (root / "data/materials/m1/items.csv", root / "data/materials/m3/final_items.csv",
                     m3_dir / "meta.json", p.parent / "meta.json", p.parent / "sets.json")})
    return frames, cov, sources


def public_summary(payload):
    """Statistical summaries for the ledger, without redistributed per-word norm values."""
    tasks = {}
    for name, result in payload["results"].items():
        unit = "nat / predictor SD" if name in ("m3_naming", "naming") else "logit / predictor SD"
        task = dict(n=result["n"], unit=unit, coefficients={}, frequency_without_semd=result["frequency_without_semd"],
                    frequency_shift=result["frequency_shift"])
        for r in result["coefficients"]:
            if r["predictor"] in ("semd", "frequency"):
                task["coefficients"].setdefault(r["state"], {})[r["predictor"]] = {
                    k: r[k] for k in ("estimate", "ci_low", "ci_high")}
        tasks[name] = task
    return dict(meta={k: v for k, v in payload.items() if k not in ("results", "coverage")}, tasks=tasks,
                coverage={name: dict(n=r["n"], matched=r["matched"], unit="items") for name, r in payload["coverage"].items()})

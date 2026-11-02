"""Run the two new batteries through existing scorers, one explicit lesion state at a time.

No lesion discovery or statistical claims live here. A completed state is reusable only with the exact same
model revision, items, prompts, means, settings and code commit. Incomplete state files are never accepted.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import gate_objective, m3_battery, m3_scoring, regional_generation, runinfo, title_gap_states


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def score_state(model, tok, naming, labels, chat=True, batches=(16, 11, 7), max_new_tokens=8, region=None,
                naming_batches=None):
    candidate_batches = batches if naming_batches is None else naming_batches
    if region is None:
        candidates = m3_scoring.averaged_naming(model, tok, naming, chat, candidate_batches)
    else:
        tables = [m3_battery.candidate_scores(model, tok, naming, chat, b, region) for b in candidate_batches]
        candidates = tables[0].copy()
        for col in ("logprob", "first_token_rank"):
            candidates[col] = np.mean([t[col].to_numpy() for t in tables], axis=0)
    choices = pd.DataFrame(columns=['item_id', 'fmt', 'variant', 'correct', 'logit_A', 'logit_B', 'margin', 'top_correct'])
    if labels:
        tables = [m3_battery.label_scores(model, tok, labels, chat, batch_size=b, region=region) for b in batches]
        choices = tables[0].copy()
        for col in [c for c in choices if c.startswith("logit_")]:
            choices[col] = np.mean([t[col].to_numpy() for t in tables], axis=0)
        # These batteries have exactly two choices. Derive decisions after averaging the logits.
        choices["margin"] = np.where(choices.correct == "A", choices.logit_A - choices.logit_B,
                                     choices.logit_B - choices.logit_A)
        choices["top_correct"] = choices.margin > 0
    generated = regional_generation.greedy_answers(model, tok, naming, chat, batches[0], max_new_tokens, region)
    generated["condition"] = [p.condition for p in naming]
    return dict(candidates=candidates, labels=choices, generated=generated)


def validate_tables(tables, naming, labels):
    expected = {
        "candidates": {(p.item_id, p.condition, r) for p in naming for r in p.candidates},
        "generated": {(p.item_id, p.condition) for p in naming},
        "labels": {(p.item_id, p.fmt, p.variant) for p in labels},
    }
    keys = dict(candidates=["item_id", "condition", "role"], generated=["item_id", "condition"],
                labels=["item_id", "fmt", "variant"])
    for name, table in tables.items():
        if name not in expected or table.duplicated(keys[name]).any():
            raise ValueError(f"{name}: unexpected table or duplicate observations")
        found = set(table[keys[name]].itertuples(index=False, name=None))
        if found != expected[name]:
            raise ValueError(f"{name}: missing or unexpected observations")
        for col in table.select_dtypes(include="number"):
            if not np.isfinite(table[col].to_numpy()).all():
                raise ValueError(f"{name}.{col}: non-finite score")
    if set(tables) != set(expected):
        raise ValueError("missing score table")


def run_states(model, tok, means, naming, labels, states, out, provenance, chat=True, batches=(16, 11, 7),
               max_new_tokens=8, neutral_sequences=(), naming_batches=None):
    title_gap_states.validate_states(states, model.config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    definition = dict(provenance=provenance, states=states, chat=chat, batches=list(batches),
                      max_new_tokens=max_new_tokens, naming=[vars(p) for p in naming], labels=[vars(p) for p in labels],
                      neutral_sequences=[list(s) for s in neutral_sequences])
    if naming_batches is not None:
        if not naming_batches or any(not isinstance(b, int) or b <= 0 for b in naming_batches):
            raise ValueError('candidate batches must be positive integers')
        definition['naming_batches'] = list(naming_batches)
    fingerprint = _fingerprint(definition)
    plan = out / "plan.json"
    if plan.exists() and json.loads(plan.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise ValueError("run directory belongs to different inputs or settings")
    plan.write_text(json.dumps(dict(fingerprint=fingerprint, **definition), indent=1) + "\n", encoding="utf-8")
    lm_rows = []
    for state in states:
        directory = out / state["name"]
        marker = directory / "complete.json"
        if marker.exists():
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record["fingerprint"] != fingerprint:
                raise ValueError("state fingerprint mismatch")
            for name, digest in record["files"].items():
                if runinfo.sha256(directory / name) != digest:
                    raise ValueError("completed state file changed")
            tables = {name: pd.read_parquet(directory / f"{name}.parquet") for name in ("candidates", "labels", "generated")}
            validate_tables(tables, naming, labels)
            lm_rows.append(record["language"])
            continue
        directory.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with title_gap_states.apply_state(model, means, state):
            tables = score_state(model, tok, naming, labels, chat, batches, max_new_tokens, state.get("region"),
                                 naming_batches=naming_batches)
            import torch
            with torch.no_grad():
                losses = [float(gate_objective.lm_loss(model, [seq], tok.eos_token_id or 0)) for seq in neutral_sequences]
        validate_tables(tables, naming, labels)
        files = {}
        for name, table in tables.items():
            path = directory / f"{name}.parquet"
            table.to_parquet(path, index=False)
            files[path.name] = runinfo.sha256(path)
        lengths = [len(seq) - 1 for seq in neutral_sequences]
        language = dict(state=state["name"], loss=float(np.average(losses, weights=lengths)) if losses else None,
                        tokens=sum(lengths), region="all",
                        proxy_for_restricted_state=state.get("region") in ("answer", "prompt"))
        record = dict(fingerprint=fingerprint, files=files, language=language, seconds=time.monotonic() - start)
        marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        lm_rows.append(language)
        print(f"{state['name']}: complete ({record['seconds']:.1f}s)", flush=True)
    pd.DataFrame(lm_rows).to_csv(out / "language.csv", index=False)
    (out / "complete.json").write_text(json.dumps(dict(fingerprint=fingerprint, states=[s["name"] for s in states]), indent=2) + "\n", encoding="utf-8")

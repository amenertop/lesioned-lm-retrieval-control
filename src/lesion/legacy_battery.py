"""Common old-item readouts for coarse and component lesions. Selection and statistics live elsewhere."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import gate_objective, m3_battery, naming, runinfo, title_gap_scoring, title_gap_states


def score(model, tok, words, control, comprehension, control_scorer, chat=True, batches=(16, 11, 7)):
    n = naming.averaged_logprobs(model, tok, words, chat, batches)
    c = control_scorer(model, tok, control, chat=chat, batch_size=32)
    choices = [m3_battery.label_scores(model, tok, comprehension, chat, b) for b in batches]
    u = choices[0].copy()
    for col in ("logit_A", "logit_B"):
        u[col] = np.mean([t[col].to_numpy() for t in choices], axis=0)
    u["margin"] = np.where(u.correct == "A", u.logit_A - u.logit_B, u.logit_B - u.logit_A)
    u["top_correct"] = u.margin > 0
    return dict(naming=n, control=c, comprehension=u)


def run(model, tok, means, words, control, comprehension, states, out, provenance, control_scorer,
        chat=True, batches=(16, 11, 7), neutral_sequences=()):
    title_gap_states.validate_states(states, model.config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    definition = dict(provenance=provenance, states=states, chat=chat, batches=list(batches),
                      words=[vars(p) for p in words], control=[vars(p) for p in control],
                      comprehension=[vars(p) for p in comprehension], neutral_sequences=list(neutral_sequences))
    fingerprint = title_gap_scoring._fingerprint(definition)
    plan = out / "plan.json"
    if plan.exists() and json.loads(plan.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise ValueError("run directory belongs to different inputs or settings")
    plan.write_text(json.dumps(dict(fingerprint=fingerprint, **definition), indent=1) + "\n", encoding="utf-8")
    language = []
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
            language.append(record["language"])
            continue
        start = time.monotonic()
        with title_gap_states.apply_state(model, means, state):
            tables = score(model, tok, words, control, comprehension, control_scorer, chat, batches)
            with torch.no_grad():
                losses = [float(gate_objective.lm_loss(model, [seq], tok.eos_token_id or 0))
                          for seq in neutral_sequences]
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, table in tables.items():
            if table.empty or not np.isfinite(table.select_dtypes(include="number").to_numpy()).all():
                raise ValueError(f"{state['name']}/{name}: missing or non-finite scores")
            path = directory / f"{name}.parquet"
            table.to_parquet(path, index=False)
            files[path.name] = runinfo.sha256(path)
        lengths = [len(s) - 1 for s in neutral_sequences]
        lm = dict(state=state["name"], loss=float(np.average(losses, weights=lengths)) if losses else None,
                  tokens=sum(lengths))
        record = dict(fingerprint=fingerprint, files=files, language=lm, seconds=time.monotonic() - start)
        marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        language.append(lm)
        pd.DataFrame(language).to_csv(out / "language.csv", index=False)
        print(f"{state['name']}: complete ({record['seconds']:.1f}s)", flush=True)
    (out / "complete.json").write_text(json.dumps(dict(fingerprint=fingerprint, states=[s["name"] for s in states]),
                                                 indent=2) + "\n", encoding="utf-8")

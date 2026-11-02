"""Source checks for the second-model replication, including completed-stage reuse."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import means, model_census, models, neutral_text, runinfo, scoring

MODEL = "qwen3-coder-30b-a3b-instruct"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preflight(means_path, census_path, tag):
    commit = runinfo.check_analysis_freeze(tag)
    census = read(census_path)
    weights = models.verified_dir(MODEL)
    hashes = {f: runinfo.sha256(scoring.M1 / f) for f in ("items.csv", "demos.csv", "templates.yaml")}
    if (weights is None or census.get("model") != MODEL
            or census.get("specification") != model_census.CANDIDATES[MODEL]
            or census.get("dtype") != "float16" or census.get("material_hashes") != hashes
            or census.get("verification_sha256") != runinfo.sha256(weights / "VERIFIED.json")
            or not census.get("pass_format") or not census.get("replication_eligible")
            or (census.get("g1") or {}).get("g1_decision") != "pass"):
        raise ValueError("Coder census failed or has different weights, materials, precision or revision")
    meta = means.Means.load(means_path).meta
    texts = neutral_text.load()
    expected = dict(model_name=MODEL, revision=models.MODELS[MODEL][1], dtype="float16",
                    text_sha256=means._text_hash(texts), n_texts=len(texts), max_tokens=512)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Coder means have different model, revision, precision or neutral text")
    return dict(commit=commit, freeze_tag=tag, model=MODEL, revision=models.MODELS[MODEL][1],
                means_sha256=runinfo.sha256(means_path), census_sha256=runinfo.sha256(census_path),
                verification_sha256=census["verification_sha256"], material_hashes=hashes,
                neutral_text_sha256=meta["text_sha256"], means_commit=meta["git_commit"])


def knowledge_screen(directory, require_lock=False, minimum=8):
    """Read a finished ranking, binding the 8-member extension to the actual screen."""
    directory = Path(directory)
    meta, selection = read(directory / "meta.json"), read(directory / "selection.json")
    if meta.get("status") != "done" or runinfo.sha256(directory / "selection.json") != meta.get("selection_sha256"):
        raise ValueError("knowledge screen unfinished or selection changed")
    want = meta.get("screen_csv_sha256")
    if (require_lock or want) and runinfo.sha256(directory / "screen.csv") != want:
        raise ValueError("knowledge ranking differs from its recorded hash")
    ranked = pd.read_csv(directory / "screen.csv").sort_values("knowledge_effect", ascending=False)
    size = selection.get("k", 4)
    if (ranked.component.duplicated().any() or len(ranked) < minimum or (require_lock and size != 4)
            or not np.isfinite(ranked.knowledge_effect).all()
            or selection.get("status", {}).get("knowledge") != "ok"
            or selection.get("knowledge") != ranked.head(size).component.tolist()):
        raise ValueError("knowledge selection missing, duplicated or inconsistent with the ranking")
    return meta, selection, ranked


def check_expansion(directory, screen_meta, keys):
    directory = Path(directory)
    meta, result = read(directory / "meta.json"), read(directory / "results.json")
    same = ("model", "revision", "dtype", "means_sha256", "materials_sha256", "resume_commit")
    if (meta.get("status") != "done" or any(meta.get(k) != screen_meta.get(k) for k in same)
            or meta.get("results_sha256") != runinfo.sha256(directory / "results.json")
            or meta.get("params", {}).get("knowledge_screen_sha256") != screen_meta.get("screen_csv_sha256")
            or result.get("sets", {}).get("k8", {}).get("components") != keys):
        raise ValueError("final knowledge set is not the completed, source-locked development extension")
    return runinfo.sha256(directory / "results.json")


def guarded_stage(directory, record, identity, command, required, inputs=()):
    """Reuse only the same request with every recorded output still present and unchanged."""
    directory, record = Path(directory), Path(record)
    request = dict(identity=identity, command=command,
                   inputs={str(p): runinfo.sha256(p) for p in inputs}, required=list(required))
    if record.exists():
        old = read(record)
        if old["request"] != request:
            raise ValueError("stage inputs changed, use a fresh output directory")
        if old.get("outputs"):
            for name, digest in old["outputs"].items():
                if not (directory / name).is_file() or runinfo.sha256(directory / name) != digest:
                    raise ValueError(f"completed stage output changed: {name}")
            return False
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("existing stage has no source record, do not infer completion from a marker")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(dict(request=request), indent=2) + "\n", encoding="utf-8")
    subprocess.run(command, check=True)
    if any(not (directory / f).is_file() for f in required):
        raise ValueError("stage did not produce all required outputs")
    if (directory / "meta.json").exists():
        meta = read(directory / "meta.json")
        if "status" in meta and meta["status"] != "done":
            raise ValueError("stage metadata is not complete")
    outputs = {str(p.relative_to(directory)): runinfo.sha256(p)
               for p in sorted(directory.rglob("*")) if p.is_file()}
    record.write_text(json.dumps(dict(request=request, outputs=outputs), indent=2) + "\n", encoding="utf-8")
    return True

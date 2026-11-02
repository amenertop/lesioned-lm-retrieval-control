"""Verified CPU loading for census models that need not occupy a permanent weight directory.

Two shards are fetched at a time. Each is checked against the pinned official manifest before loading.
The complete CPU model can wait while another experiment uses the GPU. No model files are deleted.
"""
import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "lesion-study-verified-census/1.0"}


def fetch_bytes(url, size=None, digest=None):
    stream = io.BytesIO()
    h = hashlib.sha256()
    for attempt in range(5):
        try:
            start = stream.tell()
            headers = {**UA, **({"Range": f"bytes={start}-"} if start else {})}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if start and response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-") or (size is not None and not content_range.endswith(f"/{size}")):
                        raise ValueError("resumed download has a different byte range")
                elif start:
                    # Some endpoints ignore Range. Their full response must replace the partial bytes.
                    stream.seek(0)
                    stream.truncate()
                    h = hashlib.sha256()
                for chunk in iter(lambda: response.read(8 << 20), b""):
                    stream.write(chunk)
                    h.update(chunk)
                    if size is not None and stream.tell() > size:
                        raise ValueError("download exceeds the pinned file size")
            if size is not None and stream.tell() != size:
                raise OSError("download is incomplete")
            if digest is not None and h.hexdigest() != digest:
                raise ValueError("download differs from the pinned SHA-256")
            return stream.getvalue()
        except OSError as error:
            if attempt == 4:
                raise
            print(f"download retry {attempt+1}: {type(error).__name__}, retained {stream.tell()}/{size} bytes", flush=True)
            if size is None:
                stream.seek(0)
                stream.truncate()
                h = hashlib.sha256()
            time.sleep(5)


def metadata(source, directory):
    """Only small top-level files, each checked against the fixed HF revision's git blob ID."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    repo, revision = source["repo"], source["revision"]
    info = json.loads(fetch_bytes(f"{endpoint}/api/models/{repo}/revision/{revision}?blobs=true"))
    directory.mkdir(parents=True, exist_ok=True)
    for item in info["siblings"]:
        name = item["rfilename"]
        if "/" in name or item.get("lfs"):
            continue
        data = fetch_bytes(f"{endpoint}/{repo}/resolve/{revision}/{name}")
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if item.get("blobId") != blob:
            raise ValueError(f"small-file revision mismatch: {name}")
        (directory / name).write_bytes(data)
    for name, digest in source["metadata_hashes"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"candidate metadata mismatch: {name}")


def load_cpu(config, source, fetch=None):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load
    from transformers import AutoModelForCausalLM

    if fetch is None:
        def fetch(row):
            filename = urllib.parse.quote_plus(row["file"])
            url = f"https://modelscope.cn/api/v1/models/{source['repo']}/repo?Revision=master&FilePath={filename}"
            return fetch_bytes(url, row["size"], row["sha256"])
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    expected, loaded = set(model.state_dict()), set()
    weights = list(source["weights"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(fetch, row) for row in weights[:2]]
        for i, row in enumerate(weights):
            data = pending.pop(0).result()
            if len(data) != row["size"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"shard mismatch: {row['file']}")
            shard = load(data)
            if loaded.intersection(shard) or set(shard) - expected:
                raise ValueError("duplicate or unexpected parameter in checkpoint")
            model.load_state_dict(shard, strict=False, assign=True)
            loaded.update(shard)
            del data, shard
            if i + 2 < len(weights):
                pending.append(pool.submit(fetch, weights[i+2]))
            print(f"CPU verified {i+1}/{len(weights)}: {row['file']}", flush=True)
    if loaded != expected or any(p.is_meta for p in model.parameters()):
        raise ValueError("checkpoint does not cover the complete model")
    return model.eval()

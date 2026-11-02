"""Collect and store component output means for mean ablation.

The collection hooks observe expert outputs before routing weights are applied,
attention-head inputs immediately before ``o_proj``, and dense MLP outputs.
Padding tokens are excluded from the non-expert means.  Expert collection uses
equal-length batches so the expert modules, whose hooks do not receive token
indices, are never called on padding rows.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .components import moe_block, moe_layers, n_experts as count_experts


def _model_layers(model):
    try:
        return model.model.layers
    except AttributeError as exc:  # pragma: no cover - unsupported model wrapper
        raise TypeError("model must expose model.layers") from exc


def _model_identity(model):
    """Infer a registered model name from a pinned local model path when possible."""
    name_or_path = str(getattr(model, "name_or_path", "") or getattr(model.config, "_name_or_path", ""))
    try:
        from .models import MODELS
        for name, (repo, revision, _chat) in MODELS.items():
            repo_key = repo.replace("/", "__")
            if repo_key in name_or_path and revision in name_or_path:
                return name, revision
    except Exception:
        pass
    return name_or_path or model.__class__.__name__, str(getattr(model.config, "_commit_hash", "unknown"))


def _git_commit():
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _text_hash(texts):
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class Means:
    meta: dict
    n_tokens: int
    expert_mean: torch.Tensor | None
    expert_count: torch.Tensor | None
    head_mean: torch.Tensor
    mlp_mean: torch.Tensor | None

    def routing_frequency(self):
        if self.expert_count is None:
            raise ValueError("routing frequency is unavailable for a dense model")
        if self.n_tokens <= 0:
            raise ValueError("cannot compute routing frequency with no tokens")
        return (self.expert_count.float() / float(self.n_tokens)).cpu().numpy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "meta": dict(self.meta),
            "n_tokens": int(self.n_tokens),
            "expert_mean": self.expert_mean,
            "expert_count": self.expert_count,
            "head_mean": self.head_mean,
            "mlp_mean": self.mlp_mean,
        }, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        required = {"meta", "n_tokens", "expert_mean", "expert_count", "head_mean", "mlp_mean"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"means file is missing fields: {sorted(missing)}")
        return cls(**{k: payload[k] for k in required})


def _token_batches(tokenizer, texts, max_tokens, batch_size):
    sequences = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        if not ids:
            continue
        sequences.append(torch.tensor(ids, dtype=torch.long))
    # Expert hooks receive only selected rows and no token positions. Grouping
    # equal-length sequences means every row seen by an expert is a real token.
    groups = {}
    for seq in sequences:
        groups.setdefault(len(seq), []).append(seq)
    for length in sorted(groups):
        group = groups[length]
        for start in range(0, len(group), batch_size):
            ids = torch.stack(group[start:start + batch_size], dim=0)
            mask = torch.ones_like(ids)
            positions = torch.arange(length, dtype=torch.long).expand(ids.shape[0], -1)
            yield ids, mask, positions


@torch.no_grad()
def collect_means(model, tokenizer, texts, max_tokens=512, batch_size=8,
                  model_name=None, revision=None):
    """Collect component means from neutral text using forward passes only."""
    if max_tokens <= 0 or batch_size <= 0:
        raise ValueError("max_tokens and batch_size must be positive")
    layers = _model_layers(model)
    config = model.config
    n_layers = len(layers)
    hidden = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", hidden // n_heads) or (hidden // n_heads))
    device = next(model.parameters()).device

    expert_layers = set(moe_layers(config))
    n_experts = count_experts(config)
    expert_sum = (torch.zeros((n_layers, n_experts, hidden), dtype=torch.float32, device="cpu")
                  if n_experts else None)
    expert_count = (torch.zeros((n_layers, n_experts), dtype=torch.long, device="cpu")
                    if n_experts else None)
    head_sum = torch.zeros((n_layers, n_heads, head_dim), dtype=torch.float32, device="cpu")
    mlp_sum = torch.zeros((n_layers, hidden), dtype=torch.float32, device="cpu")
    mlp_seen = torch.zeros(n_layers, dtype=torch.bool)
    handles = []

    def expert_hook(layer_index, expert_index):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.numel() == 0:
                return
            value = value.detach().float().cpu().reshape(-1, hidden)
            expert_sum[layer_index, expert_index] += value.sum(dim=0)
            expert_count[layer_index, expert_index] += value.shape[0]
        return hook

    def head_hook(layer_index):
        def hook(_module, inputs):
            value = inputs[0]
            if value is None or value.numel() == 0:
                return
            bsz, seq_len, _ = value.shape
            value = value.detach().float().reshape(bsz, seq_len, n_heads, head_dim)
            # Batches are equal-length, but retain explicit mask filtering.
            valid = current_mask.to(value.device, dtype=torch.bool).reshape(bsz, seq_len)
            head_sum[layer_index] += value[valid].sum(dim=0).cpu()
        return hook

    def mlp_hook(layer_index):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.numel() == 0:
                return
            bsz, seq_len, _ = value.shape
            valid = current_mask.to(value.device, dtype=torch.bool).reshape(bsz, seq_len)
            mlp_sum[layer_index] += value.detach().float()[valid].sum(dim=0).cpu()
            mlp_seen[layer_index] = True
        return hook

    for layer_index, layer in enumerate(layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(head_hook(layer_index)))
        block = moe_block(layer)
        if block is not None:
            for expert_index, expert in enumerate(block.experts):
                handles.append(expert.register_forward_hook(expert_hook(layer_index, expert_index)))
        else:
            handles.append(layer.mlp.register_forward_hook(mlp_hook(layer_index)))

    n_tokens = 0
    current_mask = torch.empty(0, dtype=torch.long, device=device)
    try:
        model.eval()
        for ids, mask, positions in _token_batches(tokenizer, texts, max_tokens, batch_size):
            ids, mask, positions = ids.to(device), mask.to(device), positions.to(device)
            current_mask = mask
            n_tokens += int(mask.sum().item())
            model(input_ids=ids, attention_mask=mask, position_ids=positions)
    finally:
        for handle in handles:
            handle.remove()

    if n_tokens == 0:
        raise ValueError("neutral text produced no tokens")
    head_mean = head_sum / float(n_tokens)
    if n_experts:
        safe = expert_count.clamp_min(1).unsqueeze(-1).float()
        expert_mean = expert_sum / safe
        expert_mean[expert_count == 0] = 0
    else:
        expert_mean = None
        expert_count = None
    mlp_mean = mlp_sum / float(n_tokens)
    mlp_mean[~mlp_seen] = 0
    inferred_name, inferred_revision = _model_identity(model)
    meta = {
        "model_name": model_name or inferred_name,
        "revision": revision or inferred_revision,
        "git_commit": _git_commit(),
        "text_sha256": _text_hash(texts),
        "max_tokens": int(max_tokens),
        "n_texts": int(len(texts)),
    }
    return Means(meta, n_tokens, expert_mean, expert_count, head_mean, mlp_mean)

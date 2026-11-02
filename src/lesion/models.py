"""Model registry, pinned revisions and loading (docs/plans/pilot-protocol.md §2)."""
import os
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# name -> (repo, pinned revision, uses chat template)
MODELS = {
    "olmoe-0924-base": ("allenai/OLMoE-1B-7B-0924", "6d84c48581ece794365f2b8e9cfb043c68ade9c5", False),
    "olmoe-0924-instruct": ("allenai/OLMoE-1B-7B-0924-Instruct", "7f1c97f440f06ce36705e4f2b843edb5925f4498", True),
    "olmo2-1b": ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf", False),
    # Backup model, pre-specified in the pilot protocol §2 and §6.
    "olmoe-0125-base": ("allenai/OLMoE-1B-7B-0125", "9b0c1aa87e34a20052389dce1f0cf01da783f654", False),
    "olmoe-0125-instruct": ("allenai/OLMoE-1B-7B-0125-Instruct", "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e", True),
    # Larger models, exploratory only (decisions-pending item 6.5); not part of the frozen G1 plan.
    "qwen3-30b-a3b-instruct": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", True),
    "olmo2-13b-instruct": ("allenai/OLMo-2-1124-13B-Instruct", "3a5c85baefbb1896a54d56fe2e76c0395627ddf4", True),
    # Second-model candidates (P6, user decision 2026-10-12): intact measurement check (G1) only, exploratory;
    # revisions as checked by agent2 on 2026-10-09 (reports/explore/2026-10-09-replication-model-sources.json).
    "phi35-moe-instruct": ("microsoft/Phi-3.5-MoE-instruct", "43688451b462a3351d8580625ebe1931adb3986d", True),
    "qwen3-coder-30b-a3b-instruct": ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "b2cff646eb4bb1d68355c01b18ae02e7cf42d120", True),
    # Second-model candidates (P12, user decision 2026-10-15): intact measurement check (G1) only, exploratory;
    # docs/plans/second-model-g1-p12.md. Revisions from the official metadata (reports/explore/2026-10-15-gpt-oss-20b-
    # metadata.json; 2026-10-09-replication-model-sources.json).
    "gpt-oss-20b": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", True),
    "deepseek-v2-lite-chat": ("deepseek-ai/DeepSeek-V2-Lite-Chat", "85864749cd611b4353ce1decdb286193298f64c7", True),
}

# Models whose prompts differ from "the chat template, then the bare label" (P12): name -> (prompt format, label
# prefix). Prompt format: "plain" (document start + text), "chat_template", "harmony" (gpt-oss: the official template
# with a fixed date, then an empty analysis channel and the final channel; lesion.scoring.harmony_text),
# "harmony_answer" (the same, then "Answer:"). Label prefix: "bare" ("A") or "space" (" A"). Only lesion.scoring
# interprets these; uses_chat refuses these models so that modules written for plain chat models (attention, M3,
# naming) cannot use them by mistake. gpt-oss was ("harmony", "bare") in session 14, where it failed the format gate
# (77% < 90%); since 2026-10-16 (P12b, user decision) it is ("harmony_answer", "space").
PROMPT_FORMATS = {
    "gpt-oss-20b": ("harmony_answer", "space"),
    "deepseek-v2-lite-chat": ("chat_template", "space"),
}

# Weights downloaded from ModelScope and verified file by file against the pinned Hugging Face revision
# (each folder holds a VERIFIED.json written by the download script).
VERIFIED_DIR = Path(os.environ.get("LESION_MODEL_DIR", "/autodl-fs/data/main/ms"))

# Tokenizer/config files downloaded per revision, for machines without the full weights.
LOCAL_HF = Path(os.environ.get("LESION_LOCAL_HF", "D:/hf/local"))


def uses_chat(name):
    if name in PROMPT_FORMATS:
        raise ValueError(f"{name} has its own prompt format {PROMPT_FORMATS[name]}; use prompt_format() and lesion.scoring's fmt")
    return MODELS[name][2]


def prompt_format(name):
    """(prompt format, label prefix) for any registered model (see PROMPT_FORMATS)."""
    if name in PROMPT_FORMATS:
        return PROMPT_FORMATS[name]
    return ("chat_template", "bare") if MODELS[name][2] else ("plain", "space")


def local_dir(name):
    repo, rev, _ = MODELS[name]
    return LOCAL_HF / f"{repo.replace('/', '__')}@{rev}"


def check_environment():
    """Refuse to run when torch/transformers differ from requirements.txt (a different transformers changes OLMoE internals)."""
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            pkg, ver = line.split("==")
            pins[pkg.strip().lower()] = ver.strip()
    expected = {"torch": "2.8.0", "transformers": pins["transformers"]}
    for pkg, want in expected.items():
        have = version(pkg).split("+")[0]
        if have != want:
            raise RuntimeError(f"{pkg} {have} installed, {want} required; use the `lesion` env (see README.md and env/)")


def load_tokenizer(name):
    from transformers import AutoTokenizer

    repo, rev, _ = MODELS[name]
    if local_dir(name).exists():
        return AutoTokenizer.from_pretrained(local_dir(name))
    return AutoTokenizer.from_pretrained(repo, revision=rev)


def verified_dir(name):
    """Folder with weights verified against the pinned revision, or None if there is none."""
    import json

    d = VERIFIED_DIR / name
    report = d / "VERIFIED.json"
    if not report.exists():
        return None
    info = json.loads(report.read_text())
    if not info.get("all_safetensors_match_hf") or info.get("hf_revision") != MODELS[name][1]:
        raise RuntimeError(f"{d} does not match the pinned revision of {name}; re-download it")
    return d


DTYPES = ("bfloat16", "float16", "float32")


def torch_dtype(name):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_model(name, device="cuda", dtype=None, **from_pretrained_kw):
    """from_pretrained_kw: extra loading options, e.g. attn_implementation="eager" to read attention weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    check_environment()
    repo, rev, _ = MODELS[name]
    local = verified_dir(name)
    # The server's container caps memory at about 110 GB and counts the page cache of the weight files
    # (57 GB for Qwen3-30B-A3B). Loading in bfloat16 needs no conversion in host memory; any other precision
    # is reached by converting on the GPU afterwards (bfloat16 -> float16 is exact within float16's range).
    kw = dict(dtype=torch.bfloat16, device_map=device, **from_pretrained_kw)
    if name == "gpt-oss-20b":
        # its expert weights are published in MXFP4; unpacked to bfloat16 at load (the hooks and probes need ordinary
        # tensors), as fixed in docs/plans/second-model-g1-p12.md
        from transformers import Mxfp4Config

        kw.setdefault("quantization_config", Mxfp4Config(dequantize=True))
    if local is not None:
        model = AutoModelForCausalLM.from_pretrained(local, **kw)
        tok = AutoTokenizer.from_pretrained(local)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, **kw)
        tok = load_tokenizer(name)
    if dtype is not None and dtype != torch.bfloat16:
        model = model.to(dtype)
    return model.eval(), tok


# Shrunken configs keep the architecture class and tokenizer of the real model, so code paths match
# (pilot protocol §8: develop on a tiny random model before any real model sees M1).
TINY = dict(hidden_size=32, intermediate_size=16, moe_intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=8, num_experts_per_tok=2,
            max_position_embeddings=4096, mlp_only_layers=[], decoder_sparse_step=1)


def load_tiny(name, seed=0, dtype=None, **overrides):
    """Randomly initialised model with the architecture and tokenizer of `name`, 2 layers wide 32; CPU, float32.

    overrides change the TINY settings, e.g. num_experts=32."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    src = local_dir(name) if local_dir(name).exists() else None
    repo, rev, _ = MODELS[name]
    config = AutoConfig.from_pretrained(src) if src else AutoConfig.from_pretrained(repo, revision=rev)
    settings = {**TINY, **overrides}
    for key, value in settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    if hasattr(config, "num_local_experts"):              # Phi-3.5-MoE and gpt-oss name the expert count differently
        config.num_local_experts = settings["num_experts"]
    if config.model_type == "gpt_oss":
        config.experts_per_token = settings["num_experts_per_tok"]
        config.layer_types = config.layer_types[:settings["num_hidden_layers"]]
        if hasattr(config, "quantization_config"):         # a random tiny model is not MXFP4
            del config.quantization_config
    if config.model_type == "deepseek_v2":                 # MLA and routed/shared experts have their own fields
        for key, value in dict(n_routed_experts=settings["num_experts"], n_shared_experts=1, first_k_dense_replace=1,
                               kv_lora_rank=16, q_lora_rank=None, qk_nope_head_dim=8, qk_rope_head_dim=8,
                               v_head_dim=8, n_group=1, topk_group=1,
                               num_key_value_heads=settings["num_attention_heads"]).items():  # MLA: one key per head
            setattr(config, key, value)
    for key in ("num_local_experts", "n_routed_experts"):
        if hasattr(config, key) and getattr(config, key) != settings["num_experts"]:
            raise AssertionError(f"the tiny {name} kept {key}={getattr(config, key)}")
    rope = getattr(config, "rope_scaling", None) or {}
    if rope.get("rope_type", rope.get("type")) == "longrope":
        config.rope_scaling = None                        # its per-dimension factors fit the real head size only
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype or torch.float32)
    return model.eval(), load_tokenizer(name)

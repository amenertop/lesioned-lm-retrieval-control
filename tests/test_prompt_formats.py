"""Prompt formats of the P12 second-model candidates (docs/plans/second-model-g1-p12.md): gpt-oss's harmony scoring
prompt and DeepSeek-V2's spaced labels, with the old rules unchanged. Real tokenizers; skipped when not downloaded."""
import pytest
import torch

from lesion import models, scoring


def _tok(name):
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    return models.load_tokenizer(name)


def test_formats_are_registered_and_kept_out_of_uses_chat():
    assert models.prompt_format("gpt-oss-20b") == ("harmony_answer", "space")      # P12b; "harmony" in session 14
    assert models.prompt_format("deepseek-v2-lite-chat") == ("chat_template", "space")
    assert models.prompt_format("qwen3-30b-a3b-instruct") == ("chat_template", "bare")
    assert models.prompt_format("olmoe-0924-base") == ("plain", "space")
    for name in models.PROMPT_FORMATS:
        with pytest.raises(ValueError):
            models.uses_chat(name)
    assert models.uses_chat("qwen3-30b-a3b-instruct") is True


def test_harmony_prompt_has_a_fixed_date_low_reasoning_and_opens_the_final_channel():
    tok = _tok("gpt-oss-20b")
    text = "Word: spinach\nA: club\nB: broccoli\nAnswer:"
    s = scoring.harmony_text(tok, text)
    assert s.startswith("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
                        "Knowledge cutoff: 2024-06\nCurrent date: 2026-10-15\n\nReasoning: low\n\n")
    assert s.endswith("<|start|>user<|message|>" + text + "<|end|><|start|>assistant" + scoring.HARMONY_SUFFIX)
    ids = scoring.encode(tok, text, True, "harmony")
    assert ids == tok(s, add_special_tokens=False).input_ids
    assert tok.convert_ids_to_tokens(ids[-8:]) == ["analysis", "<|message|>", "<|end|>", "<|start|>", "assistant",
                                                   "<|channel|>", "final", "<|message|>"]
    labels = scoring.label_token_ids(tok, True, prefix="bare")
    assert [tok.decode([labels[l]]) for l in "AB"] == ["A", "B"]


def test_harmony_answer_prompt_opens_the_final_message_with_answer_colon_and_reads_spaced_labels():
    tok = _tok("gpt-oss-20b")
    text = "Word: spinach\nA: club\nB: broccoli\nAnswer:"
    s = scoring.harmony_text(tok, text, answer=True)
    assert s == scoring.harmony_text(tok, text) + "Answer:"
    ids = scoring.encode(tok, text, True, "harmony_answer")
    assert ids == tok(s, add_special_tokens=False).input_ids
    assert ids[:-2] == scoring.encode(tok, text, True, "harmony")                   # only "Answer", ":" added
    assert tok.convert_ids_to_tokens(ids[-5:]) == ["<|channel|>", "final", "<|message|>", "Answer", ":"]
    labels = scoring.label_token_ids(tok, True, prefix="space")
    assert [tok.decode([labels[l]]) for l in "AB"] == [" A", " B"]
    for lab in "AB":                    # the label continues the prompt exactly as the model writes "Answer: A"
        end = tok.convert_tokens_to_ids("<|return|>")
        assert tok(s + " " + lab + "<|return|>", add_special_tokens=False).input_ids == ids + [labels[lab], end]


def test_deepseek_prompt_ends_at_assistant_colon_and_reads_spaced_labels():
    tok = _tok("deepseek-v2-lite-chat")
    ids = scoring.encode(tok, "Word: spinach", True, "chat_template")
    assert tok.decode(ids).endswith("User: Word: spinach\n\nAssistant:")
    labels = scoring.label_token_ids(tok, True, prefix="space")
    assert [tok.decode([labels[l]]) for l in "AB"] == [" A", " B"]
    assert scoring.encode(tok, "x", False)[0] == tok.bos_token_id      # no <|endoftext|> in this vocabulary


def test_old_rules_unchanged():
    tok = _tok("qwen3-30b-a3b-instruct")
    assert scoring.encode(tok, "hi", True) == scoring.encode(tok, "hi", True, "chat_template")
    assert scoring.encode(tok, "hi", False) == scoring.encode(tok, "hi", True, "plain")
    assert scoring.label_token_ids(tok, True) == scoring.label_token_ids(tok, True, prefix="bare")
    assert scoring.label_token_ids(tok, False) == scoring.label_token_ids(tok, True, prefix="space")
    with pytest.raises(ValueError):
        scoring.encode(tok, "hi", True, "unknown")


@pytest.mark.parametrize("name,key", [("gpt-oss-20b", "num_local_experts"), ("deepseek-v2-lite-chat", "n_routed_experts")])
def test_tiny_models_run_with_the_requested_expert_count(name, key):
    if not models.local_dir(name).exists():
        pytest.skip("config not downloaded")
    model, tok = models.load_tiny(name, num_experts=6)
    assert getattr(model.config, key) == 6
    ids = torch.tensor([tok("A short test sentence.", add_special_tokens=False).input_ids])
    with torch.no_grad():
        assert torch.isfinite(model(input_ids=ids).logits).all()


class _NoDocStart:
    """A tokenizer whose vocabulary lacks <|endoftext|>; unknown tokens map to an unknown-token id, not None."""
    bos_token_id, unk_token_id = 7, 3

    def get_vocab(self):
        return {"<s>": 7, "<unk>": 3, "a": 10}

    def convert_tokens_to_ids(self, tokens):
        return [self.get_vocab().get(t, self.unk_token_id) for t in tokens]

    def encode(self, text, add_special_tokens=False):
        return [10 for _ in text]


def test_plain_prompts_fall_back_to_bos_when_the_vocabulary_has_no_document_start():
    tok = _NoDocStart()
    assert scoring.encode(tok, "aa", False) == [7, 10, 10]           # not the unknown-token id 3
    tok.bos_token_id = None
    with pytest.raises(ValueError):
        scoring.encode(tok, "aa", False)


def test_tiny_deepseek_keeps_mla_and_shared_experts_and_does_not_depend_on_batching():
    if not models.local_dir("deepseek-v2-lite-chat").exists():
        pytest.skip("config not downloaded")
    model, _ = models.load_tiny("deepseek-v2-lite-chat")
    c = model.config
    assert (c.n_shared_experts, c.first_k_dense_replace, c.num_key_value_heads) == (1, 1, c.num_attention_heads)
    assert (c.kv_lora_rank, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim) == (16, 8, 8, 8)
    assert type(model.model.layers[0].mlp).__name__ == "DeepseekV2MLP"           # the dense first layer
    assert hasattr(model.model.layers[1].mlp, "shared_experts")                   # routed plus shared experts
    seqs = [[5, 9, 2, 7, 11, 3], [4, 4, 8]]
    batched = scoring.next_token_logits(model, seqs, [1, 2, 3], batch_size=2)
    single = scoring.next_token_logits(model, seqs, [1, 2, 3], batch_size=1)
    assert torch.allclose(batched, single, atol=1e-4)


def test_lesion_pipeline_refuses_the_p12_models_before_loading():
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e1_screen"))
    import common

    with pytest.raises(SystemExit, match="measurement check only"):
        common.load(SimpleNamespace(model="gpt-oss-20b", tiny=True))


def test_continuation_logprob_projects_only_the_needed_positions():
    model, _ = models.load_tiny("qwen3-30b-a3b-instruct") if models.local_dir("qwen3-30b-a3b-instruct").exists() \
        else pytest.skip("config not downloaded")
    prefixes = [[5, 9, 2, 7], [4, 4], [3, 8, 1, 6, 2], [7]]
    conts = [[11, 12], [13], [14, 15, 16], [17, 18, 19, 20, 21, 22]]   # lengths 1 to 6 in one batch
    got = scoring.continuation_logprob(model, prefixes, conts, batch_size=4)
    with torch.no_grad():
        for g, p, c in zip(got, prefixes, conts):
            logp = torch.log_softmax(model(input_ids=torch.tensor([p + c])).logits.float(), -1)[0]
            want = sum(logp[len(p) - 1 + k, t].item() for k, t in enumerate(c))
            assert abs(g - want) < 1e-4

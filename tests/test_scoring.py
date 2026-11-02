"""Checks the Experiment 0 scoring (ticket pilot 04) on a tiny random OLMoE with the real OLMoE tokenizer.

No trained model and no M1 scores are involved: the tiny model has random weights, so it only exercises
the code path (pilot protocol §8).
"""
import pytest
import torch

from lesion import models, scoring

needs_tokenizer = pytest.mark.skipif(not models.local_dir("olmoe-0924-base").exists(),
                                     reason="pinned tokenizer files not downloaded (see models.LOCAL_HF)")


@pytest.fixture(scope="module")
def materials():
    return scoring.load_materials()


@pytest.fixture(scope="module")
def tok():
    return models.load_tokenizer("olmoe-0924-base")


@pytest.fixture(scope="module")
def tiny_model(tok):
    from transformers import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(0)
    cfg = OlmoeConfig(vocab_size=50304, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, num_experts=8, num_experts_per_tok=2,
                      max_position_embeddings=4096)
    return OlmoeForCausalLM(cfg).float().eval()


def test_two_orders_and_correct_label(materials):
    items, demos, templates = materials
    prompts = scoring.build_label_prompts(items, demos, templates)
    assert len(prompts) == 2 * len(items)
    by_item = {}
    for p in prompts:
        by_item.setdefault(p.item_id, []).append(p)
    for item_id, (p0, p1) in by_item.items():
        assert (p0.order, p1.order) == (0, 1)
        assert p0.correct_label != p1.correct_label
        assert p0.label_to_option[p0.correct_label] == p1.label_to_option[p1.correct_label]
        assert p0.text.endswith("Answer:") and p1.text.endswith("Answer:")


def test_synonym_demos_swap_with_order(materials):
    items, demos, templates = materials
    syn = items[items.task == "synonym"].head(1)
    p0, p1 = scoring.build_label_prompts(syn, demos, templates)
    first_demo0 = p0.text.split("\n\n")[1]
    first_demo1 = p1.text.split("\n\n")[1]
    assert "A: similar in meaning" in first_demo0 and first_demo0.endswith("Answer: A")
    assert "A: different in meaning" in first_demo1 and first_demo1.endswith("Answer: B")


def test_matching_demos_fixed_across_orders(materials):
    items, demos, templates = materials
    col = items[items.task == "colour"].head(1)
    p0, p1 = scoring.build_label_prompts(col, demos, templates)
    assert p0.text.split("\n\n")[:-1] == p1.text.split("\n\n")[:-1]


@needs_tokenizer
def test_batched_equals_single(materials, tok, tiny_model):
    items, demos, templates = materials
    prompts = scoring.build_label_prompts(items.head(6), demos, templates)
    batched = scoring.score_label_prompts(tiny_model, tok, prompts, chat=False, batch_size=8)
    single = scoring.score_label_prompts(tiny_model, tok, prompts, chat=False, batch_size=1)
    assert torch.allclose(torch.tensor(batched[["logit_A", "logit_B"]].to_numpy()),
                          torch.tensor(single[["logit_A", "logit_B"]].to_numpy()), atol=1e-4)
    assert set(batched.columns) >= {"item_id", "task", "order", "correct_label", "logit_correct", "logit_wrong", "choice"}


@needs_tokenizer
def test_base_prompt_starts_with_doc_marker_and_labels_single(tok):
    ids = scoring.encode(tok, "Word: x\nA: y\nB: z\nAnswer:", chat=False)
    assert ids[0] == tok.convert_tokens_to_ids(scoring.DOC_START)
    lab = scoring.label_token_ids(tok, chat=False)
    assert tok.encode("Answer: A", add_special_tokens=False)[-1] == lab["A"]


@needs_tokenizer
def test_instruct_answer_position():
    tok = models.load_tokenizer("olmoe-0924-instruct")
    ids = scoring.encode(tok, "Question\nAnswer:", chat=True)
    text = tok.decode(ids)
    assert text.endswith("<|assistant|>\n"), repr(text)
    lab = scoring.label_token_ids(tok, chat=True)
    full = tok.apply_chat_template([{"role": "user", "content": "Question\nAnswer:"},
                                    {"role": "assistant", "content": "A"}], tokenize=True)
    assert full[:len(ids)] == ids and full[len(ids)] == lab["A"]


@needs_tokenizer
def test_cloze_matches_manual(materials, tok, tiny_model):
    items, demos, templates = materials
    cloze = scoring.build_cloze_prompts(items.head(2), demos, templates)
    scores = scoring.score_cloze(tiny_model, tok, cloze, batch_size=4)
    item_id, task, text, words = cloze[0]
    prefix = scoring.encode(tok, text, chat=False)
    cont = tok.encode(" " + words["option1"], add_special_tokens=False)
    with torch.no_grad():
        logp = torch.log_softmax(tiny_model(torch.tensor([prefix + cont])).logits.float(), -1)[0]
    manual = sum(logp[len(prefix) - 1 + k, t].item() for k, t in enumerate(cont))
    got = scores[(scores.item_id == item_id) & (scores.option == "option1")].logprob.item()
    assert abs(manual - got) < 1e-4


LOCAL_MODELS = [n for n in models.MODELS if models.local_dir(n).exists()]


@pytest.mark.parametrize("name", LOCAL_MODELS)
def test_every_registered_model_scores_at_the_answer_position(name):
    """For every model with local tokenizer files: labels are single tokens and the scored position is the answer."""
    tok = models.load_tokenizer(name)
    fmt, prefix = models.prompt_format(name)
    chat = fmt != "plain"
    lab = scoring.label_token_ids(tok, chat=chat, labels=("A", "B", "C", "D", "E", "F"), prefix=prefix)
    ids = scoring.encode(tok, "Question\nAnswer:", chat=chat, fmt=fmt)
    if fmt in ("harmony", "harmony_answer"):
        # the official rendering of a final-channel answer "A" ("Answer: A") has the label right after the
        # final-channel header (and "Answer:") that the scoring prompt ends with (the scoring prompt adds an empty
        # analysis block before it)
        opening = "Answer:" if fmt == "harmony_answer" else ""
        full = tok(tok.apply_chat_template([{"role": "user", "content": "Question\nAnswer:"},
                                            {"role": "assistant", "content": opening + (" A" if opening else "A")}],
                                           tokenize=False),
                   add_special_tokens=False).input_ids
        header = tok("<|channel|>final<|message|>" + opening, add_special_tokens=False).input_ids
        at = max(i for i in range(len(full) - len(header)) if full[i:i + len(header)] == header) + len(header)
        assert ids[-len(header):] == header and full[at] == lab["A"], (name, tok.decode(full[at - 3:at + 2]))
    elif chat:
        full = tok.apply_chat_template([{"role": "user", "content": "Question\nAnswer:"},
                                        {"role": "assistant", "content": "A"}], tokenize=True)
        assert full[:len(ids)] == ids, (name, tok.decode(full[len(ids) - 5:len(ids) + 3]))
        assert full[len(ids)] == lab["A"], (name, repr(tok.decode(full[len(ids):len(ids) + 3])))
    else:
        assert tok.encode("Answer: A", add_special_tokens=False)[-1] == lab["A"]

"""Actual logits agree after loading verified shards into an empty CPU model."""
import hashlib

import pytest
import torch
from safetensors.torch import save
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from lesion import streaming_weights


def test_verified_cpu_shards_preserve_logits_and_reject_corruption(tmp_path):
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.)
    torch.manual_seed(19)
    Qwen3ForCausalLM(config).save_pretrained(tmp_path)
    original = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16).eval()
    state = original.state_dict()
    keys = list(state)
    blobs = {str(i): save({k: state[k] for k in keys[i::3]}) for i in range(3)}
    source = dict(weights=[dict(file=k, size=len(v), sha256=hashlib.sha256(v).hexdigest()) for k,v in blobs.items()])
    restored = streaming_weights.load_cpu(config, source, lambda row: blobs[row['file']])
    for name, value in original.named_buffers():
        assert torch.equal(value, dict(restored.named_buffers())[name])
    ids = torch.tensor([[2,4,7]])
    with torch.no_grad():
        assert torch.equal(original(ids).logits, restored(ids).logits)
    with pytest.raises(ValueError, match='shard mismatch'):
        streaming_weights.load_cpu(config, source, lambda row: blobs[row['file']]+b'x')


@pytest.mark.parametrize('resumed_status', [206, 200])
def test_interrupted_shard_resumes_without_duplicate_bytes(monkeypatch, resumed_status):
    data = b'abcdefghijk'
    requests = []

    class Response:
        def __init__(self, chunks, status, headers):
            self.chunks, self.status, self.headers = iter(chunks), status, headers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            chunk = next(self.chunks, b'')
            if isinstance(chunk, Exception):
                raise chunk
            return chunk

    def fetch(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            return Response([data[:4], TimeoutError()], 200, {})
        return Response([data[4:] if resumed_status == 206 else data], resumed_status,
                        {'Content-Range': f'bytes 4-10/{len(data)}'})

    monkeypatch.setattr(streaming_weights.urllib.request, 'urlopen', fetch)
    monkeypatch.setattr(streaming_weights.time, 'sleep', lambda seconds: None)
    restored = streaming_weights.fetch_bytes('https://example.test/shard', len(data), hashlib.sha256(data).hexdigest())
    assert restored == data and len(requests) == 2
    assert requests[1].headers['Range'] == 'bytes=4-'

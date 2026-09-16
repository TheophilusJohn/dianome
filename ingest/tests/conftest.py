import numpy as np
import pytest

from dianome_ingest.layout import DictSource


def synthetic_model(hidden=256, inter=512, layers=2, vocab=1000, heads=4, kv=2, tied=True, seed=0):
    """A tiny Qwen2-shaped model as bf16-free fp32 arrays plus a config summary."""
    rng = np.random.default_rng(seed)
    t = {}
    t["model.embed_tokens.weight"] = rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.05
    hd = hidden // heads
    for i in range(layers):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = rng.standard_normal(hidden).astype(np.float32)
        t[p + "post_attention_layernorm.weight"] = rng.standard_normal(hidden).astype(np.float32)
        t[p + "self_attn.q_proj.weight"] = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.05
        t[p + "self_attn.q_proj.bias"] = rng.standard_normal(hidden).astype(np.float32)
        t[p + "self_attn.k_proj.weight"] = rng.standard_normal((kv * hd, hidden)).astype(np.float32) * 0.05
        t[p + "self_attn.k_proj.bias"] = rng.standard_normal(kv * hd).astype(np.float32)
        t[p + "self_attn.v_proj.weight"] = rng.standard_normal((kv * hd, hidden)).astype(np.float32) * 0.05
        t[p + "self_attn.v_proj.bias"] = rng.standard_normal(kv * hd).astype(np.float32)
        t[p + "self_attn.o_proj.weight"] = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.05
        t[p + "mlp.gate_proj.weight"] = rng.standard_normal((inter, hidden)).astype(np.float32) * 0.05
        t[p + "mlp.up_proj.weight"] = rng.standard_normal((inter, hidden)).astype(np.float32) * 0.05
        t[p + "mlp.down_proj.weight"] = rng.standard_normal((hidden, inter)).astype(np.float32) * 0.05
    t["model.norm.weight"] = rng.standard_normal(hidden).astype(np.float32)
    if not tied:
        t["lm_head.weight"] = rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.05
    config = {
        "hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": kv, "intermediate_size": inter, "vocab_size": vocab,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": tied,
        "max_position_embeddings": 4096,
    }
    return DictSource(t), config


@pytest.fixture
def tiny_model():
    return synthetic_model()


@pytest.fixture
def untied_model():
    return synthetic_model(tied=False, seed=1)

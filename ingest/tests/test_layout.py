import numpy as np
import pytest

from dianome_ingest import layout, quant
from dianome_ingest.layout import ALIGN, align, group_of, group_order, pack_model, plan_entry, role_of, segments_for, storage_kind


def test_group_order():
    assert group_order(3) == ["embed", "layer.0", "layer.1", "layer.2", "final_norm", "lm_head"]


def test_group_and_role_mapping():
    assert group_of("model.embed_tokens.weight") == "embed" and role_of("model.embed_tokens.weight") == "embed_tokens.weight"
    assert group_of("model.layers.7.self_attn.q_proj.weight") == "layer.7"
    assert role_of("model.layers.7.self_attn.q_proj.weight") == "attn.q_proj.weight"
    assert role_of("model.layers.7.self_attn.q_proj.bias") == "attn.q_proj.bias"
    assert role_of("model.layers.23.mlp.gate_proj.weight") == "mlp.gate_proj.weight"
    assert role_of("model.layers.0.input_layernorm.weight") == "input_layernorm.weight"
    assert group_of("model.norm.weight") == "final_norm" and role_of("model.norm.weight") == "norm.weight"
    assert group_of("lm_head.weight") == "lm_head" and role_of("lm_head.weight") == "lm_head.weight"
    with pytest.raises(ValueError):
        group_of("vision_tower.weight")


def test_storage_kind_rules():
    assert storage_kind("attn.q_proj.weight", (896, 896), "q4") == "q4"
    assert storage_kind("attn.q_proj.weight", (896, 896), "q8") == "q8"
    assert storage_kind("attn.q_proj.weight", (896, 896), "fp16") == "fp16"
    assert storage_kind("attn.q_proj.bias", (896,), "q4") == "fp16"
    assert storage_kind("input_layernorm.weight", (896,), "q8") == "fp16"
    assert storage_kind("embed_tokens.weight", (151936, 896), "fp16") == "fp16"
    assert storage_kind("embed_tokens.weight", (151936, 896), "q4") == "q8"
    assert storage_kind("embed_tokens.weight", (151936, 896), "q8") == "q8"
    assert storage_kind("lm_head.weight", (151936, 896), "q4") == "q8"
    assert storage_kind("lm_head.weight", (151936, 896), "q8") == "q8"


def test_plan_entry_alignment():
    p = plan_entry("q4", (4864, 896))
    assert p.parts["weights"] == (0, 4864 * 448)
    assert p.parts["scales"][0] % ALIGN == 0 and p.parts["scales"][0] >= 4864 * 448
    assert p.parts["zeros"][0] % ALIGN == 0 and p.parts["zeros"][1] == 4864 * 7
    assert p.parts["scales"][1] == 4864 * 7 * 2
    assert p.length == p.parts["zeros"][0] + p.parts["zeros"][1]
    p8 = plan_entry("q8", (10, 10))  # 100 bytes of weights -> scales at 256
    assert p8.parts == {"weights": (0, 100), "scales": (256, 20)} and p8.length == 276
    assert plan_entry("fp16", (3, 5)).length == 30
    assert align(0) == 0 and align(1) == 256 and align(256) == 256 and align(257) == 512


def test_segments_for_spans_chunks():
    chunks = [("a" * 64, 100), ("b" * 64, 100), ("c" * 64, 50)]
    assert segments_for(chunks, 0, 10) == [{"chunk": "a" * 64, "offset": 0, "length": 10}]
    assert segments_for(chunks, 90, 120) == [
        {"chunk": "a" * 64, "offset": 90, "length": 10},
        {"chunk": "b" * 64, "offset": 0, "length": 100},
        {"chunk": "c" * 64, "offset": 0, "length": 10},
    ]


def _pack(model, variants=("fp16", "q8", "q4")):
    source, cfg = model
    chunks = {}
    packed = pack_model(source, cfg["num_hidden_layers"], cfg["tie_word_embeddings"], variants, lambda s, d: chunks.__setitem__(s, d))
    return packed, chunks


def test_pack_group_order_and_entry_order(tiny_model):
    packed, _ = _pack(tiny_model)
    for v, groups in packed.items():
        assert [g.name for g in groups] == group_order(2)
        roles = [e.role for e in groups[1].entries]
        assert roles == [
            "input_layernorm.weight", "attn.q_proj.weight", "attn.q_proj.bias", "attn.k_proj.weight", "attn.k_proj.bias",
            "attn.v_proj.weight", "attn.v_proj.bias", "attn.o_proj.weight", "post_attention_layernorm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        ]


def test_pack_alignment_and_stream_bytes(tiny_model):
    packed, chunks = _pack(tiny_model)
    for v, groups in packed.items():
        for g in groups:
            stream = b"".join(chunks[sha] for sha, _ in g.chunks)
            assert len(stream) == g.bytes
            end = 0
            for e in g.entries:
                assert e.start % ALIGN == 0 and e.start >= end
                assert stream[end:e.start] == bytes(e.start - end)  # zero padding between entries
                for name, (off, n) in e.plan.parts.items():
                    assert (e.start + off) % ALIGN == 0
                end = e.start + e.plan.length
            if not g.tied:
                assert end == g.bytes  # no trailing padding


def test_pack_bytes_decode_to_reference(tiny_model):
    source, cfg = tiny_model
    packed, chunks = _pack(tiny_model)
    for v, groups in packed.items():
        for g in groups:
            ref = groups[0] if g.tied else g
            stream = b"".join(chunks[sha] for sha, _ in ref.chunks)
            for e in g.entries:
                src = "model.embed_tokens.weight" if g.tied else e.name
                w = quant.to_fp16(source.get(src))
                data = stream[e.start : e.start + e.plan.length]
                if e.plan.kind == "fp16":
                    assert np.array_equal(np.frombuffer(data, np.float16).reshape(w.shape).view(np.uint16), w.view(np.uint16))
                elif e.plan.kind == "q8":
                    q, s = quant.q8_encode(w)
                    wo, wn = e.plan.parts["weights"]
                    so, sn = e.plan.parts["scales"]
                    assert data[wo : wo + wn] == q.tobytes() and data[so : so + sn] == s.tobytes()
                else:
                    q, s, z = quant.q4_encode(w)
                    wo, wn = e.plan.parts["weights"]
                    so, sn = e.plan.parts["scales"]
                    zo, zn = e.plan.parts["zeros"]
                    assert data[wo : wo + wn] == q.tobytes() and data[so : so + sn] == s.tobytes() and data[zo : zo + zn] == z.tobytes()


def test_tied_lm_head_references_embed(tiny_model):
    packed, _ = _pack(tiny_model)
    for v, groups in packed.items():
        head, embed = groups[-1], groups[0]
        assert head.name == "lm_head" and head.tied and head.chunks == [] and head.bytes == 0
        assert len(head.entries) == 1
        e = head.entries[0]
        assert e.name == "lm_head.weight" and e.role == "lm_head.weight"
        assert e.shape == embed.entries[0].shape and e.start == embed.entries[0].start and e.plan.kind == embed.entries[0].plan.kind
        table = {}
        m = head.to_manifest(table, embed)
        assert m["tied"] is True and m["entries"][0]["tied"] is True and m["chunks"] == [] and m["bytes"] == 0
        assert m["entries"][0]["segments"] == embed.to_manifest({})["entries"][0]["segments"]
    assert packed["q8"][0].entries[0].plan.kind == "q8" and packed["q4"][0].entries[0].plan.kind == "q8"
    assert packed["fp16"][0].entries[0].plan.kind == "fp16"


def test_untied_lm_head_has_own_bytes(untied_model):
    packed, _ = _pack(untied_model)
    head = packed["fp16"][-1]
    assert not head.tied and head.bytes > 0 and head.entries[0].name == "lm_head.weight"


def test_embed_stream_identical_in_q8_and_q4(tiny_model):
    packed, _ = _pack(tiny_model)
    assert packed["q8"][0].chunks == packed["q4"][0].chunks
    assert packed["fp16"][0].chunks != packed["q8"][0].chunks


def test_pack_is_deterministic(tiny_model):
    a, ca = _pack(tiny_model)
    b, cb = _pack(tiny_model)
    for v in a:
        assert [g.chunks for g in a[v]] == [g.chunks for g in b[v]]
    assert ca.keys() == cb.keys()


def test_q4_divisibility_checked_before_writing():
    from conftest import synthetic_model
    model = synthetic_model(hidden=256, inter=900)  # down_proj is [256, 900]
    source, cfg = model
    calls = []
    with pytest.raises(ValueError) as ei:
        pack_model(source, cfg["num_hidden_layers"], True, ("fp16", "q4"), lambda s, d: calls.append(s))
    assert "mlp.down_proj.weight" in str(ei.value) and "900" in str(ei.value)
    assert calls == []  # nothing was written

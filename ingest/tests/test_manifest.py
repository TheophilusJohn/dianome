import json

import jsonschema
import pytest

from dianome_ingest import manifest as mf
from dianome_ingest.chunker import CHUNK_SIZE, LocalStore
from dianome_ingest.layout import pack_model


def build(tmp_path, model, variants=("fp16", "q8", "q4")):
    source, cfg = model
    store = LocalStore(tmp_path / "store")
    packed = pack_model(source, cfg["num_hidden_layers"], cfg["tie_word_embeddings"], variants, store.sink())
    tok = tmp_path / "tokenizer.json"
    tok.write_bytes(b'{"tok": 1}' * 1000)
    big = tmp_path / "tokenizer_config.json"
    big.write_bytes(b"x" * (CHUNK_SIZE + 10))
    files = [mf.pack_file_stream(tok, store.sink()), mf.pack_file_stream(big, store.sink())]
    m = mf.build_model_manifest(id="tiny", repo="test/tiny", revision="abc", family="qwen2", config=cfg, packed=packed, tokenizer_files=files)
    return store, m


def test_generated_manifest_validates_and_has_expected_shape(tmp_path, tiny_model):
    store, m = build(tmp_path, tiny_model)
    mf.validate(m)
    assert m["schema"] == 1 and m["chunk_size"] == CHUNK_SIZE
    assert list(m["variants"]) == ["fp16", "q8", "q4"]
    q4 = m["variants"]["q4"]["groups"][1]["entries"]
    gate = next(e for e in q4 if e["role"] == "mlp.gate_proj.weight")
    assert gate["storage"]["kind"] == "q4" and gate["storage"]["group_size"] == 128
    assert set(gate["storage"]["parts"]) == {"weights", "scales", "zeros"}
    assert m["variants"]["q4"]["groups"][-1]["tied"] is True
    assert [f["name"] for f in m["tokenizer"]["files"]] == ["tokenizer.json", "tokenizer_config.json"]
    assert len(m["tokenizer"]["files"][1]["chunks"]) == 2 and m["tokenizer"]["files"][1]["bytes"] == CHUNK_SIZE + 10


def test_byte_sums(tmp_path, tiny_model):
    store, m = build(tmp_path, tiny_model)
    table = m["chunks"]
    for v in m["variants"].values():
        assert v["bytes"] == sum(g["bytes"] for g in v["groups"])
        for g in v["groups"]:
            assert g["bytes"] == sum(table[c]["bytes"] for c in g["chunks"])
            for e in g["entries"]:
                assert all(s["chunk"] in table for s in e["segments"])
    for f in m["tokenizer"]["files"]:
        assert f["bytes"] == sum(s["length"] for s in f["segments"]) == sum(table[c]["bytes"] for c in f["chunks"])
    # every chunk in the table is on disk with the right size
    for sha, meta in table.items():
        assert store.chunk_path(sha).stat().st_size == meta["bytes"]


def test_schema_rejects_bad_manifests(tmp_path, tiny_model):
    _, m = build(tmp_path, tiny_model)
    bad = json.loads(json.dumps(m))
    bad["variants"]["fp16"]["groups"][0]["entries"][0]["storage"] = {"kind": "q2"}
    with pytest.raises(jsonschema.ValidationError):
        mf.validate(bad)
    bad = json.loads(json.dumps(m))
    bad["chunks"]["nothex"] = {"bytes": 1}
    with pytest.raises(jsonschema.ValidationError):
        mf.validate(bad)
    bad = json.loads(json.dumps(m))
    bad["files"] = []
    with pytest.raises(jsonschema.ValidationError):
        mf.validate(bad)
    bad = json.loads(json.dumps(m))
    bad["variants"]["q8"]["groups"][1]["entries"][1]["storage"]["parts"]["scales"]["offset"] = 100
    with pytest.raises(jsonschema.ValidationError):
        mf.validate(bad)


def test_save_load_hash_and_verify(tmp_path, tiny_model):
    store, m = build(tmp_path, tiny_model)
    latest, hashed, h = mf.save(store, m)
    assert hashed.name == f"{h}.json" and latest.read_bytes() == hashed.read_bytes()
    assert h == mf.manifest_hash(mf.load(store, "tiny"))
    assert mf.canonical_json(m) == json.dumps(m, sort_keys=True, separators=(",", ":")).encode()
    r = mf.Report()
    mf.verify_store(store, mf.load(store, "tiny"), r)
    assert r.ok, r.checks


def test_verify_detects_corruption(tmp_path, tiny_model):
    store, m = build(tmp_path, tiny_model)
    mf.save(store, m)
    sha = m["variants"]["fp16"]["groups"][1]["chunks"][0]
    p = store.chunk_path(sha)
    data = bytearray(p.read_bytes())
    data[0] ^= 0xFF
    p.write_bytes(bytes(data))
    r = mf.Report()
    mf.verify_store(store, m, r)
    assert not r.ok
    assert any(name == "chunk hashes & sizes" and not ok for name, ok, _ in r.checks)


def test_verify_detects_bad_byte_sum(tmp_path, tiny_model):
    store, m = build(tmp_path, tiny_model)
    mf.save(store, m)
    m["variants"]["q4"]["groups"][1]["bytes"] += 1
    r = mf.Report()
    mf.verify_store(store, m, r)
    assert any(name.startswith("variant q4") and not ok for name, ok, _ in r.checks)


def test_files_manifest(tmp_path):
    store = LocalStore(tmp_path / "store")
    d = tmp_path / "dir"
    (d / "onnx").mkdir(parents=True)
    (d / "onnx" / "model.onnx").write_bytes(b"\x01" * (CHUNK_SIZE * 2 + 3))
    (d / "config.json").write_bytes(b"{}")
    seen = set()
    files = [mf.pack_file_stream(p, store.sink(), seen, name=p.relative_to(d).as_posix()) for p in sorted(d.rglob("*")) if p.is_file()]
    m = mf.build_files_manifest(id="x-onnx", repo="local:dir", revision="", runtime="transformersjs", files=files)
    mf.validate(m)
    assert [f["name"] for f in m["files"]] == ["config.json", "onnx/model.onnx"]
    assert m["files"][1]["bytes"] == CHUNK_SIZE * 2 + 3 and len(m["files"][1]["chunks"]) == 3
    assert len(m["chunks"]) == 3  # two identical full chunks of 0x01 dedup to one
    r = mf.Report()
    mf.save(store, m)
    mf.verify_store(store, m, r)
    assert r.ok, r.checks


def test_dedup_stats(tmp_path, tiny_model):
    store, m = build(tmp_path, tiny_model)
    d = mf.dedup_stats(m)
    assert d["bytes_saved"] > 0
    names = [g[0] for g in d["shared_groups"]]
    assert "embed" in names
    embed = next(g for g in d["shared_groups"] if g[0] == "embed")
    assert set(embed[1]) == {"q8", "q4"}


def test_quantised_only_manifest_validates_and_fp16_is_optional(tmp_path, tiny_model):
    """Phase 7: 3B/7B are packed as q8+q4 only; the schema needs at least one variant, not fp16."""
    store, m = build(tmp_path, tiny_model, ("q8", "q4"))
    mf.validate(m)
    assert list(m["variants"]) == ["q8", "q4"]
    assert m["variants"]["q4"]["groups"][-1]["tied"] is True
    empty = json.loads(json.dumps(m))
    empty["variants"] = {}
    with pytest.raises(jsonschema.ValidationError):
        mf.validate(empty)


def test_inspect_summary_sizes_match_manifest(tmp_path, tiny_model):
    from dianome_ingest.cli import inspect_summary

    store, m = build(tmp_path, tiny_model, ("q8", "q4"))
    s = inspect_summary(m)
    assert s["id"] == "tiny" and s["manifest_sha256"] == mf.manifest_hash(m)
    assert s["unique_chunks"] == len(m["chunks"]) and s["unique_bytes"] == sum(c["bytes"] for c in m["chunks"].values())
    assert set(s["variants"]) == {"q8", "q4"}
    for v in ("q8", "q4"):
        assert s["variants"][v]["bytes"] == m["variants"][v]["bytes"]
        assert s["variants"][v]["chunks_listed"] == sum(len(g["chunks"]) for g in m["variants"][v]["groups"])
    assert s["tokenizer"] == {"files": 2, "bytes": sum(f["bytes"] for f in m["tokenizer"]["files"])}
    variant_chunks = {c for v in m["variants"].values() for g in v["groups"] for c in g["chunks"]}
    assert s["dedup"]["unique_chunks"] == len(variant_chunks)  # tokenizer chunks are not part of the dedup stats

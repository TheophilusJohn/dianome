"""Manifest building, canonical hashing, schema validation, store verification."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema
import numpy as np

from . import quant
from .chunker import CHUNK_SIZE, LocalStore, StreamChunker, sha256_hex
from .layout import PackedGroup, align, plan_entry, storage_kind

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "manifest.v1.json"


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def validate(manifest: dict) -> None:
    """Raise jsonschema.ValidationError if the manifest does not match schemas/manifest.v1.json."""
    jsonschema.Draft202012Validator(load_schema()).validate(manifest)


def canonical_json(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_hash(manifest: dict) -> str:
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


# ---------------------------------------------------------------- building

def pack_file_stream(path: Path, sink, seen: set[str] | None = None, name: str | None = None) -> dict:
    """Chunk one file as one stream and return its `file` manifest object."""
    c = StreamChunker(sink, seen)
    with open(path, "rb") as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if not piece:
                break
            c.write(piece)
    chunks = c.finish()
    segs, pos = [], 0
    for sha, n in chunks:
        segs.append({"chunk": sha, "offset": 0, "length": n})
        pos += n
    return {"name": name or path.name, "bytes": pos, "chunks": [sha for sha, _ in chunks], "segments": segs}


def build_model_manifest(
    *, id: str, repo: str, revision: str, family: str, config: dict,
    packed: dict[str, list[PackedGroup]], tokenizer_files: list[dict],
) -> dict:
    chunk_table: dict[str, dict] = {}
    variants = {}
    for v, groups in packed.items():
        embed = groups[0]
        gs = [g.to_manifest(chunk_table, embed if g.tied else None) for g in groups]
        variants[v] = {"bytes": sum(g["bytes"] for g in gs), "groups": gs}
    for f in tokenizer_files:
        for sha, seg in zip(f["chunks"], f["segments"]):
            chunk_table.setdefault(sha, {"bytes": seg["length"]})
    m = {
        "schema": 1,
        "id": id,
        "source": {"repo": repo, "revision": revision},
        "family": family,
        "config": config,
        "tokenizer": {"files": tokenizer_files},
        "chunk_size": CHUNK_SIZE,
        "chunks": chunk_table,
        "variants": variants,
    }
    validate(m)
    return m


def build_files_manifest(*, id: str, repo: str, revision: str, runtime: str, files: list[dict]) -> dict:
    chunk_table: dict[str, dict] = {}
    for f in files:
        for sha, seg in zip(f["chunks"], f["segments"]):
            chunk_table.setdefault(sha, {"bytes": seg["length"]})
    m = {
        "schema": 1,
        "id": id,
        "source": {"repo": repo, "revision": revision},
        "runtime": runtime,
        "chunk_size": CHUNK_SIZE,
        "chunks": chunk_table,
        "files": files,
    }
    validate(m)
    return m


# ---------------------------------------------------------------- store I/O

def save(store: LocalStore, manifest: dict) -> tuple[Path, Path, str]:
    """Write manifests/<id>/latest.json and manifests/<id>/<sha256>.json (canonical bytes). Returns paths and hash."""
    data = canonical_json(manifest)
    h = hashlib.sha256(data).hexdigest()
    d = store.manifests_dir / manifest["id"]
    d.mkdir(parents=True, exist_ok=True)
    hashed, latest = d / f"{h}.json", d / "latest.json"
    hashed.write_bytes(data)
    latest.write_bytes(data)
    return latest, hashed, h


def load(store: LocalStore, id: str) -> dict:
    p = store.manifests_dir / id / "latest.json"
    if not p.is_file():
        raise FileNotFoundError(f"no manifest at {p}")
    return json.loads(p.read_bytes())


def referenced_chunks(manifest: dict) -> set[str]:
    return set(manifest["chunks"])


# ---------------------------------------------------------------- verification

@dataclass
class Report:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def verify_store(store: LocalStore, manifest: dict, report: Report) -> None:
    """Schema, manifest hash, chunk hashes and sizes, byte sums, segment consistency, alignment."""
    try:
        validate(manifest)
        report.add("schema", True, "manifest.v1.json")
    except jsonschema.ValidationError as e:
        report.add("schema", False, e.message)
        return
    h = manifest_hash(manifest)
    hashed = store.manifests_dir / manifest["id"] / f"{h}.json"
    report.add("manifest hash", hashed.is_file() and sha256_hex(hashed.read_bytes()) == h, f"manifests/{manifest['id']}/{h}.json")

    table = manifest["chunks"]
    bad = []
    total = 0
    for sha, meta in table.items():
        p = store.chunk_path(sha)
        if not p.is_file():
            bad.append(f"{sha[:12]} missing")
            continue
        data = p.read_bytes()
        total += len(data)
        if len(data) != meta["bytes"]:
            bad.append(f"{sha[:12]} size {len(data)} != {meta['bytes']}")
        elif len(data) > CHUNK_SIZE:
            bad.append(f"{sha[:12]} larger than chunk size")
        elif sha256_hex(data) != sha:
            bad.append(f"{sha[:12]} hash mismatch")
    report.add("chunk hashes & sizes", not bad, f"{len(table)} chunks, {total} bytes" + ("; " + "; ".join(bad[:5]) if bad else ""))

    if "variants" in manifest:
        _verify_variants(manifest, report)
    if "files" in manifest:
        _verify_files("files", manifest["files"], table, report)
    if "tokenizer" in manifest:
        _verify_files("tokenizer files", manifest["tokenizer"]["files"], table, report)


def _verify_files(label: str, files: list[dict], table: dict, report: Report) -> None:
    bad = []
    for f in files:
        seg_sum = sum(s["length"] for s in f["segments"])
        chunk_sum = sum(table[c]["bytes"] for c in f["chunks"] if c in table)
        if any(c not in table for c in f["chunks"]) or any(s["chunk"] not in table for s in f["segments"]):
            bad.append(f"{f['name']}: unknown chunk")
        elif not (f["bytes"] == seg_sum == chunk_sum):
            bad.append(f"{f['name']}: bytes {f['bytes']} segments {seg_sum} chunks {chunk_sum}")
        elif any(s["length"] < table[s["chunk"]]["bytes"] for s in f["segments"][:-1]):
            bad.append(f"{f['name']}: non-final segment does not cover its chunk")
    report.add(f"{label} byte sums", not bad, f"{len(files)} files" + ("; " + "; ".join(bad[:5]) if bad else ""))


def _verify_variants(manifest: dict, report: Report) -> None:
    table = manifest["chunks"]
    from .layout import group_order
    expected_order = group_order(manifest["config"]["num_hidden_layers"])
    tied_cfg = manifest["config"]["tie_word_embeddings"]
    for vname in [v for v in ("fp16", "q8", "q4") if v in manifest["variants"]]:
        variant = manifest["variants"][vname]
        bad = []
        names = [g["name"] for g in variant["groups"]]
        if names != expected_order:
            bad.append(f"group order {names[:3]}… != expected")
        gsum = 0
        embed = variant["groups"][0]
        for g in variant["groups"]:
            csum = sum(table[c]["bytes"] for c in g["chunks"] if c in table)
            if any(c not in table for c in g["chunks"]):
                bad.append(f"{g['name']}: chunk not in table")
            if g["bytes"] != csum:
                bad.append(f"{g['name']}: bytes {g['bytes']} != chunk sum {csum}")
            gsum += g["bytes"]
            tied = g.get("tied", False)
            if g["name"] == "lm_head" and tied != tied_cfg:
                bad.append(f"lm_head tied={tied} but config tie_word_embeddings={tied_cfg}")
            if tied and (g["chunks"] or g["bytes"]):
                bad.append(f"{g['name']}: tied group must have no chunks")
            ref_chunks = embed["chunks"] if tied else g["chunks"]
            for e in g["entries"]:
                if e.get("tied", False) != tied:
                    bad.append(f"{e['name']}: entry tied flag != group")
                kind = e["storage"]["kind"]
                if kind != storage_kind(e["role"], tuple(e["shape"]), vname):
                    bad.append(f"{e['name']}: storage {kind} not expected for {vname}")
                plan = plan_entry(kind, tuple(e["shape"]))
                seg_sum = sum(s["length"] for s in e["segments"])
                if seg_sum != plan.length:
                    bad.append(f"{e['name']}: segments sum {seg_sum} != planned {plan.length}")
                if kind != "fp16":
                    got = {k: (p["offset"], p["length"]) for k, p in e["storage"]["parts"].items()}
                    if got != plan.parts:
                        bad.append(f"{e['name']}: parts {got} != planned {plan.parts}")
                # Locate the entry inside the group stream and check alignment + contiguity.
                first = e["segments"][0]
                if first["chunk"] not in ref_chunks:
                    bad.append(f"{e['name']}: segment chunk not in group")
                    continue
                idx = ref_chunks.index(first["chunk"])
                start = idx * CHUNK_SIZE + first["offset"]
                if start % align(1) != 0:
                    bad.append(f"{e['name']}: start {start} not 256-aligned")
                pos = start
                for s in e["segments"]:
                    ci, off = divmod(pos, CHUNK_SIZE)
                    if ci >= len(ref_chunks) or ref_chunks[ci] != s["chunk"] or off != s["offset"]:
                        bad.append(f"{e['name']}: segments not contiguous in stream")
                        break
                    if s["offset"] + s["length"] > table.get(s["chunk"], {"bytes": 0})["bytes"]:
                        bad.append(f"{e['name']}: segment overruns chunk")
                        break
                    pos += s["length"]
        if variant["bytes"] != gsum:
            bad.append(f"variant bytes {variant['bytes']} != group sum {gsum}")
        report.add(f"variant {vname} sums, layout, alignment", not bad, f"{len(variant['groups'])} groups, {variant['bytes']} bytes" + ("; " + "; ".join(bad[:5]) if bad else ""))


class ChunkReader:
    """Reads entry bytes from segments with a tiny chunk cache."""

    def __init__(self, store: LocalStore, cache: int = 4):
        self.store, self.n = store, cache
        self._cache: dict[str, bytes] = {}

    def chunk(self, sha: str) -> bytes:
        if sha not in self._cache:
            if len(self._cache) >= self.n:
                self._cache.pop(next(iter(self._cache)))
            self._cache[sha] = self.store.read_chunk(sha)
        return self._cache[sha]

    def entry_bytes(self, entry: dict) -> bytes:
        buf = bytearray()
        for s in entry["segments"]:
            buf += self.chunk(s["chunk"])[s["offset"] : s["offset"] + s["length"]]
        return bytes(buf)


def decode_entry(entry: dict, data: bytes) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Decode an entry's bytes. Returns (dequantized float32 or fp16 array, raw parts)."""
    shape = tuple(entry["shape"])
    st = entry["storage"]
    if st["kind"] == "fp16":
        w = np.frombuffer(data, dtype=np.float16).reshape(shape)
        return w, {"weights": w}
    p = {k: (v["offset"], v["length"]) for k, v in st["parts"].items()}
    out, inn = shape
    if st["kind"] == "q8":
        q = np.frombuffer(data, dtype=np.int8, count=p["weights"][1], offset=p["weights"][0]).reshape(out, inn)
        s = np.frombuffer(data, dtype=np.float16, count=out, offset=p["scales"][0])
        return quant.q8_decode(q, s), {"weights": q, "scales": s}
    ng = inn // quant.Q4_GROUP
    q = np.frombuffer(data, dtype=np.uint8, count=p["weights"][1], offset=p["weights"][0]).reshape(out, inn // 2)
    s = np.frombuffer(data, dtype=np.float16, count=out * ng, offset=p["scales"][0]).reshape(out, ng)
    z = np.frombuffer(data, dtype=np.uint8, count=out * ng, offset=p["zeros"][0]).reshape(out, ng)
    return quant.q4_decode(q, s, z), {"weights": q, "scales": s, "zeros": z}


@dataclass
class EntryCheck:
    variant: str
    group: str
    name: str
    role: str
    kind: str
    exact: bool | None       # fp16: bit-equal to the reference
    consistent: bool | None  # q8/q4: decoded parts == what the encoder intended
    max_abs: float | None
    rel_rms: float | None


def check_entry_against_reference(variant: str, group: str, entry: dict, data: bytes, w16: np.ndarray) -> EntryCheck:
    kind = entry["storage"]["kind"]
    deq, parts = decode_entry(entry, data)
    if kind == "fp16":
        exact = deq.shape == w16.shape and np.array_equal(deq.view(np.uint16), w16.view(np.uint16))
        return EntryCheck(variant, group, entry["name"], entry["role"], kind, exact, None, None, None)
    if kind == "q8":
        q, s = quant.q8_encode(w16)
        consistent = np.array_equal(parts["weights"], q) and np.array_equal(parts["scales"].view(np.uint16), s.view(np.uint16))
    else:
        q, s, z = quant.q4_encode(w16, entry["name"])
        consistent = (
            np.array_equal(parts["weights"], q)
            and np.array_equal(parts["scales"].view(np.uint16), s.view(np.uint16))
            and np.array_equal(parts["zeros"], z)
        )
    max_abs, rel_rms = quant.quant_error(w16, deq)
    return EntryCheck(variant, group, entry["name"], entry["role"], kind, None, bool(consistent), max_abs, rel_rms)


def aggregate_by_role(checks: list[EntryCheck]) -> list[dict]:
    """Per (variant, role) min/median/max of max_abs and rel_rms over layers."""
    buckets: dict[tuple[str, str], list[EntryCheck]] = {}
    for c in checks:
        if c.max_abs is not None:
            buckets.setdefault((c.variant, c.role), []).append(c)
    rows = []
    for (v, role), cs in sorted(buckets.items()):
        ma = np.array([c.max_abs for c in cs])
        rr = np.array([c.rel_rms for c in cs])
        rows.append({
            "variant": v, "role": role, "n": len(cs),
            "max_abs": (float(ma.min()), float(np.median(ma)), float(ma.max())),
            "rel_rms": (float(rr.min()), float(np.median(rr)), float(rr.max())),
        })
    return rows


# ---------------------------------------------------------------- inspection

def dedup_stats(manifest: dict) -> dict:
    """Chunk sharing between variants (model manifests)."""
    table = manifest["chunks"]
    per_variant: dict[str, list[str]] = {}
    for v, variant in manifest.get("variants", {}).items():
        per_variant[v] = [c for g in variant["groups"] for c in g["chunks"]]
    listed_bytes = sum(table[c]["bytes"] for cs in per_variant.values() for c in cs)
    union = set(c for cs in per_variant.values() for c in cs)
    union_bytes = sum(table[c]["bytes"] for c in union)
    owners: dict[str, set[str]] = {}
    for v, cs in per_variant.items():
        for c in cs:
            owners.setdefault(c, set()).add(v)
    shared = {c for c, vs in owners.items() if len(vs) > 1}
    pairs = {}
    vs = [v for v in ("fp16", "q8", "q4") if v in per_variant]
    for i, a in enumerate(vs):
        for b in vs[i + 1 :]:
            common = set(per_variant[a]) & set(per_variant[b])
            pairs[f"{a}∩{b}"] = (len(common), sum(table[c]["bytes"] for c in common))
    shared_groups = []
    for gi, g in enumerate(manifest["variants"][vs[0]]["groups"]) if vs else []:
        lists = {v: manifest["variants"][v]["groups"][gi]["chunks"] for v in vs}
        # Any set of variants whose chunk lists for this group are identical (not anchored on one variant).
        by_list: dict[tuple, list[str]] = {}
        for v in vs:
            if lists[v]:
                by_list.setdefault(tuple(lists[v]), []).append(v)
        for cs, same in by_list.items():
            if len(same) > 1:
                shared_groups.append((g["name"], same, len(cs), sum(table[c]["bytes"] for c in cs)))
    return {
        "listed_chunks": sum(len(cs) for cs in per_variant.values()),
        "listed_bytes": listed_bytes,
        "unique_chunks": len(union),
        "unique_bytes": union_bytes,
        "shared_chunks": len(shared),
        "shared_bytes": sum(table[c]["bytes"] for c in shared),
        "bytes_saved": listed_bytes - union_bytes,
        "pairs": pairs,
        "shared_groups": shared_groups,
    }

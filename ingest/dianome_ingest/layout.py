"""Logical layer: which tensors belong to which layer group, serialization
order, 256-byte alignment, and the packing driver that turns a tensor source
into one stream per (layer group, variant).

Layer groups, fixed order for a decoder-only model:
    embed, layer.0 … layer.L-1, final_norm, lm_head
A layer group is one stream per variant. Inside a stream every entry starts
256-byte aligned (zero padding); inside a quantized entry the sub-arrays
(weights, scales, zeros) are laid out in that order, each 256-byte aligned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import prod
from typing import Iterable, Protocol

import numpy as np

from . import quant
from .chunker import ChunkList, StreamChunker

ALIGN = 256
VARIANTS = ("fp16", "q8", "q4")
QUANT_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ROW_BLOCK_BYTES = 64 * 1024 * 1024  # fp16 bytes per row block when streaming a tensor

# Serialization order of roles inside a layer group. Unknown roles go after, by name.
LAYER_ROLE_ORDER = [
    "input_layernorm.weight",
    "attn.q_proj.weight", "attn.q_proj.bias",
    "attn.k_proj.weight", "attn.k_proj.bias",
    "attn.v_proj.weight", "attn.v_proj.bias",
    "attn.o_proj.weight", "attn.o_proj.bias",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.gate_proj.bias",
    "mlp.up_proj.weight", "mlp.up_proj.bias",
    "mlp.down_proj.weight", "mlp.down_proj.bias",
]

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def group_order(num_layers: int) -> list[str]:
    return ["embed", *[f"layer.{i}" for i in range(num_layers)], "final_norm", "lm_head"]


def group_of(name: str) -> str:
    """Layer group of a safetensors key. Raises on unknown keys (strict)."""
    if name.startswith("model.embed_tokens."):
        return "embed"
    m = _LAYER_RE.match(name)
    if m:
        return f"layer.{int(m.group(1))}"
    if name.startswith("model.norm."):
        return "final_norm"
    if name.startswith("lm_head."):
        return "lm_head"
    raise ValueError(f"tensor {name!r} does not belong to a known layer group")


def role_of(name: str) -> str:
    """Position-independent role: strip the group prefix, self_attn -> attn."""
    if name.startswith("model.embed_tokens."):
        return name[len("model."):]
    m = _LAYER_RE.match(name)
    if m:
        rest = m.group(2)
        return "attn." + rest[len("self_attn."):] if rest.startswith("self_attn.") else rest
    if name.startswith("model.norm."):
        return name[len("model."):]
    if name.startswith("lm_head."):
        return name
    raise ValueError(f"tensor {name!r} has no known role")


def role_sort_key(role: str, name: str) -> tuple[int, str]:
    try:
        return LAYER_ROLE_ORDER.index(role), name
    except ValueError:
        return len(LAYER_ROLE_ORDER), name


def storage_kind(role: str, shape: tuple[int, ...], variant: str) -> str:
    """fp16 | q8 | q4 for an entry in a variant.

    Only 2-D linear weights are quantized. embed_tokens and lm_head are fp16 in
    the fp16 variant and q8 in both the q8 and q4 variants (never q4). Norms and
    biases stay fp16.
    """
    if variant == "fp16" or len(shape) != 2:
        return "fp16"
    if role in ("embed_tokens.weight", "lm_head.weight"):
        return "q8"
    base = role.rsplit(".", 1)
    if len(base) == 2 and base[1] == "weight" and base[0].split(".")[-1] in QUANT_LINEAR:
        return variant
    return "fp16"


@dataclass
class EntryPlan:
    kind: str
    length: int                               # total entry length incl. internal padding
    parts: dict[str, tuple[int, int]]         # name -> (offset within entry, length)

    def storage(self) -> dict:
        if self.kind == "fp16":
            return {"kind": "fp16"}
        parts = {k: {"offset": o, "length": n} for k, (o, n) in self.parts.items()}
        if self.kind == "q8":
            return {"kind": "q8", "parts": parts}
        return {"kind": "q4", "group_size": quant.Q4_GROUP, "parts": parts}


def plan_entry(kind: str, shape: tuple[int, ...]) -> EntryPlan:
    """Byte layout of one entry from its storage kind and shape."""
    n = prod(shape)
    if kind == "fp16":
        return EntryPlan("fp16", n * 2, {})
    out, inn = shape
    if kind == "q8":
        sizes = [("weights", out * inn), ("scales", out * 2)]
    elif kind == "q4":
        ng = inn // quant.Q4_GROUP
        sizes = [("weights", out * inn // 2), ("scales", out * ng * 2), ("zeros", out * ng)]
    else:
        raise ValueError(kind)
    parts: dict[str, tuple[int, int]] = {}
    pos = 0
    for name, size in sizes:
        pos = align(pos)
        parts[name] = (pos, size)
        pos += size
    return EntryPlan(kind, pos, parts)


def segments_for(chunks: ChunkList, start: int, length: int) -> list[dict]:
    """Map [start, start+length) of a stream onto its chunk list."""
    segs = []
    pos = 0
    end = start + length
    for sha, clen in chunks:
        s, e = max(start, pos), min(end, pos + clen)
        if e > s:
            segs.append({"chunk": sha, "offset": s - pos, "length": e - s})
        pos += clen
    return segs


class TensorSource(Protocol):
    def names(self) -> list[str]: ...
    def shape(self, name: str) -> tuple[int, ...]: ...
    def get(self, name: str) -> np.ndarray: ...
    def get_rows(self, name: str, start: int, stop: int) -> np.ndarray: ...


class DictSource:
    """In-memory TensorSource (tests, synthetic models)."""

    def __init__(self, tensors: dict[str, np.ndarray]):
        self._t = tensors

    def names(self) -> list[str]:
        return list(self._t)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._t[name].shape)

    def get(self, name: str) -> np.ndarray:
        return self._t[name]

    def get_rows(self, name: str, start: int, stop: int) -> np.ndarray:
        return self._t[name][start:stop]


def iter_fp16_blocks(source: TensorSource, name: str, block_bytes: int = ROW_BLOCK_BYTES) -> Iterable[np.ndarray]:
    """Yield the tensor as fp16 row blocks (whole tensor if not 2-D)."""
    shape = source.shape(name)
    if len(shape) != 2:
        yield quant.to_fp16(source.get(name)).reshape(-1)
        return
    out, inn = shape
    rows = max(1, block_bytes // max(1, inn * 2))
    for a in range(0, out, rows):
        yield quant.to_fp16(source.get_rows(name, a, min(out, a + rows)))


class StreamWriter:
    def __init__(self, chunker: StreamChunker):
        self.chunker = chunker
        self.pos = 0

    def write(self, data) -> None:
        self.chunker.write(data)
        self.pos += len(memoryview(data).cast("B"))

    def pad_to(self, target: int) -> None:
        if target > self.pos:
            self.write(bytes(target - self.pos))

    def finish(self) -> ChunkList:
        return self.chunker.finish()


@dataclass
class PackedEntry:
    name: str
    role: str
    shape: tuple[int, ...]
    plan: EntryPlan
    start: int  # offset in the group stream


@dataclass
class PackedGroup:
    name: str
    chunks: ChunkList
    entries: list[PackedEntry]
    tied: bool = False

    @property
    def bytes(self) -> int:
        return sum(n for _, n in self.chunks)

    def to_manifest(self, chunk_table: dict[str, dict], source_group: "PackedGroup | None" = None) -> dict:
        for sha, n in self.chunks:
            chunk_table.setdefault(sha, {"bytes": n})
        ref = source_group if source_group is not None else self
        entries = []
        for e in self.entries:
            d = {
                "name": e.name,
                "role": e.role,
                "shape": list(e.shape),
                "storage": e.plan.storage(),
                "segments": segments_for(ref.chunks, e.start, e.plan.length),
            }
            if self.tied:
                d["tied"] = True
            entries.append(d)
        g = {"name": self.name, "bytes": self.bytes, "chunks": [sha for sha, _ in self.chunks], "entries": entries}
        if self.tied:
            g["tied"] = True
        return g


def assign_groups(names: Iterable[str], num_layers: int, tied: bool) -> dict[str, list[str]]:
    """Group -> ordered tensor names. Validates coverage of the fixed group order."""
    groups: dict[str, list[str]] = {g: [] for g in group_order(num_layers)}
    for n in names:
        g = group_of(n)
        if g not in groups:
            raise ValueError(f"tensor {n!r} maps to group {g!r} outside the {num_layers}-layer model")
        groups[g].append(n)
    for g, ns in groups.items():
        ns.sort(key=lambda n: role_sort_key(role_of(n), n))
        if not ns and not (g == "lm_head" and tied):
            raise ValueError(f"layer group {g!r} has no tensors")
    if tied and groups["lm_head"]:
        # Honour the config: the tied head reuses embed; a stray lm_head tensor is ignored.
        groups["lm_head"] = []
    if not tied and len(groups["lm_head"]) != 1:
        raise ValueError("model is not tied but has no single lm_head.weight tensor")
    if len(groups["embed"]) != 1:
        raise ValueError("expected exactly one embed_tokens tensor")
    return groups


def pack_model(
    source: TensorSource,
    num_layers: int,
    tied: bool,
    variants: tuple[str, ...],
    sink,
    progress=None,
) -> dict[str, list[PackedGroup]]:
    """Pack every layer group into one stream per variant.

    Each source tensor is read once (in row blocks), converted to fp16 once,
    and the q8/q4 encodings are derived from that fp16 block and written to the
    per-variant streams in lock-step. Returns variant -> ordered groups.
    """
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"unknown variant {v!r}")
    groups = assign_groups(source.names(), num_layers, tied)
    # q4 divisibility: assert before writing anything.
    if "q4" in variants:
        for names in groups.values():
            for n in names:
                shape = source.shape(n)
                if storage_kind(role_of(n), shape, "q4") == "q4":
                    quant.check_q4_shape(n, shape)

    seen: set[str] = set()
    result: dict[str, list[PackedGroup]] = {v: [] for v in variants}
    for gname, names in groups.items():
        if gname == "lm_head" and tied:
            for v in variants:
                embed = result[v][0]
                e = embed.entries[0]
                head = PackedEntry("lm_head.weight", "lm_head.weight", e.shape, e.plan, e.start)
                result[v].append(PackedGroup("lm_head", [], [head], tied=True))
            continue
        writers = {v: StreamWriter(StreamChunker(sink, seen)) for v in variants}
        entries: dict[str, list[PackedEntry]] = {v: [] for v in variants}
        for n in names:
            role, shape = role_of(n), source.shape(n)
            plans = {v: plan_entry(storage_kind(role, shape, v), shape) for v in variants}
            starts = {}
            for v in variants:
                writers[v].pad_to(align(writers[v].pos))
                starts[v] = writers[v].pos
            params: dict[str, dict[str, list[np.ndarray]]] = {v: {"scales": [], "zeros": []} for v in variants}
            for block in iter_fp16_blocks(source, n):
                for v in variants:
                    kind = plans[v].kind
                    if kind == "fp16":
                        writers[v].write(block.tobytes())
                    elif kind == "q8":
                        q, s = quant.q8_encode(block)
                        writers[v].write(q.tobytes())
                        params[v]["scales"].append(s)
                    else:
                        q, s, z = quant.q4_encode(block, n)
                        writers[v].write(q.tobytes())
                        params[v]["scales"].append(s)
                        params[v]["zeros"].append(z)
            for v in variants:
                w, plan, start = writers[v], plans[v], starts[v]
                if plan.kind != "fp16":
                    assert w.pos == start + plan.parts["weights"][1], (n, v, "weights length")
                    for part in ("scales", "zeros"):
                        if part in plan.parts:
                            w.pad_to(start + plan.parts[part][0])
                            w.write(np.concatenate(params[v][part]).tobytes())
                assert w.pos == start + plan.length, (n, v, "entry length")
                entries[v].append(PackedEntry(n, role, shape, plan, start))
            if progress:
                progress(gname, n)
        for v in variants:
            result[v].append(PackedGroup(gname, writers[v].finish(), entries[v]))
    return result

"""Binary WebSocket frames.

Frame = u32 magic 0x444E4D31 ("DNM1") | u32 header_len | JSON header | payload.
Both u32 are little-endian, so the first four bytes on the wire are 31 4D 4E 44.
The JSON header always carries `"type"`. Payloads are little-endian, row-major:
fp16 `[T, d_model]` hidden states, or int32 `[T]` token ids when N = 0.

| type    | dir | header                                              | payload                          |
| open    | c→s | {model, N, max_ctx, sampling{temperature,top_p,seed}, logits_topk?} | —                  |
| opened  | s→c | {session, L, d_model, boundary: "input_to_block_N"} | —                                |
| prefill | c→s | {T, positions: [start, end)}                        | fp16 [T, d_model] or int32 [T]   |
| decode  | c→s | {position}                                          | fp16 [1, d_model] or int32 [1]   |
| token   | s→c | {id, position, busy_ms, done, topk_ids?}            | optional fp16 [k] top-k logits   |
| stats   | c→s / s→c | {} / {tokens, busy_seconds, gpu_seconds_per_token} | —                          |
| ping    | c→s | {t?}  (RTT probe; allowed before open)              | —                                |
| pong    | s→c | {t?}  (echoes the ping header)                      | —                                |
| close   | any | {}                                                  | —                                |
| error   | s→c | {code, message}                                     | —                                |
"""

from __future__ import annotations

import dataclasses
import struct
from typing import Any, Optional

import numpy as np
import orjson

MAGIC = 0x444E4D31  # "DNM1" read big-endian; sent as little-endian u32
_HEAD = struct.Struct("<II")

TYPES = ("open", "opened", "prefill", "decode", "token", "stats", "close", "error", "ping", "pong")
REQUIRED: dict[str, tuple[str, ...]] = {
    "open": ("model", "N", "max_ctx"),
    "opened": ("session", "L", "d_model", "boundary"),
    "prefill": ("T", "positions"),
    "decode": ("position",),
    "token": ("id", "position", "busy_ms", "done"),
    "stats": (),
    "ping": (),
    "pong": (),
    "close": (),
    "error": ("code", "message"),
}

HIDDEN_DTYPE = np.dtype("<f2")
IDS_DTYPE = np.dtype("<i4")


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclasses.dataclass
class Message:
    type: str
    header: dict[str, Any]
    payload: bytes = b""

    def __getitem__(self, key: str) -> Any:
        return self.header[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.header.get(key, default)


def encode(type: str, header: Optional[dict[str, Any]] = None, payload: bytes | memoryview = b"") -> bytes:
    if type not in TYPES:
        raise ProtocolError("bad_type", f"unknown message type {type!r}")
    h = dict(header or {})
    h["type"] = type
    hb = orjson.dumps(h)
    return _HEAD.pack(MAGIC, len(hb)) + hb + bytes(payload)


def decode(frame: bytes | bytearray | memoryview) -> Message:
    frame = memoryview(frame)
    if len(frame) < _HEAD.size:
        raise ProtocolError("bad_frame", f"frame too short ({len(frame)} bytes)")
    magic, hlen = _HEAD.unpack_from(frame, 0)
    if magic != MAGIC:
        raise ProtocolError("bad_magic", f"bad magic 0x{magic:08X}")
    if _HEAD.size + hlen > len(frame):
        raise ProtocolError("bad_frame", "header_len exceeds frame")
    try:
        header = orjson.loads(bytes(frame[_HEAD.size : _HEAD.size + hlen]))
    except orjson.JSONDecodeError as e:
        raise ProtocolError("bad_header", f"header is not JSON: {e}") from e
    if not isinstance(header, dict) or "type" not in header:
        raise ProtocolError("bad_header", "header must be a JSON object with a 'type'")
    t = header["type"]
    if t not in TYPES:
        raise ProtocolError("bad_type", f"unknown message type {t!r}")
    missing = [k for k in REQUIRED[t] if k not in header]
    if missing:
        raise ProtocolError("bad_header", f"{t}: missing {', '.join(missing)}")
    return Message(t, header, bytes(frame[_HEAD.size + hlen :]))


# -- payload helpers ----------------------------------------------------------------


def hidden_to_bytes(x) -> bytes:
    """torch or numpy [T, d] -> little-endian fp16 row-major bytes."""
    a = np.ascontiguousarray(_to_numpy(x), dtype=np.float16).astype(HIDDEN_DTYPE, copy=False)
    if a.ndim != 2:
        raise ProtocolError("bad_payload", f"hidden must be 2-D, got shape {a.shape}")
    return a.tobytes()


def bytes_to_hidden(b: bytes, T: int, d_model: int) -> np.ndarray:
    want = T * d_model * 2
    if len(b) != want:
        raise ProtocolError("bad_payload", f"expected {want} bytes of fp16 [{T}, {d_model}], got {len(b)}")
    return np.frombuffer(b, dtype=HIDDEN_DTYPE).reshape(T, d_model).astype(np.float16, copy=False)


def ids_to_bytes(ids) -> bytes:
    a = np.ascontiguousarray(_to_numpy(ids)).astype(IDS_DTYPE)
    if a.ndim != 1:
        raise ProtocolError("bad_payload", f"ids must be 1-D, got shape {a.shape}")
    return a.tobytes()


def bytes_to_ids(b: bytes, T: int) -> np.ndarray:
    want = T * 4
    if len(b) != want:
        raise ProtocolError("bad_payload", f"expected {want} bytes of int32 [{T}], got {len(b)}")
    return np.frombuffer(b, dtype=IDS_DTYPE).astype(np.int32, copy=False)


def _to_numpy(x):
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

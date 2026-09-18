import numpy as np
import pytest
import torch

from dianome_server import protocol as p


def test_magic_bytes_on_wire():
    frame = p.encode("close", {})
    assert frame[:4] == bytes.fromhex("314D4E44")  # 0x444E4D31 little-endian
    assert int.from_bytes(frame[:4], "little") == 0x444E4D31


CASES = {
    "open": ({"model": "m", "N": 8, "max_ctx": 4096, "sampling": {"temperature": 0.7, "top_p": 0.9, "seed": 1}}, b""),
    "opened": ({"session": "abc", "L": 24, "d_model": 896, "boundary": "input_to_block_8"}, b""),
    "prefill": ({"T": 3, "positions": [0, 3]}, None),
    "decode": ({"position": 3}, None),
    "token": ({"id": 42, "position": 4, "busy_ms": 1.5, "done": False}, b""),
    "stats": ({"tokens": 4, "busy_seconds": 0.01, "gpu_seconds_per_token": 0.0025}, b""),
    "close": ({}, b""),
    "error": ({"code": "bad_N", "message": "nope"}, b""),
    "ping": ({"t": 12.5}, b""),
    "pong": ({"t": 12.5}, b""),
}


@pytest.mark.parametrize("type", list(CASES))
def test_round_trip_every_type(type):
    header, payload = CASES[type]
    if payload is None:
        payload = b"\x01\x02\x03\x04" * 3
    m = p.decode(p.encode(type, header, payload))
    assert m.type == type
    assert {k: v for k, v in m.header.items() if k != "type"} == header
    assert m.payload == payload


def test_hidden_payload_dtype_shape_and_endianness():
    x = torch.randn(5, 896).half()
    b = p.hidden_to_bytes(x)
    assert len(b) == 5 * 896 * 2
    y = p.bytes_to_hidden(b, 5, 896)
    assert y.dtype == np.float16 and y.shape == (5, 896)
    assert np.array_equal(y, x.numpy())
    # first element is little-endian fp16 on the wire
    assert b[:2] == np.array([x[0, 0].item()], dtype="<f2").tobytes()
    with pytest.raises(p.ProtocolError):
        p.bytes_to_hidden(b, 4, 896)


def test_ids_payload_int32_little_endian():
    ids = torch.tensor([1, 151935, 7])
    b = p.ids_to_bytes(ids)
    assert len(b) == 12 and b[:4] == (1).to_bytes(4, "little")
    y = p.bytes_to_ids(b, 3)
    assert y.dtype == np.int32 and y.tolist() == [1, 151935, 7]
    with pytest.raises(p.ProtocolError):
        p.bytes_to_ids(b, 2)


@pytest.mark.parametrize("bad,code", [
    (b"\x00" * 4, "bad_frame"),
    (b"\x00\x00\x00\x00" + (2).to_bytes(4, "little") + b"{}", "bad_magic"),
    (p.encode("close", {})[:8] + b"{", "bad_frame"),
    (bytes.fromhex("314D4E44") + (5).to_bytes(4, "little") + b"nope!", "bad_header"),
    (bytes.fromhex("314D4E44") + (16).to_bytes(4, "little") + b'{"type":"nope"} ', "bad_type"),
    (bytes.fromhex("314D4E44") + (16).to_bytes(4, "little") + b'{"type":"open"}  ', "bad_header"),
])
def test_rejects_bad_frames(bad, code):
    with pytest.raises(p.ProtocolError) as e:
        p.decode(bad)
    assert e.value.code == code

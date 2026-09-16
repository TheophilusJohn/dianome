import numpy as np
import pytest

from dianome_ingest import quant


def w16(shape, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * scale).astype(np.float16)


def test_to_fp16_from_bf16_bits():
    import ml_dtypes  # noqa
    f32 = np.array([1.0, -2.5, 3.14159, 1e-3], dtype=np.float32)
    bf = f32.astype(ml_dtypes.bfloat16)
    out = quant.to_fp16(bf)
    assert out.dtype == np.float16
    assert np.array_equal(out, bf.astype(np.float32).astype(np.float16))


def test_q8_encode_decode_consistency_and_layout():
    w = w16((64, 256))
    q, s = quant.q8_encode(w)
    assert q.dtype == np.int8 and q.shape == (64, 256)
    assert s.dtype == np.float16 and s.shape == (64,)
    assert q.min() >= -127 and q.max() <= 127
    # What the encoder intended: q == round(W/scale)
    expect = np.clip(np.round(w.astype(np.float32) / s.astype(np.float32)[:, None]), -127, 127)
    assert np.array_equal(q.astype(np.float32), expect)
    # Round-trip through raw bytes reproduces q and s exactly
    q2 = np.frombuffer(q.tobytes(), dtype=np.int8).reshape(64, 256)
    s2 = np.frombuffer(s.tobytes(), dtype=np.float16)
    assert np.array_equal(q2, q) and np.array_equal(s2.view(np.uint16), s.view(np.uint16))
    deq = quant.q8_decode(q2, s2)
    max_abs, rel = quant.quant_error(w, deq)
    assert rel < 0.01 and max_abs <= float(s.max()) / 2 + 1e-6


def test_q8_zero_row_scale_is_one():
    w = w16((4, 128))
    w[2] = 0
    q, s = quant.q8_encode(w)
    assert s[2] == np.float16(1.0) and not q[2].any()


def test_q8_scale_matches_spec():
    w = w16((8, 128))
    _, s = quant.q8_encode(w)
    amax = np.max(np.abs(w), axis=1)
    assert np.array_equal(s, (amax / np.float16(127)).astype(np.float16))


def test_q4_encode_decode_consistency_and_packing():
    w = w16((32, 512))
    packed, s, z = quant.q4_encode(w)
    assert packed.dtype == np.uint8 and packed.shape == (32, 256)
    assert s.dtype == np.float16 and s.shape == (32, 4)
    assert z.dtype == np.uint8 and z.shape == (32, 4) and z.max() <= 15
    q = quant.q4_unpack(packed)
    assert q.shape == (32, 512) and q.max() <= 15
    # nibble order: element i even -> low nibble of byte i//2
    assert np.array_equal(packed[:, 0] & 0xF, q[:, 0]) and np.array_equal(packed[:, 0] >> 4, q[:, 1])
    # encoder intent reproduced by unpacking
    g = w.reshape(32, 4, 128).astype(np.float32)
    intended = np.clip(np.round(g / s.astype(np.float32)[:, :, None]) + z[:, :, None], 0, 15).reshape(32, 512)
    assert np.array_equal(q, intended.astype(np.uint8))
    deq = quant.q4_decode(packed, s, z)
    max_abs, rel = quant.quant_error(w, deq)
    assert rel < 0.2 and max_abs <= float(s.max()) / 2 + 1e-6  # ~0.1 is expected for 4-bit/128 on Gaussian data


def test_q4_constant_group_round_trips_exactly():
    # max == min: scale = |min| (or 1.0 when min == 0) so a constant group decodes exactly.
    w = w16((3, 256))
    w[0, :128] = np.float16(0)
    w[1, :128] = np.float16(0.5)
    w[2, :128] = np.float16(-0.75)
    packed, s, z = quant.q4_encode(w)
    assert s[0, 0] == np.float16(1.0) and s[1, 0] == np.float16(0.5) and s[2, 0] == np.float16(0.75)
    deq = quant.q4_decode(packed, s, z)
    assert not deq[0, :128].any()
    assert np.array_equal(deq[1, :128], np.full(128, 0.5, np.float32))
    assert np.array_equal(deq[2, :128], np.full(128, -0.75, np.float32))


def test_q4_divisibility_assertion_fires_on_900_wide():
    w = w16((16, 900))
    with pytest.raises(ValueError) as ei:
        quant.q4_encode(w, "model.layers.3.mlp.down_proj.weight")
    msg = str(ei.value)
    assert "model.layers.3.mlp.down_proj.weight" in msg and "[16, 900]" in msg and "128" in msg


def test_determinism_on_synthetic_tensor():
    w = w16((128, 1024), seed=42)
    a = quant.q8_encode(w)
    b = quant.q8_encode(w.copy())
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
    a4 = quant.q4_encode(w)
    b4 = quant.q4_encode(np.ascontiguousarray(w[::1]))
    assert all(x.tobytes() == y.tobytes() for x, y in zip(a4, b4))
    # Row blocks give the same bytes as the whole array (row-independent schemes)
    parts = [quant.q4_encode(w[i : i + 50]) for i in range(0, 128, 50)]
    assert b"".join(p[0].tobytes() for p in parts) == a4[0].tobytes()
    assert b"".join(p[1].tobytes() for p in parts) == a4[1].tobytes()
    assert b"".join(p[2].tobytes() for p in parts) == a4[2].tobytes()


def test_rounding_is_half_to_even():
    # W/scale == 2.5 exactly -> 2 (half to even), 3.5 -> 4
    s = np.float16(1.0)
    w = np.array([[2.5, 3.5, -2.5, 0.5, 1.5, 4.5] + [0] * 122], dtype=np.float16) * s
    w[0, 6] = 127  # fixes scale at 1.0
    q, sc = quant.q8_encode(w)
    assert sc[0] == np.float16(1.0)
    assert q[0, :6].tolist() == [2, 4, -2, 0, 2, 4]

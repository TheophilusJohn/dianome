"""fp16 / q8 / q4 encoders and decoders. numpy only, deterministic.

fp16 is the reference: every quantized variant is derived from the fp16 array,
never from the original bf16/fp32.

q8: symmetric per-output-channel int8. W[out, in]:
    scale[o] = max(|W[o, :]|) / 127            (fp16; 1.0 if the row is all zeros)
    q = round_half_even(W / scale) clipped to [-127, 127]
    stored: int8 q[out, in] row-major, then fp16 scale[out]

q4: asymmetric group-wise, group size 128 along `in`. Per group:
    min, max over the 128 values (fp16)
    scale = (max - min) / 15                     (fp16; if max == min: |min|, or 1.0 when min == 0,
                                                  so a constant group round-trips exactly)
    zero  = clip(round_half_even(-min / scale), 0, 15)
    q     = clip(round_half_even(W / scale) + zero, 0, 15)
    stored: uint8 packed[out, in/2] (element i in the low nibble of byte i//2
            when i is even, high nibble when odd), then fp16 scale[out, in/128],
            then uint8 zero[out, in/128] (value in the low 4 bits)

Arithmetic: min/max/abs are exact in fp16; scale is computed with np.float16
arithmetic (numpy computes half ops in float32 and rounds to nearest even);
the ratio W / scale and the decoders use float32. np.round is half-to-even.
"""
from __future__ import annotations

import numpy as np

Q4_GROUP = 128


def to_fp16(arr: np.ndarray) -> np.ndarray:
    """Convert a source tensor (bf16, fp16, fp32, fp64) to a contiguous fp16 array.

    bf16 is widened bit-exactly to fp32 and then rounded to fp16 (nearest even).
    """
    if arr.dtype == np.float16:
        return np.ascontiguousarray(arr)
    if arr.dtype.name == "bfloat16":
        u32 = np.ascontiguousarray(arr).view(np.uint16).astype(np.uint32) << np.uint32(16)
        return u32.view(np.float32).astype(np.float16)
    if arr.dtype in (np.float32, np.float64):
        return arr.astype(np.float16)
    raise TypeError(f"unsupported source dtype {arr.dtype}")


def _check2d(w: np.ndarray) -> None:
    if w.dtype != np.float16 or w.ndim != 2:
        raise ValueError(f"expected a 2-D float16 array, got {w.dtype} shape {w.shape}")


# ---------------------------------------------------------------- q8

def q8_encode(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (q int8 [out, in], scale fp16 [out])."""
    _check2d(w)
    amax = np.max(np.abs(w), axis=1)  # fp16, exact
    scale = (amax / np.float16(127)).astype(np.float16)
    scale = np.where(amax == np.float16(0), np.float16(1.0), scale).astype(np.float16)
    ratio = w.astype(np.float32) / scale.astype(np.float32)[:, None]
    q = np.clip(np.round(ratio), -127, 127).astype(np.int8)
    return q, scale


def q8_decode(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize to float32 [out, in]."""
    return q.astype(np.float32) * scale.astype(np.float32)[:, None]


# ---------------------------------------------------------------- q4

def check_q4_shape(name: str, shape: tuple[int, ...]) -> None:
    """Assert, do not assume: the input dimension must be a multiple of 128."""
    if len(shape) != 2 or shape[1] % Q4_GROUP != 0:
        raise ValueError(
            f"q4 needs in % {Q4_GROUP} == 0 but tensor {name!r} has shape {list(shape)}"
            f" (in = {shape[-1] if shape else 'n/a'}); refusing to pad"
        )


def q4_encode(w: np.ndarray, name: str = "<tensor>") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (packed uint8 [out, in/2], scale fp16 [out, in/128], zero uint8 [out, in/128])."""
    _check2d(w)
    check_q4_shape(name, w.shape)
    out, inn = w.shape
    g = w.reshape(out, inn // Q4_GROUP, Q4_GROUP)
    mn = g.min(axis=2)  # fp16, exact
    mx = g.max(axis=2)
    scale = ((mx - mn) / np.float16(15)).astype(np.float16)
    const = np.where(mn == np.float16(0), np.float16(1.0), np.abs(mn)).astype(np.float16)
    scale = np.where(mx == mn, const, scale).astype(np.float16)
    s32 = scale.astype(np.float32)
    zero = np.clip(np.round(-mn.astype(np.float32) / s32), 0, 15).astype(np.uint8)
    q = np.round(g.astype(np.float32) / s32[:, :, None]) + zero[:, :, None].astype(np.float32)
    q = np.clip(q, 0, 15).astype(np.uint8).reshape(out, inn)
    packed = q4_pack(q)
    return packed, scale, zero


def q4_pack(q: np.ndarray) -> np.ndarray:
    """uint8 nibbles [out, in] -> packed uint8 [out, in/2]; even element low nibble."""
    return (q[:, 0::2] | (q[:, 1::2] << np.uint8(4))).astype(np.uint8)


def q4_unpack(packed: np.ndarray) -> np.ndarray:
    """packed uint8 [out, in/2] -> nibbles uint8 [out, in]."""
    out, half = packed.shape
    q = np.empty((out, half * 2), dtype=np.uint8)
    q[:, 0::2] = packed & np.uint8(0x0F)
    q[:, 1::2] = packed >> np.uint8(4)
    return q


def q4_decode(packed: np.ndarray, scale: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """Dequantize to float32 [out, in]."""
    q = q4_unpack(packed)
    out, inn = q.shape
    g = q.reshape(out, inn // Q4_GROUP, Q4_GROUP).astype(np.float32)
    deq = (g - zero.astype(np.float32)[:, :, None]) * scale.astype(np.float32)[:, :, None]
    return deq.reshape(out, inn)


# ---------------------------------------------------------------- errors

def quant_error(w16: np.ndarray, deq32: np.ndarray) -> tuple[float, float]:
    """(max absolute error, relative RMS error ||W - deq|| / ||W||), in float32."""
    w = w16.astype(np.float32)
    d = w - deq32
    max_abs = float(np.max(np.abs(d))) if d.size else 0.0
    norm = float(np.linalg.norm(w))
    rel_rms = float(np.linalg.norm(d) / norm) if norm > 0 else 0.0
    return max_abs, rel_rms

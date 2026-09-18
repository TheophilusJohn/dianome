// IEEE-754 binary16 <-> binary32, bit-exact. f32→f16 rounds to nearest, ties to even, like numpy/torch casts
// and like WGSL's pack2x16float on the backends we measured. No dependency on Float16Array.

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function f16ToF32(h: number): number {
  const s = (h & 0x8000) << 16;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) {
    if (m === 0) { u32[0] = s; return f32[0]!; }
    // subnormal: m * 2^-24
    u32[0] = s;
    return f32[0]! + (s ? -m : m) * 5.960464477539063e-8;
  }
  if (e === 31) { u32[0] = s | 0x7f800000 | (m << 13); return f32[0]!; }
  u32[0] = s | ((e + 112) << 23) | (m << 13);
  return f32[0]!;
}

export function f32ToF16(v: number): number {
  f32[0] = v;
  const x = u32[0]!;
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 | (mant >>> 13) : 0);
  let e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7c00; // overflow -> inf
  if (e <= 0) {
    if (e < -10) return sign; // underflow -> 0
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1))) half++;
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++; // may carry into exponent, which is correct
  return sign | half;
}

/** Round an f32 value to the nearest fp16 value (as f32). */
export function roundF16(v: number): number { return f16ToF32(f32ToF16(v)); }

export function f16ArrayToF32(src: Uint16Array, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) o[i] = f16ToF32(src[i]!);
  return o;
}

export function f32ArrayToF16(src: ArrayLike<number>, out?: Uint16Array): Uint16Array {
  const o = out ?? new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) o[i] = f32ToF16(src[i]!);
  return o;
}

/** fp16 little-endian bytes (any offset) -> Float32Array. */
export function f16BytesToF32(bytes: Uint8Array, count = bytes.byteLength >>> 1): Float32Array {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = f16ToF32(dv.getUint16(i * 2, true));
  return out;
}

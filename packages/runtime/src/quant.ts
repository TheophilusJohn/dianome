// TS port of ingest/dianome_ingest/quant.py (q8 / q4 encoders and decoders), bit-exact with numpy's fp16
// arithmetic: half ops are computed in f32 and rounded to nearest-even half; ratios and decoders use f32;
// rounding is half-to-even. Used by the unit tests to quantise random tensors for the q8/q4 kernels.

import { f16ToF32, f32ToF16, roundF16 } from "./f16";

export const Q4_GROUP = 128;

function roundHalfEven(v: number): number {
  const f = Math.floor(v);
  const d = v - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

const fr = Math.fround;

export interface Q8 { q: Int8Array; scale: Uint16Array /* fp16 bits [out] */; out: number; in: number }
export interface Q4 { packed: Uint8Array; scale: Uint16Array /* fp16 bits [out, in/128] */; zero: Uint8Array; out: number; in: number }

/** `w` holds fp16-representable f32 values, row-major [out, in]. */
export function q8Encode(w: Float32Array, out: number, inn: number): Q8 {
  const q = new Int8Array(out * inn);
  const scale = new Uint16Array(out);
  for (let o = 0; o < out; o++) {
    let amax = 0;
    for (let i = 0; i < inn; i++) amax = Math.max(amax, Math.abs(w[o * inn + i]!));
    const s = amax === 0 ? 1 : roundF16(fr(amax / 127));
    scale[o] = f32ToF16(s);
    for (let i = 0; i < inn; i++) {
      const r = fr(w[o * inn + i]! / s);
      q[o * inn + i] = Math.max(-127, Math.min(127, roundHalfEven(r)));
    }
  }
  return { q, scale, out, in: inn };
}

export function q8Decode(z: Q8): Float32Array {
  const w = new Float32Array(z.out * z.in);
  for (let o = 0; o < z.out; o++) {
    const s = f16ToF32(z.scale[o]!);
    for (let i = 0; i < z.in; i++) w[o * z.in + i] = fr(z.q[o * z.in + i]! * s);
  }
  return w;
}

export function q4Encode(w: Float32Array, out: number, inn: number): Q4 {
  if (inn % Q4_GROUP !== 0) throw new Error(`q4 needs in % ${Q4_GROUP} == 0, got in = ${inn}`);
  const ng = inn / Q4_GROUP;
  const packed = new Uint8Array(out * inn / 2);
  const scale = new Uint16Array(out * ng);
  const zero = new Uint8Array(out * ng);
  for (let o = 0; o < out; o++) {
    for (let g = 0; g < ng; g++) {
      const base = o * inn + g * Q4_GROUP;
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < Q4_GROUP; i++) { const v = w[base + i]!; if (v < mn) mn = v; if (v > mx) mx = v; }
      let s: number;
      if (mx === mn) s = mn === 0 ? 1 : Math.abs(mn);
      else s = roundF16(fr(roundF16(fr(mx - mn)) / 15));
      scale[o * ng + g] = f32ToF16(s);
      const z = Math.max(0, Math.min(15, roundHalfEven(fr(-mn / s))));
      zero[o * ng + g] = z;
      for (let i = 0; i < Q4_GROUP; i++) {
        const q = Math.max(0, Math.min(15, roundHalfEven(fr(w[base + i]! / s)) + z));
        const idx = base + i;
        const byte = idx >> 1;
        if ((idx & 1) === 0) packed[byte] = (packed[byte]! & 0xf0) | q;
        else packed[byte] = (packed[byte]! & 0x0f) | (q << 4);
      }
    }
  }
  return { packed, scale, zero, out, in: inn };
}

export function q4Decode(z: Q4): Float32Array {
  const ng = z.in / Q4_GROUP;
  const w = new Float32Array(z.out * z.in);
  for (let o = 0; o < z.out; o++) {
    for (let g = 0; g < ng; g++) {
      const s = f16ToF32(z.scale[o * ng + g]!);
      const zp = z.zero[o * ng + g]!;
      for (let i = 0; i < Q4_GROUP; i++) {
        const idx = o * z.in + g * Q4_GROUP + i;
        const byte = z.packed[idx >> 1]!;
        const q = (idx & 1) === 0 ? byte & 0x0f : byte >> 4;
        w[idx] = fr((q - zp) * s);
      }
    }
  }
  return w;
}

export interface Part { offset: number; length: number }

/** Serialize like ingest's entry layout: weights, then scales, then zeros, each 256-byte aligned. */
export function q8Bytes(z: Q8): { bytes: Uint8Array; parts: { weights: Part; scales: Part } } {
  const wl = z.q.length, so = align(wl), sl = z.scale.length * 2;
  const bytes = new Uint8Array(so + sl);
  bytes.set(new Uint8Array(z.q.buffer, z.q.byteOffset, wl), 0);
  bytes.set(new Uint8Array(z.scale.buffer, z.scale.byteOffset, sl), so);
  return { bytes, parts: { weights: { offset: 0, length: wl }, scales: { offset: so, length: sl } } };
}

export function q4Bytes(z: Q4): { bytes: Uint8Array; parts: { weights: Part; scales: Part; zeros: Part } } {
  const wl = z.packed.length, so = align(wl), sl = z.scale.length * 2, zo = align(so + sl), zl = z.zero.length;
  const bytes = new Uint8Array(zo + zl);
  bytes.set(z.packed, 0);
  bytes.set(new Uint8Array(z.scale.buffer, z.scale.byteOffset, sl), so);
  bytes.set(z.zero, zo);
  return { bytes, parts: { weights: { offset: 0, length: wl }, scales: { offset: so, length: sl }, zeros: { offset: zo, length: zl } } };
}

function align(n: number): number { return Math.ceil(n / 256) * 256; }

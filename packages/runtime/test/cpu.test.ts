// CPU reference ops against naive, independently written formulas (these are what the GPU kernels are checked
// against in Chrome, so they must be right on their own), plus fp16 conversion and the npy reader.
import { describe, expect, it } from "vitest";
import { CpuOps, cpuKv } from "../src/cpu";
import { f16ToF32, f32ToF16, roundF16 } from "../src/f16";
import { parseNpy } from "../src/npy";
import { ropeTable } from "../src/rope";
import type { ModelConfig } from "../src/block";

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const rand = (n: number, scale: number, seed: number): Float32Array => { const r = rng(seed); return Float32Array.from({ length: n }, () => (r() * 2 - 1) * scale); };

const cfg: ModelConfig = { d: 64, heads: 2, kvHeads: 1, headDim: 32, inter: 96, eps: 1e-6, ropeTheta: 10000, layers: 2 };

describe("fp16 conversion", () => {
  it("round-trips representable values and rounds to nearest even", () => {
    for (const v of [0, 1, -1, 0.5, 65504, 6.103515625e-5, 5.960464477539063e-8, 3.14159]) {
      const bits = f32ToF16(v);
      expect(f16ToF32(bits)).toBe(roundF16(v));
    }
    expect(f16ToF32(f32ToF16(1 + 2 ** -11))).toBe(1);            // tie -> even (1.0)
    expect(f16ToF32(f32ToF16(1 + 3 * 2 ** -11))).toBe(1 + 2 ** -9); // tie -> even (odd mantissa rounds up)
    expect(f16ToF32(f32ToF16(1e5))).toBe(Infinity);
    expect(f16ToF32(0x3c00)).toBe(1);
    expect(f16ToF32(0xc000)).toBe(-2);
  });
});

describe("npy reader", () => {
  it("parses a v1 header with fp16 payload", () => {
    const header = "{'descr': '<f2', 'fortran_order': False, 'shape': (2, 3), }";
    const pad = 64 - (10 + header.length) % 64;
    const h = header + " ".repeat(pad - 1) + "\n";
    const buf = new Uint8Array(10 + h.length + 12);
    buf.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, h.length & 0xff, h.length >> 8]);
    buf.set(new TextEncoder().encode(h), 10);
    const dv = new DataView(buf.buffer);
    const vals = [1, -2, 0.5, 4, 0.25, -8];
    vals.forEach((v, i) => dv.setUint16(10 + h.length + i * 2, f32ToF16(v), true));
    const n = parseNpy(buf.buffer);
    expect(n.shape).toEqual([2, 3]);
    expect([...n.data]).toEqual(vals);
  });
});

describe("CPU reference ops", () => {
  const ops = new CpuOps(cfg, { round16: false, maxCtx: 16 });

  it("rmsnorm matches the formula", () => {
    const x = rand(3 * 64, 2, 1), w = rand(64, 1, 2), out = new Float32Array(3 * 64);
    ops.rmsnorm(x, { data: w, shape: [64] }, out, 3);
    for (let t = 0; t < 3; t++) {
      let ss = 0; for (let i = 0; i < 64; i++) ss += x[t * 64 + i]! ** 2;
      const inv = 1 / Math.sqrt(ss / 64 + 1e-6);
      for (let i = 0; i < 64; i++) expect(out[t * 64 + i]).toBeCloseTo(x[t * 64 + i]! * inv * w[i]!, 5);
    }
  });

  it("matmul computes x · W^T + b", () => {
    const x = rand(2 * 64, 1, 3), w = rand(96 * 64, 0.1, 4), b = rand(96, 0.1, 5), out = new Float32Array(2 * 96);
    ops.matmul(x, { data: w, shape: [96, 64] }, { data: b, shape: [96] }, out, 2);
    for (let t = 0; t < 2; t++) for (let o = 0; o < 96; o++) {
      let acc = b[o]!; for (let i = 0; i < 64; i++) acc += x[t * 64 + i]! * w[o * 64 + i]!;
      expect(out[t * 96 + o]).toBeCloseTo(acc, 5);
    }
  });

  it("rope rotates pairs (i, i+half) by the position angle", () => {
    const q = rand(2 * 64, 1, 6), k = rand(2 * 32, 1, 7), v = rand(2 * 32, 1, 8);
    const before = q.slice();
    const kv = cpuKv(cfg, 16);
    ops.ropeKv(q, k, v, 2, 3, kv);
    const table = ropeTable(10000, 32, 16, false);
    for (let t = 0; t < 2; t++) for (let h = 0; h < 2; h++) for (let i = 0; i < 16; i++) {
      const p = 3 + t, c = table.cos[p * 32 + i]!, s = table.sin[p * 32 + i]!;
      const x1 = before[t * 64 + h * 32 + i]!, x2 = before[t * 64 + h * 32 + i + 16]!;
      expect(q[t * 64 + h * 32 + i]).toBeCloseTo(x1 * c - x2 * s, 5);
      expect(q[t * 64 + h * 32 + i + 16]).toBeCloseTo(x2 * c + x1 * s, 5);
    }
    // k lands in the cache at rows 3 and 4, v is copied.
    for (let t = 0; t < 2; t++) for (let i = 0; i < 32; i++) expect(kv.v[(3 + t) * 32 + i]).toBe(v[t * 32 + i]);
    expect(kv.k[3 * 32]).not.toBe(0);
  });

  it("attention is causal softmax(QK^T/sqrt(d)) V with GQA", () => {
    const kv = cpuKv(cfg, 16);
    kv.k.set(rand(5 * 32, 1, 9)); kv.v.set(rand(5 * 32, 1, 10));
    const q = rand(2 * 64, 1, 11), out = new Float32Array(2 * 64);
    ops.attention(q, kv, out, 2, 3); // queries at positions 3, 4
    for (let t = 0; t < 2; t++) for (let h = 0; h < 2; h++) {
      const n = 3 + t + 1;
      const sc = Array.from({ length: n }, (_, j) => { let s = 0; for (let i = 0; i < 32; i++) s += q[t * 64 + h * 32 + i]! * kv.k[j * 32 + i]!; return s / Math.sqrt(32); });
      const m = Math.max(...sc), e = sc.map((s) => Math.exp(s - m)), z = e.reduce((a, b) => a + b);
      for (let i = 0; i < 32; i++) {
        let acc = 0; for (let j = 0; j < n; j++) acc += (e[j]! / z) * kv.v[j * 32 + i]!;
        expect(out[t * 64 + h * 32 + i]).toBeCloseTo(acc, 5);
      }
    }
  });

  it("siluMul and add", () => {
    const g = rand(96, 3, 12), u = rand(96, 1, 13), out = new Float32Array(96);
    ops.siluMul(g, u, out, 1);
    for (let i = 0; i < 96; i++) expect(out[i]).toBeCloseTo((g[i]! / (1 + Math.exp(-g[i]!))) * u[i]!, 5);
    ops.add(g.subarray(0, 64), u.subarray(0, 64), out, 1);
    for (let i = 0; i < 64; i++) expect(out[i]).toBeCloseTo(g[i]! + u[i]!, 6);
  });

  it("round16 rounds every materialised value to fp16", () => {
    const r = new CpuOps(cfg, { round16: true, maxCtx: 16 });
    const x = rand(64, 1, 14), w = rand(64, 1, 15), out = new Float32Array(64);
    r.rmsnorm(x, { data: w, shape: [64] }, out, 1);
    for (const v of out) expect(v).toBe(roundF16(v));
  });
});

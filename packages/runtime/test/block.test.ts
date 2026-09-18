// Block wiring: forwardBlock (the op sequence the GPU runs) driven by the CPU backend on a tiny synthetic model
// (d = 64, 2 heads, 1 kv head, 2 blocks) against a monolithic, independently written reference of a Qwen2
// block, over a prefill and two cached decode steps. Plus the KV state bookkeeping.
import { describe, expect, it } from "vitest";
import { forwardBlock, type BlockWeights, type ModelConfig, type Workspace } from "../src/block";
import { CpuOps, cpuKv, type CpuKv, type CpuWeight } from "../src/cpu";
import { KvState } from "../src/kv";
import { ropeTable } from "../src/rope";

const cfg: ModelConfig = { d: 64, heads: 2, kvHeads: 1, headDim: 32, inter: 96, eps: 1e-6, ropeTheta: 10000, layers: 2 };
const MAXCTX = 16;

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const rand = (n: number, scale: number, seed: number): Float32Array => { const r = rng(seed); return Float32Array.from({ length: n }, () => (r() * 2 - 1) * scale); };
const W = (shape: number[], scale: number, seed: number): CpuWeight => ({ data: rand(shape.reduce((a, b) => a * b, 1), scale, seed), shape });

function synthBlock(seed: number): BlockWeights<CpuWeight> {
  return {
    inputNorm: W([64], 1, seed), q: W([64, 64], 0.2, seed + 1), qBias: W([64], 0.1, seed + 2), k: W([32, 64], 0.2, seed + 3), kBias: W([32], 0.1, seed + 4),
    v: W([32, 64], 0.2, seed + 5), vBias: W([32], 0.1, seed + 6), o: W([64, 64], 0.2, seed + 7), postNorm: W([64], 1, seed + 8),
    gate: W([96, 64], 0.2, seed + 9), up: W([96, 64], 0.2, seed + 10), down: W([64, 96], 0.2, seed + 11),
  };
}

/** Independent reference: whole-sequence Qwen2 block (no cache), returns outputs for all `T` positions. */
function referenceBlock(w: BlockWeights<CpuWeight>, x: Float32Array, T: number): Float32Array {
  const { d, heads, kvHeads, headDim: hd, inter } = cfg;
  const table = ropeTable(cfg.ropeTheta, hd, MAXCTX, false);
  const norm = (row: Float32Array, g: Float32Array) => { let ss = 0; for (const v of row) ss += v * v; const inv = 1 / Math.sqrt(ss / d + cfg.eps); return row.map((v, i) => v * inv * g[i]!); };
  const lin = (row: Float32Array, wt: CpuWeight, b?: CpuWeight) => { const [o, i] = [wt.shape[0]!, wt.shape[1]!]; const out = new Float32Array(o); for (let r = 0; r < o; r++) { let a = b ? b.data[r]! : 0; for (let c = 0; c < i; c++) a += row[c]! * wt.data[r * i + c]!; out[r] = a; } return out; };
  const rope = (vec: Float32Array, nh: number, p: number) => { const out = vec.slice(); for (let h = 0; h < nh; h++) for (let i = 0; i < hd / 2; i++) { const c = table.cos[p * hd + i]!, s = table.sin[p * hd + i]!; const a = vec[h * hd + i]!, b = vec[h * hd + i + hd / 2]!; out[h * hd + i] = a * c - b * s; out[h * hd + i + hd / 2] = b * c + a * s; } return out; };
  const qs: Float32Array[] = [], ks: Float32Array[] = [], vs: Float32Array[] = [], hs: Float32Array[] = [];
  for (let t = 0; t < T; t++) {
    const xt = x.subarray(t * d, (t + 1) * d);
    const h = norm(xt, w.inputNorm.data);
    hs.push(h);
    qs.push(rope(lin(h, w.q, w.qBias), heads, t));
    ks.push(rope(lin(h, w.k, w.kBias), kvHeads, t));
    vs.push(lin(h, w.v, w.vBias));
  }
  const out = new Float32Array(T * d);
  for (let t = 0; t < T; t++) {
    const attn = new Float32Array(heads * hd);
    for (let h = 0; h < heads; h++) {
      const kh = Math.floor(h / (heads / kvHeads));
      const sc: number[] = [];
      for (let j = 0; j <= t; j++) { let s = 0; for (let i = 0; i < hd; i++) s += qs[t]![h * hd + i]! * ks[j]![kh * hd + i]!; sc.push(s / Math.sqrt(hd)); }
      const m = Math.max(...sc), e = sc.map((s) => Math.exp(s - m)), z = e.reduce((a, b) => a + b);
      for (let i = 0; i < hd; i++) { let a = 0; for (let j = 0; j <= t; j++) a += (e[j]! / z) * vs[j]![kh * hd + i]!; attn[h * hd + i] = a; }
    }
    const o = lin(attn, w.o);
    const x2 = x.subarray(t * d, (t + 1) * d).map((v, i) => v + o[i]!);
    const h2 = norm(x2, w.postNorm.data);
    const g = lin(h2, w.gate), u = lin(h2, w.up);
    const act = g.map((v, i) => (v / (1 + Math.exp(-v))) * u[i]!);
    const m = lin(act, w.down);
    for (let i = 0; i < d; i++) out[t * d + i] = x2[i]! + m[i]!;
    void inter;
  }
  return out;
}

function workspace(): Workspace<Float32Array> {
  const c = (cols: number) => new Float32Array(MAXCTX * cols);
  return { x: c(64), x2: c(64), h: c(64), q: c(64), k: c(32), v: c(32), attn: c(64), o: c(64), gate: c(96), up: c(96), act: c(96), mlp: c(64) };
}

describe("block wiring (forwardBlock + CPU backend vs monolithic reference)", () => {
  const ops = new CpuOps(cfg, { round16: false, maxCtx: MAXCTX });
  const blocks = [synthBlock(100), synthBlock(200)];
  const T = 7;
  const x0 = rand(T * 64, 1, 300);

  // Reference: 2 blocks over the full 7-token sequence.
  const ref1 = referenceBlock(blocks[0]!, x0, T);
  const ref2 = referenceBlock(blocks[1]!, ref1, T);

  it("prefill of 7 tokens through 2 blocks matches", () => {
    const ws = workspace();
    const kvs: CpuKv[] = [cpuKv(cfg, MAXCTX), cpuKv(cfg, MAXCTX)];
    ws.x.set(x0);
    forwardBlock(ops, blocks[0]!, ws, kvs[0]!, T, 0);
    forwardBlock(ops, blocks[1]!, ws, kvs[1]!, T, 0);
    for (let i = 0; i < T * 64; i++) expect(ws.x[i]).toBeCloseTo(ref2[i]!, 4);
  });

  it("prefill 5 + decode 2 with the KV cache equals the 7-token prefill", () => {
    const ws = workspace();
    const kvs: CpuKv[] = [cpuKv(cfg, MAXCTX), cpuKv(cfg, MAXCTX)];
    const st = new KvState(MAXCTX);
    ws.x.set(x0.subarray(0, 5 * 64));
    st.advance(0, 5);
    forwardBlock(ops, blocks[0]!, ws, kvs[0]!, 5, 0);
    forwardBlock(ops, blocks[1]!, ws, kvs[1]!, 5, 0);
    for (let i = 0; i < 5 * 64; i++) expect(ws.x[i]).toBeCloseTo(ref2[i]!, 4);
    for (const pos of [5, 6]) {
      ws.x.set(x0.subarray(pos * 64, (pos + 1) * 64));
      st.advance(pos, 1);
      forwardBlock(ops, blocks[0]!, ws, kvs[0]!, 1, pos);
      forwardBlock(ops, blocks[1]!, ws, kvs[1]!, 1, pos);
      for (let i = 0; i < 64; i++) expect(ws.x[i]).toBeCloseTo(ref2[pos * 64 + i]!, 4);
    }
    expect(st.length).toBe(7);
  });

  it("`until` stops after the named stage and leaves the residual untouched", () => {
    const ws = workspace();
    ws.x.set(x0);
    const last = forwardBlock(ops, blocks[0]!, ws, cpuKv(cfg, MAXCTX), T, 0, "o_proj");
    expect(last).toBe("o_proj");
    for (let i = 0; i < T * 64; i++) expect(ws.x[i]).toBe(x0[i]);
    expect(ws.o.subarray(0, 64).some((v) => v !== 0)).toBe(true);
    expect(ws.gate.every((v) => v === 0)).toBe(true);
  });
});

describe("KV state", () => {
  it("requires contiguous positions and respects maxCtx", () => {
    const st = new KvState(8);
    st.advance(0, 5);
    expect(() => st.advance(4, 1)).toThrow(/contiguous/);
    expect(() => st.advance(5, 4)).toThrow(/maxCtx/);
    st.advance(5, 3);
    expect(st.length).toBe(8);
    st.reset();
    expect(st.length).toBe(0);
  });
});

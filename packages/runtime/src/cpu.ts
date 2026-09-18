// CPU reference backend: plain loops in f32 storage with f64 accumulation, the same op contract the GPU
// implements. With `round16` it also rounds to fp16 where an fp16 PyTorch model materialises an fp16 tensor
// (after every matmul, norm, RoPE product/sum, attention output, silu, product, residual add), which is what the
// fixtures were made with. The unit tests compare the GPU kernels against this on random inputs.

import type { BlockOps, ModelConfig } from "./block";
import { roundF16 } from "./f16";
import { ropeTable, type RopeTable } from "./rope";

export interface CpuWeight { data: Float32Array; shape: number[] }
export type CpuTensor = Float32Array; // row-major, `cols` implied by the op, rows = maxCtx capacity
export interface CpuKv { k: Float32Array; v: Float32Array; rows: number; cols: number }

export interface CpuOpts { round16: boolean; rope?: RopeTable; maxCtx: number }

export function cpuKv(cfg: ModelConfig, maxCtx: number): CpuKv {
  const cols = cfg.kvHeads * cfg.headDim;
  return { k: new Float32Array(maxCtx * cols), v: new Float32Array(maxCtx * cols), rows: maxCtx, cols };
}

export class CpuOps implements BlockOps<CpuWeight, CpuTensor, CpuKv> {
  readonly round16: boolean;
  readonly rope: RopeTable;
  constructor(readonly cfg: ModelConfig, opts: CpuOpts) {
    this.round16 = opts.round16;
    this.rope = opts.rope ?? ropeTable(cfg.ropeTheta, cfg.headDim, opts.maxCtx, opts.round16);
  }
  private r(v: number): number { return this.round16 ? roundF16(v) : Math.fround(v); }

  rmsnorm(x: CpuTensor, w: CpuWeight, out: CpuTensor, T: number): void {
    const d = w.data.length;
    for (let t = 0; t < T; t++) {
      let ss = 0;
      for (let i = 0; i < d; i++) { const v = x[t * d + i]!; ss += v * v; }
      const inv = Math.fround(1 / Math.sqrt(Math.fround(ss / d) + this.cfg.eps));
      for (let i = 0; i < d; i++) {
        const n = this.r(x[t * d + i]! * inv);
        out[t * d + i] = this.r(n * w.data[i]!);
      }
    }
  }

  matmul(x: CpuTensor, w: CpuWeight, bias: CpuWeight | null, out: CpuTensor, T: number): void {
    const [o, i] = [w.shape[0]!, w.shape[1]!];
    for (let t = 0; t < T; t++) {
      for (let r = 0; r < o; r++) {
        let acc = 0;
        const xb = t * i, wb = r * i;
        for (let c = 0; c < i; c++) acc += x[xb + c]! * w.data[wb + c]!;
        if (bias) acc += bias.data[r]!;
        out[t * o + r] = this.r(acc);
      }
    }
  }

  private rotate(buf: CpuTensor, T: number, pos0: number, heads: number, dst?: { arr: Float32Array; cols: number }): void {
    const hd = this.cfg.headDim, half = hd / 2, cols = heads * hd;
    for (let t = 0; t < T; t++) {
      const p = pos0 + t;
      for (let h = 0; h < heads; h++) {
        const base = t * cols + h * hd;
        for (let i = 0; i < half; i++) {
          const c = this.rope.cos[p * hd + i]!, s = this.rope.sin[p * hd + i]!;
          const x1 = buf[base + i]!, x2 = buf[base + i + half]!;
          const o1 = this.r(this.r(x1 * c) + this.r(-x2 * s));
          const o2 = this.r(this.r(x2 * c) + this.r(x1 * s));
          if (dst) { dst.arr[p * dst.cols + h * hd + i] = o1; dst.arr[p * dst.cols + h * hd + i + half] = o2; }
          else { buf[base + i] = o1; buf[base + i + half] = o2; }
        }
      }
    }
  }

  ropeKv(q: CpuTensor, k: CpuTensor, v: CpuTensor, T: number, pos0: number, kv: CpuKv): void {
    this.rotate(q, T, pos0, this.cfg.heads);
    this.rotate(k, T, pos0, this.cfg.kvHeads, { arr: kv.k, cols: kv.cols });
    for (let t = 0; t < T; t++) for (let c = 0; c < kv.cols; c++) kv.v[(pos0 + t) * kv.cols + c] = this.r(v[t * kv.cols + c]!);
  }

  attention(q: CpuTensor, kv: CpuKv, out: CpuTensor, T: number, pos0: number): void {
    const { heads, kvHeads, headDim: hd } = this.cfg;
    const qcols = heads * hd, group = heads / kvHeads, scale = 1 / Math.sqrt(hd);
    const scores = new Float64Array(pos0 + T);
    for (let t = 0; t < T; t++) {
      const nKeys = pos0 + t + 1;
      for (let h = 0; h < heads; h++) {
        const kh = Math.floor(h / group);
        const qb = t * qcols + h * hd;
        let max = -Infinity;
        for (let j = 0; j < nKeys; j++) {
          let s = 0;
          const kb = j * kv.cols + kh * hd;
          for (let i = 0; i < hd; i++) s += q[qb + i]! * kv.k[kb + i]!;
          s *= scale;
          scores[j] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j < nKeys; j++) { const e = Math.exp(scores[j]! - max); scores[j] = e; sum += e; }
        for (let i = 0; i < hd; i++) {
          let acc = 0;
          for (let j = 0; j < nKeys; j++) acc += (scores[j]! / sum) * kv.v[j * kv.cols + kh * hd + i]!;
          out[t * qcols + h * hd + i] = this.r(acc);
        }
      }
    }
  }

  siluMul(gate: CpuTensor, up: CpuTensor, out: CpuTensor, T: number): void {
    const n = T * this.cfg.inter;
    for (let i = 0; i < n; i++) {
      const g = gate[i]!;
      const s = this.r(g / (1 + Math.exp(-g)));
      out[i] = this.r(s * up[i]!);
    }
  }

  add(a: CpuTensor, b: CpuTensor, out: CpuTensor, T: number): void {
    const n = T * this.cfg.d;
    for (let i = 0; i < n; i++) out[i] = this.r(a[i]! + b[i]!);
  }
}

/** Embedding lookup: rows of an fp16-valued [vocab, d] table. */
export function cpuEmbed(table: Float32Array, d: number, tokens: ArrayLike<number>, out: Float32Array): void {
  for (let t = 0; t < tokens.length; t++) {
    const id = tokens[t]!;
    out.set(table.subarray(id * d, id * d + d), t * d);
  }
}

// lm_head + sampler (Phase 5b). The head is one more matmul over an existing entry: final RMSNorm (the
// `final_norm` group's norm.weight) then `logits = normed · W^T` with W = the lm_head entry, which the manifest
// marks `tied` on Qwen2.5-0.5B, so the embed entry's bytes (fp16 or q8) are uploaded once more as a weight.
// The head runs on the last row only (one matvec of [1, d] x [vocab, d]^T); sampling is on the CPU over the
// vocab-sized f32 logits read back from the GPU.

import type { GpuOps, GpuTensor } from "./gpu/ops";
import { createStorage, type WeightBuffer } from "./gpu/buffers";

export type HeadMatvec = "seq" | "lanes" | "tiled";

export class LmHead {
  readonly vocab: number;
  readonly d: number;
  private readonly lastRow: GpuTensor;
  private readonly normed: GpuTensor;
  private readonly logits: GpuTensor;
  private allRows: GpuTensor | null = null;
  readonly tally = { bytes: 0 };

  constructor(readonly ops: GpuOps, readonly norm: WeightBuffer, readonly weight: WeightBuffer, readonly matvec: HeadMatvec | "auto" = "auto") {
    this.vocab = weight.shape[0]!;
    this.d = weight.shape[1]!;
    if (norm.shape[0] !== this.d) throw new Error(`final_norm width ${norm.shape[0]} != lm_head in ${this.d}`);
    this.lastRow = this.tensor("head.x", 1, this.d);
    this.normed = this.tensor("head.normed", 1, this.d);
    this.logits = this.tensor("head.logits", 1, this.vocab);
  }

  private tensor(label: string, rows: number, cols: number): GpuTensor {
    return { buffer: createStorage(this.ops.device, label, rows * cols * 4, this.tally), rows, cols, label };
  }

  /** Records final norm + lm_head for row T-1 of `x` into the logits tensor (inside an open pass). */
  record(x: GpuTensor, T: number): void {
    this.ops.copy(x.buffer, (T - 1) * this.d * 4, this.lastRow.buffer, 0, this.d * 4);
    this.ops.rmsnorm(this.lastRow, this.norm, this.normed, 1);
    this.ops.matmulWith(this.normed, this.weight, null, this.logits, 1, this.matvec === "auto" ? undefined : this.matvec);
  }

  /** Reads the last recorded logits ([vocab] f32). */
  read(): Promise<Float32Array> {
    return this.ops.read(this.logits, 1);
  }

  /** Every row's logits (tests / gates): final norm + lm_head on rows 0..T-1 with the tiled kernel. */
  async logitsAll(x: GpuTensor, T: number): Promise<Float32Array> {
    if (!this.allRows || this.allRows.rows < T) {
      this.allRows?.buffer.destroy();
      this.allRows = this.tensor("head.logits.all", T, this.vocab);
    }
    const normed = this.tensor("head.normed.all", T, this.d);
    this.ops.run(() => {
      this.ops.rmsnorm(x, this.norm, normed, T);
      this.ops.matmulWith(normed, this.weight, null, this.allRows!, T, "tiled");
    }, `lm_head all rows T=${T}`);
    const out = await this.ops.read(this.allRows, T);
    normed.buffer.destroy();
    return out;
  }

  destroy(): void {
    this.lastRow.buffer.destroy(); this.normed.buffer.destroy(); this.logits.buffer.destroy(); this.allRows?.buffer.destroy();
    this.norm.buffer.destroy(); this.weight.buffer.destroy();
  }
}

// -- sampler ---------------------------------------------------------------------------------------

export interface SamplingOptions {
  /** 0 (default) = greedy. */
  temperature?: number;
  /** Nucleus threshold in (0, 1]; 1 (default) = off. */
  topP?: number;
  /** Seed for the PRNG when temperature > 0; omitted = a random seed. */
  seed?: number;
}

/** xoshiro128** seeded from a 32-bit integer through splitmix32; `next()` is a double in [0, 1). */
export class Prng {
  private s = new Uint32Array(4);
  constructor(seed: number) {
    let x = (seed >>> 0) || 0x9e3779b9;
    for (let i = 0; i < 4; i++) {
      x = (x + 0x9e3779b9) >>> 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      this.s[i] = (z ^ (z >>> 16)) >>> 0;
    }
  }
  private u32(): number {
    const s = this.s;
    const r = Math.imul(rotl(Math.imul(s[1]!, 5) >>> 0, 7), 9) >>> 0;
    const t = (s[1]! << 9) >>> 0;
    s[2] = (s[2]! ^ s[0]!) >>> 0; s[3] = (s[3]! ^ s[1]!) >>> 0; s[1] = (s[1]! ^ s[2]!) >>> 0; s[0] = (s[0]! ^ s[3]!) >>> 0;
    s[2] = (s[2]! ^ t) >>> 0; s[3] = rotl(s[3]!, 11);
    return r;
  }
  next(): number { return (this.u32() * 2 ** 21 + (this.u32() >>> 11)) / 2 ** 53; }
}
function rotl(x: number, k: number): number { return ((x << k) | (x >>> (32 - k))) >>> 0; }

export class Sampler {
  readonly temperature: number;
  readonly topP: number;
  readonly seed: number;
  private readonly rng: Prng;

  constructor(opts: SamplingOptions = {}) {
    this.temperature = Math.max(0, opts.temperature ?? 0);
    this.topP = Math.min(1, Math.max(0, opts.topP ?? 1));
    this.seed = opts.seed ?? Math.floor(Math.random() * 2 ** 32);
    this.rng = new Prng(this.seed);
  }

  /** Same rules as the server's Sampler: greedy at temperature 0; else softmax(logits / T), nucleus `topP`, multinomial. */
  sample(logits: Float32Array): number {
    if (this.temperature <= 0) return argmax(logits);
    const n = logits.length;
    let max = -Infinity;
    for (let i = 0; i < n; i++) if (logits[i]! > max) max = logits[i]!;
    const p = new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) { const e = Math.exp((logits[i]! - max) / this.temperature); p[i] = e; sum += e; }
    for (let i = 0; i < n; i++) p[i]! /= sum;
    if (this.topP < 1) {
      // Exact nucleus without sorting the whole vocab: a token with p < (1 - topP) / n can never be inside the
      // nucleus (the tail above it already sums past topP), so only candidates at or above that bound are sorted.
      const bound = (1 - this.topP) / n;
      const idx: number[] = [];
      for (let i = 0; i < n; i++) if (p[i]! >= bound) idx.push(i);
      idx.sort((a, b) => p[b]! - p[a]! || a - b);
      let cum = 0, kept = 0;
      for (const i of idx) { if (cum < this.topP) { kept += p[i]!; cum += p[i]!; } else break; }
      let u = this.rng.next() * kept;
      cum = 0;
      for (const i of idx) { if (cum >= this.topP) break; u -= p[i]!; if (u < 0) return i; cum += p[i]!; }
      return idx[0]!;
    }
    let u = this.rng.next();
    for (let i = 0; i < n; i++) { u -= p[i]!; if (u < 0) return i; }
    return argmax(logits);
  }
}

/** First index of the maximum (torch.argmax's tie rule). */
export function argmax(v: ArrayLike<number>): number {
  let best = 0, bv = -Infinity;
  for (let i = 0; i < v.length; i++) if (v[i]! > bv) { bv = v[i]!; best = i; }
  return best;
}

/** Top-k (id, logit) pairs, descending. */
export function topk(v: ArrayLike<number>, k: number): { id: number; logit: number }[] {
  const out: { id: number; logit: number }[] = [];
  for (let i = 0; i < v.length; i++) {
    const x = v[i]!;
    if (out.length < k || x > out[out.length - 1]!.logit) {
      let j = out.length;
      while (j > 0 && out[j - 1]!.logit < x) j--;
      out.splice(j, 0, { id: i, logit: x });
      if (out.length > k) out.pop();
    }
  }
  return out;
}

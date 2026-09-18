// RoPE cos/sin tables, computed the way transformers' Qwen2RotaryEmbedding does (default rope, attention
// scaling 1): inv_freq = 1 / theta^(2i/d) in f32, freqs = pos * inv_freq in f32, cos/sin in f32, then cast to
// the model dtype (fp16) — that last cast is the `round16` option. Layout [positions, headDim] with the second
// half repeating the first (emb = cat(freqs, freqs)).

import { roundF16 } from "./f16";

export interface RopeTable { cos: Float32Array; sin: Float32Array; positions: number; headDim: number }

export function ropeTable(theta: number, headDim: number, positions: number, round16: boolean): RopeTable {
  const half = headDim / 2;
  const fr = Math.fround;
  const invFreq = new Float32Array(half);
  for (let i = 0; i < half; i++) invFreq[i] = fr(1 / fr(Math.pow(theta, fr((2 * i) / headDim))));
  const cos = new Float32Array(positions * headDim);
  const sin = new Float32Array(positions * headDim);
  for (let p = 0; p < positions; p++) {
    for (let i = 0; i < half; i++) {
      const f = fr(p * invFreq[i]!);
      let c = fr(Math.cos(f)), s = fr(Math.sin(f));
      if (round16) { c = roundF16(c); s = roundF16(s); }
      cos[p * headDim + i] = c; cos[p * headDim + i + half] = c;
      sin[p * headDim + i] = s; sin[p * headDim + i + half] = s;
    }
  }
  return { cos, sin, positions, headDim };
}

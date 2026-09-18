import { describe, expect, it } from "vitest";
import { Prng, Sampler, argmax, topk } from "../src/lmhead";

function logits(seed: number, n = 4096): Float32Array {
  const r = new Prng(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (r.next() - 0.5) * 12;
  return out;
}

describe("sampler", () => {
  it("greedy = first argmax (torch tie rule)", () => {
    expect(new Sampler().sample(new Float32Array([1, 3, 3, 2]))).toBe(1);
    expect(argmax([-1, -1, 0.5, 0.5])).toBe(2);
  });

  it("is deterministic for a seed and differs across seeds", () => {
    const l = logits(1);
    const a = new Sampler({ temperature: 0.8, seed: 42 }), b = new Sampler({ temperature: 0.8, seed: 42 }), c = new Sampler({ temperature: 0.8, seed: 43 });
    const sa = Array.from({ length: 64 }, () => a.sample(l)), sb = Array.from({ length: 64 }, () => b.sample(l)), sc = Array.from({ length: 64 }, () => c.sample(l));
    expect(sa).toEqual(sb);
    expect(sa).not.toEqual(sc);
    expect(new Set(sa).size).toBeGreaterThan(1);
  });

  it("top-p never returns a token outside the nucleus and matches the full sort", () => {
    const l = logits(7);
    const p = new Float64Array(l.length);
    let max = -Infinity; for (const v of l) max = Math.max(max, v);
    let sum = 0; for (let i = 0; i < l.length; i++) { p[i] = Math.exp((l[i]! - max) / 0.7); sum += p[i]!; }
    for (let i = 0; i < l.length; i++) p[i]! /= sum;
    const order = [...p.keys()].sort((a, b) => p[b]! - p[a]! || a - b);
    const nucleus = new Set<number>();
    let cum = 0; for (const i of order) { if (cum < 0.5) { nucleus.add(i); cum += p[i]!; } else break; }
    const s = new Sampler({ temperature: 0.7, topP: 0.5, seed: 3 });
    for (let k = 0; k < 500; k++) expect(nucleus.has(s.sample(l))).toBe(true);
    // top-p 1 with a tiny temperature is greedy in practice
    const g = new Sampler({ temperature: 1e-3, seed: 1 });
    expect(g.sample(l)).toBe(argmax(l));
  });

  it("topk is descending and stable", () => {
    expect(topk([1, 5, 3, 5, 2], 3)).toEqual([{ id: 1, logit: 5 }, { id: 3, logit: 5 }, { id: 2, logit: 3 }]);
  });

  it("prng draws in [0, 1) and is reproducible", () => {
    const a = new Prng(9), b = new Prng(9);
    for (let i = 0; i < 1000; i++) { const x = a.next(); expect(x).toBeGreaterThanOrEqual(0); expect(x).toBeLessThan(1); expect(b.next()).toBe(x); }
  });
});

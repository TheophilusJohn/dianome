import { describe, expect, it } from "vitest";
import { join } from "node:path";
import { f16ToF32 } from "../src/f16";
import { q4Decode, q4Encode, q8Decode, q8Encode } from "../src/quant";
import { readJson } from "./helpers/paths";

interface Cases {
  out: number; in: number; w_f16_bits: number[];
  q8: number[]; q8_scale_bits: number[]; q4_packed: number[]; q4_scale_bits: number[]; q4_zero: number[];
  q8_dec: number[]; q4_dec: number[];
}

// test/fixtures/quant_cases.json was written by ingest's quant.py (see the Phase 5a notes) on a 6x256 tensor with
// an all-zero row, a constant group and a zero group, so every branch of the encoders is exercised.
describe("quant encoder port (byte-identical with quant.py)", () => {
  const c = readJson<Cases>(join(import.meta.dirname, "fixtures/quant_cases.json"));
  const w = new Float32Array(c.w_f16_bits.map(f16ToF32));

  it("q8 matches the Python encoder", () => {
    const q = q8Encode(w, c.out, c.in);
    expect([...q.q]).toEqual(c.q8);
    expect([...q.scale]).toEqual(c.q8_scale_bits);
    const dec = q8Decode(q);
    for (let i = 0; i < dec.length; i++) expect(dec[i]).toBeCloseTo(c.q8_dec[i]!, 7);
  });

  it("q4 matches the Python encoder", () => {
    const q = q4Encode(w, c.out, c.in);
    expect([...q.packed]).toEqual(c.q4_packed);
    expect([...q.scale]).toEqual(c.q4_scale_bits);
    expect([...q.zero]).toEqual(c.q4_zero);
    const dec = q4Decode(q);
    for (let i = 0; i < dec.length; i++) expect(dec[i]).toBeCloseTo(c.q4_dec[i]!, 7);
  });

  it("q4 refuses in % 128 != 0", () => {
    expect(() => q4Encode(new Float32Array(2 * 100), 2, 100)).toThrow(/128/);
  });
});

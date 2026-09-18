import { describe, expect, it } from "vitest";
import { join } from "node:path";
import { Tokenizer, type TokenizerJson } from "../src/tokenizer";
import { FIXTURES, readJson, tokenizerJsonPath } from "./helpers/paths";

interface Cases { count: number; cases: { text: string; ids: number[]; decoded: string }[] }

describe("tokenizer (gate 1)", () => {
  const tok = new Tokenizer(readJson<TokenizerJson>(tokenizerJsonPath()));
  const cases = readJson<Cases>(join(FIXTURES, "tokenizer_cases.json"));

  it("has 200 cases", () => { expect(cases.count).toBe(200); expect(cases.cases.length).toBe(200); });

  it("encodes all 200 cases exactly like the HF tokenizer", () => {
    const failures: string[] = [];
    for (const c of cases.cases) {
      const got = tok.encode(c.text);
      if (got.length !== c.ids.length || got.some((v, i) => v !== c.ids[i])) failures.push(`${JSON.stringify(c.text)}\n  want ${JSON.stringify(c.ids)}\n  got  ${JSON.stringify(got)}`);
    }
    expect(failures, failures.join("\n")).toEqual([]);
  });

  it("decodes all 200 cases exactly like the HF tokenizer", () => {
    const failures: string[] = [];
    for (const c of cases.cases) {
      const got = tok.decode(c.ids);
      if (got !== c.decoded) failures.push(`${JSON.stringify(c.ids.slice(0, 12))}\n  want ${JSON.stringify(c.decoded)}\n  got  ${JSON.stringify(got)}`);
    }
    expect(failures, failures.join("\n")).toEqual([]);
  });

  it("matches the fixture prompt's 32 token ids", () => {
    const p = readJson<{ text: string; token_ids: number[] }>(join(FIXTURES, "prompt.json"));
    expect(tok.encode(p.text)).toEqual(p.token_ids);
  });

  it("round-trips unicode through encode/decode", () => {
    for (const s of ["héllo wörld", "日本語 テスト", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "tab\tnew\nline", "<|im_start|>x<|im_end|>"]) expect(tok.decode(tok.encode(s))).toBe(s);
  });
});

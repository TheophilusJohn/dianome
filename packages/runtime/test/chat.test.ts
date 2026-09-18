import { describe, expect, it } from "vitest";
import { join } from "node:path";
import { renderChat, type ChatMessage } from "../src/chat";
import { Tokenizer, type TokenizerJson } from "../src/tokenizer";
import { FIXTURES, readJson, tokenizerJsonPath } from "./helpers/paths";

interface Cases { count: number; cases: { messages: ChatMessage[]; add_generation_prompt: boolean; text: string; ids: number[] }[] }

describe("chat template (Qwen2.5 ChatML) vs HF apply_chat_template", () => {
  const cases = readJson<Cases>(join(FIXTURES, "chat_template_cases.json"));
  const tok = new Tokenizer(readJson<TokenizerJson>(tokenizerJsonPath()));

  it("has 50 cases", () => { expect(cases.count).toBe(50); expect(cases.cases.length).toBe(50); });

  it("renders all 50 message lists to the HF string", () => {
    const failures: string[] = [];
    for (const [i, c] of cases.cases.entries()) {
      const got = renderChat(c.messages, { addGenerationPrompt: c.add_generation_prompt });
      if (got !== c.text) failures.push(`case ${i}: want ${JSON.stringify(c.text)}\n  got  ${JSON.stringify(got)}`);
    }
    expect(failures, failures.join("\n")).toEqual([]);
  });

  it("tokenises all 50 rendered strings to the HF ids", () => {
    const failures: string[] = [];
    for (const [i, c] of cases.cases.entries()) {
      const got = tok.encode(renderChat(c.messages, { addGenerationPrompt: c.add_generation_prompt }));
      if (got.length !== c.ids.length || got.some((v, j) => v !== c.ids[j])) failures.push(`case ${i}: want ${JSON.stringify(c.ids)}\n  got  ${JSON.stringify(got)}`);
    }
    expect(failures, failures.join("\n")).toEqual([]);
  });

  it("rejects unknown roles", () => {
    expect(() => renderChat([{ role: "tool" as ChatMessage["role"], content: "x" }])).toThrow(/unsupported chat role/);
  });
});

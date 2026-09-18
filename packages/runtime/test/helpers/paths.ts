import { readFileSync, readdirSync, existsSync } from "node:fs";
import { resolve, join } from "node:path";
import { homedir } from "node:os";

export const REPO = resolve(import.meta.dirname, "../../../..");
export const FIXTURES = join(REPO, "fixtures/qwen2.5-0.5b-instruct");

/** tokenizer.json from the HF cache (the store carries the same bytes; the unit test avoids the chunk store). */
export function tokenizerJsonPath(): string {
  const snaps = join(homedir(), ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots");
  for (const s of readdirSync(snaps)) { const p = join(snaps, s, "tokenizer.json"); if (existsSync(p)) return p; }
  throw new Error("tokenizer.json not found in the HF cache");
}

export function readJson<T = unknown>(p: string): T { return JSON.parse(readFileSync(p, "utf8")) as T; }

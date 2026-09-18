// A tiny schema-shaped manifest, stored the way Phase 1 stores it: canonical JSON (sorted keys, no whitespace)
// at manifests/<id>/latest.json and manifests/<id>/<sha256>.json.
import { env } from "cloudflare:test";
import { sha256Hex } from "../src/http";

export const MODEL_ID = "test-model";

export function canonical(obj: unknown): string {
  if (Array.isArray(obj)) return `[${obj.map(canonical).join(",")}]`;
  if (obj && typeof obj === "object") {
    return `{${Object.keys(obj).sort().map((k) => `${JSON.stringify(k)}:${canonical((obj as Record<string, unknown>)[k])}`).join(",")}}`;
  }
  return JSON.stringify(obj);
}

export async function seedManifest(id = MODEL_ID): Promise<{ bytes: Uint8Array; sha: string }> {
  const manifest = {
    schema: 1, id, source: { repo: "local:test", revision: "" }, chunk_size: 8388608,
    chunks: { ["a".repeat(64)]: { bytes: 1024 } },
    runtime: "other", files: [{ name: "x.bin", bytes: 1024, chunks: ["a".repeat(64)], segments: [{ chunk: "a".repeat(64), offset: 0, length: 1024 }] }],
  };
  const bytes = new TextEncoder().encode(canonical(manifest));
  const sha = await sha256Hex(bytes);
  await env.STORE.put(`manifests/${id}/${sha}.json`, bytes, { httpMetadata: { contentType: "application/json", cacheControl: "public, max-age=31536000, immutable" } });
  await env.STORE.put(`manifests/${id}/latest.json`, bytes, { httpMetadata: { contentType: "application/json", cacheControl: "public, max-age=60" } });
  return { bytes, sha };
}

export const validReport = () => ({
  schema: 1, model: "qwen2.5-0.5b-instruct", variant: "q4", bytes: 323893760, chunks: 42, ms: 8420,
  source: "network", cache_hits: 0, browser: "chrome", webgpu: true,
});

export const validReportV2 = () => ({
  ...validReport(), schema: 2, source: "mixed", cache_hits: 17,
  bytes_per_second: 38467000, verify_ms: 210, transfer_ms: 0, quota_bytes: 10760000000, max_buffer_size: 4294967292, cache_mode: "per-site",
});

export const validSession = () => ({
  schema: 3, model: "qwen2.5-0.5b-instruct", variant: "q4", mode: "split", N: 12, L: 24, prompt_tokens: 40, new_tokens: 64,
  client_ms: 620.5, server_busy_ms: 710.2, rtt_ms: 3.8, tok_per_s: 41.7, plan_policy: "cost", cache_mode: "per-site", browser: "chrome", webgpu: true,
});

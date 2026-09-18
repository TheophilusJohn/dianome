// Tokenizer from the store: the manifest lists tokenizer.json as a plain file stream in the same chunk store
// (`manifest.tokenizer.files`), so its bytes are fetched from `${cdn}/chunks/<sha>` segments, hashed and
// assembled here. The SDK's stream() only yields weight groups, which is why this is separate.

import type { ModelManifest } from "dianome";
import { Tokenizer, type TokenizerJson } from "./tokenizer";

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const h = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function fetchStoreFile(cdn: string, manifest: ModelManifest, name: string, fetchImpl: typeof fetch = fetch): Promise<Uint8Array> {
  const file = manifest.tokenizer.files.find((f) => f.name === name);
  if (!file) throw new Error(`manifest ${manifest.id} has no tokenizer file ${name}`);
  const chunks = new Map<string, ArrayBuffer>();
  await Promise.all([...new Set(file.segments.map((s) => s.chunk))].map(async (sha) => {
    const r = await fetchImpl(`${cdn.replace(/\/+$/, "")}/chunks/${sha}`);
    if (!r.ok) throw new Error(`chunk ${sha.slice(0, 12)}: ${r.status}`);
    const buf = await r.arrayBuffer();
    if ((await sha256Hex(buf)) !== sha) throw new Error(`chunk ${sha.slice(0, 12)}: hash mismatch`);
    chunks.set(sha, buf);
  }));
  const total = file.segments.reduce((n, s) => n + s.length, 0);
  const out = new Uint8Array(total);
  let pos = 0;
  for (const s of file.segments) { out.set(new Uint8Array(chunks.get(s.chunk)!, s.offset, s.length), pos); pos += s.length; }
  return out;
}

export async function loadTokenizer(cdn: string, manifest: ModelManifest, fetchImpl: typeof fetch = fetch): Promise<Tokenizer> {
  const bytes = await fetchStoreFile(cdn, manifest, "tokenizer.json", fetchImpl);
  return new Tokenizer(JSON.parse(new TextDecoder().decode(bytes)) as TokenizerJson);
}

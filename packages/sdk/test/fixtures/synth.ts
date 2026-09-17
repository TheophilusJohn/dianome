// Synthetic manifests with real chunk bytes and real SHA-256s, built the way ingest lays streams out: every entry
// 256-byte aligned within its group stream, streams cut into CHUNK_SIZE chunks (chunk boundaries restart per
// stream), tied lm_head pointing into embed's chunks. Small by default; `chunkSize` can be shrunk so a group spans
// several chunks without allocating 8 MB per chunk.

import { CHUNK_SIZE, type FilesManifest, type ModelManifest, type Segment, type Storage } from "../../src/manifest";
import { sha256 } from "../../src/sha";

export const ALIGN = 256;

/** Deterministic bytes: a tiny xorshift seeded by `seed` (fast, no crypto needed). */
export function fill(n: number, seed: number): Uint8Array {
  const out = new Uint8Array(n);
  let x = (seed | 0) || 1;
  for (let i = 0; i < n; i++) {
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    out[i] = x & 0xff;
  }
  return out;
}

const pad = (n: number): number => Math.ceil(n / ALIGN) * ALIGN;

export interface EntryDef { name: string; role: string; shape: number[]; storage: Storage; seed: number }

export interface StreamLayout { bytes: Uint8Array; chunks: string[]; entrySegments: Map<string, Segment[]> }

/** Serializes entries into one stream: each entry padded to 256, parts already padded per the storage's part table. */
export function layoutStream(entries: { name: string; length: number; seed: number }[]): { bytes: Uint8Array; offsets: Map<string, number> } {
  const offsets = new Map<string, number>();
  let total = 0;
  for (const e of entries) { offsets.set(e.name, total); total += pad(e.length); }
  const bytes = new Uint8Array(total);
  for (const e of entries) bytes.set(fill(e.length, e.seed), offsets.get(e.name)!);
  return { bytes, offsets };
}

export async function chunkStream(bytes: Uint8Array, chunkSize: number, store: Map<string, Uint8Array>): Promise<{ shas: string[]; segFor: (offset: number, length: number) => Segment[] }> {
  const shas: string[] = [];
  const lens: number[] = [];
  for (let off = 0; off < bytes.length; off += chunkSize) {
    const piece = bytes.slice(off, Math.min(off + chunkSize, bytes.length));
    const sha = await sha256(piece);
    shas.push(sha); lens.push(piece.length);
    store.set(sha, piece);
  }
  const segFor = (offset: number, length: number): Segment[] => {
    const out: Segment[] = [];
    let remaining = length, pos = offset;
    while (remaining > 0) {
      const ci = Math.floor(pos / chunkSize);
      const within = pos - ci * chunkSize;
      const take = Math.min(remaining, lens[ci]! - within);
      out.push({ chunk: shas[ci]!, offset: within, length: take });
      pos += take; remaining -= take;
    }
    return out;
  };
  return { shas, segFor };
}

/** Byte length of an entry from its storage's part table (or fp16 element count). */
export function entryLength(storage: Storage, shape: number[]): number {
  if (storage.kind === "fp16") return shape.reduce((a, b) => a * b, 1) * 2;
  const parts = Object.values(storage.parts);
  return Math.max(...parts.map((p) => p.offset + p.length));
}

export function q4Storage(out: number, inn: number): Storage {
  const w = (out * inn) / 2, s = out * (inn / 128) * 2, z = out * (inn / 128);
  return { kind: "q4", group_size: 128, parts: { weights: { offset: 0, length: w }, scales: { offset: pad(w), length: s }, zeros: { offset: pad(w) + pad(s), length: z } } };
}
export function q8Storage(out: number, inn: number): Storage {
  const w = out * inn, s = out * 2;
  return { kind: "q8", parts: { weights: { offset: 0, length: w }, scales: { offset: pad(w), length: s } } };
}

export interface SynthModel {
  manifest: ModelManifest;
  chunks: Map<string, Uint8Array>;
  /** Raw (unpadded) entry bytes per variant/entry name, for assertions. */
  raw: Map<string, Uint8Array>;
}

export interface SynthOptions {
  id?: string;
  layers?: number;
  hidden?: number;
  vocab?: number;
  chunkSize?: number;
  variants?: ("fp16" | "q4")[];
  tie?: boolean;
}

/**
 * A tiny decoder-only model: embed (fp16 in fp16, q8 in q4), L layers (norm fp16 + q_proj fp16/q4 + bias fp16),
 * final_norm, tied lm_head. Group streams are laid out and chunked exactly like ingest does.
 */
export async function synthModel(o: SynthOptions = {}): Promise<SynthModel> {
  const id = o.id ?? "synth-model", L = o.layers ?? 2, H = o.hidden ?? 256, V = o.vocab ?? 64, chunkSize = o.chunkSize ?? CHUNK_SIZE;
  const variants = o.variants ?? ["fp16", "q4"];
  const tie = o.tie ?? true;
  const chunks = new Map<string, Uint8Array>();
  const raw = new Map<string, Uint8Array>();
  const table: Record<string, { bytes: number }> = {};
  const out: ModelManifest = {
    schema: 1, id, source: { repo: "local:synth", revision: "" }, family: "qwen2",
    config: { hidden_size: H, num_hidden_layers: L, num_attention_heads: 2, num_key_value_heads: 1, intermediate_size: H * 2, vocab_size: V, rms_norm_eps: 1e-6, rope_theta: 10000, tie_word_embeddings: tie, max_position_embeddings: 1024 },
    tokenizer: { files: [] }, chunk_size: CHUNK_SIZE, chunks: table, variants: { fp16: { bytes: 0, groups: [] } },
  };
  const tokBytes = new TextEncoder().encode(JSON.stringify({ model_type: "qwen2", vocab: V }));
  const tokSha = await sha256(tokBytes);
  chunks.set(tokSha, tokBytes); table[tokSha] = { bytes: tokBytes.length };
  out.tokenizer.files.push({ name: "config.json", bytes: tokBytes.length, chunks: [tokSha], segments: [{ chunk: tokSha, offset: 0, length: tokBytes.length }] });

  for (const vn of variants) {
    const groups: ModelManifest["variants"]["fp16"]["groups"] = [];
    const embedStorage: Storage = vn === "fp16" ? { kind: "fp16" } : q8Storage(V, H);
    const defs: { name: string; entries: EntryDef[] }[] = [
      { name: "embed", entries: [{ name: "model.embed_tokens.weight", role: "embed_tokens.weight", shape: [V, H], storage: embedStorage, seed: 11 }] },
    ];
    for (let l = 0; l < L; l++) {
      defs.push({ name: `layer.${l}`, entries: [
        { name: `model.layers.${l}.input_layernorm.weight`, role: "input_layernorm.weight", shape: [H], storage: { kind: "fp16" }, seed: 100 + l },
        { name: `model.layers.${l}.self_attn.q_proj.weight`, role: "attn.q_proj.weight", shape: [H, H], storage: vn === "fp16" ? { kind: "fp16" } : q4Storage(H, H), seed: 200 + l },
        { name: `model.layers.${l}.self_attn.q_proj.bias`, role: "attn.q_proj.bias", shape: [H], storage: { kind: "fp16" }, seed: 300 + l },
      ] });
    }
    defs.push({ name: "final_norm", entries: [{ name: "model.norm.weight", role: "norm.weight", shape: [H], storage: { kind: "fp16" }, seed: 400 }] });
    let vbytes = 0;
    let embedSegs: Segment[] = [];
    for (const d of defs) {
      const lay = layoutStream(d.entries.map((e) => ({ name: e.name, length: entryLength(e.storage, e.shape), seed: e.seed })));
      const { shas, segFor } = await chunkStream(lay.bytes, chunkSize, chunks);
      for (const sha of shas) table[sha] = { bytes: chunks.get(sha)!.length };
      const entries = d.entries.map((e) => {
        const len = entryLength(e.storage, e.shape);
        raw.set(`${vn}/${e.name}`, fill(len, e.seed));
        return { name: e.name, role: e.role, shape: e.shape, storage: e.storage, segments: segFor(lay.offsets.get(e.name)!, len) };
      });
      if (d.name === "embed") embedSegs = entries[0]!.segments;
      groups.push({ name: d.name, bytes: lay.bytes.length, chunks: shas, entries });
      vbytes += lay.bytes.length;
    }
    if (tie) {
      groups.push({ name: "lm_head", bytes: 0, chunks: [], tied: true, entries: [{ name: "lm_head.weight", role: "lm_head.weight", shape: [V, H], storage: embedStorage, segments: embedSegs, tied: true }] });
      raw.set(`${vn}/lm_head.weight`, raw.get(`${vn}/model.embed_tokens.weight`)!);
    } else {
      const lay = layoutStream([{ name: "lm_head.weight", length: entryLength({ kind: "fp16" }, [V, H]), seed: 500 }]);
      const { shas, segFor } = await chunkStream(lay.bytes, chunkSize, chunks);
      for (const sha of shas) table[sha] = { bytes: chunks.get(sha)!.length };
      raw.set(`${vn}/lm_head.weight`, fill(V * H * 2, 500));
      groups.push({ name: "lm_head", bytes: lay.bytes.length, chunks: shas, entries: [{ name: "lm_head.weight", role: "lm_head.weight", shape: [V, H], storage: { kind: "fp16" }, segments: segFor(0, V * H * 2) }] });
      vbytes += lay.bytes.length;
    }
    out.variants[vn] = { bytes: vbytes, groups };
  }
  return { manifest: out, chunks, raw };
}

export interface SynthFiles { manifest: FilesManifest; chunks: Map<string, Uint8Array>; raw: Map<string, Uint8Array> }

export async function synthFiles(files: { name: string; length: number; seed: number }[], o: { id?: string; runtime?: FilesManifest["runtime"]; chunkSize?: number } = {}): Promise<SynthFiles> {
  const chunks = new Map<string, Uint8Array>();
  const raw = new Map<string, Uint8Array>();
  const table: Record<string, { bytes: number }> = {};
  const list: FilesManifest["files"] = [];
  for (const f of files) {
    const bytes = fill(f.length, f.seed);
    raw.set(f.name, bytes);
    const { shas, segFor } = await chunkStream(bytes, o.chunkSize ?? CHUNK_SIZE, chunks);
    for (const sha of shas) table[sha] = { bytes: chunks.get(sha)!.length };
    list.push({ name: f.name, bytes: f.length, chunks: shas, segments: f.length ? segFor(0, f.length) : [] });
  }
  return { manifest: { schema: 1, id: o.id ?? "synth-files", source: { repo: "local:synth", revision: "" }, runtime: o.runtime ?? "other", chunk_size: CHUNK_SIZE, chunks: table, files: list }, chunks, raw };
}

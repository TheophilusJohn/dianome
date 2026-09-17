// Entries from chunks. A segment that fits in one chunk becomes a view into that chunk's buffer (no copy); an entry
// spread over several chunks is concatenated once into a fresh buffer. Quantized parts are sub-views of the entry.
// Every entry starts 256-byte aligned in its stream (ingest pads), and chunk length is a multiple of 256, so the
// view's byteOffset is 256-aligned too; a concatenated entry starts at offset 0 of its own buffer.

import { DianomeError } from "./errors";
import type { Segment } from "./manifest";
import type { EntrySpec, EntryStorage, PlanGroup } from "./plan";

export const ALIGN = 256;

export interface EntryParts { weights: Uint8Array; scales: Uint8Array; zeros?: Uint8Array }

export interface LoadedEntry {
  name: string;
  role: string;
  shape: number[];
  storage: EntryStorage;
  /**
   * The entry's bytes. May alias a chunk buffer shared with other entries: treat it as read-only. Copy it
   * (`bytes.slice()`) before transferring its buffer to a worker or mutating it.
   */
  bytes: Uint8Array;
  /** Sub-views of `bytes` for q8/q4 storage. */
  parts?: EntryParts;
  /** True when `bytes` was concatenated from several chunks (owns its buffer); false when it aliases a chunk. */
  copied: boolean;
  tied?: true;
}

export type ChunkBuffers = Pick<Map<string, ArrayBuffer>, "get">;

function bufFor(seg: Segment, chunks: ChunkBuffers, entry: string): ArrayBuffer {
  const buf = chunks.get(seg.chunk);
  if (!buf) throw new DianomeError("chunk_missing", `${entry}: chunk ${seg.chunk.slice(0, 12)} not loaded`);
  if (seg.offset + seg.length > buf.byteLength) throw new DianomeError("manifest_invalid", `${entry}: segment exceeds chunk ${seg.chunk.slice(0, 12)}`);
  return buf;
}

/** Bytes for an ordered list of segments: a view when there is one segment, otherwise one concatenation. */
export function assembleBytes(segments: Segment[], chunks: ChunkBuffers, entry = "entry"): { bytes: Uint8Array; copied: boolean } {
  if (segments.length === 0) throw new DianomeError("manifest_invalid", `${entry}: no segments`);
  if (segments.length === 1) {
    const s = segments[0]!;
    return { bytes: new Uint8Array(bufFor(s, chunks, entry), s.offset, s.length), copied: false };
  }
  const total = segments.reduce((n, s) => n + s.length, 0);
  const out = new Uint8Array(new ArrayBuffer(total));
  let pos = 0;
  for (const s of segments) {
    out.set(new Uint8Array(bufFor(s, chunks, entry), s.offset, s.length), pos);
    pos += s.length;
  }
  return { bytes: out, copied: true };
}

function partView(bytes: Uint8Array, p: { offset: number; length: number }, entry: string, what: string): Uint8Array {
  if (p.offset + p.length > bytes.byteLength) throw new DianomeError("manifest_invalid", `${entry}: part ${what} exceeds entry`);
  return bytes.subarray(p.offset, p.offset + p.length);
}

export function assembleEntry(spec: EntrySpec, chunks: ChunkBuffers): LoadedEntry {
  const { bytes, copied } = assembleBytes(spec.segments, chunks, spec.name);
  const e: LoadedEntry = { name: spec.name, role: spec.role, shape: spec.shape, storage: spec.storage, bytes, copied };
  if (spec.tied) e.tied = true;
  const st = spec.storage;
  if (st.kind === "q8") {
    e.parts = { weights: partView(bytes, st.parts.weights, spec.name, "weights"), scales: partView(bytes, st.parts.scales, spec.name, "scales") };
  } else if (st.kind === "q4") {
    e.parts = {
      weights: partView(bytes, st.parts.weights, spec.name, "weights"),
      scales: partView(bytes, st.parts.scales, spec.name, "scales"),
      zeros: partView(bytes, st.parts.zeros, spec.name, "zeros"),
    };
  }
  return e;
}

export function assembleGroup(group: PlanGroup, chunks: ChunkBuffers): Map<string, LoadedEntry> {
  const out = new Map<string, LoadedEntry>();
  for (const spec of group.entries) out.set(spec.name, assembleEntry(spec, chunks));
  return out;
}

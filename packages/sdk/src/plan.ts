// A LoadPlan is the manifest reduced to what the loader needs: ordered groups, each with the chunks it needs and
// the entries it yields, plus the flat download order. Model variants and files manifests both map onto it.

import { DianomeError } from "./errors";
import type { Entry, FilesManifest, ModelManifest, Segment, Storage, VariantName } from "./manifest";

export type EntryStorage = Storage | { kind: "raw" };

export interface EntrySpec {
  name: string;
  role: string;
  shape: number[];
  storage: EntryStorage;
  segments: Segment[];
  tied?: true;
}

export interface PlanGroup {
  name: string;
  index: number;
  /** Chunks this group needs (its own stream plus anything its entries point into), in download order. */
  needs: string[];
  /** Bytes of the group's own stream (0 for a tied group). */
  bytes: number;
  entries: EntrySpec[];
  tied: boolean;
}

export interface PlanChunk { sha: string; bytes: number; group: string; groupIndex: number }

export interface LoadPlan {
  modelId: string;
  /** Variant name for a model manifest, "files" for a files manifest (goes into the telemetry `variant`). */
  label: string;
  groups: PlanGroup[];
  /** Flat download order: manifest group order, chunk order within a group, each chunk once. */
  order: PlanChunk[];
  bytesTotal: number;
  /** Index of the last group that needs each chunk, so the loader can drop its own reference after that group. */
  lastUse: Map<string, number>;
}

function finish(modelId: string, label: string, groups: PlanGroup[], chunkBytes: (sha: string) => number): LoadPlan {
  const seen = new Set<string>();
  const order: PlanChunk[] = [];
  const lastUse = new Map<string, number>();
  for (const g of groups) {
    for (const sha of g.needs) {
      lastUse.set(sha, g.index);
      if (seen.has(sha)) continue;
      seen.add(sha);
      order.push({ sha, bytes: chunkBytes(sha), group: g.name, groupIndex: g.index });
    }
  }
  return { modelId, label, groups, order, bytesTotal: order.reduce((s, c) => s + c.bytes, 0), lastUse };
}

function needsOf(chunks: string[], entries: { segments: Segment[] }[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  const add = (sha: string) => { if (!seen.has(sha)) { seen.add(sha); out.push(sha); } };
  for (const sha of chunks) add(sha);
  for (const e of entries) for (const s of e.segments) add(s.chunk);
  return out;
}

export function planVariant(manifest: ModelManifest, variant: VariantName): LoadPlan {
  const v = manifest.variants[variant];
  if (!v) throw new DianomeError("manifest_invalid", `variant ${variant} not in manifest ${manifest.id}`);
  const groups: PlanGroup[] = v.groups.map((g, index) => ({
    name: g.name, index, bytes: g.bytes, tied: g.tied === true,
    needs: needsOf(g.chunks, g.entries),
    entries: g.entries.map((e: Entry): EntrySpec => ({ name: e.name, role: e.role, shape: e.shape, storage: e.storage, segments: e.segments, ...(e.tied ? { tied: true as const } : {}) })),
  }));
  return finish(manifest.id, variant, groups, (sha) => manifest.chunks[sha]!.bytes);
}

/** One group per file (in manifest order, or the order of `only` when given), each with a single raw entry named by its path. */
export function planFiles(manifest: FilesManifest, only?: string[]): LoadPlan {
  let files = manifest.files;
  if (only) {
    const byName = new Map(files.map((f) => [f.name, f]));
    files = only.map((n) => {
      const f = byName.get(n);
      if (!f) throw new DianomeError("manifest_invalid", `file ${n} not in manifest ${manifest.id}`);
      return f;
    });
  }
  const groups: PlanGroup[] = files.map((f, index) => ({
    name: f.name, index, bytes: f.bytes, tied: false,
    needs: needsOf(f.chunks, [f]),
    entries: [{ name: f.name, role: "file", shape: [f.bytes], storage: { kind: "raw" }, segments: f.segments }],
  }));
  return finish(manifest.id, "files", groups, (sha) => manifest.chunks[sha]!.bytes);
}

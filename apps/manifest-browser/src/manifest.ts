// Types mirroring schemas/manifest.v1.json, plus the derived statistics the page shows.

export type Sha256 = string;
export interface Segment { chunk: Sha256; offset: number; length: number }
export interface Part { offset: number; length: number }
export type Storage =
  | { kind: "fp16" }
  | { kind: "q8"; parts: { weights: Part; scales: Part } }
  | { kind: "q4"; group_size: 128; parts: { weights: Part; scales: Part; zeros: Part } };
export interface Entry { name: string; role: string; shape: number[]; storage: Storage; segments: Segment[]; tied?: true }
export interface Group { name: string; bytes: number; chunks: Sha256[]; entries: Entry[]; tied?: true }
export interface Variant { bytes: number; groups: Group[] }
export interface FileEntry { name: string; bytes: number; chunks: Sha256[]; segments: Segment[] }
export interface ModelConfig {
  hidden_size: number; num_hidden_layers: number; num_attention_heads: number; num_key_value_heads: number;
  intermediate_size: number; vocab_size: number; rms_norm_eps: number; rope_theta: number;
  tie_word_embeddings: boolean; max_position_embeddings: number;
}
export interface ManifestBase {
  schema: 1; id: string; source: { repo: string; revision: string }; chunk_size: 8388608;
  chunks: Record<Sha256, { bytes: number }>;
}
export interface ModelManifest extends ManifestBase {
  family: string; config: ModelConfig; tokenizer: { files: FileEntry[] };
  variants: Partial<Record<"fp16" | "q8" | "q4", Variant>>;
}
export interface FilesManifest extends ManifestBase { runtime: "webllm" | "transformersjs" | "other"; files: FileEntry[] }
export type Manifest = ModelManifest | FilesManifest;

export const VARIANT_ORDER = ["fp16", "q8", "q4"] as const;
export type VariantName = (typeof VARIANT_ORDER)[number];

export function isModel(m: Manifest): m is ModelManifest {
  return "variants" in m;
}

export function variantNames(m: ModelManifest): VariantName[] {
  return VARIANT_ORDER.filter((v) => m.variants[v] !== undefined);
}

export interface DedupStats {
  listedChunks: number; listedBytes: number;
  uniqueChunks: number; uniqueBytes: number;
  sharedChunks: number; sharedBytes: number;
  bytesSaved: number;
  pairs: { a: VariantName; b: VariantName; chunks: number; bytes: number }[];
  sharedGroups: { group: string; variants: VariantName[]; chunks: number; bytes: number }[];
}

/** Chunk sharing between variants: a chunk is shared when more than one variant lists it. */
export function dedupStats(m: ModelManifest): DedupStats {
  const size = (c: Sha256) => m.chunks[c]?.bytes ?? 0;
  const names = variantNames(m);
  const lists = new Map<VariantName, Sha256[]>();
  for (const v of names) lists.set(v, m.variants[v]!.groups.flatMap((g) => g.chunks));
  const owners = new Map<Sha256, Set<VariantName>>();
  let listedChunks = 0, listedBytes = 0;
  for (const [v, cs] of lists) for (const c of cs) {
    listedChunks++; listedBytes += size(c);
    (owners.get(c) ?? owners.set(c, new Set()).get(c)!).add(v);
  }
  let uniqueBytes = 0, sharedChunks = 0, sharedBytes = 0;
  for (const [c, vs] of owners) { uniqueBytes += size(c); if (vs.size > 1) { sharedChunks++; sharedBytes += size(c); } }
  const pairs: DedupStats["pairs"] = [];
  for (let i = 0; i < names.length; i++) for (let j = i + 1; j < names.length; j++) {
    const a = names[i]!, b = names[j]!;
    const sb = new Set(lists.get(b));
    const common = new Set(lists.get(a)!.filter((c) => sb.has(c)));
    pairs.push({ a, b, chunks: common.size, bytes: [...common].reduce((s, c) => s + size(c), 0) });
  }
  const sharedGroups: DedupStats["sharedGroups"] = [];
  const first = m.variants[names[0]!]!;
  first.groups.forEach((g, gi) => {
    // Any set of variants whose chunk lists for this group are identical (not anchored on one variant).
    const byList = new Map<string, VariantName[]>();
    for (const v of names) {
      const grp = m.variants[v]!.groups[gi];
      if (!grp || grp.chunks.length === 0) continue;
      const key = grp.chunks.join(",");
      (byList.get(key) ?? byList.set(key, []).get(key)!).push(v);
    }
    for (const [key, same] of byList) {
      if (same.length < 2) continue;
      const chunks = key.split(",");
      sharedGroups.push({ group: g.name, variants: same, chunks: chunks.length, bytes: chunks.reduce((s, c) => s + size(c), 0) });
    }
  });
  return { listedChunks, listedBytes, uniqueChunks: owners.size, uniqueBytes, sharedChunks, sharedBytes, bytesSaved: listedBytes - uniqueBytes, pairs, sharedGroups };
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const mib = n / (1024 * 1024);
  return mib >= 1024 ? `${(mib / 1024).toFixed(2)} GiB` : `${mib.toFixed(1)} MiB`;
}

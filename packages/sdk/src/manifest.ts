// Manifest fetch + validation. The validator hand-mirrors schemas/manifest.v1.json (the SDK has no runtime
// dependencies, so no JSON Schema library); test/manifest.test.ts cross-checks it against the real schema with ajv.
// Types are generated from the schema into manifest.gen.ts.

import { AbortedError, isAbort, ManifestError } from "./errors";
import type { Entry, File, FilesManifest, Group, Manifest, ModelManifest, Part, Segment, Storage, Variant } from "./manifest.gen";
import { SHA_RE } from "./sha";

export type { Config, Entry, File, FilesManifest, Group, Manifest, ManifestBase, ModelManifest, Part, Segment, Storage, Variant } from "./manifest.gen";

export const CHUNK_SIZE = 8388608;
export const VARIANTS = ["fp16", "q8", "q4"] as const;
export type VariantName = (typeof VARIANTS)[number];
export const RUNTIMES = ["webllm", "transformersjs", "other"] as const;
export const ROLES_Q = ["weights", "scales", "zeros"] as const;

const ID_RE = /^[a-z0-9][a-z0-9._-]*$/;
const GROUP_RE = /^(embed|layer\.[0-9]+|final_norm|lm_head)$/;

class Ctx {
  constructor(readonly path: string) {}
  at(k: string | number): Ctx { return new Ctx(typeof k === "number" ? `${this.path}[${k}]` : `${this.path}.${k}`); }
  fail(msg: string): never { throw new ManifestError("manifest_invalid", `${this.path}: ${msg}`); }
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isInt = (v: unknown, min: number, max = Number.MAX_SAFE_INTEGER): v is number => typeof v === "number" && Number.isInteger(v) && v >= min && v <= max;
function int(v: unknown, c: Ctx, min: number, max = Number.MAX_SAFE_INTEGER): number {
  if (!isInt(v, min, max)) c.fail(`expected integer in [${min}, ${max}]`);
  return v;
}

function obj(v: unknown, c: Ctx, required: string[], allowed: string[]): Record<string, unknown> {
  if (!isObj(v)) c.fail("expected object");
  for (const k of required) if (!(k in v)) c.fail(`missing ${k}`);
  for (const k of Object.keys(v)) if (!allowed.includes(k)) c.fail(`unknown property ${k}`);
  return v;
}
function str(v: unknown, c: Ctx, re?: RegExp, minLength = 0): string {
  if (typeof v !== "string") c.fail("expected string");
  if (v.length < minLength) c.fail("empty string");
  if (re && !re.test(v)) c.fail(`does not match ${re}`);
  return v;
}
function arr(v: unknown, c: Ctx, minItems = 0): unknown[] {
  if (!Array.isArray(v)) c.fail("expected array");
  if (v.length < minItems) c.fail(`expected at least ${minItems} items`);
  return v;
}

function segment(v: unknown, c: Ctx): Segment {
  const o = obj(v, c, ["chunk", "offset", "length"], ["chunk", "offset", "length"]);
  return { chunk: str(o.chunk, c.at("chunk"), SHA_RE), offset: int(o.offset, c.at("offset"), 0, CHUNK_SIZE - 1), length: int(o.length, c.at("length"), 1, CHUNK_SIZE) };
}
function part(v: unknown, c: Ctx): Part {
  const o = obj(v, c, ["offset", "length"], ["offset", "length"]);
  const offset = int(o.offset, c.at("offset"), 0);
  if (offset % 256 !== 0) c.at("offset").fail("must be a multiple of 256");
  return { offset, length: int(o.length, c.at("length"), 1) };
}
function storage(v: unknown, c: Ctx): Storage {
  if (!isObj(v)) return c.fail("expected object");
  switch (v.kind) {
    case "fp16": obj(v, c, ["kind"], ["kind"]); return { kind: "fp16" };
    case "q8": {
      obj(v, c, ["kind", "parts"], ["kind", "parts"]);
      const p = obj(v.parts, c.at("parts"), ["weights", "scales"], ["weights", "scales"]);
      return { kind: "q8", parts: { weights: part(p.weights, c.at("parts.weights")), scales: part(p.scales, c.at("parts.scales")) } };
    }
    case "q4": {
      obj(v, c, ["kind", "group_size", "parts"], ["kind", "group_size", "parts"]);
      if (v.group_size !== 128) c.at("group_size").fail("must be 128");
      const p = obj(v.parts, c.at("parts"), ["weights", "scales", "zeros"], ["weights", "scales", "zeros"]);
      return { kind: "q4", group_size: 128, parts: { weights: part(p.weights, c.at("parts.weights")), scales: part(p.scales, c.at("parts.scales")), zeros: part(p.zeros, c.at("parts.zeros")) } };
    }
    default: return c.at("kind").fail("expected fp16 | q8 | q4");
  }
}
function entry(v: unknown, c: Ctx): Entry {
  const o = obj(v, c, ["name", "role", "shape", "storage", "segments"], ["name", "role", "shape", "storage", "segments", "tied"]);
  const shape = arr(o.shape, c.at("shape"), 1).map((d, i) => (isInt(d, 1) ? d : c.at("shape").at(i).fail("bad dim")));
  const e: Entry = {
    name: str(o.name, c.at("name"), undefined, 1),
    role: str(o.role, c.at("role"), undefined, 1),
    shape,
    storage: storage(o.storage, c.at("storage")),
    segments: arr(o.segments, c.at("segments"), 1).map((s, i) => segment(s, c.at("segments").at(i))),
  };
  if ("tied" in o) { if (o.tied !== true) c.at("tied").fail("must be true"); e.tied = true; }
  return e;
}
function group(v: unknown, c: Ctx): Group {
  const o = obj(v, c, ["name", "bytes", "chunks", "entries"], ["name", "bytes", "chunks", "entries", "tied"]);
  const g: Group = {
    name: str(o.name, c.at("name"), GROUP_RE),
    bytes: int(o.bytes, c.at("bytes"), 0),
    chunks: arr(o.chunks, c.at("chunks")).map((s, i) => str(s, c.at("chunks").at(i), SHA_RE)),
    entries: arr(o.entries, c.at("entries"), 1).map((e, i) => entry(e, c.at("entries").at(i))),
  };
  if ("tied" in o) { if (o.tied !== true) c.at("tied").fail("must be true"); g.tied = true; }
  return g;
}
function variant(v: unknown, c: Ctx): Variant {
  const o = obj(v, c, ["bytes", "groups"], ["bytes", "groups"]);
  return { bytes: int(o.bytes, c.at("bytes"), 0), groups: arr(o.groups, c.at("groups"), 1).map((g, i) => group(g, c.at("groups").at(i))) };
}
function file(v: unknown, c: Ctx): File {
  const o = obj(v, c, ["name", "bytes", "chunks", "segments"], ["name", "bytes", "chunks", "segments"]);
  return {
    name: str(o.name, c.at("name"), undefined, 1),
    bytes: int(o.bytes, c.at("bytes"), 0),
    chunks: arr(o.chunks, c.at("chunks")).map((s, i) => str(s, c.at("chunks").at(i), SHA_RE)),
    segments: arr(o.segments, c.at("segments")).map((s, i) => segment(s, c.at("segments").at(i))),
  };
}

/**
 * Validates an untrusted value against manifest schema v1 and returns it typed. Beyond the schema it checks the
 * cross-references the schema cannot express: every referenced chunk exists in the table, every segment fits its
 * chunk, and every `bytes` sum matches the chunks listed.
 */
export function validateManifest(input: unknown): Manifest {
  const c = new Ctx("manifest");
  const top = obj(input, c, ["schema", "id", "source", "chunk_size", "chunks"],
    ["schema", "id", "source", "family", "config", "tokenizer", "runtime", "chunk_size", "chunks", "variants", "files"]);
  if (top.schema !== 1) c.at("schema").fail("unsupported schema (expected 1)");
  if (top.chunk_size !== CHUNK_SIZE) c.at("chunk_size").fail(`expected ${CHUNK_SIZE}`);
  const id = str(top.id, c.at("id"), ID_RE);
  const src = obj(top.source, c.at("source"), ["repo", "revision"], ["repo", "revision"]);
  const source = { repo: str(src.repo, c.at("source.repo"), undefined, 1), revision: str(src.revision, c.at("source.revision")) };

  if (!isObj(top.chunks)) return c.at("chunks").fail("expected object");
  const chunks: Record<string, { bytes: number }> = {};
  for (const [sha, cv] of Object.entries(top.chunks)) {
    const cc = c.at("chunks").at(sha);
    if (!SHA_RE.test(sha)) cc.fail("bad sha256 key");
    const o = obj(cv, cc, ["bytes"], ["bytes"]);
    chunks[sha] = { bytes: int(o.bytes, cc.at("bytes"), 1, CHUNK_SIZE) };
  }
  const chunkBytes = (sha: string, cc: Ctx): number => {
    const e = chunks[sha];
    if (!e) cc.fail(`chunk ${sha.slice(0, 12)} not in chunk table`);
    return e.bytes;
  };
  const checkSegments = (segs: Segment[], cc: Ctx): number => {
    let total = 0;
    segs.forEach((s, i) => {
      const n = chunkBytes(s.chunk, cc.at(i));
      if (s.offset + s.length > n) cc.at(i).fail("segment exceeds chunk");
      total += s.length;
    });
    return total;
  };
  const checkFile = (f: File, cc: Ctx): void => {
    const sum = f.chunks.reduce((s, sha, i) => s + chunkBytes(sha, cc.at("chunks").at(i)), 0);
    if (sum !== f.bytes) cc.at("bytes").fail(`is ${f.bytes} but chunks sum to ${sum}`);
    if (checkSegments(f.segments, cc.at("segments")) !== f.bytes) cc.at("segments").fail("segment lengths do not sum to bytes");
  };

  const hasModel = "variants" in top, hasFiles = "files" in top || "runtime" in top;
  if (hasModel && hasFiles) c.fail("a manifest has either variants or files, not both");

  if (hasModel) {
    for (const k of ["family", "config", "tokenizer"]) if (!(k in top)) c.fail(`model manifest missing ${k}`);
    const cfg = obj(top.config, c.at("config"),
      ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size", "rms_norm_eps", "rope_theta", "tie_word_embeddings", "max_position_embeddings"],
      Object.keys(top.config as object));
    for (const k of ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size", "max_position_embeddings"]) if (!isInt(cfg[k], 1)) c.at("config").at(k).fail("bad integer");
    for (const k of ["rms_norm_eps", "rope_theta"]) if (typeof cfg[k] !== "number" || !((cfg[k] as number) > 0)) c.at("config").at(k).fail("bad number");
    if (typeof cfg.tie_word_embeddings !== "boolean") c.at("config.tie_word_embeddings").fail("bad boolean");
    const tok = obj(top.tokenizer, c.at("tokenizer"), ["files"], ["files"]);
    const tokenizer = { files: arr(tok.files, c.at("tokenizer.files")).map((f, i) => file(f, c.at("tokenizer.files").at(i))) };
    tokenizer.files.forEach((f, i) => checkFile(f, c.at("tokenizer.files").at(i)));
    const vs = obj(top.variants, c.at("variants"), ["fp16"], [...VARIANTS]);
    const variants: ModelManifest["variants"] = { fp16: variant(vs.fp16, c.at("variants.fp16")) };
    if ("q8" in vs) variants.q8 = variant(vs.q8, c.at("variants.q8"));
    if ("q4" in vs) variants.q4 = variant(vs.q4, c.at("variants.q4"));
    for (const [vn, v] of Object.entries(variants)) {
      const vc = c.at("variants").at(vn);
      let vsum = 0;
      v.groups.forEach((g, gi) => {
        const gc = vc.at("groups").at(gi);
        const sum = g.chunks.reduce((s, sha, i) => s + chunkBytes(sha, gc.at("chunks").at(i)), 0);
        if (sum !== g.bytes) gc.at("bytes").fail(`is ${g.bytes} but chunks sum to ${sum}`);
        if (g.tied && g.chunks.length !== 0) gc.fail("tied group must list no chunks");
        vsum += g.bytes;
        g.entries.forEach((e, ei) => checkSegments(e.segments, gc.at("entries").at(ei).at("segments")));
      });
      if (vsum !== v.bytes) vc.at("bytes").fail(`is ${v.bytes} but groups sum to ${vsum}`);
    }
    const m: ModelManifest = {
      schema: 1, id, source, family: str(top.family, c.at("family"), undefined, 1), config: cfg as ModelManifest["config"],
      tokenizer, chunk_size: CHUNK_SIZE, chunks, variants,
    };
    return m;
  }
  if (hasFiles) {
    for (const k of ["runtime", "files"]) if (!(k in top)) c.fail(`files manifest missing ${k}`);
    if (!(RUNTIMES as readonly unknown[]).includes(top.runtime)) c.at("runtime").fail("expected webllm | transformersjs | other");
    const files = arr(top.files, c.at("files")).map((f, i) => file(f, c.at("files").at(i)));
    files.forEach((f, i) => checkFile(f, c.at("files").at(i)));
    const m: FilesManifest = { schema: 1, id, source, runtime: top.runtime as FilesManifest["runtime"], chunk_size: CHUNK_SIZE, chunks, files };
    return m;
  }
  return c.fail("manifest has neither variants nor files");
}

export function isModelManifest(m: Manifest): m is ModelManifest { return "variants" in m && m.variants !== undefined; }
export function isFilesManifest(m: Manifest): m is FilesManifest { return "files" in m && m.files !== undefined; }

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export interface FetchedManifest { manifest: Manifest; sha: string | null; etag: string | null }

/** GET `${api}/v1/models/<id>/manifest`, validate, and return the manifest with the immutable hash the API reports. */
export async function fetchManifest(api: string, id: string, opts: { signal?: AbortSignal; fetch?: FetchLike } = {}): Promise<FetchedManifest> {
  if (!ID_RE.test(id)) throw new ManifestError("manifest_invalid", `bad model id ${JSON.stringify(id)}`);
  const url = `${api.replace(/\/+$/, "")}/v1/models/${id}/manifest`;
  const f = opts.fetch ?? fetch;
  let res: Response;
  try {
    res = await f(url, { signal: opts.signal ?? null, headers: { Accept: "application/json" } });
  } catch (e) {
    if (opts.signal?.aborted || isAbort(e)) throw new AbortedError();
    throw new ManifestError("manifest_http", `manifest ${url}: ${(e as Error).message}`, { cause: e });
  }
  if (!res.ok) throw new ManifestError("manifest_http", `manifest ${url}: HTTP ${res.status}`);
  let body: unknown;
  try { body = await res.json(); } catch (e) {
    if (opts.signal?.aborted || isAbort(e)) throw new AbortedError();
    throw new ManifestError("manifest_invalid", `manifest ${url}: not JSON`, { cause: e });
  }
  const manifest = validateManifest(body);
  if (manifest.id !== id) throw new ManifestError("manifest_invalid", `manifest id ${manifest.id} does not match requested ${id}`);
  return { manifest, sha: res.headers.get("X-Dianome-Manifest-Sha"), etag: res.headers.get("ETag") };
}

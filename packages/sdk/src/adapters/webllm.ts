// `dianome/webllm`: pre-warms WebLLM's own Cache API caches from a Dianome MLC artifact, so `engine.reload()`
// finds every shard already cached and fetches nothing from the model URL.
//
//   import { prewarmWebLLM } from "dianome/webllm";
//   const modelUrl = "https://cdn.dianome.dev/mlc/qwen2.5-0.5b-instruct-mlc/";   // any URL; it is only a cache key
//   await prewarmWebLLM(new Dianome(), "qwen2.5-0.5b-instruct-mlc", { modelUrl });
//   const engine = await CreateMLCEngine("Qwen2.5-0.5B-Instruct-q4f16_1-MLC", { appConfig: { model_list: [{ model: modelUrl, model_id: "…", model_lib: "…" }] } });
//
// WebLLM keys its Cache API entries by the full URL `new URL(<file>, cleanModelUrl(modelUrl))`, where
// cleanModelUrl adds a trailing slash and then `resolve/main/` unless the URL already contains `/resolve/<rev>/`
// (Hugging Face layout). Shards, `ndarray-cache.json` / `tensor-cache.json` and `tokenizer.json` live under the
// cache named "webllm/model", `mlc-chat-config.json` under "webllm/config". The model library (wasm) is not part
// of the artifact and still comes from WebLLM's CDN.

import type { Dianome } from "../index";
import { isFilesManifest } from "../manifest";

export const WEBLLM_MODEL_CACHE = "webllm/model";
export const WEBLLM_CONFIG_CACHE = "webllm/config";

export interface PrewarmOptions {
  /** The `model` URL of the WebLLM model record; normalised to end with "/". */
  modelUrl: string;
  /** Only these files (default: every file of the artifact). */
  files?: string[];
  signal?: AbortSignal;
  onFile?: (file: string, bytes: number, source: string) => void;
  onProgress?: (p: { file: string; bytesDone: number; bytesTotal: number; source: string }) => void;
  /** Injected CacheStorage (tests). */
  caches?: CacheStorage;
}

export interface PrewarmResult { files: string[]; bytes: number; ms: number; source: string; skipped: string[] }

/** Mirrors WebLLM's cleanModelUrl(): trailing slash, then `resolve/main/` unless a `/resolve/<rev>/` segment exists. */
export function normalizeModelUrl(url: string): string {
  let u = url.endsWith("/") ? url : `${url}/`;
  if (!/.+\/resolve\/.+\//.test(u)) u += "resolve/main/";
  return new URL(u).href;
}

/** Cache API caches a file must be put into, keyed by WebLLM's URL for it. */
export function cacheTargets(file: string, modelUrl: string): { cache: string; url: string }[] {
  const url = new URL(file, normalizeModelUrl(modelUrl)).href;
  if (file === "mlc-chat-config.json") return [{ cache: WEBLLM_CONFIG_CACHE, url }, { cache: WEBLLM_MODEL_CACHE, url }];
  return [{ cache: WEBLLM_MODEL_CACHE, url }];
}

/** Files WebLLM will ask for: everything except repo housekeeping that it never requests. */
export function isWebLLMFile(name: string): boolean {
  return !/^(\.|README|LICENSE)/i.test(name);
}

export async function prewarmWebLLM(d: Dianome, artifactId: string, opts: PrewarmOptions): Promise<PrewarmResult> {
  const cs = opts.caches ?? caches;
  const manifest = await d.manifest(artifactId, opts.signal);
  if (!isFilesManifest(manifest)) throw new Error(`dianome/webllm: ${artifactId} is not a files artifact`);
  const wanted = (opts.files ?? manifest.files.map((f) => f.name)).filter(isWebLLMFile);
  const opened = new Map<string, Cache>();
  const open = async (name: string): Promise<Cache> => opened.get(name) ?? (opened.set(name, await cs.open(name)), opened.get(name)!);
  const t0 = Date.now();
  // Skip files every target already holds, then stream the rest one file at a time.
  const skipped: string[] = [];
  const todo: string[] = [];
  for (const f of wanted) {
    const targets = cacheTargets(f, opts.modelUrl);
    let present = true;
    for (const t of targets) if (!(await (await open(t.cache)).match(t.url))) { present = false; break; }
    (present ? skipped : todo).push(f);
  }
  let bytes = 0;
  let source = "none";
  if (todo.length > 0) {
    const gen = d.stream(artifactId, { files: todo, ...(opts.signal ? { signal: opts.signal } : {}), onProgress: (p) => opts.onProgress?.({ file: p.group, bytesDone: p.bytesDone, bytesTotal: p.bytesTotal, source: p.source }) });
    for (;;) {
      const n = await gen.next();
      if (n.done) { source = n.value.source; break; }
      const entry = n.value.entries.get(n.value.name)!;
      const body = entry.copied ? entry.bytes : entry.bytes.slice(); // a chunk-aliasing view would pin the whole chunk in the Response
      for (const t of cacheTargets(n.value.name, opts.modelUrl)) {
        await (await open(t.cache)).put(t.url, new Response(body as BodyInit, { headers: { "Content-Type": n.value.name.endsWith(".json") ? "application/json" : "application/octet-stream", "Content-Length": String(body.byteLength) } }));
      }
      bytes += body.byteLength;
      opts.onFile?.(n.value.name, body.byteLength, "streamed");
    }
  }
  return { files: todo, bytes, ms: Date.now() - t0, source, skipped };
}

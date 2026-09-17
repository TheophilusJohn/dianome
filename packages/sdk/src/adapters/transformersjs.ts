// `dianome/transformersjs`: a custom cache for Transformers.js so model files resolve from the Dianome store.
//
//   import { env } from "@huggingface/transformers";
//   import { dianomeCache } from "dianome/transformersjs";
//   env.useCustomCache = true;
//   env.customCache = dianomeCache(new Dianome(), "qwen2.5-0.5b-instruct-onnx");
//   const pipe = await pipeline("text-generation", "onnx-community/Qwen2.5-0.5B-Instruct", { dtype: "q4" });
//
// Transformers.js calls `match()` with its candidate keys (the local path `/models/<model>/<file>`, then the
// remote URL `https://huggingface.co/<model>/resolve/<rev>/<file>`) and `put()` after a network download. A key
// whose trailing path is a file of the artifact is answered from Dianome (verified chunks, per-site or cross-site
// cache); anything else falls through to an ordinary Cache API cache so unrelated files keep working.

import type { Dianome } from "../index";
import { isFilesManifest, type FilesManifest } from "../manifest";

export interface DianomeCacheOptions {
  /** Cache API cache used for keys the artifact does not cover (default "transformers-cache", Transformers.js's own). */
  fallbackCacheName?: string | null;
  /** Called per file served from Dianome with the load summary's source. */
  onFile?: (file: string, source: string) => void;
}

export interface TransformersCache {
  match(request: RequestInfo | URL): Promise<Response | undefined>;
  put(request: RequestInfo | URL, response: Response): Promise<void>;
  /** Files of the artifact, once the manifest has been fetched. */
  files(): Promise<string[]>;
}

export function keyPath(request: RequestInfo | URL): string {
  const s = typeof request === "string" ? request : request instanceof URL ? request.href : request.url;
  try { return decodeURIComponent(new URL(s, "http://x").pathname); } catch { return s; }
}

/** The artifact file whose name is the longest path-boundary suffix of `path`, or null. */
export function resolveFile(path: string, files: readonly string[]): string | null {
  let best: string | null = null;
  for (const f of files) {
    if ((path === f || path.endsWith(`/${f}`)) && (best === null || f.length > best.length)) best = f;
  }
  return best;
}

export function dianomeCache(d: Dianome, artifactId: string, opts: DianomeCacheOptions = {}): TransformersCache {
  let manifest: Promise<FilesManifest> | null = null;
  const files = async (): Promise<string[]> => (await getManifest()).files.map((f) => f.name);
  const getManifest = (): Promise<FilesManifest> => {
    manifest ??= d.manifest(artifactId).then((m) => {
      if (!isFilesManifest(m)) throw new Error(`dianome/transformersjs: ${artifactId} is not a files artifact`);
      return m;
    });
    manifest.catch(() => { manifest = null; });
    return manifest;
  };
  const fallbackName = opts.fallbackCacheName === undefined ? "transformers-cache" : opts.fallbackCacheName;
  const fallback = async (): Promise<Cache | null> => (fallbackName && typeof caches !== "undefined" ? caches.open(fallbackName) : null);

  return {
    files,
    async match(request) {
      const path = keyPath(request);
      const file = resolveFile(path, await files());
      if (file === null) return (await fallback())?.match(request);
      let bytes: Uint8Array | null = null;
      let source = "unknown";
      for await (const group of d.stream(artifactId, { files: [file] })) bytes = group.entries.get(file)!.bytes;
      source = d.lastSummary?.source ?? source;
      opts.onFile?.(file, source);
      // Transformers.js reads the body once and uses Content-Length for progress; the bytes may alias a chunk,
      // which is fine for a Response body it only reads.
      return new Response(bytes as BodyInit, { status: 200, headers: { "Content-Type": file.endsWith(".json") ? "application/json" : "application/octet-stream", "Content-Length": String(bytes!.byteLength), "X-Dianome-Source": source } });
    },
    async put(request, response) {
      const path = keyPath(request);
      if (resolveFile(path, await files()) !== null) return; // already served from Dianome; nothing to store
      const fb = await fallback();
      if (fb) await fb.put(request, response);
    },
  };
}

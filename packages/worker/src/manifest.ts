// Manifest endpoints, read through the R2 binding (no S3 keys).
//
// Store layout (Phase 1): manifests/<id>/latest.json and manifests/<id>/<sha256>.json hold the same
// canonical bytes (sorted keys, no whitespace); the immutable name is the sha256 of those bytes.
// The manifest has no `sha256` field of its own, so X-Dianome-Manifest-Sha is computed from the
// body and memoised in KV keyed by R2's ETag so a 304 never has to re-read the object.

import { error, json, sha256Hex } from "./http";
import type { Env } from "./types";

export const ID_RE = /^[a-z0-9][a-z0-9._-]*$/;
export const SHA_RE = /^[0-9a-f]{64}$/;
const CACHE_LATEST = "public, max-age=60";
const CACHE_IMMUTABLE = "public, max-age=31536000, immutable";
const SHA_MEMO_TTL = 7 * 24 * 3600;

export async function listModels(env: Env): Promise<Response> {
  const ids: string[] = [];
  let cursor: string | undefined;
  do {
    const page = await env.STORE.list({ prefix: "manifests/", delimiter: "/", cursor });
    for (const p of page.delimitedPrefixes) ids.push(p.slice("manifests/".length).replace(/\/$/, ""));
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);
  ids.sort();
  return json({ models: ids }, { headers: { "Cache-Control": CACHE_LATEST } });
}

/** GET /v1/models/:id/manifest — latest.json, ETag passthrough, 304 on If-None-Match, sha header. */
export async function latestManifest(request: Request, env: Env, ctx: ExecutionContext, id: string): Promise<Response> {
  if (!ID_RE.test(id)) return error(400, "bad_id");
  const key = `manifests/${id}/latest.json`;
  const obj = await env.STORE.get(key, { onlyIf: request.headers });
  if (obj === null) return error(404, "not_found");

  const headers = new Headers({ ETag: obj.httpEtag, "Cache-Control": CACHE_LATEST, "Content-Type": "application/json" });
  const memoKey = `msha:${id}:${obj.etag}`;

  if (!("body" in obj)) {
    // Precondition failed (If-None-Match matched): 304, no body. Still expose the sha when it is cheap.
    const memo = await env.STATS_CACHE.get(memoKey);
    if (memo) headers.set("X-Dianome-Manifest-Sha", memo);
    headers.delete("Content-Type");
    return new Response(null, { status: 304, headers });
  }

  let sha = await env.STATS_CACHE.get(memoKey);
  let body: ArrayBuffer | ReadableStream = obj.body;
  if (!sha) {
    const bytes = await obj.arrayBuffer();
    sha = await sha256Hex(bytes);
    body = bytes;
    ctx.waitUntil(env.STATS_CACHE.put(memoKey, sha, { expirationTtl: SHA_MEMO_TTL }));
  }
  headers.set("X-Dianome-Manifest-Sha", sha);
  headers.set("Content-Length", String(obj.size));
  return new Response(request.method === "HEAD" ? null : body, { status: 200, headers });
}

/** GET /v1/models/:id/manifest/:sha — the immutable manifest. */
export async function hashedManifest(request: Request, env: Env, id: string, sha: string): Promise<Response> {
  if (!ID_RE.test(id)) return error(400, "bad_id");
  if (!SHA_RE.test(sha)) return error(400, "bad_sha");
  const obj = await env.STORE.get(`manifests/${id}/${sha}.json`, { onlyIf: request.headers });
  if (obj === null) return error(404, "not_found");
  const headers = new Headers({ ETag: obj.httpEtag, "Cache-Control": CACHE_IMMUTABLE, "X-Dianome-Manifest-Sha": sha });
  if (!("body" in obj)) return new Response(null, { status: 304, headers });
  headers.set("Content-Type", "application/json");
  headers.set("Content-Length", String(obj.size));
  return new Response(request.method === "HEAD" ? null : obj.body, { status: 200, headers });
}

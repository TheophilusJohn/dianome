// POST /v1/telemetry/load: validate one load report and write one Analytics Engine point.
// Privacy: the IP address and user agent string are never read; only request.cf.country / colo are kept,
// and the timestamp we attach is rounded to the hour.

import { error } from "./http";
import { BROWSERS, LOAD_SOURCES, type Env, type LoadReport } from "./types";

export const MAX_BODY_BYTES = 4096;
export const RATE_LIMIT_PER_MINUTE = 600;

const FIELDS: Record<keyof LoadReport, true> = {
  schema: true, model: true, variant: true, bytes: true, chunks: true, ms: true,
  source: true, cache_hits: true, browser: true, webgpu: true,
};
const MODEL_RE = /^[a-z0-9][a-z0-9._-]*$/;
const VARIANT_RE = /^[a-z0-9][a-z0-9._/-]*$/;

type Invalid = { ok: false; reason: string };
type Valid = { ok: true; report: LoadReport };

function isInt(v: unknown, min: number, max: number): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= min && v <= max;
}

/** Mirrors schemas/telemetry.v1.json: unknown fields, missing fields, wrong types, bad enums and out-of-range numbers all fail. */
export function validateLoadReport(input: unknown): Valid | Invalid {
  if (typeof input !== "object" || input === null || Array.isArray(input)) return { ok: false, reason: "body must be a JSON object" };
  const o = input as Record<string, unknown>;
  for (const k of Object.keys(o)) if (!(k in FIELDS)) return { ok: false, reason: `unknown field: ${k}` };
  for (const k of Object.keys(FIELDS)) if (!(k in o)) return { ok: false, reason: `missing field: ${k}` };
  if (o.schema !== 1) return { ok: false, reason: "schema must be 1" };
  if (typeof o.model !== "string" || o.model.length > 128 || !MODEL_RE.test(o.model)) return { ok: false, reason: "bad model" };
  if (typeof o.variant !== "string" || o.variant.length > 128 || !VARIANT_RE.test(o.variant)) return { ok: false, reason: "bad variant" };
  if (!isInt(o.bytes, 0, 1099511627776)) return { ok: false, reason: "bad bytes" };
  if (!isInt(o.chunks, 0, 1_000_000)) return { ok: false, reason: "bad chunks" };
  if (!isInt(o.ms, 0, 86_400_000)) return { ok: false, reason: "bad ms" };
  if (!(LOAD_SOURCES as readonly unknown[]).includes(o.source)) return { ok: false, reason: "bad source" };
  if (!isInt(o.cache_hits, 0, 1_000_000) || o.cache_hits > o.chunks) return { ok: false, reason: "bad cache_hits" };
  if (!(BROWSERS as readonly unknown[]).includes(o.browser)) return { ok: false, reason: "bad browser" };
  if (typeof o.webgpu !== "boolean") return { ok: false, reason: "bad webgpu" };
  return { ok: true, report: o as unknown as LoadReport };
}

/** Reads at most MAX_BODY_BYTES; returns null when the body is larger. */
async function readBounded(request: Request): Promise<Uint8Array | null> {
  const declared = Number(request.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > MAX_BODY_BYTES) return null;
  if (!request.body) return new Uint8Array(0);
  const chunks: Uint8Array[] = [];
  let total = 0;
  const reader = request.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > MAX_BODY_BYTES) { await reader.cancel(); return null; }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out;
}

/** Crude abuse guard: a KV counter per country:colo:minute (not per IP). KV is eventually consistent, so this is approximate by design. */
async function overLimit(env: Env, country: string, colo: string): Promise<boolean> {
  const minute = Math.floor(Date.now() / 60_000);
  const key = `rl:${country}:${colo}:${minute}`;
  const n = Number((await env.STATS_CACHE.get(key)) ?? "0");
  if (n >= RATE_LIMIT_PER_MINUTE) return true;
  await env.STATS_CACHE.put(key, String(n + 1), { expirationTtl: 120 });
  return false;
}

export async function ingestLoad(request: Request, env: Env): Promise<Response> {
  const ct = request.headers.get("content-type") ?? "";
  if (!ct.toLowerCase().startsWith("application/json")) return error(415, "unsupported_media_type", "send application/json");
  const raw = await readBounded(request);
  if (raw === null) return error(413, "payload_too_large", `body must be at most ${MAX_BODY_BYTES} bytes`);
  let parsed: unknown;
  try { parsed = JSON.parse(new TextDecoder().decode(raw)); } catch { return error(400, "bad_json"); }
  const v = validateLoadReport(parsed);
  if (!v.ok) return error(400, "invalid_report", v.reason);

  const cf = (request as Request & { cf?: IncomingRequestCfProperties }).cf;
  const country = typeof cf?.country === "string" ? cf.country : "XX";
  const colo = typeof cf?.colo === "string" ? cf.colo : "UNK";
  if (await overLimit(env, country, colo)) return error(429, "rate_limited");

  const r = v.report;
  const hour = new Date(Math.floor(Date.now() / 3_600_000) * 3_600_000).toISOString();
  env.TELEMETRY.writeDataPoint({
    blobs: [r.model, r.variant, r.source, r.browser, country, colo, hour],
    doubles: [r.bytes, r.chunks, r.ms, r.cache_hits, r.webgpu ? 1 : 0],
    indexes: [r.model],
  });
  return new Response(null, { status: 202 });
}

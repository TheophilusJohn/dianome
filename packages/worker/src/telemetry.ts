// POST /v1/telemetry/load: validate one load report and write one Analytics Engine point.
// Privacy: the IP address and user agent string are never read; only request.cf.country / colo are kept,
// and the timestamp we attach is rounded to the hour.
// Phase 6: telemetry stays keyless, but a report posted with `Authorization: Bearer dk_live_…` records the key's id
// (never the key) in the data point and meters the key's hourly usage bucket (metering.ts). A bad or revoked key is
// refused so a misconfigured app notices; a schema-3 `key_id` in the body must match the bearer.

import { error } from "./http";
import { KEY_ID_RE, authenticateKey, keyError } from "./keys";
import { recordLoad, recordSession } from "./metering";
import { BROWSERS, CACHE_MODES, LOAD_SOURCES, PLAN_POLICIES, SESSION_MODES, type Env, type LoadReport, type LoadReportV1, type LoadReportV2, type SessionReportV3 } from "./types";

export const MAX_BODY_BYTES = 4096;
export const RATE_LIMIT_PER_MINUTE = 600;

const FIELDS: Record<keyof LoadReportV1, true> = {
  schema: true, model: true, variant: true, bytes: true, chunks: true, ms: true,
  source: true, cache_hits: true, browser: true, webgpu: true,
};
/** Optional schema-2 fields: name → [min, max] for integers, or the enum for cache_mode. */
const V2_INTS: Record<Exclude<keyof LoadReportV2, keyof LoadReportV1 | "cache_mode">, [number, number]> = {
  bytes_per_second: [0, 1099511627776], verify_ms: [0, 86_400_000], transfer_ms: [0, 86_400_000],
  quota_bytes: [0, 1099511627776], max_buffer_size: [0, 1099511627776],
};
const V2_FIELDS: Record<Exclude<keyof LoadReportV2, keyof LoadReportV1>, true> = {
  bytes_per_second: true, verify_ms: true, transfer_ms: true, quota_bytes: true, max_buffer_size: true, cache_mode: true,
};

/** Analytics Engine column map: blobs 1-8, doubles 1-10. Absent schema-2 doubles are written as -1, never 0. */
export const ABSENT = -1;
const MODEL_RE = /^[a-z0-9][a-z0-9._-]*$/;
const VARIANT_RE = /^[a-z0-9][a-z0-9._/-]*$/;

type Invalid = { ok: false; reason: string };
type Valid = { ok: true; report: LoadReport };
type ValidSession = { ok: true; session: SessionReportV3 };

function isInt(v: unknown, min: number, max: number): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= min && v <= max;
}
function isNum(v: unknown, min: number, max: number): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= min && v <= max;
}

/** Schema 3 (schemas/telemetry.v3.json): every field required, no prompt content. */
const V3_INTS: Record<"N" | "L" | "prompt_tokens" | "new_tokens", [number, number]> = { N: [0, 1000], L: [1, 1000], prompt_tokens: [0, 1_000_000], new_tokens: [0, 1_000_000] };
const V3_NUMS: Record<"client_ms" | "server_busy_ms" | "rtt_ms" | "tok_per_s", [number, number]> = { client_ms: [0, 86_400_000], server_busy_ms: [0, 86_400_000], rtt_ms: [0, 86_400_000], tok_per_s: [0, 1_000_000] };
const V3_FIELDS: Record<Exclude<keyof SessionReportV3, "key_id">, true> = {
  schema: true, model: true, variant: true, mode: true, N: true, L: true, prompt_tokens: true, new_tokens: true,
  client_ms: true, server_busy_ms: true, rtt_ms: true, tok_per_s: true, plan_policy: true, cache_mode: true, browser: true, webgpu: true,
};
const V3_OPTIONAL: Record<"key_id", true> = { key_id: true };

export function validateSessionReport(o: Record<string, unknown>): ValidSession | Invalid {
  for (const k of Object.keys(o)) if (!(k in V3_FIELDS) && !(k in V3_OPTIONAL)) return { ok: false, reason: `unknown field: ${k}` };
  for (const k of Object.keys(V3_FIELDS)) if (!(k in o)) return { ok: false, reason: `missing field: ${k}` };
  if (typeof o.model !== "string" || o.model.length > 128 || !MODEL_RE.test(o.model)) return { ok: false, reason: "bad model" };
  if (typeof o.variant !== "string" || o.variant.length > 128 || !VARIANT_RE.test(o.variant)) return { ok: false, reason: "bad variant" };
  if (!(SESSION_MODES as readonly unknown[]).includes(o.mode)) return { ok: false, reason: "bad mode" };
  for (const [k, [min, max]] of Object.entries(V3_INTS)) if (!isInt(o[k], min, max)) return { ok: false, reason: `bad ${k}` };
  for (const [k, [min, max]] of Object.entries(V3_NUMS)) if (!isNum(o[k], min, max)) return { ok: false, reason: `bad ${k}` };
  if ((o.N as number) > (o.L as number)) return { ok: false, reason: "bad N" };
  if ((o.mode === "server" && o.N !== 0) || (o.mode === "local" && o.N !== o.L) || (o.mode === "split" && o.N === 0)) return { ok: false, reason: "mode/N mismatch" };
  if (!(PLAN_POLICIES as readonly unknown[]).includes(o.plan_policy)) return { ok: false, reason: "bad plan_policy" };
  if (!(CACHE_MODES as readonly unknown[]).includes(o.cache_mode)) return { ok: false, reason: "bad cache_mode" };
  if (!(BROWSERS as readonly unknown[]).includes(o.browser)) return { ok: false, reason: "bad browser" };
  if (typeof o.webgpu !== "boolean") return { ok: false, reason: "bad webgpu" };
  if ("key_id" in o && (typeof o.key_id !== "string" || !KEY_ID_RE.test(o.key_id))) return { ok: false, reason: "bad key_id" };
  return { ok: true, session: o as unknown as SessionReportV3 };
}

/** Mirrors schemas/telemetry.v1.json and v2.json: unknown fields, missing fields, wrong types, bad enums and out-of-range numbers all fail. */
export function validateLoadReport(input: unknown): Valid | Invalid {
  if (typeof input !== "object" || input === null || Array.isArray(input)) return { ok: false, reason: "body must be a JSON object" };
  const o = input as Record<string, unknown>;
  if (o.schema !== 1 && o.schema !== 2) return { ok: false, reason: "schema must be 1 or 2" };
  const v2 = o.schema === 2;
  for (const k of Object.keys(o)) if (!(k in FIELDS) && !(v2 && k in V2_FIELDS)) return { ok: false, reason: `unknown field: ${k}` };
  for (const k of Object.keys(FIELDS)) if (!(k in o)) return { ok: false, reason: `missing field: ${k}` };
  if (v2) {
    for (const [k, [min, max]] of Object.entries(V2_INTS)) if (k in o && !isInt(o[k], min, max)) return { ok: false, reason: `bad ${k}` };
    if ("cache_mode" in o && !(CACHE_MODES as readonly unknown[]).includes(o.cache_mode)) return { ok: false, reason: "bad cache_mode" };
  }
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
  const isV3 = typeof parsed === "object" && parsed !== null && !Array.isArray(parsed) && (parsed as Record<string, unknown>).schema === 3;
  const v = isV3 ? validateSessionReport(parsed as Record<string, unknown>) : validateLoadReport(parsed);
  if (!v.ok) return error(400, "invalid_report", v.reason);
  const auth = await authenticateKey(request, env);
  const authErr = keyError(auth);
  if (authErr) return authErr;
  const keyId = auth.status === "ok" ? auth.record.id : "";
  if ("session" in v && v.session.key_id !== undefined && v.session.key_id !== keyId) return error(400, "invalid_report", keyId ? "key_id does not match the bearer key" : "key_id given without the key as bearer");

  const cf = (request as Request & { cf?: IncomingRequestCfProperties }).cf;
  const country = typeof cf?.country === "string" ? cf.country : "XX";
  const colo = typeof cf?.colo === "string" ? cf.colo : "UNK";
  if (await overLimit(env, country, colo)) return error(429, "rate_limited");

  const hour = new Date(Math.floor(Date.now() / 3_600_000) * 3_600_000).toISOString();
  if ("session" in v) env.SESSIONS.writeDataPoint(sessionDataPoint(v.session, country, colo, hour, keyId));
  else env.TELEMETRY.writeDataPoint(dataPoint(v.report, country, colo, hour, keyId));
  if (keyId) {
    // Metering is the key's own usage record: a failed bucket write is a 500 the SDK can see, never a silent miss.
    if ("session" in v) await recordSession(env, keyId, v.session);
    else await recordLoad(env, keyId, v.report);
  }
  return new Response(null, { status: 202 });
}

/**
 * dianome_sessions: blobs 1 model, 2 variant, 3 mode, 4 plan_policy, 5 cache_mode, 6 browser, 7 country, 8 colo, 9 hour,
 * 10 key_id ("" when the report came without a key; blob 9 is taken here, so the key id sits one column later than
 * in dianome_loads); doubles 1 N, 2 L, 3 prompt_tokens, 4 new_tokens, 5 client_ms, 6 server_busy_ms, 7 rtt_ms, 8 tok_per_s, 9 webgpu.
 */
export function sessionDataPoint(s: SessionReportV3, country: string, colo: string, hour: string, keyId = ""): AnalyticsEngineDataPoint {
  return {
    blobs: [s.model, s.variant, s.mode, s.plan_policy, s.cache_mode, s.browser, country, colo, hour, keyId],
    doubles: [s.N, s.L, s.prompt_tokens, s.new_tokens, s.client_ms, s.server_busy_ms, s.rtt_ms, s.tok_per_s, s.webgpu ? 1 : 0],
    indexes: [s.model],
  };
}

/**
 * blobs: 1 model, 2 variant, 3 source, 4 browser, 5 country, 6 colo, 7 hour, 8 cache_mode ("" for schema 1),
 * 9 key_id ("" when the report came without a key).
 * doubles: 1 bytes, 2 chunks, 3 ms, 4 cache_hits, 5 webgpu, 6 bytes_per_second, 7 verify_ms, 8 transfer_ms,
 * 9 quota_bytes, 10 max_buffer_size (6-10 are ABSENT for schema 1 or when the client omitted them).
 */
export function dataPoint(r: LoadReport, country: string, colo: string, hour: string, keyId = ""): AnalyticsEngineDataPoint {
  const v2: Partial<LoadReportV2> = r.schema === 2 ? r : {};
  const d = (n: number | undefined): number => (typeof n === "number" ? n : ABSENT);
  return {
    blobs: [r.model, r.variant, r.source, r.browser, country, colo, hour, v2.cache_mode ?? "", keyId],
    doubles: [r.bytes, r.chunks, r.ms, r.cache_hits, r.webgpu ? 1 : 0, d(v2.bytes_per_second), d(v2.verify_ms), d(v2.transfer_ms), d(v2.quota_bytes), d(v2.max_buffer_size)],
    indexes: [r.model],
  };
}

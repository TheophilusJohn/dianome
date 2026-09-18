// Phase 5b split endpoints; Phase 7 made them per model.
//   GET  /v1/split/servers → { default, models: [{ model, ws, plan }] }: the SPLIT_SERVERS map (public; a session token
//       still gates every WebSocket). The demo's model selector reads this.
//   POST /v1/split/session[?model=] → { url, token, expires_at }: a short-lived HMAC session token for that model's server.
//       token = base64url(payload) + "." + base64url(HMAC-SHA256(SPLIT_SIGNING_KEY, payload)),
//       payload = { sid, model, exp (unix, now + 3600), max_ctx, origin }. The model comes from ?model=, else the JSON
//       body's `model`, else the map's first entry. Phase 6: the request needs an API key (`Authorization: Bearer
//       dk_live_…`, see keys.ts) OR an allowed demo Origin (SPLIT_ALLOWED_ORIGINS); keyed requests skip the origin
//       allowlist (a key with `allowed_origins` is limited to those) and the response names the key id (`key_id`), so the
//       SDK can put the id, never the key, in its session report. Rate-limited per country:colo:minute like telemetry.
//   GET  /v1/split/rates   → server/bench/rates.json (the stated GPU rate; the client computes cost from it).
//   GET  /v1/split/plan[?model=] → that model's server's public /plan, cached 5 s per server.
//
// SPLIT_SERVERS is a JSON map model id -> { ws, plan }, e.g.
//   {"qwen2.5-0.5b-instruct":{"ws":"wss://split.dianome.dev","plan":"https://split.dianome.dev/plan"},
//    "qwen2.5-7b-instruct":{"ws":"wss://gpu.dianome.dev","plan":"https://gpu.dianome.dev/plan"}}

import { error, json } from "./http";
import { authenticateKey, keyError } from "./keys";
import { ID_RE } from "./manifest";
import ratesJson from "../../../server/bench/rates.json";
import type { Env } from "./types";

export const SESSION_TTL_SECONDS = 3600;
export const SESSION_RATE_LIMIT_PER_MINUTE = 60;
export const PLAN_CACHE_MS = 5000;
export const PLAN_TIMEOUT_MS = 3000;
export const MAX_CTX = 8192; // Phase 7: the summariser takes ~6k-token documents + 256 new tokens; the server caps at the same value

export interface SplitServer { ws: string; plan: string }

/** Parsed SPLIT_SERVERS in insertion order (the first entry is the default model); null when unset or malformed. */
export function splitServers(env: Env): Record<string, SplitServer> | null {
  if (!env.SPLIT_SERVERS) return null;
  let parsed: unknown;
  try { parsed = JSON.parse(env.SPLIT_SERVERS); } catch { return null; }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
  const out: Record<string, SplitServer> = {};
  for (const [model, v] of Object.entries(parsed as Record<string, unknown>)) {
    if (!ID_RE.test(model) || typeof v !== "object" || v === null || Array.isArray(v)) return null;
    const { ws, plan } = v as Record<string, unknown>;
    if (typeof ws !== "string" || !/^wss?:\/\//.test(ws) || typeof plan !== "string" || !/^https?:\/\//.test(plan)) return null;
    out[model] = { ws, plan };
  }
  return Object.keys(out).length ? out : null;
}

const NOT_CONFIGURED = "SPLIT_SIGNING_KEY / SPLIT_SERVERS are not set (or SPLIT_SERVERS is not a JSON map of model -> { ws, plan })";

/** GET /v1/split/servers */
export function servers(env: Env): Response {
  const map = splitServers(env);
  if (!map) return error(503, "split_not_configured", NOT_CONFIGURED);
  const models = Object.entries(map).map(([model, s]) => ({ model, ws: s.ws, plan: s.plan }));
  return json({ default: models[0]!.model, models }, { headers: { "Cache-Control": "public, max-age=60" } });
}

export interface TokenPayload { sid: string; model: string; exp: number; max_ctx: number; origin: string }

const enc = new TextEncoder();

function b64u(bytes: Uint8Array): string {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function unb64u(s: string): Uint8Array {
  const b = atob(s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4));
  return Uint8Array.from(b, (c) => c.charCodeAt(0));
}

async function hmacKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

/** Canonical JSON (sorted keys, no whitespace), the same bytes the server's `auth.mint` produces. */
export function canonicalPayload(p: TokenPayload): string {
  return JSON.stringify({ exp: p.exp, max_ctx: p.max_ctx, model: p.model, origin: p.origin, sid: p.sid });
}

export async function mintToken(secret: string, payload: TokenPayload): Promise<string> {
  const bytes = enc.encode(canonicalPayload(payload));
  const sig = new Uint8Array(await crypto.subtle.sign("HMAC", await hmacKey(secret), bytes));
  return `${b64u(bytes)}.${b64u(sig)}`;
}

/** Signature + expiry check (the server does the same in Python); null when invalid. */
export async function verifyToken(token: string, secret: string, now = Date.now() / 1000): Promise<TokenPayload | null> {
  const parts = token.split(".");
  if (parts.length !== 2 || !parts[0] || !parts[1]) return null;
  let payload: Uint8Array, sig: Uint8Array;
  try { payload = unb64u(parts[0]); sig = unb64u(parts[1]); } catch { return null; }
  const ok = await crypto.subtle.verify("HMAC", await hmacKey(secret), sig as BufferSource, payload as BufferSource);
  if (!ok) return null;
  let obj: unknown;
  try { obj = JSON.parse(new TextDecoder().decode(payload)); } catch { return null; }
  if (typeof obj !== "object" || obj === null) return null;
  const o = obj as Record<string, unknown>;
  if (typeof o.sid !== "string" || typeof o.model !== "string" || typeof o.exp !== "number" || typeof o.max_ctx !== "number" || typeof o.origin !== "string") return null;
  if (o.exp <= now) return null;
  return { sid: o.sid, model: o.model, exp: o.exp, max_ctx: o.max_ctx, origin: o.origin };
}

function allowedOrigins(env: Env): string[] {
  return (env.SPLIT_ALLOWED_ORIGINS ?? "").split(",").map((s) => s.trim().replace(/\/+$/, "")).filter(Boolean);
}

/** Same shape as telemetry's guard: a KV counter per country:colo:minute (approximate by design). */
async function overLimit(env: Env, country: string, colo: string): Promise<boolean> {
  const minute = Math.floor(Date.now() / 60_000);
  const key = `rl:split:${country}:${colo}:${minute}`;
  const n = Number((await env.STATS_CACHE.get(key)) ?? "0");
  if (n >= SESSION_RATE_LIMIT_PER_MINUTE) return true;
  await env.STATS_CACHE.put(key, String(n + 1), { expirationTtl: 120 });
  return false;
}

export async function createSession(request: Request, env: Env): Promise<Response> {
  const map = splitServers(env);
  if (!env.SPLIT_SIGNING_KEY || !map) return error(503, "split_not_configured", NOT_CONFIGURED);
  const origin = (request.headers.get("Origin") ?? "").replace(/\/+$/, "");
  const auth = await authenticateKey(request, env);
  const authErr = keyError(auth);
  if (authErr) return authErr;
  const keyed = auth.status === "ok" ? auth.record : null;
  if (keyed) {
    const allowed = keyed.allowed_origins?.map((o) => o.replace(/\/+$/, ""));
    if (allowed && !allowed.includes(origin)) return error(403, "origin_not_allowed", `key ${keyed.id} is limited to ${allowed.join(", ")}`);
  } else if (!origin || !allowedOrigins(env).includes(origin)) {
    return error(403, "origin_not_allowed", "send an API key (Authorization: Bearer dk_live_…) or call from an allowed demo origin");
  }
  let body: Record<string, unknown> = {};
  const ct = request.headers.get("content-type") ?? "";
  if (ct.toLowerCase().startsWith("application/json")) {
    try { const parsed: unknown = await request.json(); if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) body = parsed as Record<string, unknown>; else return error(400, "bad_json"); }
    catch { return error(400, "bad_json"); }
  }
  const query = new URL(request.url).searchParams.get("model");
  const model = query ?? (typeof body.model === "string" ? body.model : Object.keys(map)[0]!);
  const server = Object.prototype.hasOwnProperty.call(map, model) ? map[model]! : null;
  if (!server) return error(400, "model_not_allowed");
  let maxCtx = MAX_CTX;
  if (body.max_ctx !== undefined) {
    if (typeof body.max_ctx !== "number" || !Number.isInteger(body.max_ctx) || body.max_ctx < 1) return error(400, "bad_max_ctx");
    maxCtx = Math.min(MAX_CTX, body.max_ctx);
  }
  const cf = (request as Request & { cf?: IncomingRequestCfProperties }).cf;
  const country = typeof cf?.country === "string" ? cf.country : "XX";
  const colo = typeof cf?.colo === "string" ? cf.colo : "UNK";
  if (await overLimit(env, country, colo)) return error(429, "rate_limited");
  const exp = Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS;
  const sid = b64u(crypto.getRandomValues(new Uint8Array(12)));
  const token = await mintToken(env.SPLIT_SIGNING_KEY, { sid, model, exp, max_ctx: maxCtx, origin });
  return json({ url: server.ws, token, expires_at: new Date(exp * 1000).toISOString(), sid, model, max_ctx: maxCtx, ...(keyed ? { key_id: keyed.id } : {}) }, { headers: { "Cache-Control": "no-store" } });
}

export interface Rates { gpu: string; usd_per_hour: number | null; source: string; retrieved: string }

export function rates(): Response {
  const r = ratesJson as Rates & { _note?: string };
  return json({ gpu: r.gpu, usd_per_hour: r.usd_per_hour, source: r.source, retrieved: r.retrieved, rate_available: typeof r.usd_per_hour === "number" }, { headers: { "Cache-Control": "public, max-age=300" } });
}

const planMemo = new Map<string, { at: number; status: number; body: string }>();
/** Tests: forget the cached plans. */
export function resetPlanCache(): void { planMemo.clear(); }

export async function planProxy(request: Request, env: Env): Promise<Response> {
  const map = splitServers(env);
  if (!map) return error(503, "split_not_configured", NOT_CONFIGURED);
  const model = new URL(request.url).searchParams.get("model") ?? Object.keys(map)[0]!;
  const server = Object.prototype.hasOwnProperty.call(map, model) ? map[model]! : null;
  if (!server) return error(400, "model_not_allowed");
  const url = server.plan;
  const now = Date.now();
  let memo = planMemo.get(url);
  if (!memo || now - memo.at > PLAN_CACHE_MS) {
    let status = 503, body = JSON.stringify({ error: "server_unreachable", reachable: false, model });
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(PLAN_TIMEOUT_MS), headers: { Accept: "application/json" } });
      if (res.ok) { const text = await res.text(); JSON.parse(text); status = 200; body = text; }
      else body = JSON.stringify({ error: "server_error", reachable: false, model, status: res.status });
    } catch (e) {
      body = JSON.stringify({ error: "server_unreachable", reachable: false, model, detail: e instanceof Error ? e.message : String(e) });
    }
    memo = { at: now, status, body };
    planMemo.set(url, memo);
  }
  const age = Math.floor((now - memo.at) / 1000);
  return new Response(memo.body, { status: memo.status, headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": `public, max-age=${Math.ceil(PLAN_CACHE_MS / 1000)}`, Age: String(age) } });
}

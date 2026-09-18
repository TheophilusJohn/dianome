// Phase 6 API keys. Format `dk_live_<24 random bytes base64url>` (32 characters after the prefix). A key is stored
// only as its SHA-256 (`key:<hash>` → KeyRecord); the plaintext is returned once at creation and never logged.
// There are no accounts in this phase: a key's owner is the key's own id, and keys created through the dashboard
// (POST /v1/me/keys) inherit the caller's owner so one developer can hold several keys. KV layout (STATS_CACHE):
//   key:<sha256 hex>   → KeyRecord               lookup by the presented key
//   keyid:<id>         → sha256 hex              lookup by id (revoke, list)
//   owner:<owner>      → string[] of key ids     the dashboard's key list
//   rl:keys:<country>:<colo>:<YYYYMMDD> → count  POST /v1/keys is open but limited to 10 per day per country:colo

import { error, json, sha256Hex } from "./http";
import type { Env } from "./types";

export const KEY_PREFIX = "dk_live_";
export const KEY_RE = /^dk_live_[A-Za-z0-9_-]{32}$/;
export const KEY_ID_RE = /^k_[0-9a-f]{16}$/;
export const KEYS_PER_DAY = 10;

export interface KeyRecord {
  id: string;
  owner: string;
  /** ISO 8601. */
  created: string;
  /** ISO 8601 when revoked, else null. */
  revoked: string | null;
  /** When set, only these Origins may mint sessions with the key (unset: any origin, or none for non-browser clients). */
  allowed_origins?: string[];
  plan: "free";
}

const enc = new TextEncoder();

function b64u(bytes: Uint8Array): string {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** A fresh key: the prefix plus 24 random bytes, base64url (32 characters). */
export function generateKey(): string {
  return KEY_PREFIX + b64u(crypto.getRandomValues(new Uint8Array(24)));
}

export function generateKeyId(): string {
  return "k_" + Array.from(crypto.getRandomValues(new Uint8Array(8)), (b) => b.toString(16).padStart(2, "0")).join("");
}

export function hashKey(key: string): Promise<string> {
  return sha256Hex(enc.encode(key));
}

/** The bearer key from an Authorization header; null when absent or not shaped like a key. */
export function bearerKey(request: Request): string | null {
  const h = request.headers.get("Authorization") ?? "";
  const m = /^Bearer\s+(\S+)$/i.exec(h);
  if (!m) return null;
  return KEY_RE.test(m[1]!) ? m[1]! : null;
}

/** True when the header carries some bearer, well-formed or not (so a malformed key gets 401 rather than "no key"). */
export function hasBearer(request: Request): boolean {
  return /^Bearer\s+\S+/i.test(request.headers.get("Authorization") ?? "");
}

export async function lookupKey(env: Env, key: string): Promise<KeyRecord | null> {
  if (!KEY_RE.test(key)) return null;
  const raw = await env.STATS_CACHE.get(`key:${await hashKey(key)}`);
  if (!raw) return null;
  try { return JSON.parse(raw) as KeyRecord; } catch { return null; }
}

export async function lookupKeyById(env: Env, id: string): Promise<KeyRecord | null> {
  if (!KEY_ID_RE.test(id)) return null;
  const hash = await env.STATS_CACHE.get(`keyid:${id}`);
  if (!hash) return null;
  const raw = await env.STATS_CACHE.get(`key:${hash}`);
  if (!raw) return null;
  try { return JSON.parse(raw) as KeyRecord; } catch { return null; }
}

export async function ownerKeyIds(env: Env, owner: string): Promise<string[]> {
  const raw = await env.STATS_CACHE.get(`owner:${owner}`);
  if (!raw) return [];
  try { const v: unknown = JSON.parse(raw); return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []; } catch { return []; }
}

/**
 * Creates a key. `owner` defaults to the new key's own id (the open endpoint); the dashboard passes the caller's
 * owner. Returns the plaintext key exactly once.
 */
export async function createKey(env: Env, owner?: string, now = new Date()): Promise<{ key: string; record: KeyRecord }> {
  const key = generateKey();
  const id = generateKeyId();
  const record: KeyRecord = { id, owner: owner ?? id, created: now.toISOString(), revoked: null, plan: "free" };
  const hash = await hashKey(key);
  await env.STATS_CACHE.put(`key:${hash}`, JSON.stringify(record));
  await env.STATS_CACHE.put(`keyid:${id}`, hash);
  const ids = await ownerKeyIds(env, record.owner);
  if (!ids.includes(id)) await env.STATS_CACHE.put(`owner:${record.owner}`, JSON.stringify([...ids, id]));
  return { key, record };
}

/** Marks the key revoked (idempotent). Returns the updated record, or null when there is no such key. */
export async function revokeKey(env: Env, id: string, now = new Date()): Promise<KeyRecord | null> {
  const record = await lookupKeyById(env, id);
  if (!record) return null;
  if (record.revoked) return record;
  const hash = await env.STATS_CACHE.get(`keyid:${id}`);
  const updated: KeyRecord = { ...record, revoked: now.toISOString() };
  await env.STATS_CACHE.put(`key:${hash}`, JSON.stringify(updated));
  return updated;
}

export type KeyAuth =
  | { status: "none" }
  | { status: "ok"; record: KeyRecord; key: string }
  | { status: "invalid" }
  | { status: "revoked"; record: KeyRecord };

/** Resolves the bearer key on a request: none, ok, invalid (unknown or malformed), or revoked. */
export async function authenticateKey(request: Request, env: Env): Promise<KeyAuth> {
  if (!hasBearer(request)) return { status: "none" };
  const key = bearerKey(request);
  if (!key) return { status: "invalid" };
  const record = await lookupKey(env, key);
  if (!record) return { status: "invalid" };
  if (record.revoked) return { status: "revoked", record };
  return { status: "ok", record, key };
}

/** The error response for a key that is not usable; null when the auth outcome is fine (ok or none). */
export function keyError(auth: KeyAuth): Response | null {
  if (auth.status === "invalid") return error(401, "invalid_key", "the Authorization bearer is not a known dk_live_ key");
  if (auth.status === "revoked") return error(403, "key_revoked", `key ${auth.record.id} was revoked ${auth.record.revoked}`);
  return null;
}

/** 401 unless the request carries a live key. */
export async function requireKey(request: Request, env: Env): Promise<{ record: KeyRecord; key: string } | Response> {
  const auth = await authenticateKey(request, env);
  if (auth.status === "none") return error(401, "key_required", "send Authorization: Bearer dk_live_…");
  const err = keyError(auth);
  if (err) return err;
  return { record: (auth as { record: KeyRecord }).record, key: (auth as { key: string }).key };
}

function dayStamp(now: Date): string {
  return now.toISOString().slice(0, 10).replace(/-/g, "");
}

/** KV counter per country:colo:day (approximate by design, like the other limits). */
export async function overKeyLimit(env: Env, country: string, colo: string, now = new Date()): Promise<boolean> {
  const key = `rl:keys:${country}:${colo}:${dayStamp(now)}`;
  const n = Number((await env.STATS_CACHE.get(key)) ?? "0");
  if (n >= KEYS_PER_DAY) return true;
  await env.STATS_CACHE.put(key, String(n + 1), { expirationTtl: 2 * 86400 });
  return false;
}

export function publicRecord(r: KeyRecord): Omit<KeyRecord, "allowed_origins"> & { allowed_origins?: string[] } {
  return { id: r.id, owner: r.owner, created: r.created, revoked: r.revoked, plan: r.plan, ...(r.allowed_origins ? { allowed_origins: r.allowed_origins } : {}) };
}

/**
 * POST /v1/keys: open, rate-limited. The response is the only time the key is shown.
 * There are no accounts: keep the key, it is the identity.
 */
export async function createKeyOpen(request: Request, env: Env): Promise<Response> {
  const cf = (request as Request & { cf?: IncomingRequestCfProperties }).cf;
  const country = typeof cf?.country === "string" ? cf.country : "XX";
  const colo = typeof cf?.colo === "string" ? cf.colo : "UNK";
  if (await overKeyLimit(env, country, colo)) return error(429, "rate_limited", `at most ${KEYS_PER_DAY} keys per day from this location`);
  const { key, record } = await createKey(env);
  return json({ key, ...publicRecord(record), note: "Keep this key: it is shown once and is your only identity (no accounts in this phase)." }, { status: 201, headers: { "Cache-Control": "no-store" } });
}

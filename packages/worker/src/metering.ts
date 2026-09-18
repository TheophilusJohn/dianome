// Phase 6 metering: per-key counters in hourly KV buckets, `usage:<key_id>:<YYYYMMDDHH>` → UsageBucket.
//
// Source of truth is the schema-3 session report the SDK posts when a run() call finishes (it carries the session's
// `stats`: N, L, new_tokens, server_busy_ms, mode) with the key as the bearer; load reports (schema 1–2) posted with a
// key add `bytes_served`. Layer-tokens: the client ran N blocks per new token, the server ran L − N blocks plus the
// language-model head, counted as one layer-equivalent per token whenever the server ran it (split and server modes).
//
// KV has no transactions: writes are read-modify-write with a re-read between attempts, so two reports for the same
// key landing in the same hour on different colos can still lose one increment (KV is eventually consistent).
// That is accepted for a free-tier control plane and said so in the notes; Analytics Engine keeps every raw row.

import type { Env, LoadReport, SessionReportV3 } from "./types";

export const BUCKET_TTL_SECONDS = 8 * 86400; // the dashboard window is 168 h; keep a day of slack
export const MAX_USAGE_HOURS = 168;
export const WRITE_ATTEMPTS = 3;

export interface ModeCounts { local: number; split: number; server: number }

export interface UsageCounters {
  sessions: number;
  /** N × new_tokens summed over sessions: block-tokens the client ran. */
  client_layer_tokens: number;
  /** (L − N) × new_tokens, plus new_tokens for lm_head when the server ran it: block-tokens the server ran. */
  server_layer_tokens: number;
  new_tokens: number;
  server_busy_ms: number;
  /** Bytes reported by load reports carrying this key. */
  bytes_served: number;
  modes: ModeCounts;
}

export interface UsageBucket extends UsageCounters {
  /** ISO 8601 hour start, e.g. 2026-09-18T14:00:00.000Z. */
  hour: string;
}

export const emptyCounters = (): UsageCounters => ({ sessions: 0, client_layer_tokens: 0, server_layer_tokens: 0, new_tokens: 0, server_busy_ms: 0, bytes_served: 0, modes: { local: 0, split: 0, server: 0 } });

/** `YYYYMMDDHH` in UTC for the hour containing `at`. */
export function hourStamp(at: Date): string {
  return at.toISOString().slice(0, 13).replace(/[-T]/g, "");
}
export function hourStart(at: Date): Date {
  return new Date(Math.floor(at.getTime() / 3_600_000) * 3_600_000);
}
export function stampToIso(stamp: string): string {
  return `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)}T${stamp.slice(8, 10)}:00:00.000Z`;
}

export const bucketKey = (keyId: string, stamp: string): string => `usage:${keyId}:${stamp}`;

/** The layer-token split for one session report. */
export function layerTokens(s: Pick<SessionReportV3, "mode" | "N" | "L" | "new_tokens">): { client: number; server: number } {
  const client = s.N * s.new_tokens;
  const headOnServer = s.mode === "local" ? 0 : s.new_tokens;
  const server = (s.L - s.N) * s.new_tokens + headOnServer;
  return { client, server };
}

function addSession(c: UsageCounters, s: SessionReportV3): UsageCounters {
  const lt = layerTokens(s);
  return {
    ...c,
    sessions: c.sessions + 1,
    client_layer_tokens: c.client_layer_tokens + lt.client,
    server_layer_tokens: c.server_layer_tokens + lt.server,
    new_tokens: c.new_tokens + s.new_tokens,
    server_busy_ms: Math.round((c.server_busy_ms + s.server_busy_ms) * 10) / 10,
    modes: { ...c.modes, [s.mode]: c.modes[s.mode] + 1 },
  };
}

function addLoad(c: UsageCounters, r: LoadReport): UsageCounters {
  return { ...c, bytes_served: c.bytes_served + r.bytes };
}

function parseCounters(raw: string | null): UsageCounters {
  if (!raw) return emptyCounters();
  try {
    const v = JSON.parse(raw) as Partial<UsageCounters>;
    const base = emptyCounters();
    return { ...base, ...v, modes: { ...base.modes, ...(v.modes ?? {}) } };
  } catch { return emptyCounters(); }
}

/**
 * Read-modify-write of one bucket. Each attempt re-reads before applying `update`; the last write wins on a race
 * (see the header). Resolves to what was written.
 */
export async function updateBucket(env: Env, keyId: string, at: Date, update: (c: UsageCounters) => UsageCounters): Promise<UsageCounters> {
  const key = bucketKey(keyId, hourStamp(at));
  let written: UsageCounters = emptyCounters();
  let lastErr: unknown = null;
  for (let attempt = 0; attempt < WRITE_ATTEMPTS; attempt++) {
    try {
      const current = parseCounters(await env.STATS_CACHE.get(key));
      written = update(current);
      await env.STATS_CACHE.put(key, JSON.stringify(written), { expirationTtl: BUCKET_TTL_SECONDS });
      return written;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}

export function recordSession(env: Env, keyId: string, s: SessionReportV3, at = new Date()): Promise<UsageCounters> {
  return updateBucket(env, keyId, at, (c) => addSession(c, s));
}

export function recordLoad(env: Env, keyId: string, r: LoadReport, at = new Date()): Promise<UsageCounters> {
  return updateBucket(env, keyId, at, (c) => addLoad(c, r));
}

export function sumCounters(buckets: UsageCounters[]): UsageCounters {
  const t = emptyCounters();
  for (const b of buckets) {
    t.sessions += b.sessions; t.client_layer_tokens += b.client_layer_tokens; t.server_layer_tokens += b.server_layer_tokens;
    t.new_tokens += b.new_tokens; t.server_busy_ms += b.server_busy_ms; t.bytes_served += b.bytes_served;
    t.modes.local += b.modes.local; t.modes.split += b.modes.split; t.modes.server += b.modes.server;
  }
  t.server_busy_ms = Math.round(t.server_busy_ms * 10) / 10;
  return t;
}

/** The buckets of the last `hours` hours (the current hour included), oldest first; absent hours are omitted. */
export async function readUsage(env: Env, keyId: string, hours: number, now = new Date()): Promise<UsageBucket[]> {
  const h = Math.max(1, Math.min(MAX_USAGE_HOURS, Math.floor(hours)));
  const end = hourStart(now);
  const firstStamp = hourStamp(new Date(end.getTime() - (h - 1) * 3_600_000));
  const prefix = `usage:${keyId}:`;
  const names: string[] = [];
  let cursor: string | undefined;
  do {
    const page = await env.STATS_CACHE.list({ prefix, ...(cursor ? { cursor } : {}) });
    for (const k of page.keys) names.push(k.name);
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);
  const stamps = names.map((n) => n.slice(prefix.length)).filter((s) => /^\d{10}$/.test(s) && s >= firstStamp).sort();
  const values = await Promise.all(stamps.map((s) => env.STATS_CACHE.get(bucketKey(keyId, s))));
  return stamps.map((s, i) => ({ hour: stampToIso(s), ...parseCounters(values[i] ?? null) }));
}

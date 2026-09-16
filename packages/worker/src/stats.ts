// GET /v1/stats/loads: 7-day aggregates from the Analytics Engine SQL API, cached in KV for 5 minutes.
// Column map (see telemetry.ts): blob1 model, blob2 variant, blob3 source, blob4 browser, blob5 country, blob6 colo;
// double1 bytes, double2 chunks, double3 ms, double4 cache_hits, double5 webgpu. Rows are weighted by _sample_interval.

import { json } from "./http";
import type { CountryStats, Env, LoadStats, ModelVariantStats, SourceStats } from "./types";

export const WINDOW_HOURS = 168;
export const STATS_CACHE_KEY = "stats:loads:v1";
export const STATS_TTL_SECONDS = 300;
const DATASET = "dianome_loads";
const SQL_API = (account: string) => `https://api.cloudflare.com/client/v4/accounts/${account}/analytics_engine/sql`;

const WINDOW = `WHERE timestamp > NOW() - INTERVAL '${WINDOW_HOURS}' HOUR`;
const P50 = "quantileExactWeighted(0.5)(double3, _sample_interval)";
const P90 = "quantileExactWeighted(0.9)(double3, _sample_interval)";

export const QUERIES = {
  by_country: `SELECT blob5 AS country, sum(_sample_interval) AS loads, ${P50} AS p50_ms, ${P90} AS p90_ms,
    sum(_sample_interval * double4) / sum(_sample_interval * double2) AS cache_hit_rate
    FROM ${DATASET} ${WINDOW} GROUP BY country ORDER BY loads DESC LIMIT 250 FORMAT JSON`,
  by_model_variant: `SELECT blob1 AS model, blob2 AS variant, sum(_sample_interval) AS loads, ${P50} AS p50_ms, max(double1) AS bytes
    FROM ${DATASET} ${WINDOW} GROUP BY model, variant ORDER BY loads DESC LIMIT 250 FORMAT JSON`,
  by_source: `SELECT blob3 AS source, sum(_sample_interval) AS loads, ${P50} AS p50_ms
    FROM ${DATASET} ${WINDOW} GROUP BY source ORDER BY loads DESC LIMIT 16 FORMAT JSON`,
} as const;

type Row = Record<string, unknown>;

const num = (v: unknown): number => { const n = typeof v === "string" ? Number(v) : (v as number); return Number.isFinite(n) ? n : 0; };
const str = (v: unknown): string => (typeof v === "string" ? v : String(v ?? ""));
const round = (n: number, d = 3): number => Math.round(n * 10 ** d) / 10 ** d;

export function emptyStats(now = new Date()): LoadStats {
  return {
    since: new Date(now.getTime() - WINDOW_HOURS * 3_600_000).toISOString().replace(/\.\d{3}Z$/, "Z"),
    window_hours: WINDOW_HOURS,
    by_country: [], by_model_variant: [], by_source: [],
  };
}

async function runQuery(env: Env, sql: string): Promise<Row[]> {
  const res = await fetch(SQL_API(env.CF_ACCOUNT_ID!), {
    method: "POST",
    headers: { Authorization: `Bearer ${env.CF_ANALYTICS_TOKEN}`, "Content-Type": "text/plain" },
    body: sql,
  });
  if (!res.ok) throw new Error(`analytics sql api ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const body = (await res.json()) as { data?: Row[] };
  return Array.isArray(body.data) ? body.data : [];
}

/** Runs the three queries; throws when the SQL API is unreachable or rejects a query. */
export async function computeStats(env: Env, now = new Date()): Promise<LoadStats> {
  const [country, mv, source] = await Promise.all([
    runQuery(env, QUERIES.by_country), runQuery(env, QUERIES.by_model_variant), runQuery(env, QUERIES.by_source),
  ]);
  const out = emptyStats(now);
  out.by_country = country.map((r): CountryStats => ({
    country: str(r.country), loads: Math.round(num(r.loads)), p50_ms: Math.round(num(r.p50_ms)), p90_ms: Math.round(num(r.p90_ms)),
    cache_hit_rate: round(Math.min(1, Math.max(0, num(r.cache_hit_rate)))),
  }));
  out.by_model_variant = mv.map((r): ModelVariantStats => ({
    model: str(r.model), variant: str(r.variant), loads: Math.round(num(r.loads)), p50_ms: Math.round(num(r.p50_ms)), bytes: Math.round(num(r.bytes)),
  }));
  out.by_source = source.map((r): SourceStats => ({ source: str(r.source), loads: Math.round(num(r.loads)), p50_ms: Math.round(num(r.p50_ms)) }));
  return out;
}

export async function loadStats(env: Env, ctx: ExecutionContext): Promise<Response> {
  const headers = { "Cache-Control": `public, max-age=${STATS_TTL_SECONDS}` };
  if (!env.CF_ACCOUNT_ID || !env.CF_ANALYTICS_TOKEN) {
    return json({ ...emptyStats(), degraded: true }, { headers: { ...headers, "X-Dianome-Stats": "degraded:no-token" } });
  }
  const cached = await env.STATS_CACHE.get(STATS_CACHE_KEY);
  if (cached) return new Response(cached, { headers: { ...headers, "Content-Type": "application/json; charset=utf-8", "X-Dianome-Stats": "kv-hit" } });
  try {
    const stats = await computeStats(env);
    const text = JSON.stringify(stats);
    ctx.waitUntil(env.STATS_CACHE.put(STATS_CACHE_KEY, text, { expirationTtl: STATS_TTL_SECONDS }));
    return new Response(text, { headers: { ...headers, "Content-Type": "application/json; charset=utf-8", "X-Dianome-Stats": "computed" } });
  } catch (e) {
    console.error("stats degraded:", (e as Error).message);
    return json({ ...emptyStats(), degraded: true }, { headers: { "Cache-Control": "no-store", "X-Dianome-Stats": "degraded:sql-error" } });
  }
}

// GET /v1/stats/sessions: 7-day aggregates of the schema-3 session reports (Phase 5b run() telemetry), from the
// dianome_sessions dataset through the Analytics Engine SQL API, cached in KV for 5 minutes like /v1/stats/loads.
// Column map (telemetry.ts sessionDataPoint): blob1 model, blob2 variant, blob3 mode, blob4 plan_policy, blob5 cache_mode,
// blob6 browser; double1 N, double2 L, double3 prompt_tokens, double4 new_tokens, double5 client_ms, double6 server_busy_ms,
// double7 rtt_ms, double8 tok_per_s, double9 webgpu. Never any prompt or output content: none is ever written.

import { json } from "./http";
import { STATS_TTL_SECONDS, WINDOW_HOURS, iso, num, round, runQuery, str } from "./stats";
import type { Env, SessionStats } from "./types";

export const SESSIONS_CACHE_KEY = "stats:sessions:v1";
const DATASET = "dianome_sessions";
const WINDOW = `WHERE timestamp > NOW() - INTERVAL '${WINDOW_HOURS / 24}' DAY`;
const W = (col: string, q = 0.5) => `quantileExactWeighted(${q})(${col}, _sample_interval)`;
const SESSIONS = "sum(_sample_interval) AS sessions";

export const SESSION_QUERIES = {
  by_model_mode: `SELECT blob1 AS model, blob3 AS mode, ${SESSIONS}, ${W("double8")} AS p50_tok_per_s, ${W("double1")} AS p50_n, max(double2) AS L,
    ${W("double6")} AS p50_server_busy_ms, ${W("double7")} AS p50_rtt_ms, sum(_sample_interval * double4) AS new_tokens
    FROM ${DATASET} ${WINDOW} GROUP BY blob1, blob3 ORDER BY sessions DESC LIMIT 250 FORMAT JSON`,
  by_policy: `SELECT blob4 AS policy, blob3 AS mode, ${SESSIONS} FROM ${DATASET} ${WINDOW} GROUP BY blob4, blob3 ORDER BY sessions DESC LIMIT 64 FORMAT JSON`,
  by_browser: `SELECT blob6 AS browser, ${SESSIONS}, ${W("double8")} AS p50_tok_per_s, avg(double9) AS webgpu_share
    FROM ${DATASET} ${WINDOW} GROUP BY blob6 ORDER BY sessions DESC LIMIT 16 FORMAT JSON`,
} as const;

export function emptySessionStats(now = new Date()): SessionStats {
  return {
    since: iso(new Date(now.getTime() - WINDOW_HOURS * 3_600_000)), window_hours: WINDOW_HOURS, computed_at: iso(now),
    by_model_mode: [], by_policy: [], by_browser: [],
  };
}
export const hasSessionRows = (s: SessionStats): boolean => s.by_model_mode.length > 0 || s.by_policy.length > 0 || s.by_browser.length > 0;

export async function computeSessionStats(env: Env, now = new Date()): Promise<SessionStats> {
  const [mm, pol, br] = await Promise.all([runQuery(env, SESSION_QUERIES.by_model_mode), runQuery(env, SESSION_QUERIES.by_policy), runQuery(env, SESSION_QUERIES.by_browser)]);
  const out = emptySessionStats(now);
  out.by_model_mode = mm.map((r) => ({
    model: str(r.model), mode: str(r.mode), sessions: Math.round(num(r.sessions)), p50_tok_per_s: round(num(r.p50_tok_per_s), 1), p50_n: Math.round(num(r.p50_n)),
    L: Math.round(num(r.L)), p50_server_busy_ms: Math.round(num(r.p50_server_busy_ms)), p50_rtt_ms: round(num(r.p50_rtt_ms), 1), new_tokens: Math.round(num(r.new_tokens)),
  }));
  out.by_policy = pol.map((r) => ({ policy: str(r.policy), mode: str(r.mode), sessions: Math.round(num(r.sessions)) }));
  out.by_browser = br.map((r) => ({ browser: str(r.browser), sessions: Math.round(num(r.sessions)), p50_tok_per_s: round(num(r.p50_tok_per_s), 1), webgpu_share: round(Math.min(1, Math.max(0, num(r.webgpu_share)))) }));
  return out;
}

function isCurrentShape(text: string): boolean {
  try { const v = JSON.parse(text) as Partial<SessionStats>; return typeof v.computed_at === "string" && Array.isArray(v.by_model_mode); } catch { return false; }
}

export async function sessionStats(env: Env, ctx: ExecutionContext): Promise<Response> {
  const headers = { "Cache-Control": `public, max-age=${STATS_TTL_SECONDS}` };
  if (!env.CF_ACCOUNT_ID || !env.CF_ANALYTICS_TOKEN) return json({ ...emptySessionStats(), degraded: true }, { headers: { ...headers, "X-Dianome-Stats": "degraded:no-token" } });
  const cached = await env.STATS_CACHE.get(SESSIONS_CACHE_KEY);
  if (cached && isCurrentShape(cached)) return new Response(cached, { headers: { ...headers, "Content-Type": "application/json; charset=utf-8", "X-Dianome-Stats": "kv-hit" } });
  try {
    const stats = await computeSessionStats(env);
    const text = JSON.stringify(stats);
    const rows = hasSessionRows(stats);
    if (rows) ctx.waitUntil(env.STATS_CACHE.put(SESSIONS_CACHE_KEY, text, { expirationTtl: STATS_TTL_SECONDS }));
    return new Response(text, { headers: { ...(rows ? headers : { "Cache-Control": "no-store" }), "Content-Type": "application/json; charset=utf-8", "X-Dianome-Stats": rows ? "computed" : "computed:empty-not-cached" } });
  } catch (e) {
    console.error("session stats degraded:", (e as Error).message);
    return json({ ...emptySessionStats(), degraded: true }, { headers: { "Cache-Control": "no-store", "X-Dianome-Stats": "degraded:sql-error" } });
  }
}

// Phase 6 dashboard API, all behind `Authorization: Bearer dk_live_…`:
//   GET    /v1/me/usage?hours=168 → hourly buckets and totals for the key's owner, plus a cost estimate for the
//                                   server share: server_busy_ms at the rate in /v1/split/rates (an estimate, labelled).
//   GET    /v1/me/keys            → every key with this owner (ids, dates, revoked), the caller's marked `current`.
//   POST   /v1/me/keys            → a new key under the same owner (shown once).
//   DELETE /v1/me/keys/:id        → revoke (the caller's own key included: the next session mint is refused).
// Usage is aggregated over all of the owner's keys so a rotated key keeps its history; each bucket also names its key.

import { error, json } from "./http";
import { KEY_ID_RE, createKey, lookupKeyById, ownerKeyIds, publicRecord, requireKey, revokeKey, type KeyRecord } from "./keys";
import { MAX_USAGE_HOURS, hourStart, readUsage, sumCounters, type UsageBucket, type UsageCounters } from "./metering";
import ratesJson from "../../../server/bench/rates.json";
import type { Env } from "./types";

interface RatesFile { gpu: string; usd_per_hour: number | null; source: string; retrieved: string }

export interface CostEstimate {
  estimate: true;
  basis: string;
  usd: number | null;
  usd_per_hour: number | null;
  gpu: string;
  source: string;
  retrieved: string;
}

export interface UsageResponse {
  key_id: string;
  owner: string;
  keys: string[];
  hours: number;
  since: string;
  until: string;
  buckets: (UsageBucket & { key_id: string })[];
  totals: UsageCounters;
  last_24h: UsageCounters;
  cost_estimate: CostEstimate;
}

/** Server-share cost from busy milliseconds at the stated GPU rate; null when rates.json has no rate. */
export function costEstimate(serverBusyMs: number, rates: RatesFile = ratesJson as RatesFile): CostEstimate {
  const usd = typeof rates.usd_per_hour === "number" ? Math.round((serverBusyMs / 3_600_000) * rates.usd_per_hour * 1e6) / 1e6 : null;
  return {
    estimate: true,
    basis: "server_busy_ms / 3.6e6 h × usd_per_hour from /v1/split/rates; busy time is model execution only, so this is a lower bound on what a server actually costs",
    usd, usd_per_hour: rates.usd_per_hour, gpu: rates.gpu, source: rates.source, retrieved: rates.retrieved,
  };
}

export async function usage(request: Request, env: Env): Promise<Response> {
  const auth = await requireKey(request, env);
  if (auth instanceof Response) return auth;
  const q = new URL(request.url).searchParams.get("hours");
  let hours = MAX_USAGE_HOURS;
  if (q !== null) {
    const n = Number(q);
    if (!Number.isInteger(n) || n < 1) return error(400, "bad_hours", `hours must be an integer in 1..${MAX_USAGE_HOURS}`);
    hours = Math.min(MAX_USAGE_HOURS, n);
  }
  const now = new Date();
  const ids = await ownerKeyIds(env, auth.record.owner);
  const keyIds = ids.includes(auth.record.id) ? ids : [auth.record.id, ...ids];
  const perKey = await Promise.all(keyIds.map(async (id) => (await readUsage(env, id, hours, now)).map((b) => ({ ...b, key_id: id }))));
  const buckets = perKey.flat().sort((a, b) => (a.hour < b.hour ? -1 : a.hour > b.hour ? 1 : a.key_id < b.key_id ? -1 : 1));
  const end = hourStart(now);
  const since = new Date(end.getTime() - (hours - 1) * 3_600_000).toISOString();
  const dayCut = new Date(end.getTime() - 23 * 3_600_000).toISOString();
  const totals = sumCounters(buckets);
  const last24 = sumCounters(buckets.filter((b) => b.hour >= dayCut));
  const body: UsageResponse = {
    key_id: auth.record.id, owner: auth.record.owner, keys: keyIds, hours, since, until: new Date(end.getTime() + 3_600_000).toISOString(),
    buckets, totals, last_24h: last24, cost_estimate: costEstimate(totals.server_busy_ms),
  };
  return json(body, { headers: { "Cache-Control": "no-store" } });
}

export async function listKeys(request: Request, env: Env): Promise<Response> {
  const auth = await requireKey(request, env);
  if (auth instanceof Response) return auth;
  const ids = await ownerKeyIds(env, auth.record.owner);
  const keyIds = ids.includes(auth.record.id) ? ids : [auth.record.id, ...ids];
  const records = (await Promise.all(keyIds.map((id) => lookupKeyById(env, id)))).filter((r): r is KeyRecord => r !== null);
  return json({ owner: auth.record.owner, keys: records.map((r) => ({ ...publicRecord(r), current: r.id === auth.record.id })) }, { headers: { "Cache-Control": "no-store" } });
}

export async function createOwnedKey(request: Request, env: Env): Promise<Response> {
  const auth = await requireKey(request, env);
  if (auth instanceof Response) return auth;
  const { key, record } = await createKey(env, auth.record.owner);
  return json({ key, ...publicRecord(record) }, { status: 201, headers: { "Cache-Control": "no-store" } });
}

export async function revokeOwnedKey(request: Request, env: Env, id: string): Promise<Response> {
  const auth = await requireKey(request, env);
  if (auth instanceof Response) return auth;
  if (!KEY_ID_RE.test(id)) return error(400, "bad_key_id");
  const target = await lookupKeyById(env, id);
  if (!target || target.owner !== auth.record.owner) return error(404, "key_not_found");
  const updated = await revokeKey(env, id);
  return json({ ...publicRecord(updated ?? target), current: id === auth.record.id }, { headers: { "Cache-Control": "no-store" } });
}

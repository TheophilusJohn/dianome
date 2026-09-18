// Phase 6: API keys, session gating (key OR allowed origin), metering buckets, the dashboard API, rate limits.
import { createExecutionContext, env, waitOnExecutionContext } from "cloudflare:test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import worker from "../src/index";
import { KEYS_PER_DAY, KEY_RE, createKey, generateKey, hashKey, lookupKey, lookupKeyById, revokeKey } from "../src/keys";
import { BUCKET_TTL_SECONDS, bucketKey, emptyCounters, hourStamp, layerTokens, readUsage, recordLoad, recordSession, sumCounters } from "../src/metering";
import { costEstimate } from "../src/dashboard-api";
import { dataPoint, sessionDataPoint } from "../src/telemetry";
import { verifyToken } from "../src/split";
import { validReport, validSession } from "./fixtures";

const BASE = "https://api.dianome.dev";
const ORIGIN = "http://localhost:5177";

async function call(path: string, init: RequestInit = {}, envOverride: Partial<typeof env> = {}, cf?: Record<string, unknown>) {
  const ctx = createExecutionContext();
  const req = new Request(BASE + path, init) as Request<unknown, IncomingRequestCfProperties>;
  if (cf) Object.defineProperty(req, "cf", { value: cf });
  const res = await worker.fetch(req, { ...env, ...envOverride }, ctx);
  await waitOnExecutionContext(ctx);
  return res;
}
const bearer = (key: string, extra: Record<string, string> = {}) => ({ Authorization: `Bearer ${key}`, ...extra });
const postJson = (path: string, body: unknown, headers: Record<string, string> = {}) =>
  call(path, { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) });

/** Wipes the KV keys a test may have left (the pool gives every file its own isolated storage, but tests in a file share it). */
async function wipe(prefix: string) {
  const l = await env.STATS_CACHE.list({ prefix });
  await Promise.all(l.keys.map((k) => env.STATS_CACHE.delete(k.name)));
}
beforeEach(async () => { for (const p of ["key:", "keyid:", "owner:", "usage:", "rl:"]) await wipe(p); });

describe("keys: create, hash, lookup, revoke", () => {
  it("generates dk_live_ keys with 32 base64url characters, never two alike", () => {
    const a = generateKey(), b = generateKey();
    expect(a).toMatch(KEY_RE); expect(b).toMatch(KEY_RE);
    expect(a).not.toBe(b);
    expect(a.length).toBe("dk_live_".length + 32);
  });
  it("stores only the SHA-256 and finds the record by key or by id; the owner is the key's own id", async () => {
    const { key, record } = await createKey(env);
    expect(record).toMatchObject({ owner: record.id, revoked: null, plan: "free" });
    expect(record.id).toMatch(/^k_[0-9a-f]{16}$/);
    const hash = await hashKey(key);
    expect(await env.STATS_CACHE.get(`key:${hash}`)).toBe(JSON.stringify(record));
    expect(await env.STATS_CACHE.get(`keyid:${record.id}`)).toBe(hash);
    // the plaintext appears nowhere in KV
    const all = await env.STATS_CACHE.list({});
    for (const k of all.keys) { expect(k.name).not.toContain(key); expect(await env.STATS_CACHE.get(k.name)).not.toContain(key); }
    expect(await lookupKey(env, key)).toEqual(record);
    expect(await lookupKeyById(env, record.id)).toEqual(record);
    expect(await lookupKey(env, "dk_live_" + "A".repeat(32))).toBeNull();
    expect(await lookupKey(env, "not-a-key")).toBeNull();
    expect(await lookupKeyById(env, "k_0000000000000000")).toBeNull();
  });
  it("revoke stamps a time, is idempotent, and lookups then report it", async () => {
    const { key, record } = await createKey(env);
    const r1 = await revokeKey(env, record.id);
    expect(r1?.revoked).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    const r2 = await revokeKey(env, record.id);
    expect(r2?.revoked).toBe(r1?.revoked);
    expect((await lookupKey(env, key))?.revoked).toBe(r1?.revoked);
    expect(await revokeKey(env, "k_ffffffffffffffff")).toBeNull();
  });
  it("POST /v1/keys is open, returns the key once with its id, and is limited to 10 per day per country:colo", async () => {
    const res = await call("/v1/keys", { method: "POST" });
    expect(res.status).toBe(201);
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    const body = await res.json() as { key: string; id: string; owner: string; plan: string; revoked: null };
    expect(body.key).toMatch(KEY_RE);
    expect(body).toMatchObject({ owner: body.id, plan: "free", revoked: null });
    expect((await lookupKey(env, body.key))?.id).toBe(body.id);
    expect((await call("/v1/keys")).status).toBe(405);
    for (let i = 1; i < KEYS_PER_DAY; i++) expect((await call("/v1/keys", { method: "POST" })).status).toBe(201);
    const over = await call("/v1/keys", { method: "POST" });
    expect(over.status).toBe(429);
    expect(await over.json()).toMatchObject({ error: "rate_limited" });
    // another location has its own counter
    expect((await call("/v1/keys", { method: "POST" }, {}, { country: "DE", colo: "FRA" })).status).toBe(201);
  });
});

describe("split session: key OR allowed origin", () => {
  const session = (headers: Record<string, string>, body: unknown = {}) => postJson("/v1/split/session", body, headers);

  it("a keyed request needs no origin, gets a token for that origin-less client, and names the key id", async () => {
    const { key, record } = await createKey(env);
    const res = await session(bearer(key));
    expect(res.status).toBe(200);
    const body = await res.json() as { token: string; key_id: string; sid: string };
    expect(body.key_id).toBe(record.id);
    expect(JSON.stringify(body)).not.toContain(key);
    const payload = await verifyToken(body.token, "test-signing-key");
    expect(payload).toMatchObject({ sid: body.sid, model: "qwen2.5-0.5b-instruct", origin: "" });
  });
  it("a keyed request from any origin is accepted (the demo allowlist does not apply), the origin goes into the token", async () => {
    const { key } = await createKey(env);
    const res = await session(bearer(key, { Origin: "https://customer.example" }));
    expect(res.status).toBe(200);
    const body = await res.json() as { token: string };
    expect((await verifyToken(body.token, "test-signing-key"))?.origin).toBe("https://customer.example");
  });
  it("a key with allowed_origins is limited to them", async () => {
    const { key, record } = await createKey(env);
    const hash = await env.STATS_CACHE.get(`keyid:${record.id}`);
    await env.STATS_CACHE.put(`key:${hash}`, JSON.stringify({ ...record, allowed_origins: ["https://ok.example/"] }));
    expect((await session(bearer(key, { Origin: "https://ok.example" }))).status).toBe(200);
    expect((await session(bearer(key, { Origin: "https://other.example" }))).status).toBe(403);
    expect((await session(bearer(key))).status).toBe(403);
  });
  it("without a key the demo origin allowlist still gates, and the response carries no key id", async () => {
    const ok = await session({ Origin: ORIGIN });
    expect(ok.status).toBe(200);
    expect("key_id" in (await ok.json() as Record<string, unknown>)).toBe(false);
    expect((await session({ Origin: "https://evil.example" })).status).toBe(403);
    expect((await session({})).status).toBe(403);
  });
  it("an unknown, malformed or revoked key is refused even from an allowed origin", async () => {
    const { key, record } = await createKey(env);
    const unknown = await session(bearer("dk_live_" + "B".repeat(32), { Origin: ORIGIN }));
    expect(unknown.status).toBe(401);
    expect(await unknown.json()).toMatchObject({ error: "invalid_key" });
    expect((await session({ Authorization: "Bearer nope", Origin: ORIGIN })).status).toBe(401);
    await revokeKey(env, record.id);
    const revoked = await session(bearer(key, { Origin: ORIGIN }));
    expect(revoked.status).toBe(403);
    expect(await revoked.json()).toMatchObject({ error: "key_revoked" });
  });
});

describe("metering", () => {
  afterEach(() => vi.useRealTimers());

  it("layer-tokens: client N × new, server (L − N) × new plus the head when the server ran it", () => {
    expect(layerTokens({ mode: "split", N: 12, L: 24, new_tokens: 64 })).toEqual({ client: 768, server: 12 * 64 + 64 });
    expect(layerTokens({ mode: "server", N: 0, L: 24, new_tokens: 10 })).toEqual({ client: 0, server: 250 });
    expect(layerTokens({ mode: "local", N: 24, L: 24, new_tokens: 10 })).toEqual({ client: 240, server: 0 });
    expect(layerTokens({ mode: "split", N: 24, L: 24, new_tokens: 10 })).toEqual({ client: 240, server: 10 });
  });
  it("read-modify-write accumulates into the hour bucket with a TTL; a second session and a load add up", async () => {
    vi.useFakeTimers({ now: Date.UTC(2026, 8, 18, 14, 25), toFake: ["Date"] });
    const at = new Date();
    const s = validSession() as never;
    const first = await recordSession(env, "k_0123456789abcdef", s, at);
    expect(first).toMatchObject({ sessions: 1, client_layer_tokens: 768, server_layer_tokens: 12 * 64 + 64, new_tokens: 64, server_busy_ms: 710.2, bytes_served: 0, modes: { local: 0, split: 1, server: 0 } });
    const second = await recordSession(env, "k_0123456789abcdef", { ...validSession(), mode: "server", N: 0, new_tokens: 10, server_busy_ms: 100 } as never, at);
    expect(second).toMatchObject({ sessions: 2, client_layer_tokens: 768, server_layer_tokens: 12 * 64 + 64 + 250, new_tokens: 74, server_busy_ms: 810.2, modes: { split: 1, server: 1 } });
    const withLoad = await recordLoad(env, "k_0123456789abcdef", validReport() as never, at);
    expect(withLoad.bytes_served).toBe(323893760);
    expect(withLoad.sessions).toBe(2);
    const key = bucketKey("k_0123456789abcdef", hourStamp(at));
    expect(key).toBe("usage:k_0123456789abcdef:2026091814");
    const meta = await env.STATS_CACHE.getWithMetadata(key);
    expect(JSON.parse(meta.value!)).toEqual(withLoad);
    const list = await env.STATS_CACHE.list({ prefix: "usage:k_0123456789abcdef:" });
    // KV stamps expiry from its own (real) clock while Date is faked here: allow a day either way
    const realNow = performance.timeOrigin + performance.now();
    expect(list.keys[0]?.expiration).toBeGreaterThan(realNow / 1000 + BUCKET_TTL_SECONDS - 86400);
    expect(list.keys[0]?.expiration).toBeLessThan(realNow / 1000 + BUCKET_TTL_SECONDS + 86400);
  });
  it("retries a failed KV write with a fresh read and gives up after three attempts", async () => {
    const real = env.STATS_CACHE.put.bind(env.STATS_CACHE);
    let fails = 2;
    const spy = vi.spyOn(env.STATS_CACHE, "put").mockImplementation(async (...args) => { if (fails-- > 0) throw new Error("kv write failed"); return real(...(args as Parameters<typeof real>)); });
    const out = await recordSession(env, "k_00000000000000aa", validSession() as never);
    expect(out.sessions).toBe(1);
    expect(spy).toHaveBeenCalledTimes(3);
    spy.mockRestore();
    const always = vi.spyOn(env.STATS_CACHE, "put").mockRejectedValue(new Error("kv down"));
    await expect(recordSession(env, "k_00000000000000aa", validSession() as never)).rejects.toThrow("kv down");
    expect(always).toHaveBeenCalledTimes(3);
    always.mockRestore();
  });
  it("readUsage returns the window's buckets oldest first and omits hours with nothing", async () => {
    const id = "k_00000000000000bb";
    const t = Date.UTC(2026, 8, 18, 14, 0);
    await recordSession(env, id, validSession() as never, new Date(t));
    await recordSession(env, id, validSession() as never, new Date(t - 5 * 3_600_000));
    await recordSession(env, id, validSession() as never, new Date(t - 200 * 3_600_000)); // outside 168 h
    const all = await readUsage(env, id, 168, new Date(t + 60_000));
    expect(all.map((b) => b.hour)).toEqual(["2026-09-18T09:00:00.000Z", "2026-09-18T14:00:00.000Z"]);
    const recent = await readUsage(env, id, 2, new Date(t + 60_000));
    expect(recent.map((b) => b.hour)).toEqual(["2026-09-18T14:00:00.000Z"]);
    expect(sumCounters(all).sessions).toBe(2);
    expect(sumCounters([])).toEqual(emptyCounters());
  });
  it("cost estimate: busy ms at the stated rate, labelled an estimate; null without a rate", () => {
    const c = costEstimate(3_600_000, { gpu: "g", usd_per_hour: 0.49, source: "s", retrieved: "2026-09-17" });
    expect(c).toMatchObject({ estimate: true, usd: 0.49, usd_per_hour: 0.49, gpu: "g" });
    expect(costEstimate(1_000, { gpu: "g", usd_per_hour: 0.49, source: "s", retrieved: "r" }).usd).toBeCloseTo(0.49 / 3600, 6);
    expect(costEstimate(1_000, { gpu: "g", usd_per_hour: null, source: "s", retrieved: "r" }).usd).toBeNull();
  });
});

describe("telemetry with a key", () => {
  it("records the key id (never the key) in blob 9 of loads and blob 10 of sessions, and meters the bucket", async () => {
    const { key, record } = await createKey(env);
    const s = await postJson("/v1/telemetry/load", validSession(), bearer(key));
    expect(s.status).toBe(202);
    const l = await postJson("/v1/telemetry/load", validReport(), bearer(key));
    expect(l.status).toBe(202);
    const buckets = await readUsage(env, record.id, 1);
    expect(buckets).toHaveLength(1);
    expect(buckets[0]).toMatchObject({ sessions: 1, client_layer_tokens: 768, bytes_served: 323893760, modes: { split: 1 } });
    expect(sessionDataPoint(validSession() as never, "US", "ATL", "h", record.id).blobs?.[9]).toBe(record.id);
    expect(sessionDataPoint(validSession() as never, "US", "ATL", "h").blobs?.[9]).toBe("");
    expect(dataPoint(validReport() as never, "US", "ATL", "h", record.id).blobs?.[8]).toBe(record.id);
    expect(dataPoint(validReport() as never, "US", "ATL", "h").blobs?.[8]).toBe("");
  });
  it("accepts a body key_id that matches the bearer and rejects one that does not, or one without a bearer", async () => {
    const { key, record } = await createKey(env);
    expect((await postJson("/v1/telemetry/load", { ...validSession(), key_id: record.id }, bearer(key))).status).toBe(202);
    const mismatch = await postJson("/v1/telemetry/load", { ...validSession(), key_id: "k_0000000000000000" }, bearer(key));
    expect(mismatch.status).toBe(400);
    expect(await mismatch.json()).toMatchObject({ detail: "key_id does not match the bearer key" });
    expect((await postJson("/v1/telemetry/load", { ...validSession(), key_id: record.id })).status).toBe(400);
    expect((await postJson("/v1/telemetry/load", { ...validSession(), key_id: "bad" }, bearer(key))).status).toBe(400);
    expect((await postJson("/v1/telemetry/load", { ...validSession(), key: key }, bearer(key))).status).toBe(400); // the key itself is never a field
  });
  it("a keyless report is still accepted and meters nothing; a bad or revoked key is refused", async () => {
    expect((await postJson("/v1/telemetry/load", validSession())).status).toBe(202);
    expect((await env.STATS_CACHE.list({ prefix: "usage:" })).keys).toHaveLength(0);
    expect((await postJson("/v1/telemetry/load", validSession(), bearer("dk_live_" + "C".repeat(32)))).status).toBe(401);
    const { key, record } = await createKey(env);
    await revokeKey(env, record.id);
    expect((await postJson("/v1/telemetry/load", validSession(), bearer(key))).status).toBe(403);
  });
});

describe("dashboard API", () => {
  it("GET /v1/me/usage: shape, totals, last 24 h, cost estimate, hours clamp; 401 without a key", async () => {
    const { key, record } = await createKey(env);
    const now = new Date();
    await recordSession(env, record.id, validSession() as never, now);
    await recordSession(env, record.id, { ...validSession(), mode: "server", N: 0, new_tokens: 10, server_busy_ms: 100 } as never, new Date(now.getTime() - 30 * 3_600_000));
    const res = await call("/v1/me/usage?hours=168", { headers: bearer(key) });
    expect(res.status).toBe(200);
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    const body = await res.json() as Record<string, unknown> & { buckets: unknown[]; totals: Record<string, unknown>; last_24h: Record<string, unknown>; cost_estimate: Record<string, unknown> };
    expect(Object.keys(body).sort()).toEqual(["buckets", "cost_estimate", "hours", "key_id", "keys", "last_24h", "owner", "since", "totals", "until"]);
    expect(body).toMatchObject({ key_id: record.id, owner: record.id, keys: [record.id], hours: 168 });
    expect(body.buckets).toHaveLength(2);
    expect(body.buckets[0]).toMatchObject({ key_id: record.id, sessions: 1, modes: { server: 1 } });
    expect(body.totals).toMatchObject({ sessions: 2, client_layer_tokens: 768, server_layer_tokens: 12 * 64 + 64 + 250, server_busy_ms: 810.2, modes: { local: 0, split: 1, server: 1 } });
    expect(body.last_24h).toMatchObject({ sessions: 1, modes: { split: 1 } });
    expect(body.cost_estimate).toMatchObject({ estimate: true });
    expect(typeof body.cost_estimate.basis).toBe("string");
    expect(JSON.stringify(body)).not.toContain(key);
    const clamped = await call("/v1/me/usage?hours=9999", { headers: bearer(key) });
    expect(((await clamped.json()) as { hours: number }).hours).toBe(168);
    expect((await call("/v1/me/usage?hours=0", { headers: bearer(key) })).status).toBe(400);
    expect((await call("/v1/me/usage?hours=abc", { headers: bearer(key) })).status).toBe(400);
    const none = await call("/v1/me/usage");
    expect(none.status).toBe(401);
    expect(await none.json()).toMatchObject({ error: "key_required" });
    expect((await call("/v1/me/usage", { headers: bearer("dk_live_" + "D".repeat(32)) })).status).toBe(401);
  });
  it("keys: list marks the caller, POST creates a sibling under the same owner, DELETE revokes and the next session is refused", async () => {
    const { key, record } = await createKey(env);
    const list1 = await call("/v1/me/keys", { headers: bearer(key) });
    expect(list1.status).toBe(200);
    expect(await list1.json()).toEqual({ owner: record.id, keys: [{ id: record.id, owner: record.id, created: record.created, revoked: null, plan: "free", current: true }] });
    const made = await call("/v1/me/keys", { method: "POST", headers: bearer(key) });
    expect(made.status).toBe(201);
    const sibling = await made.json() as { key: string; id: string; owner: string };
    expect(sibling.owner).toBe(record.id);
    expect(sibling.key).toMatch(KEY_RE);
    const list2 = await (await call("/v1/me/keys", { headers: bearer(sibling.key) })).json() as { keys: { id: string; current: boolean }[] };
    expect(list2.keys.map((k) => [k.id, k.current])).toEqual([[record.id, false], [sibling.id, true]]);
    // usage is per owner: a session on the sibling shows up for the first key too, with its own key_id
    await postJson("/v1/telemetry/load", validSession(), bearer(sibling.key));
    const usage = await (await call("/v1/me/usage", { headers: bearer(key) })).json() as { keys: string[]; buckets: { key_id: string }[]; totals: { sessions: number } };
    expect(usage.keys).toEqual([record.id, sibling.id]);
    expect(usage.buckets.map((b) => b.key_id)).toEqual([sibling.id]);
    expect(usage.totals.sessions).toBe(1);
    // revoke the sibling from the first key, then the sibling can mint no session and reach no dashboard route
    const del = await call(`/v1/me/keys/${sibling.id}`, { method: "DELETE", headers: bearer(key) });
    expect(del.status).toBe(200);
    expect(await del.json()).toMatchObject({ id: sibling.id, current: false });
    expect(((await del.json().catch(() => null)) ?? true)).toBeTruthy();
    const refused = await postJson("/v1/split/session", {}, bearer(sibling.key, { Origin: ORIGIN }));
    expect(refused.status).toBe(403);
    expect((await call("/v1/me/keys", { headers: bearer(sibling.key) })).status).toBe(403);
    // a foreign key cannot be revoked (404, not 403, so ids are not confirmed), nor a malformed id
    const { record: other } = await createKey(env);
    expect((await call(`/v1/me/keys/${other.id}`, { method: "DELETE", headers: bearer(key) })).status).toBe(404);
    expect((await call("/v1/me/keys/zzz", { method: "DELETE", headers: bearer(key) })).status).toBe(400);
    expect((await call(`/v1/me/keys/${record.id}`, { method: "GET", headers: bearer(key) })).status).toBe(405);
    // revoking one's own key works and the next session mint is refused
    expect((await call(`/v1/me/keys/${record.id}`, { method: "DELETE", headers: bearer(key) })).status).toBe(200);
    expect((await postJson("/v1/split/session", {}, bearer(key))).status).toBe(403);
  });
  it("preflight allows Authorization and DELETE", async () => {
    const res = await call("/v1/me/keys", { method: "OPTIONS", headers: { Origin: "https://dianome.dev", "Access-Control-Request-Method": "DELETE", "Access-Control-Request-Headers": "authorization" } });
    expect(res.status).toBe(204);
    expect(res.headers.get("Access-Control-Allow-Methods")).toContain("DELETE");
    expect(res.headers.get("Access-Control-Allow-Headers")).toContain("Authorization");
  });
});

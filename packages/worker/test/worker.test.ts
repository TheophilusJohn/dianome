import { createExecutionContext, env, waitOnExecutionContext } from "cloudflare:test";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import worker from "../src/index";
import { STATS_CACHE_KEY } from "../src/stats";
import { SESSIONS_CACHE_KEY } from "../src/sessions";
import { dataPoint, sessionDataPoint } from "../src/telemetry";
import { PLAN_CACHE_MS, SESSION_TTL_SECONDS, canonicalPayload, mintToken, resetPlanCache, verifyToken } from "../src/split";
import { MODEL_ID, seedManifest, validReport, validReportV2, validSession } from "./fixtures";

const BASE = "https://api.dianome.dev";

async function call(path: string, init: RequestInit = {}, envOverride: Partial<typeof env> = {}) {
  const ctx = createExecutionContext();
  const req = new Request(BASE + path, init) as Request<unknown, IncomingRequestCfProperties>;
  const res = await worker.fetch(req, { ...env, ...envOverride }, ctx);
  await waitOnExecutionContext(ctx);
  return res;
}

const post = (body: unknown, headers: Record<string, string> = {}) =>
  call("/v1/telemetry/load", { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: typeof body === "string" ? body : JSON.stringify(body) });

let seeded: { bytes: Uint8Array; sha: string };
beforeAll(async () => { seeded = await seedManifest(); });

describe("CORS", () => {
  it("answers preflight with 204 and the allow headers", async () => {
    const res = await call("/v1/telemetry/load", { method: "OPTIONS", headers: { Origin: "https://example.com", "Access-Control-Request-Method": "POST" } });
    expect(res.status).toBe(204);
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe("*");
    expect(res.headers.get("Access-Control-Allow-Methods")).toContain("POST");
    expect(res.headers.get("Access-Control-Allow-Headers")).toContain("Content-Type");
    expect(res.headers.get("Access-Control-Max-Age")).toBe("86400");
  });
  it("adds Access-Control-Allow-Origin to every response, errors included", async () => {
    const res = await call("/nope");
    expect(res.status).toBe(404);
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe("*");
    expect(await res.json()).toEqual({ error: "not_found" });
  });
});

describe("healthz", () => {
  it("reports ok and the baked version", async () => {
    const res = await call("/healthz");
    expect(await res.json()).toEqual({ ok: true, version: "test" });
  });
});

describe("models", () => {
  it("lists manifest ids from the manifests/ prefix", async () => {
    await seedManifest("another-model");
    const res = await call("/v1/models");
    expect(res.status).toBe(200);
    expect(res.headers.get("Cache-Control")).toBe("public, max-age=60");
    expect(await res.json()).toEqual({ models: ["another-model", MODEL_ID] });
  });
});

describe("latest manifest", () => {
  it("200: body, ETag passthrough, sha header, max-age=60", async () => {
    const res = await call(`/v1/models/${MODEL_ID}/manifest`);
    expect(res.status).toBe(200);
    expect(res.headers.get("Cache-Control")).toBe("public, max-age=60");
    expect(res.headers.get("ETag")).toMatch(/^"[0-9a-f]+"$/);
    expect(res.headers.get("X-Dianome-Manifest-Sha")).toBe(seeded.sha);
    expect(res.headers.get("Access-Control-Expose-Headers")).toContain("X-Dianome-Manifest-Sha");
    expect(new Uint8Array(await res.arrayBuffer())).toEqual(seeded.bytes);
  });
  it("304 on a matching If-None-Match, sha header still present", async () => {
    const first = await call(`/v1/models/${MODEL_ID}/manifest`);
    const etag = first.headers.get("ETag")!;
    const res = await call(`/v1/models/${MODEL_ID}/manifest`, { headers: { "If-None-Match": etag } });
    expect(res.status).toBe(304);
    expect(res.headers.get("ETag")).toBe(etag);
    expect(res.headers.get("X-Dianome-Manifest-Sha")).toBe(seeded.sha);
    expect(await res.text()).toBe("");
  });
  it("200 again on a stale If-None-Match", async () => {
    const res = await call(`/v1/models/${MODEL_ID}/manifest`, { headers: { "If-None-Match": '"deadbeef"' } });
    expect(res.status).toBe(200);
  });
  it("404 for an unknown id, 400 for a malformed one", async () => {
    expect((await call("/v1/models/nope/manifest")).status).toBe(404);
    expect((await call("/v1/models/..%2Fetc/manifest")).status).toBe(400);
  });
});

describe("hashed manifest", () => {
  it("serves the immutable manifest with immutable headers", async () => {
    const res = await call(`/v1/models/${MODEL_ID}/manifest/${seeded.sha}`);
    expect(res.status).toBe(200);
    expect(res.headers.get("Cache-Control")).toBe("public, max-age=31536000, immutable");
    expect(res.headers.get("ETag")).toMatch(/^".+"$/);
    expect(res.headers.get("X-Dianome-Manifest-Sha")).toBe(seeded.sha);
    expect(new Uint8Array(await res.arrayBuffer())).toEqual(seeded.bytes);
  });
  it("304 on If-None-Match, 400 on a malformed sha, 404 when absent", async () => {
    const first = await call(`/v1/models/${MODEL_ID}/manifest/${seeded.sha}`);
    const res = await call(`/v1/models/${MODEL_ID}/manifest/${seeded.sha}`, { headers: { "If-None-Match": first.headers.get("ETag")! } });
    expect(res.status).toBe(304);
    expect((await call(`/v1/models/${MODEL_ID}/manifest/abc`)).status).toBe(400);
    expect((await call(`/v1/models/${MODEL_ID}/manifest/${"0".repeat(64)}`)).status).toBe(404);
  });
});

describe("telemetry", () => {
  it("accepts a valid report with 202 and an empty body", async () => {
    const res = await post(validReport());
    expect(res.status).toBe(202);
    expect(await res.text()).toBe("");
  });
  it("rejects a body over 4 KB", async () => {
    const res = await post({ ...validReport(), variant: "q4" + "x".repeat(5000) });
    expect(res.status).toBe(413);
  });
  it("rejects an oversize declared Content-Length before reading", async () => {
    const res = await post(validReport(), { "Content-Length": "999999" });
    expect(res.status).toBe(413);
  });
  it("rejects an unknown field", async () => {
    const res = await post({ ...validReport(), ip: "1.2.3.4" });
    expect(res.status).toBe(400);
    expect(await res.json()).toMatchObject({ error: "invalid_report", detail: "unknown field: ip" });
  });
  it("rejects a bad enum value", async () => {
    expect((await post({ ...validReport(), source: "disk" })).status).toBe(400);
    expect((await post({ ...validReport(), browser: "opera" })).status).toBe(400);
  });
  it("rejects missing fields, wrong types, wrong schema and cache_hits > chunks", async () => {
    const { webgpu: _w, ...missing } = validReport();
    expect((await post(missing)).status).toBe(400);
    expect((await post({ ...validReport(), ms: "8420" })).status).toBe(400);
    expect((await post({ ...validReport(), schema: 3 })).status).toBe(400);
    expect((await post({ ...validReport(), cache_hits: 43 })).status).toBe(400);
    expect((await post("not json")).status).toBe(400);
  });
  it("accepts a schema-2 report with every optional field, and one with none", async () => {
    expect((await post(validReportV2())).status).toBe(202);
    expect((await post({ ...validReport(), schema: 2 })).status).toBe(202);
    expect((await post({ ...validReport(), schema: 2, cache_mode: "none" })).status).toBe(202);
  });
  it("rejects schema-2 fields on a schema-1 report, bad v2 values, and schema 3", async () => {
    const r1 = await post({ ...validReport(), cache_mode: "per-site" });
    expect(r1.status).toBe(400);
    expect(await r1.json()).toMatchObject({ detail: "unknown field: cache_mode" });
    expect((await post({ ...validReportV2(), cache_mode: "disk" })).status).toBe(400);
    expect((await post({ ...validReportV2(), verify_ms: -1 })).status).toBe(400);
    expect((await post({ ...validReportV2(), bytes_per_second: 1.5 })).status).toBe(400);
    expect((await post({ ...validReportV2(), quota_bytes: "lots" })).status).toBe(400);
    expect((await post({ ...validReportV2(), ip: "1.2.3.4" })).status).toBe(400);
    expect((await post({ ...validReport(), schema: 3 })).status).toBe(400);
  });
  it("maps schema 1 and 2 onto the Analytics Engine columns (absent doubles are -1, cache_mode is blob8, key id blob9)", () => {
    const p1 = dataPoint(validReport() as never, "US", "ATL", "2026-09-16T23:00:00.000Z");
    expect(p1.blobs).toEqual(["qwen2.5-0.5b-instruct", "q4", "network", "chrome", "US", "ATL", "2026-09-16T23:00:00.000Z", "", ""]);
    expect(p1.doubles).toEqual([323893760, 42, 8420, 0, 1, -1, -1, -1, -1, -1]);
    expect(p1.indexes).toEqual(["qwen2.5-0.5b-instruct"]);
    const p2 = dataPoint(validReportV2() as never, "US", "ATL", "h");
    expect(p2.blobs?.[7]).toBe("per-site");
    expect(p2.doubles).toEqual([323893760, 42, 8420, 17, 1, 38467000, 210, 0, 10760000000, 4294967292]);
    const p3 = dataPoint({ ...validReportV2(), transfer_ms: undefined, cache_mode: undefined } as never, "US", "ATL", "h");
    expect(p3.doubles?.[7]).toBe(-1);
    expect(p3.blobs?.[7]).toBe("");
  });
  it("rejects non-JSON content types and GET", async () => {
    expect((await call("/v1/telemetry/load", { method: "POST", headers: { "Content-Type": "text/plain" }, body: "{}" })).status).toBe(415);
    expect((await call("/v1/telemetry/load")).status).toBe(405);
  });
});

describe("telemetry v3 (session reports)", () => {
  it("accepts a valid session report with 202", async () => {
    expect((await post(validSession())).status).toBe(202);
    expect((await post({ ...validSession(), mode: "local", N: 24, server_busy_ms: 0, rtt_ms: 0 })).status).toBe(202);
    expect((await post({ ...validSession(), mode: "server", N: 0, client_ms: 0 })).status).toBe(202);
  });
  it("rejects unknown fields, missing fields, bad enums, mode/N mismatches and prompt content", async () => {
    const r = await post({ ...validSession(), prompt: "secret" });
    expect(r.status).toBe(400);
    expect(await r.json()).toMatchObject({ detail: "unknown field: prompt" });
    const { tok_per_s: _t, ...missing } = validSession();
    expect((await post(missing)).status).toBe(400);
    expect((await post({ ...validSession(), mode: "hybrid" })).status).toBe(400);
    expect((await post({ ...validSession(), plan_policy: "cheap" })).status).toBe(400);
    expect((await post({ ...validSession(), mode: "server", N: 3 })).status).toBe(400);
    expect((await post({ ...validSession(), mode: "local", N: 12 })).status).toBe(400);
    expect((await post({ ...validSession(), N: 30 })).status).toBe(400);
    expect((await post({ ...validSession(), client_ms: -1 })).status).toBe(400);
    expect((await post({ ...validSession(), tok_per_s: "fast" })).status).toBe(400);
  });
  it("maps schema 3 onto the sessions dataset columns", () => {
    const p = sessionDataPoint(validSession() as never, "US", "ATL", "h");
    expect(p.blobs).toEqual(["qwen2.5-0.5b-instruct", "q4", "split", "cost", "per-site", "chrome", "US", "ATL", "h", ""]);
    expect(p.doubles).toEqual([12, 24, 40, 64, 620.5, 710.2, 3.8, 41.7, 1]);
    expect(p.indexes).toEqual(["qwen2.5-0.5b-instruct"]);
  });
});

describe("split: session tokens", () => {
  const ORIGIN = "http://localhost:5177";
  const session = (body: unknown = {}, headers: Record<string, string> = { Origin: ORIGIN }, envOverride = {}) =>
    call("/v1/split/session", { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) }, envOverride);

  it("mints a token that verifies, with the payload fields and a 1 h expiry", async () => {
    const res = await session({ max_ctx: 512 });
    expect(res.status).toBe(200);
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    const body = await res.json() as { url: string; token: string; expires_at: string; sid: string; model: string; max_ctx: number };
    expect(body.url).toBe("ws://127.0.0.1:8765");
    expect(body.model).toBe("qwen2.5-0.5b-instruct");
    expect(body.max_ctx).toBe(512);
    const exp = Date.parse(body.expires_at) / 1000;
    expect(exp - Date.now() / 1000).toBeGreaterThan(SESSION_TTL_SECONDS - 5);
    const payload = await verifyToken(body.token, "test-signing-key");
    expect(payload).toMatchObject({ sid: body.sid, model: "qwen2.5-0.5b-instruct", max_ctx: 512, origin: ORIGIN });
    expect(payload!.exp).toBe(Math.floor(exp));
    // the payload half is canonical JSON (sorted keys, no whitespace), the same bytes the Python server signs
    const raw = atob(body.token.split(".")[0]!.replace(/-/g, "+").replace(/_/g, "/"));
    expect(raw).toBe(canonicalPayload(payload!));
    expect(raw.startsWith('{"exp":')).toBe(true);
  });
  it("round trip: mint → verify; expiry and a bad key are refused", async () => {
    const now = 1_800_000_000;
    const t = await mintToken("k", { sid: "s", model: "m", exp: now + 60, max_ctx: 64, origin: "https://a" });
    expect(await verifyToken(t, "k", now)).toEqual({ sid: "s", model: "m", exp: now + 60, max_ctx: 64, origin: "https://a" });
    expect(await verifyToken(t, "k", now + 61)).toBeNull();
    expect(await verifyToken(t, "other", now)).toBeNull();
    expect(await verifyToken(t.slice(0, -2) + "zz", "k", now)).toBeNull();
    expect(await verifyToken("garbage", "k", now)).toBeNull();
    expect(await verifyToken("a.b.c", "k", now)).toBeNull();
  });
  it("refuses a missing or foreign Origin, an unknown model, a bad max_ctx, and non-POST", async () => {
    expect((await session({}, {})).status).toBe(403);
    expect((await session({}, { Origin: "https://evil.example" })).status).toBe(403);
    expect((await session({ model: "other-model" })).status).toBe(400);
    expect((await session({ max_ctx: 0 })).status).toBe(400);
    expect((await session({ max_ctx: "big" })).status).toBe(400);
    expect((await session({ max_ctx: 99999 })).status).toBe(200);
    expect(((await (await session({ max_ctx: 99999 })).json()) as { max_ctx: number }).max_ctx).toBe(8192);
    expect((await call("/v1/split/session")).status).toBe(405);
  });
  it("picks the server per model: ?model= wins over the body, the body over the map's first entry", async () => {
    const q = await call("/v1/split/session?model=qwen2.5-7b-instruct", { method: "POST", headers: { "Content-Type": "application/json", Origin: ORIGIN }, body: JSON.stringify({ model: "qwen2.5-0.5b-instruct" }) });
    expect(q.status).toBe(200);
    const qb = await q.json() as { url: string; model: string; token: string };
    expect(qb).toMatchObject({ url: "ws://127.0.0.1:8766", model: "qwen2.5-7b-instruct" });
    expect((await verifyToken(qb.token, "test-signing-key"))?.model).toBe("qwen2.5-7b-instruct");
    const b = await (await session({ model: "qwen2.5-7b-instruct" })).json() as { url: string; model: string };
    expect(b).toMatchObject({ url: "ws://127.0.0.1:8766", model: "qwen2.5-7b-instruct" });
    const none = await (await call("/v1/split/session", { method: "POST", headers: { Origin: ORIGIN } })).json() as { url: string; model: string };
    expect(none).toMatchObject({ url: "ws://127.0.0.1:8765", model: "qwen2.5-0.5b-instruct" });
    expect((await call("/v1/split/session?model=nope", { method: "POST", headers: { Origin: ORIGIN } })).status).toBe(400);
  });
  it("503 when the signing key or the server map is not configured, or the map is malformed", async () => {
    expect((await session({}, { Origin: ORIGIN }, { SPLIT_SIGNING_KEY: undefined })).status).toBe(503);
    expect((await session({}, { Origin: ORIGIN }, { SPLIT_SERVERS: undefined })).status).toBe(503);
    expect((await session({}, { Origin: ORIGIN }, { SPLIT_SERVERS: "not json" })).status).toBe(503);
    expect((await session({}, { Origin: ORIGIN }, { SPLIT_SERVERS: JSON.stringify({ m: { ws: "http://x", plan: "http://x/plan" } }) })).status).toBe(503);
    expect((await session({}, { Origin: ORIGIN }, { SPLIT_SERVERS: JSON.stringify({ "Bad Id": { ws: "ws://x", plan: "http://x/plan" } }) })).status).toBe(503);
    expect((await session({}, { Origin: ORIGIN }, { SPLIT_SERVERS: "{}" })).status).toBe(503);
  });
});

describe("split: servers", () => {
  it("lists the SPLIT_SERVERS map in order with the default first", async () => {
    const res = await call("/v1/split/servers");
    expect(res.status).toBe(200);
    expect(res.headers.get("Cache-Control")).toBe("public, max-age=60");
    expect(await res.json()).toEqual({
      default: "qwen2.5-0.5b-instruct",
      models: [
        { model: "qwen2.5-0.5b-instruct", ws: "ws://127.0.0.1:8765", plan: "http://127.0.0.1:8765/plan" },
        { model: "qwen2.5-7b-instruct", ws: "ws://127.0.0.1:8766", plan: "http://127.0.0.1:8766/plan" },
      ],
    });
    expect((await call("/v1/split/servers", {}, { SPLIT_SERVERS: undefined })).status).toBe(503);
    expect((await call("/v1/split/servers", { method: "POST" })).status).toBe(404);
  });
});

describe("split: rates and plan proxy", () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); resetPlanCache(); });

  it("serves rates.json fields and says whether a rate is available", async () => {
    const res = await call("/v1/split/rates");
    expect(res.status).toBe(200);
    expect(res.headers.get("Cache-Control")).toBe("public, max-age=300");
    const body = await res.json() as Record<string, unknown>;
    expect(Object.keys(body).sort()).toEqual(["gpu", "rate_available", "retrieved", "source", "usd_per_hour"]);
    expect(body.rate_available).toBe(typeof body.usd_per_hour === "number");
  });
  it("proxies the server's /plan per model and caches each for 5 s", async () => {
    vi.useFakeTimers({ now: 1_800_000_000_000, toFake: ["Date"] });
    const plan = { model: "qwen2.5-0.5b-instruct", L: 24, d_model: 896, active_sessions: 0, busy_fraction_60s: 0.01, ms_per_block_decode: 0.63, ms_per_block_prefill: 0.9, lm_head_ms: 3.0 };
    const plan7b = { ...plan, model: "qwen2.5-7b-instruct", L: 28, d_model: 3584 };
    const fetchMock = vi.fn(async (url: string) => new Response(JSON.stringify(url.startsWith("http://127.0.0.1:8766") ? plan7b : plan), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    const a = await call("/v1/split/plan");
    expect(a.status).toBe(200);
    expect(await a.json()).toEqual(plan);
    expect(a.headers.get("Cache-Control")).toBe("public, max-age=5");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect((fetchMock.mock.calls[0] as unknown[])[0]).toBe("http://127.0.0.1:8765/plan");
    vi.setSystemTime(1_800_000_000_000 + PLAN_CACHE_MS - 1);
    expect(await (await call("/v1/split/plan")).json()).toEqual(plan);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    // another model is another cache entry
    expect(await (await call("/v1/split/plan?model=qwen2.5-7b-instruct")).json()).toEqual(plan7b);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect((fetchMock.mock.calls[1] as unknown[])[0]).toBe("http://127.0.0.1:8766/plan");
    expect(await (await call("/v1/split/plan?model=qwen2.5-0.5b-instruct")).json()).toEqual(plan);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    vi.setSystemTime(1_800_000_000_000 + PLAN_CACHE_MS + 1);
    await call("/v1/split/plan");
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect((await call("/v1/split/plan?model=nope")).status).toBe(400);
    expect((await call("/v1/split/plan", {}, { SPLIT_SERVERS: undefined })).status).toBe(503);
  });
  it("503 with reachable: false when the server is down, and that is cached too", async () => {
    const fetchMock = vi.fn(async () => { throw new TypeError("connect ECONNREFUSED"); });
    vi.stubGlobal("fetch", fetchMock);
    const res = await call("/v1/split/plan");
    expect(res.status).toBe(503);
    expect(await res.json()).toMatchObject({ error: "server_unreachable", reachable: false, model: "qwen2.5-0.5b-instruct" });
    await call("/v1/split/plan");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("stats", () => {
  const SQL = /api\.cloudflare\.com\/client\/v4\/accounts\/acct\/analytics_engine\/sql$/;
  const over = { CF_ANALYTICS_TOKEN: "t", CF_ACCOUNT_ID: "acct" };
  /** Replaces the global fetch (what src/stats.ts calls) for the SQL API only. */
  function stubSql(handler: (sql: string, init: RequestInit) => Response | Promise<Response>) {
    const real = globalThis.fetch;
    vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (SQL.test(url)) return handler(String(init?.body), init ?? {});
      return real(input, init);
    });
  }
  afterEach(async () => { await env.STATS_CACHE.delete(STATS_CACHE_KEY); vi.unstubAllGlobals(); });

  it("degrades to the empty shape without the SQL token", async () => {
    const res = await call("/v1/stats/loads", {}, { CF_ANALYTICS_TOKEN: undefined, CF_ACCOUNT_ID: undefined });
    expect(res.status).toBe(200);
    const body = await res.json() as Record<string, unknown>;
    expect(body).toMatchObject({ window_hours: 168, by_country: [], by_model_variant: [], by_source: [], degraded: true });
    expect(body.since).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
    expect(body.computed_at).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
    expect(await env.STATS_CACHE.get(STATS_CACHE_KEY)).toBeNull();
  });

  it("does not cache an empty (no rows) result and recomputes on the next request", async () => {
    let calls = 0;
    stubSql(() => { calls++; return Response.json({ meta: [], data: [], rows: 0 }); });
    const res = await call("/v1/stats/loads", {}, over);
    expect(res.status).toBe(200);
    expect(res.headers.get("X-Dianome-Stats")).toBe("computed:empty-not-cached");
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    const body = await res.json() as Record<string, unknown>;
    expect(body).toMatchObject({ by_country: [], by_model_variant: [], by_source: [] });
    expect(body.degraded).toBeUndefined();
    expect(body.computed_at).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
    expect(await env.STATS_CACHE.get(STATS_CACHE_KEY)).toBeNull();
    const again = await call("/v1/stats/loads", {}, over);
    expect(again.headers.get("X-Dianome-Stats")).toBe("computed:empty-not-cached");
    expect(calls).toBe(6);
  });

  it("includes computed_at on both the fresh-compute and the KV-hit path", async () => {
    stubSql((sql) => Response.json({ meta: [], data: sql.includes("AS source") ? [{ source: "network", loads: 1, p50_ms: 15629 }] : [], rows: 1 }));
    const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;
    const fresh = await call("/v1/stats/loads", {}, over);
    expect(fresh.headers.get("X-Dianome-Stats")).toBe("computed");
    const freshBody = await fresh.json() as Record<string, unknown>;
    expect(freshBody.computed_at).toMatch(ISO);
    const hit = await call("/v1/stats/loads", {}, over);
    expect(hit.headers.get("X-Dianome-Stats")).toBe("kv-hit");
    const hitBody = await hit.json() as Record<string, unknown>;
    expect(hitBody.computed_at).toMatch(ISO);
    expect(hitBody.computed_at).toBe(freshBody.computed_at);
  });

  it("ignores a cached entry without computed_at (written by an older Worker) and recomputes", async () => {
    await env.STATS_CACHE.put(STATS_CACHE_KEY, JSON.stringify({ since: "2026-09-10T00:01:22Z", window_hours: 168, by_country: [{ country: "US", loads: 1, p50_ms: 1, p90_ms: 1, cache_hit_rate: 0 }], by_model_variant: [], by_source: [] }));
    stubSql((sql) => Response.json({ meta: [], data: sql.includes("AS source") ? [{ source: "network", loads: 1, p50_ms: 15629 }] : [], rows: 1 }));
    const res = await call("/v1/stats/loads", {}, over);
    expect(res.headers.get("X-Dianome-Stats")).toBe("computed");
    const body = await res.json() as Record<string, unknown>;
    expect(body.computed_at).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
    expect(body.by_source).toEqual([{ source: "network", loads: 1, p50_ms: 15629 }]);
    expect(JSON.parse((await env.STATS_CACHE.get(STATS_CACHE_KEY))!).computed_at).toBe(body.computed_at);
  });

  it("degrades when the SQL API answers 200 with a non-JSON or shapeless body", async () => {
    stubSql(() => new Response("<html>login</html>", { status: 200 }));
    const res = await call("/v1/stats/loads", {}, over);
    expect(res.headers.get("X-Dianome-Stats")).toBe("degraded:sql-error");
    expect(await env.STATS_CACHE.get(STATS_CACHE_KEY)).toBeNull();
  });

  it("degrades (not 500) when the SQL API errors", async () => {
    stubSql(() => new Response("no", { status: 401 }));
    const res = await call("/v1/stats/loads", {}, over);
    expect(res.status).toBe(200);
    expect(res.headers.get("X-Dianome-Stats")).toBe("degraded:sql-error");
    expect(await res.json()).toMatchObject({ degraded: true, by_country: [] });
    expect(await env.STATS_CACHE.get(STATS_CACHE_KEY)).toBeNull();
  });

  it("maps SQL rows into the stats shape and caches in KV for 5 minutes", async () => {
    const reply = (rows: unknown[]) => Response.json({ meta: [], data: rows, rows: rows.length });
    const seen: string[] = [];
    stubSql((sql, init) => {
      seen.push(sql);
      expect(new Headers(init.headers).get("Authorization")).toBe("Bearer t");
      if (sql.includes("AS country")) return reply([{ country: "US", loads: "12", p50_ms: 8100.4, p90_ms: 15200, cache_hit_rate: 0.41 }]);
      if (sql.includes("AS variant")) return reply([{ model: "qwen2.5-0.5b-instruct", variant: "q4", loads: 9, p50_ms: 7900, bytes: "323893760" }]);
      return reply([{ source: "cross-site-cache", loads: 3, p50_ms: 140 }]);
    });
    const res = await call("/v1/stats/loads", {}, over);
    expect(res.status).toBe(200);
    expect(res.headers.get("X-Dianome-Stats")).toBe("computed");
    expect(res.headers.get("Cache-Control")).toBe("public, max-age=300");
    expect(seen).toHaveLength(3);
    expect(seen.every((q) => q.includes("FROM dianome_loads") && q.includes("INTERVAL '7' DAY") && q.endsWith("FORMAT JSON"))).toBe(true);
    const body = await res.json() as Record<string, unknown>;
    expect(body.computed_at).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
    expect(body).toMatchObject({
      window_hours: 168,
      by_country: [{ country: "US", loads: 12, p50_ms: 8100, p90_ms: 15200, cache_hit_rate: 0.41 }],
      by_model_variant: [{ model: "qwen2.5-0.5b-instruct", variant: "q4", loads: 9, p50_ms: 7900, bytes: 323893760 }],
      by_source: [{ source: "cross-site-cache", loads: 3, p50_ms: 140 }],
    });
    const again = await call("/v1/stats/loads", {}, over);
    expect(again.headers.get("X-Dianome-Stats")).toBe("kv-hit");
    expect(seen).toHaveLength(3);
    expect(((await again.json()) as Record<string, unknown>).computed_at).toBe(body.computed_at);
    expect(await env.STATS_CACHE.get(STATS_CACHE_KEY)).not.toBeNull();
  });
});

describe("stats: sessions", () => {
  const SQL = /api\.cloudflare\.com\/client\/v4\/accounts\/acct\/analytics_engine\/sql$/;
  const over = { CF_ANALYTICS_TOKEN: "t", CF_ACCOUNT_ID: "acct" };
  function stubSql(handler: (sql: string) => Response) {
    const real = globalThis.fetch;
    vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (SQL.test(url)) return handler(String(init?.body));
      return real(input, init);
    });
  }
  afterEach(async () => { await env.STATS_CACHE.delete(SESSIONS_CACHE_KEY); vi.unstubAllGlobals(); });

  it("degrades to the empty shape without the SQL token", async () => {
    const res = await call("/v1/stats/sessions", {}, { CF_ANALYTICS_TOKEN: undefined, CF_ACCOUNT_ID: undefined });
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({ window_hours: 168, by_model_mode: [], by_policy: [], by_browser: [], degraded: true });
  });
  it("maps the dianome_sessions rows, never touches the loads dataset, and caches in KV", async () => {
    const reply = (rows: unknown[]) => Response.json({ meta: [], data: rows, rows: rows.length });
    const seen: string[] = [];
    stubSql((sql) => {
      seen.push(sql);
      expect(sql).toContain("FROM dianome_sessions");
      if (sql.includes("AS model")) return reply([{ model: "qwen2.5-0.5b-instruct", mode: "split", sessions: "7", p50_tok_per_s: 41.66, p50_n: 12, L: 24, p50_server_busy_ms: 710.2, p50_rtt_ms: 3.84, new_tokens: 448 }]);
      if (sql.includes("AS policy")) return reply([{ policy: "cost", mode: "local", sessions: 5 }]);
      return reply([{ browser: "chrome", sessions: 7, p50_tok_per_s: 44.1, webgpu_share: 1 }]);
    });
    const res = await call("/v1/stats/sessions", {}, over);
    expect(res.status).toBe(200);
    expect(res.headers.get("X-Dianome-Stats")).toBe("computed");
    const body = await res.json() as Record<string, unknown>;
    expect(body.by_model_mode).toEqual([{ model: "qwen2.5-0.5b-instruct", mode: "split", sessions: 7, p50_tok_per_s: 41.7, p50_n: 12, L: 24, p50_server_busy_ms: 710, p50_rtt_ms: 3.8, new_tokens: 448 }]);
    expect(body.by_policy).toEqual([{ policy: "cost", mode: "local", sessions: 5 }]);
    expect(body.by_browser).toEqual([{ browser: "chrome", sessions: 7, p50_tok_per_s: 44.1, webgpu_share: 1 }]);
    expect(seen.length).toBe(3);
    expect(seen.every((q) => !q.includes("dianome_loads"))).toBe(true);
    const hit = await call("/v1/stats/sessions", {}, over);
    expect(hit.headers.get("X-Dianome-Stats")).toBe("kv-hit");
    expect(seen.length).toBe(3);
  });
  it("degrades (not 500) when the SQL API errors and does not cache", async () => {
    stubSql(() => new Response("no", { status: 401 }));
    const res = await call("/v1/stats/sessions", {}, over);
    expect(res.status).toBe(200);
    expect(res.headers.get("X-Dianome-Stats")).toBe("degraded:sql-error");
    expect(await env.STATS_CACHE.get(SESSIONS_CACHE_KEY)).toBeNull();
  });
});

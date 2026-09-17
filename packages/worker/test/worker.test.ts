import { createExecutionContext, env, waitOnExecutionContext } from "cloudflare:test";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import worker from "../src/index";
import { STATS_CACHE_KEY } from "../src/stats";
import { dataPoint } from "../src/telemetry";
import { MODEL_ID, seedManifest, validReport, validReportV2 } from "./fixtures";

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
  it("maps schema 1 and 2 onto the Analytics Engine columns (absent doubles are -1, cache_mode is blob8)", () => {
    const p1 = dataPoint(validReport() as never, "US", "ATL", "2026-09-16T23:00:00.000Z");
    expect(p1.blobs).toEqual(["qwen2.5-0.5b-instruct", "q4", "network", "chrome", "US", "ATL", "2026-09-16T23:00:00.000Z", ""]);
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

import { createExecutionContext, env, waitOnExecutionContext } from "cloudflare:test";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import worker from "../src/index";
import { STATS_CACHE_KEY } from "../src/stats";
import { MODEL_ID, seedManifest, validReport } from "./fixtures";

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
    expect((await post({ ...validReport(), schema: 2 })).status).toBe(400);
    expect((await post({ ...validReport(), cache_hits: 43 })).status).toBe(400);
    expect((await post("not json")).status).toBe(400);
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
    expect(seen.every((q) => q.includes("FROM dianome_loads") && q.includes("INTERVAL '168' HOUR") && q.endsWith("FORMAT JSON"))).toBe(true);
    expect(await res.json()).toMatchObject({
      window_hours: 168,
      by_country: [{ country: "US", loads: 12, p50_ms: 8100, p90_ms: 15200, cache_hit_rate: 0.41 }],
      by_model_variant: [{ model: "qwen2.5-0.5b-instruct", variant: "q4", loads: 9, p50_ms: 7900, bytes: 323893760 }],
      by_source: [{ source: "cross-site-cache", loads: 3, p50_ms: 140 }],
    });
    const again = await call("/v1/stats/loads", {}, over);
    expect(again.headers.get("X-Dianome-Stats")).toBe("kv-hit");
    expect(seen).toHaveLength(3);
    expect(await env.STATS_CACHE.get(STATS_CACHE_KEY)).not.toBeNull();
  });
});

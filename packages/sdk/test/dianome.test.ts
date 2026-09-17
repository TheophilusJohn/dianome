import { describe, expect, it, vi } from "vitest";
import { PerSiteStore } from "../src/cache/persite";
import { AbortedError, ChunkError } from "../src/errors";
import { Dianome, type LoadedGroup, type Progress } from "../src/index";
import { synthFiles, synthModel } from "./fixtures/synth";
import { makeStore } from "./helpers/env";
import { FakeFetch } from "./helpers/fakeFetch";

const API = "https://api.test", CDN = "https://cdn.test";

async function setup(o: { chunkSize?: number; store?: PerSiteStore | null; telemetry?: boolean; cache?: "auto" | "per-site" | "none" } = {}) {
  const model = await synthModel({ chunkSize: o.chunkSize ?? 4096 });
  const ff = new FakeFetch(API, CDN, model.chunks, new Map([[model.manifest.id, model.manifest]]), "cd".repeat(32));
  const d = new Dianome({ api: API, cdn: CDN, fetch: ff.fetch, telemetry: o.telemetry ?? true, cache: o.cache ?? "per-site", retries: 0 });
  // Inject the fake store where the class would open a real one.
  const store = o.store === undefined ? makeStore({ cdn: CDN }).store : o.store;
  if (o.cache !== "none") (d as unknown as { storePromise: Promise<PerSiteStore | null> }).storePromise = Promise.resolve(store);
  return { model, ff, d, store };
}

describe("Dianome.stream", () => {
  it("yields groups in manifest order with entries, then returns the summary and posts a v2 report", async () => {
    const { model, ff, d } = await setup();
    const progress: Progress[] = [];
    const gen = d.stream("synth-model", { variant: "q4", onProgress: (p) => progress.push(p) });
    const names: string[] = [];
    let summary;
    for (;;) {
      const n = await gen.next();
      if (n.done) { summary = n.value; break; }
      names.push(n.value.name);
      expect(n.value.entries.size).toBeGreaterThan(0);
    }
    const q4 = model.manifest.variants.q4!;
    expect(names).toEqual(q4.groups.map((g) => g.name));
    expect(summary.chunks).toBe(new Set(q4.groups.flatMap((g) => g.chunks)).size);
    expect(summary.bytes).toBe(q4.bytes);
    expect(summary.source).toBe("network");
    expect(summary.cacheMode).toBe("per-site");
    expect(summary.telemetryStatus).toBe(202);
    expect(ff.telemetry).toHaveLength(1);
    expect(ff.telemetry[0]).toMatchObject({ schema: 2, model: "synth-model", variant: "q4", source: "network", cache_hits: 0, chunks: summary.chunks, bytes: q4.bytes, cache_mode: "per-site" });
    expect(progress).toHaveLength(summary.chunks);
    expect(progress.at(-1)).toMatchObject({ bytesDone: q4.bytes, bytesTotal: q4.bytes, chunksDone: summary.chunks, chunksTotal: summary.chunks, source: "network" });
    expect(d.lastSummary).toBe(summary);
  });

  it("second load comes entirely from the per-site cache with zero chunk requests", async () => {
    const { ff, d, store } = await setup();
    const first = await d.load("synth-model", { variant: "q4" });
    expect(first.summary.source).toBe("network");
    const ff2 = new FakeFetch(API, CDN, ff.chunks, ff.manifests);
    const d2 = new Dianome({ api: API, cdn: CDN, fetch: ff2.fetch, cache: "per-site" });
    (d2 as unknown as { storePromise: unknown }).storePromise = Promise.resolve(store);
    const second = await d2.load("synth-model", { variant: "q4" });
    expect(ff2.chunkRequests()).toBe(0);
    expect(second.summary.source).toBe("per-site-cache");
    expect(second.summary.cacheHits).toBe(second.summary.chunks);
    expect(ff2.telemetry[0]).toMatchObject({ source: "per-site-cache", cache_hits: second.summary.chunks });
    expect(second.entry("model.layers.1.self_attn.q_proj.weight").bytes).toEqual(first.entry("model.layers.1.self_attn.q_proj.weight").bytes);
  });

  it("a group that completes early waits for earlier groups; tied lm_head arrives last from embed's chunks", async () => {
    const { model, ff, d } = await setup({ chunkSize: 4096 });
    // Make embed's first chunk the slowest so layer groups finish before embed.
    const embedSha = model.manifest.variants.q4!.groups[0]!.chunks[0]!;
    const orig = ff.fetch;
    const slow = async (url: string, init?: RequestInit) => {
      if (url.endsWith(embedSha)) await new Promise((r) => setTimeout(r, 30));
      return orig(url, init);
    };
    const d2 = new Dianome({ api: API, cdn: CDN, fetch: slow, cache: "none", telemetry: false });
    const seen: string[] = [];
    const arrivals: string[] = [];
    for await (const g of d2.stream("synth-model", { variant: "q4", onProgress: (p) => arrivals.push(p.group) })) seen.push(g.name);
    expect(seen).toEqual(["embed", "layer.0", "layer.1", "final_norm", "lm_head"]);
    expect(arrivals.lastIndexOf("embed")).toBeGreaterThan(arrivals.indexOf("layer.0")); // a layer.0 chunk arrived before embed completed
    void d;
  });

  it("load() exposes groups and entries; entry lookup errors are DianomeErrors", async () => {
    const { d, model } = await setup({ telemetry: false });
    const m = await d.load("synth-model", { variant: "fp16" });
    expect(m.variant).toBe("fp16");
    expect(m.manifest).toEqual(model.manifest);
    expect(m.group("layer.1").entries.size).toBe(3);
    expect(m.entry("lm_head.weight").tied).toBe(true);
    expect(m.entry("lm_head.weight").bytes).toEqual(model.raw.get("fp16/lm_head.weight"));
    expect(() => m.group("layer.9")).toThrow(/no group/);
    expect(() => m.entry("nope")).toThrow(/no entry/);
    expect(m.summary.report).toBeNull();
    expect(m.summary.telemetryStatus).toBeNull();
  });

  it("defaults to the smallest variant present", async () => {
    const { d } = await setup({ telemetry: false });
    expect((await d.load("synth-model")).variant).toBe("q4");
  });

  it("loads files manifests one group per file, honouring `files` order", async () => {
    const files = await synthFiles([{ name: "a.bin", length: 10_000, seed: 1 }, { name: "dir/b.bin", length: 5, seed: 2 }, { name: "c.bin", length: 7, seed: 3 }], { chunkSize: 4096, runtime: "webllm" });
    const ff = new FakeFetch(API, CDN, files.chunks, new Map([[files.manifest.id, files.manifest]]));
    const d = new Dianome({ api: API, cdn: CDN, fetch: ff.fetch, cache: "none" });
    const got: LoadedGroup[] = [];
    for await (const g of d.stream("synth-files", { files: ["c.bin", "a.bin"] })) got.push(g);
    expect(got.map((g) => g.name)).toEqual(["c.bin", "a.bin"]);
    expect(got[1]!.entries.get("a.bin")!.bytes).toEqual(files.raw.get("a.bin"));
    expect(got[1]!.entries.get("a.bin")!.storage).toEqual({ kind: "raw" });
    expect(ff.telemetry[0]).toMatchObject({ model: "synth-files", variant: "files", chunks: 4 });
    await expect(d.load("synth-files", { files: ["zzz"] })).rejects.toThrow(/not in manifest/);
  });

  it("breaking out of the stream aborts the fetch; an aborted signal rejects with AbortedError and sends no telemetry", async () => {
    const { ff, d } = await setup();
    ff.delayMs = 2;
    for await (const g of d.stream("synth-model", { variant: "q4" })) { if (g.name === "embed") break; }
    await new Promise((r) => setTimeout(r, 20));
    const after = ff.chunkRequests();
    await new Promise((r) => setTimeout(r, 20));
    expect(ff.chunkRequests()).toBe(after);
    expect(ff.telemetry).toHaveLength(0);

    const ac = new AbortController();
    const p = d.load("synth-model", { variant: "q4", signal: ac.signal });
    setTimeout(() => ac.abort(), 3);
    await expect(p).rejects.toBeInstanceOf(AbortedError);
    expect(ff.telemetry).toHaveLength(0);
  });

  it("propagates chunk errors and sends no telemetry", async () => {
    const { model, ff, d } = await setup();
    ff.fail(model.manifest.variants.q4!.groups[1]!.chunks[0]!, { kind: "status", status: 404 });
    await expect(d.load("synth-model", { variant: "q4" })).rejects.toBeInstanceOf(ChunkError);
    expect(ff.telemetry).toHaveLength(0);
  });

  it("quota exhaustion switches the session to cache none, the load still completes, and cache_mode reports none", async () => {
    const model = await synthModel({ chunkSize: 4096 });
    const ff = new FakeFetch(API, CDN, model.chunks, new Map([[model.manifest.id, model.manifest]]));
    const { store } = makeStore({ cdn: CDN, capacity: 6000 });
    const d = new Dianome({ api: API, cdn: CDN, fetch: ff.fetch, cache: "per-site" });
    (d as unknown as { storePromise: unknown }).storePromise = Promise.resolve(store);
    const m = await d.load("synth-model", { variant: "q4" });
    expect(m.summary.cacheDisabled).toBe(true);
    expect(m.summary.cacheMode).toBe("none");
    expect(ff.telemetry[0]).toMatchObject({ cache_mode: "none", source: "network" });
    expect(await d.cache.status()).toEqual({ mode: "none", quotaBytes: null, usageBytes: 0, chunks: 0 });
    // The next load in the same session does not touch the cache at all.
    const ff2 = new FakeFetch(API, CDN, model.chunks, ff.manifests);
    (d as unknown as { fetchImpl: unknown }).fetchImpl = ff2.fetch;
    await d.load("synth-model", { variant: "q4" });
    expect(ff2.chunkRequests()).toBe(new Set(model.manifest.variants.q4!.groups.flatMap((g) => g.chunks)).size);
  });

  it("cache controls: status/evict/clear go to the store; mode none reports empty", async () => {
    const { d, store } = await setup({ telemetry: false });
    await d.load("synth-model", { variant: "q4" });
    const st = await d.cache.status();
    expect(st.mode).toBe("per-site");
    expect(st.chunks).toBeGreaterThan(0);
    await d.cache.evict("synth-model");
    expect((await store!.status()).chunks).toBe(0);
    await d.load("synth-model", { variant: "q4" });
    await d.cache.clear();
    expect((await d.cache.status()).chunks).toBe(0);

    const none = new Dianome({ api: API, cdn: CDN, cache: "none" });
    expect(await none.cache.status()).toEqual({ mode: "none", quotaBytes: null, usageBytes: 0, chunks: 0 });
    expect(await none.cache.mode()).toBe("none");
  });

  it("telemetry: false sends nothing; a failed POST does not fail the load", async () => {
    const { ff, d } = await setup({ telemetry: false });
    await d.load("synth-model", { variant: "q4" });
    expect(ff.telemetry).toHaveLength(0);
    const { ff: ff2, d: d2 } = await setup();
    ff2.telemetryStatus = 500;
    const m = await d2.load("synth-model", { variant: "q4" });
    expect(m.summary.telemetryStatus).toBe(500);
    expect(ff2.telemetry).toHaveLength(1);
  });

  it("uses default endpoints and strips trailing slashes", () => {
    const d = new Dianome();
    expect(d.api).toBe("https://api.dianome.dev");
    expect(d.cdn).toBe("https://cdn.dianome.dev");
    expect(new Dianome({ api: "http://localhost:8787/", cdn: "http://localhost:8788//" })).toMatchObject({ api: "http://localhost:8787", cdn: "http://localhost:8788" });
    vi.restoreAllMocks();
  });
});
